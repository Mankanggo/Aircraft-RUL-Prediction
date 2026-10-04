"""
End-to-end inference path of the frozen final models (training data only - these
tests never read the official test files):

raw text file -> read_raw_file -> validate_input -> FoldPreprocessor.transform
-> last observed row -> model -> RUL >= 0
"""

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.components.validation import last_observed_rows, make_cv_plan, make_truncated_samples
from src.data_schema import RAW_COLUMNS, SUBSETS
from src.pipelines.prediction_pipeline import InputValidationError, RULPredictor, validate_input
from src.pipelines.training_pipeline import MODEL_DIR, make_final_preprocessor, verify_config_frozen

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "manifest.json")),
                                reason="final models not trained (python -m src.pipelines.training_pipeline --step train)")


@pytest.fixture(scope="module")
def predictors():
    return {s: RULPredictor.load(s) for s in SUBSETS}


def test_frozen_config_and_artifacts_consistent():
    cfg = verify_config_frozen()
    manifest = json.load(open(os.path.join(MODEL_DIR, "manifest.json"), encoding="utf-8"))
    assert manifest["config_body_sha256"] == cfg["frozen"]["config_body_sha256"]
    for s in SUBSETS:
        meta = manifest["subsets"][s]
        assert meta["family"] == cfg["models"][s]["family"] and meta["rul_cap"] == cfg["models"][s]["rul_cap"]
        assert meta["params"] == cfg["models"][s]["params"]
        assert meta["n_features"] == cfg["feature_set"]["n_features"]


@pytest.mark.parametrize("subset", SUBSETS)
def test_raw_text_file_to_prediction(cmapss, predictors, subset, tmp_path):
    """A raw whitespace file and the equivalent DataFrame give identical predictions."""
    train = cmapss[subset]["train"]
    cuts = make_cv_plan(train, n_cuts=1)[1].head(30)
    samples = make_truncated_samples(train, cuts)
    # one 'engine' per truncated sample, renumbered like a test file
    samples["unit_id"] = samples["sample_id"] + 1
    path = tmp_path / "engines.txt"
    np.savetxt(path, samples[RAW_COLUMNS].to_numpy(float), fmt="%.6g")
    from_file = predictors[subset].predict_file(str(path))
    from_df = predictors[subset].predict_engines(samples[RAW_COLUMNS])
    pd.testing.assert_frame_equal(from_file, from_df, rtol=1e-5)
    assert len(from_df) == len(cuts) and (from_df.predicted_rul >= 0).all()
    assert (from_df.last_cycle.to_numpy() == cuts.cut_cycle.to_numpy()).all()


@pytest.mark.parametrize("subset", ["FD001", "FD004"])
def test_prediction_uses_only_history(cmapss, predictors, subset):
    """Prediction for a truncated engine == prediction from full-trajectory features at that cycle."""
    p = predictors[subset]
    train = cmapss[subset]["train"]
    engine = train[train.unit_id == train.unit_id.iloc[0]][RAW_COLUMNS]
    full = p.features(engine)
    for cut in (1, 15, 60, engine.cycle.max() - 10):
        pred = p.predict_engines(engine[engine.cycle <= cut]).predicted_rul.item()
        row = full[full.cycle == cut][p.preprocessor.feature_names_].to_numpy(np.float32)
        assert np.isclose(pred, max(float(p.model.predict(row)[0]), 0.0), rtol=1e-5)


def test_loaded_artifacts_match_freshly_trained_model(cmapss, predictors):
    """Artifact round-trip: reloaded FD001 model == the same frozen config retrained in this process."""
    import src.components.model_trainer as mt
    from src.components.model_trainer import make_model
    from src.utils import cap_rul

    cfg = verify_config_frozen()
    m = cfg["models"]["FD001"]
    train = cmapss["FD001"]["train"].drop(columns="RUL")
    pre = make_final_preprocessor()
    X = pre.fit_transform(train)
    rul = (train.groupby("unit_id").cycle.transform("max") - train.cycle).reindex(X.index)
    assert mt.N_JOBS == cfg["training_protocol"]["n_jobs"]
    model = make_model(m["family"], m["params"]).fit(X[pre.feature_names_].to_numpy(np.float32), cap_rul(rul, m["rul_cap"]).to_numpy(float))
    cuts = make_cv_plan(train, n_cuts=2)[1]
    samples = make_truncated_samples(train, cuts)
    Xs = last_observed_rows(pre.transform(samples, group_cols=("sample_id",)))[pre.feature_names_].to_numpy(np.float32)
    fresh = np.maximum(model.predict(Xs), 0)
    samples["unit_id"] = samples["sample_id"] + 1
    loaded = predictors["FD001"].predict_engines(samples[RAW_COLUMNS]).predicted_rul.to_numpy()
    np.testing.assert_allclose(loaded, fresh, rtol=1e-6)


def test_tampered_artifact_is_rejected(tmp_path):
    shutil.copytree(MODEL_DIR, tmp_path / "final")
    with open(tmp_path / "final" / "FD001" / "model.joblib", "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(RuntimeError, match="hash"):
        RULPredictor.load("FD001", model_dir=str(tmp_path / "final"))


def test_input_validation(cmapss, predictors):
    df = cmapss["FD001"]["train"][RAW_COLUMNS].head(400)
    with pytest.raises(InputValidationError, match="missing columns"):
        validate_input(df.drop(columns="T50"))
    bad = df.copy()
    bad.loc[bad.index[5], "Ps30"] = np.nan
    with pytest.raises(InputValidationError, match="non-numeric"):
        validate_input(bad)
    with pytest.raises(InputValidationError, match="duplicate"):
        validate_input(pd.concat([df, df.head(1)]))
    with pytest.raises(InputValidationError, match="non-contiguous"):
        validate_input(df.drop(index=df.index[3]))
    with pytest.raises(ValueError, match="operating condition"):
        validate_input(df.assign(op_setting_1=33.0))
    with pytest.raises(ValueError, match="not present"):
        predictors["FD001"].predict_engines(cmapss["FD002"]["train"][RAW_COLUMNS].head(300))  # FD001 model: sea level only
