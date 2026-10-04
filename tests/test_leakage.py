"""
Leakage checks:
  * features at cycle t never depend on cycles > t (prefix invariance) or on other engines;
  * preprocessing statistics come from training engines only and are never refitted;
  * the CV loop never lets a validation engine into fitting.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.components.data_transformation import ConditionNormalizer, FoldPreprocessor, add_causal_features
from src.components.validation import cross_validate, iter_folds, make_cv_plan, make_engine_folds
from src.data_schema import CANDIDATE_SENSORS, SENSOR_COLUMNS
from src.utils import fit_condition_stats

SENSORS = ["T24", "T50", "Ps30", "phi", "epr", "htBleed"]


def _engines(df, n, seed=0):
    units = np.random.default_rng(seed).choice(df["unit_id"].unique(), size=n, replace=False)
    return df[df["unit_id"].isin(units)].reset_index(drop=True)


def _features_close(a, b, cols):
    np.testing.assert_allclose(a[cols].to_numpy(), b[cols].to_numpy(), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("subset", ["FD001", "FD004"])
def test_features_are_prefix_invariant(cmapss, subset):
    """Features computed on the full trajectory == features computed on a truncated prefix."""
    df = _engines(cmapss[subset]["train"], 12, seed=1)
    pre = FoldPreprocessor(sensors=SENSORS).fit(cmapss[subset]["train"])
    full = pre.transform(df)
    rng = np.random.default_rng(0)
    for unit in df["unit_id"].unique():
        life = df.loc[df.unit_id == unit, "cycle"].max()
        for cut in rng.integers(1, life, size=3):
            prefix = df[(df.unit_id == unit) & (df.cycle <= cut)]
            on_prefix = pre.transform(prefix)
            on_full = full[(full.unit_id == unit) & (full.cycle <= cut)]
            _features_close(on_prefix.sort_values("cycle"), on_full.sort_values("cycle"), pre.feature_names_)


def test_future_rows_cannot_change_past_features(cmapss):
    df = _engines(cmapss["FD002"]["train"], 5, seed=2)
    pre = FoldPreprocessor(sensors=SENSORS).fit(cmapss["FD002"]["train"])
    base = pre.transform(df)
    tampered = df.copy()
    tampered[SENSOR_COLUMNS] = tampered[SENSOR_COLUMNS].astype(float)
    future = tampered["cycle"] > 50
    tampered.loc[future, SENSOR_COLUMNS] *= np.random.default_rng(0).uniform(0.5, 2.0, size=(future.sum(), len(SENSOR_COLUMNS)))
    after = pre.transform(tampered)
    past = df["cycle"] <= 50
    _features_close(base[past], after[past], pre.feature_names_)
    assert not np.allclose(base.loc[future, pre.feature_names_], after.loc[future, pre.feature_names_])


def test_features_do_not_mix_engines(cmapss):
    df = _engines(cmapss["FD003"]["train"], 6, seed=3)
    pre = FoldPreprocessor(sensors=SENSORS).fit(cmapss["FD003"]["train"])
    base = pre.transform(df)
    victim = df["unit_id"].unique()[0]
    tampered = df.copy()
    tampered.loc[tampered.unit_id == victim, SENSORS] += 50.0
    after = pre.transform(tampered)
    others = df["unit_id"] != victim
    _features_close(base[others], after[others], pre.feature_names_)


def test_row_order_does_not_matter(cmapss):
    df = _engines(cmapss["FD001"]["train"], 5, seed=4)
    pre = FoldPreprocessor(sensors=SENSORS).fit(cmapss["FD001"]["train"])
    a = pre.transform(df)
    b = pre.transform(df.sample(frac=1.0, random_state=0)).loc[a.index]
    _features_close(a, b, pre.feature_names_)


def test_preprocessor_uses_training_statistics_only(cmapss):
    train = cmapss["FD004"]["train"]
    folds = make_engine_folds(train)
    _, tr, va = next(iter_folds(train, folds))
    pre = FoldPreprocessor().fit(tr)

    # condition statistics equal those of the training-fold rows, not of all engines
    expected = fit_condition_stats(tr, CANDIDATE_SENSORS)
    pd.testing.assert_frame_equal(pre.normalizer_.stats_, expected)
    assert not np.allclose(fit_condition_stats(train, CANDIDATE_SENSORS).to_numpy(), expected.to_numpy())
    assert pre.fit_units_.isdisjoint(set(va["unit_id"]))

    # transform never refits: state is identical after transforming (even absurd) validation data
    snapshot = (pre.normalizer_.stats_.copy(), pre.mean_.copy(), pre.std_.copy())
    weird = va.copy()
    weird[SENSOR_COLUMNS] *= 3.0
    pre.transform(weird)
    pd.testing.assert_frame_equal(pre.normalizer_.stats_, snapshot[0])
    pd.testing.assert_series_equal(pre.mean_, snapshot[1])
    pd.testing.assert_series_equal(pre.std_, snapshot[2])


def test_normalizer_rejects_unseen_operating_condition(cmapss):
    norm = ConditionNormalizer().fit(cmapss["FD001"]["train"])  # sea level only
    with pytest.raises(ValueError, match="not present"):
        norm.transform(cmapss["FD002"]["train"].head(500))


def test_unfitted_preprocessor_refuses_to_transform(cmapss):
    with pytest.raises(RuntimeError):
        FoldPreprocessor().transform(cmapss["FD001"]["train"].head(50))


def test_no_target_or_identifier_in_features(cmapss):
    pre = FoldPreprocessor(include_cycle=True).fit(cmapss["FD001"]["train"])
    banned = {"RUL", "unit_id", "sample_id", "lifespan", "cut_cycle", "rul_at_cut", "op_condition"}
    assert banned.isdisjoint(pre.feature_names_)
    assert all(not f.startswith(("op_setting", "RUL")) for f in pre.feature_names_)


class _SpyPreprocessor(FoldPreprocessor):
    """Records exactly which rows reach fit() and transform()."""

    log = []

    def fit(self, train_df, group_cols=("unit_id",)):
        _SpyPreprocessor.log.append(("fit", train_df[["unit_id", "cycle"]].copy(), None))
        return super().fit(train_df, group_cols)

    def transform(self, df, group_cols=("unit_id",)):
        cols = ["unit_id", "cycle"] + (["sample_id"] if "sample_id" in df else [])
        _SpyPreprocessor.log.append(("transform", df[cols].copy(), tuple(group_cols)))
        return super().transform(df, group_cols)


def test_cross_validate_never_fits_on_validation_engines(cmapss):
    train = _engines(cmapss["FD001"]["train"], 30, seed=5)
    ref = cmapss["FD001"]["test"].groupby("unit_id")["cycle"].max().to_numpy()
    folds, cuts = make_cv_plan(train, ref, n_splits=3, n_cuts=3)
    _SpyPreprocessor.log = []
    preds = cross_validate(train, folds, cuts, lambda: _SpyPreprocessor(sensors=SENSORS), lambda: Ridge(), caps=[None, 125])

    fold_of = folds.set_index("unit_id")["fold"]
    fits = [rows for kind, rows, _ in _SpyPreprocessor.log if kind == "fit"]
    assert len(fits) == 3
    for k, rows in enumerate(fits):
        assert (rows["unit_id"].map(fold_of) != k).all(), "validation engine reached fit()"

    # validation transforms only ever see rows up to each sample's cut
    cut_of = cuts.set_index("sample_id")["cut_cycle"]
    for kind, rows, groups in _SpyPreprocessor.log:
        if kind == "transform" and groups == ("sample_id",):
            assert (rows["cycle"] <= rows["sample_id"].map(cut_of)).all()

    # exactly one prediction per (cap, validation sample), scored against the uncapped truth
    assert len(preds) == 2 * len(cuts)
    merged = preds.merge(cuts[["sample_id", "rul_at_cut"]], on="sample_id")
    assert (merged["y_true"] == merged["rul_at_cut"]).all()
