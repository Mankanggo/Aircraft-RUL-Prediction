"""
Final (frozen) classical-ML training pipeline.

  freeze     write configs/final_config.json (+ feature list) and FINAL_CONFIG.md from the
             selections made in the classical experiments (training data only)
  verify_cv  re-run the frozen configuration through the fixed CV plan and check it
             reproduces the validated CV scores exactly (no test data involved)
  train      fit one preprocessor + model per subset on ALL training engines and save
             them under models/final/<subset>/ with SHA-256 hashes

The official test set is not read by this module.

Usage: python -m src.pipelines.training_pipeline --step freeze|verify_cv|train
"""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import time
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.components.data_transformation import FeatureConfig, FoldPreprocessor
import src.components.model_trainer as model_trainer
from src.components.model_trainer import make_model
from src.components.validation import (
    DEFAULT_MAX_RUL, DEFAULT_MIN_RUL, DEFAULT_N_CUTS, DEFAULT_N_SPLITS, DEFAULT_SEED, DEFAULT_STRATEGY,
    WIDE_MAX_RUL, cross_validate, make_cv_plan, summarise_cv,
)
from src.data_schema import CANDIDATE_SENSORS, SUBSETS
from src.utils import add_train_rul, cap_rul, read_cmapss_file

CONFIG_DIR = "configs"
CONFIG_PATH = os.path.join(CONFIG_DIR, "final_config.json")
FEATURE_LIST_PATH = os.path.join(CONFIG_DIR, "final_feature_list.json")
MODEL_DIR = os.path.join("models", "final")
EXP_DIR = os.path.join("reports", "experiments", "classical")
RAW_DIR = os.path.join("data", "raw")
CONFIG_VERSION = "classical-v1.0"
SELECTED_FEATURE_SET = "R2_F7 +cycle (minimal)"

# The selected feature set R2, built directly (not as a column selection from the
# experiment superset). verify_cv proves both routes give identical CV results.
FINAL_FEATURE_CONFIG = FeatureConfig(
    groups=("current", "mean", "std", "minmax", "slope", "delta", "baseline"),
    mean_windows=(5, 10, 20, 30), std_windows=(10, 30), slope_windows=(10, 20, 30), minmax_windows=(10, 30),
    delta_lags=(10,), momentum_long=30, smooth_window=5, baseline_cycles=10, sensors=tuple(CANDIDATE_SENSORS),
    derived=False, normalize=True, health_index=False, include_cycle=True,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_final_preprocessor() -> FoldPreprocessor:
    return FoldPreprocessor(config=FINAL_FEATURE_CONFIG)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def verify_config_frozen() -> dict:
    """Raise if configs/final_config.json or the feature list changed after freezing.
    Also applies the frozen thread count (needed for bit-exact XGBoost reproduction)."""
    cfg = load_config()
    frozen = cfg["frozen"]
    if sha256_file(FEATURE_LIST_PATH) != frozen["feature_list_sha256"]:
        raise RuntimeError("feature list changed after freezing")
    body = {k: v for k, v in cfg.items() if k != "frozen"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != frozen["config_body_sha256"]:
        raise RuntimeError("final_config.json changed after freezing")
    model_trainer.N_JOBS = cfg["training_protocol"]["n_jobs"]
    return cfg


# ------------------------------------------------------------------------------------------
def step_freeze():
    if os.path.exists(CONFIG_PATH):
        raise RuntimeError(f"{CONFIG_PATH} already exists - the configuration is frozen; delete it deliberately to re-freeze")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    models, cv = {}, {}
    finalize = pd.concat([pd.read_csv(os.path.join(EXP_DIR, f"finalize_results_{s}.csv")) for s in SUBSETS])
    for s in SUBSETS:
        sel = pd.read_csv(os.path.join(EXP_DIR, f"selected_models_{s}.csv"))
        assert (sel.feature_set == SELECTED_FEATURE_SET).all()
        prim = finalize[(finalize.subset == s) & (finalize.stage == "final_primary")].set_index("model")
        best = prim.rmse.idxmin()  # protocol: lowest primary-plan CV RMSE among the tuned families
        row = sel[sel.model == best].iloc[0]
        models[s] = {"family": best, "rul_cap": float(row.cap), "params": json.loads(row.params)}
        cv[s] = {plan: {k: round(float(finalize[(finalize.subset == s) & (finalize.stage == f"final_{plan}") & (finalize.model == best)][k].iloc[0]), 4)
                        for k in ("rmse", "mae", "bias", "phm08_score_per_sample", "rmse_fold_std")}
                 for plan in ("primary", "confirm", "wide")}

    # feature list: names do not depend on the data, but fit on FD001 to materialise them
    pre = make_final_preprocessor().fit(add_train_rul(read_cmapss_file("FD001", "train", RAW_DIR)))
    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as fh:
        json.dump({"version": CONFIG_VERSION, "feature_set": SELECTED_FEATURE_SET, "n_features": len(pre.feature_names_),
                   "features": pre.feature_meta_.to_dict(orient="records")}, fh, indent=1)

    body = {
        "version": CONFIG_VERSION,
        "feature_set": {"name": SELECTED_FEATURE_SET, "n_features": len(pre.feature_names_),
                        "feature_config": dataclasses.asdict(FINAL_FEATURE_CONFIG), "feature_list_file": FEATURE_LIST_PATH},
        "preprocessing": {
            "input_columns": "unit_id, cycle, op_setting_1..3, 21 Table-2 sensors (src/data_schema.py)",
            "operating_condition": "settings rounded to (0, 2, 0) decimals -> 6 known centres (stateless)",
            "normalisation": "per-condition z-score of the 17 candidate sensors, statistics from training engines only",
            "excluded_sensors": ["T2", "P2", "Nf_dmd", "PCNfR_dmd"],
            "feature_scaling": "per-feature standardisation with training statistics (irrelevant for trees, kept for parity with CV)",
            "causality": "every feature at cycle t uses only cycles <= t of the same engine",
        },
        "target": {"definition": "RUL = last cycle of the training engine - current cycle (failure row = 0)",
                   "training_target": "min(RUL, rul_cap)", "evaluation_target": "uncapped true RUL"},
        "models": models,
        "training_protocol": {
            "data": "all rows (every cycle) of ALL training engines of the subset; one model per subset",
            "prediction": "one prediction per engine at its last observed cycle, clipped at >= 0",
            "seed": 42, "n_jobs": 4,
            "deterministic": "fixed seeds; LightGBM deterministic=True; XGBoost 'hist' results depend on the thread "
                             "count, so models are fitted with exactly n_jobs threads (the setting used in validation)",
        },
        "validation_protocol": {
            "folds": f"{DEFAULT_N_SPLITS} engine-level folds per subset, lifespan-stratified, seed {DEFAULT_SEED}",
            "validation_samples": f"{DEFAULT_N_CUTS} truncated samples per engine, strategy={DEFAULT_STRATEGY}, true RUL at cut in [{DEFAULT_MIN_RUL}, {DEFAULT_MAX_RUL}]",
            "confirmation_plan": "same, seed 2024", "wide_plan": f"RUL at cut in [{DEFAULT_MIN_RUL}, {WIDE_MAX_RUL}]",
            "cap_rule": "lowest CV RMSE; tie-break largest cap within 1 SE",
            "documents": ["reports/target_and_validation_protocol.md", "reports/classical_ml_stage_report.md"],
        },
        "cv_results": cv,
        "official_test_set": "not used for any selection; to be read only by src/pipelines/evaluate_test.py after training",
    }
    body_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    cfg = body | {"frozen": {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "config_body_sha256": body_hash,
                             "feature_list_sha256": sha256_file(FEATURE_LIST_PATH)}}
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"frozen {CONFIG_PATH}: body sha256 {body_hash}")


def step_verify_cv(subsets=SUBSETS):
    """The frozen pipeline must reproduce the validated CV RMSE (primary plan) exactly."""
    cfg = verify_config_frozen()
    rows = []
    for s in subsets:
        m = cfg["models"][s]
        train = add_train_rul(read_cmapss_file(s, "train", RAW_DIR))
        folds, cuts = make_cv_plan(train)
        preds = cross_validate(train, folds, cuts, make_final_preprocessor,
                               lambda: make_model(m["family"], m["params"]), caps=[m["rul_cap"]])
        got = summarise_cv(preds).iloc[0]
        want = cfg["cv_results"][s]["primary"]
        rows.append({"subset": s, "model": m["family"], "cap": m["rul_cap"], "rmse_reproduced": round(got.rmse, 4),
                     "rmse_validated": want["rmse"], "match": abs(got.rmse - want["rmse"]) < 1e-3})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(MODEL_DIR, exist_ok=True)
    out.to_csv(os.path.join(MODEL_DIR, "cv_reproduction_check.csv"), index=False)
    if not out.match.all():
        raise RuntimeError("frozen pipeline does not reproduce the validated CV scores")


def step_train(subsets=SUBSETS):
    cfg = verify_config_frozen()
    manifest = {"config_version": cfg["version"], "config_body_sha256": cfg["frozen"]["config_body_sha256"],
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
                "libraries": {m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "joblib")},
                "subsets": {}}
    for s in subsets:
        m = cfg["models"][s]
        t0 = time.time()
        train = read_cmapss_file(s, "train", RAW_DIR)
        pre = make_final_preprocessor()
        X = pre.fit_transform(train)
        rul = (train.groupby("unit_id")["cycle"].transform("max") - train["cycle"]).reindex(X.index)
        model = make_model(m["family"], m["params"])
        model.fit(X[pre.feature_names_].to_numpy(np.float32), cap_rul(rul, m["rul_cap"]).to_numpy(float))
        with open(FEATURE_LIST_PATH, encoding="utf-8") as fh:
            frozen_names = [f["feature"] for f in json.load(fh)["features"]]
        assert pre.feature_names_ == frozen_names, "feature list differs from the frozen list"

        out = os.path.join(MODEL_DIR, s)
        os.makedirs(out, exist_ok=True)
        joblib.dump(pre, os.path.join(out, "preprocessor.joblib"))
        joblib.dump(model, os.path.join(out, "model.joblib"))
        meta = {"subset": s, "family": m["family"], "rul_cap": m["rul_cap"], "params": m["params"],
                "n_features": len(pre.feature_names_), "train_engines": int(train.unit_id.nunique()), "train_rows": len(train),
                "train_file_sha256": sha256_file(os.path.join(RAW_DIR, f"train_{s}.txt")),
                "fit_seconds": round(time.time() - t0, 1)}
        with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        meta["artifact_sha256"] = {f: sha256_file(os.path.join(out, f)) for f in ("preprocessor.joblib", "model.joblib", "metadata.json")}
        manifest["subsets"][s] = meta
        print(f"trained {s}: {m['family']} cap {m['rul_cap']:g} on {meta['train_engines']} engines / {meta['train_rows']} rows "
              f"({meta['fit_seconds']}s)", flush=True)
    with open(os.path.join(MODEL_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["freeze", "verify_cv", "train"], required=True)
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    a = ap.parse_args(argv)
    {"freeze": lambda: step_freeze(), "verify_cv": lambda: step_verify_cv(a.subsets), "train": lambda: step_train(a.subsets)}[a.step]()


if __name__ == "__main__":
    main()
