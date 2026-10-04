"""
Feature-engineering ablations and classical-ML baselines on the fixed CV plan.

Stages (all on the training data only; the official test set is never touched):
  ablation  feature-group / normalisation / sensor ablations with a fixed LightGBM
            and a Ridge reference at cap 125 -> selects the feature set
  tune      per subset and model family: hyper-parameter search at cap 125, then the
            RUL-cap grid for the best config; cap chosen by the protocol rule
            (lowest RMSE, tie-break largest cap within 1 SE)
  finalize  selected models re-scored on the wide plan and on an independent
            confirmation plan (new folds + cuts, seed 2024), and permutation
            importance (feature groups, sensors) on the validation samples

Every model sees exactly the same validation samples (data/processed/cv_plan logic,
seed 42). Results: reports/experiments/classical/.

Usage: python -m src.pipelines.classical_experiments --stage all
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning

from src.components.data_transformation import DERIVED_SENSORS, FEATURE_GROUPS, FeatureConfig, FoldPreprocessor
from src.components.model_trainer import ABLATION_MODELS, model_factory, search_space
from src.components.validation import (
    DEFAULT_SEED,
    WIDE_MAX_RUL,
    build_fold_data,
    evaluate_on_folds,
    make_cv_plan,
    select_cap,
    summarise_cv,
)
from src.data_schema import CANDIDATE_SENSORS, SUBSETS
from src.utils import add_train_rul, read_cmapss_file

# Ridge on deliberately collinear rolling features: ill-conditioning is handled by the penalty.
warnings.filterwarnings("ignore", category=LinAlgWarning)

OUT_DIR = os.path.join("reports", "experiments", "classical")
RAW_DIR = os.path.join("data", "raw")
ABLATION_CAP = 125  # protocol default (diagnostic optimum, 02_target_and_validation)
CAP_GRID = (100, 110, 120, 125, 130, 140, 150)
MODELS = ("ridge", "elasticnet", "random_forest", "xgboost", "lightgbm")
CONFIRM_SEED = 2024

# Superset of all features; ablations select columns from it.
FULL_CONFIG = FeatureConfig(
    groups=FEATURE_GROUPS, mean_windows=(5, 10, 20, 30), std_windows=(10, 30), slope_windows=(10, 20, 30),
    minmax_windows=(10, 30), delta_lags=(10,), momentum_long=30, smooth_window=5, baseline_cycles=10,
    derived=True, health_index=True, include_cycle=True,
)
BASE = list(CANDIDATE_SENSORS)
DERIVED = list(DERIVED_SENSORS)
GENERIC = ["current", "mean", "std", "minmax", "slope", "delta", "baseline", "expanding"]
CORE5 = ["T24", "T30", "T50", "Ps30", "htBleed"]
FAULT_MODE = ["P30", "phi", "W31", "W32", "BPR", "epr"]
WEAK = ["P15", "farB", "epr"]


def _fs(groups, sensors, hi=False, cycle=False, raw=False):
    return {"groups": list(groups), "sensors": list(sensors), "hi": hi, "cycle": cycle, "raw": raw}


def feature_sets() -> Dict[str, dict]:
    """Named feature sets: forward (cumulative), leave-one-group-out, normalisation, sensor ablations."""
    fs = {}
    for i in range(1, len(GENERIC) + 1):
        fs[f"F{i}_" + "+".join(GENERIC[:i]) if i == 1 else f"F{i}_+{GENERIC[i - 1]}"] = _fs(GENERIC[:i], BASE)
    fs["F9_+derived"] = _fs(GENERIC, BASE + DERIVED)
    fs["F10_+health_index (FULL)"] = _fs(GENERIC, BASE + DERIVED, hi=True)
    fs["F11_+cycle"] = _fs(GENERIC, BASE + DERIVED, hi=True, cycle=True)
    for g in GENERIC:
        fs[f"LOGO_-{g}"] = _fs([x for x in GENERIC if x != g], BASE + DERIVED, hi=True)
    fs["LOGO_-derived"] = _fs(GENERIC, BASE, hi=True)
    fs["LOGO_-health_index"] = _fs(GENERIC, BASE + DERIVED)
    fs["RAW_FULL (no condition normalisation)"] = _fs(GENERIC, BASE + DERIVED, hi=True, raw=True)
    # Sensor ablations: without the health index (it would mix excluded sensors back in);
    # compare against LOGO_-health_index.
    fs["S_core5_only"] = _fs(GENERIC, CORE5)
    fs["S_-fault_mode_sensors"] = _fs(GENERIC, [s for s in BASE if s not in FAULT_MODE] + ["dT_HPC", "Nc_over_Nf", "T50_over_T30"])
    fs["S_-weak_sensors"] = _fs(GENERIC, [s for s in BASE if s not in WEAK] + DERIVED)
    fs["S_-corrected_speeds"] = _fs(GENERIC, [s for s in BASE if s not in ("NRf", "NRc")] + DERIVED)
    # Refinement pass, added after the first ablation run showed (a) a large gain from the cycle
    # feature and (b) that the expanding group hurts (LOGO). Selection-bias check: confirmation plan.
    no_exp = [g for g in GENERIC if g != "expanding"]
    fs["R1_F11 -expanding"] = _fs(no_exp, BASE + DERIVED, hi=True, cycle=True)
    fs["R2_F7 +cycle (minimal)"] = _fs(GENERIC[:7], BASE, cycle=True)
    fs["R3_R1 -health_index"] = _fs(no_exp, BASE + DERIVED, cycle=True)
    return fs


def select_columns(meta: pd.DataFrame, spec: dict) -> List[str]:
    m = meta
    keep = m["group"].isin(spec["groups"]) & m["sensor"].isin(spec["sensors"])
    if spec["hi"]:
        keep |= m["group"] == "health_index"
    if spec["cycle"]:
        keep |= m["group"] == "cycle"
    keep |= m["group"] == "condition"  # one-hot condition (raw variant only)
    return list(m.loc[keep, "feature"])


def config_id(model: str, params: dict) -> str:
    return hashlib.md5(json.dumps([model, params], sort_keys=True).encode()).hexdigest()[:10]


class Logger:
    """One results + predictions file per (stage, subset): subsets can run in parallel processes."""

    def __init__(self, stage: str):
        os.makedirs(OUT_DIR, exist_ok=True)
        self.stage = stage

    def paths(self, subset: str):
        return (os.path.join(OUT_DIR, f"{self.stage}_results_{subset}.csv"),
                os.path.join(OUT_DIR, f"{self.stage}_predictions_{subset}.csv.gz"))

    def is_done(self, subset: str) -> bool:
        return os.path.exists(self.paths(subset)[0])

    def write(self, subset: str, results: pd.DataFrame, preds: pd.DataFrame):
        res_path, pred_path = self.paths(subset)
        preds.to_csv(pred_path, index=False)
        results.to_csv(res_path, index=False)  # written last: marks the subset as done


def load_stage(stage: str, kind: str = "results") -> pd.DataFrame:
    """Concatenate the per-subset files of a stage (kind = 'results' | 'predictions')."""
    ext = "csv" if kind == "results" else "csv.gz"
    parts = [os.path.join(OUT_DIR, f"{stage}_{kind}_{s}.{ext}") for s in SUBSETS]
    return pd.concat([pd.read_csv(p) for p in parts if os.path.exists(p)], ignore_index=True)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    with open(os.path.join(OUT_DIR, f"run_log_{os.getpid()}.txt"), "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def run_eval(fold_data, model: str, params: dict, caps, features, tags: dict):
    """Evaluate one (model, params) for several caps on the fold data; returns result rows + predictions."""
    t0 = time.time()
    preds = evaluate_on_folds(fold_data, model_factory(model, params), caps=caps, features=features)
    secs = time.time() - t0
    summ = summarise_cv(preds)
    fold_rmse = preds.groupby(["cap", "fold"]).apply(lambda g: np.sqrt(np.mean((g.y_pred - g.y_true) ** 2))).unstack()
    rows = []
    for _, r in summ.iterrows():
        rows.append(tags | {"model": model, "config_id": config_id(model, params), "params": json.dumps(params, sort_keys=True),
                            "n_features": len(features) if features is not None else len(fold_data[0].feature_names),
                            "fold_rmse": json.dumps([round(v, 3) for v in fold_rmse.loc[r["cap"]].tolist()]),
                            "runtime_s": round(secs / max(len(list(caps)), 1), 1)} | r.to_dict())
    preds = preds.assign(**tags, model=model, config_id=config_id(model, params))
    return pd.DataFrame(rows), preds


def load_train(subset: str) -> pd.DataFrame:
    return add_train_rul(read_cmapss_file(subset, "train", RAW_DIR))


# --------------------------------------------------------------------------------------
def stage_ablation(subsets):
    logger = Logger("ablation")
    sets = feature_sets()
    for s in subsets:
        if logger.is_done(s):
            log(f"ablation {s}: already done, skipping")
            continue
        train = load_train(s)
        folds, cuts = make_cv_plan(train)
        results, preds = [], []
        for raw in (False, True):
            t0 = time.time()
            fd = build_fold_data(train, folds, cuts, lambda: FoldPreprocessor(config=FULL_CONFIG.with_(normalize=not raw)))
            meta = fd[0].preprocessor.feature_meta_
            log(f"ablation {s}: fold data (raw={raw}) built in {time.time() - t0:.0f}s, {len(meta)} features")
            for name, spec in sets.items():
                if spec["raw"] != raw:
                    continue
                cols = select_columns(meta, spec)
                for model, params in ABLATION_MODELS.items():
                    r, p = run_eval(fd, model, params, [ABLATION_CAP], cols, {"stage": "ablation", "subset": s, "feature_set": name})
                    results.append(r)
                    preds.append(p)
                log(f"ablation {s} {name}: {len(cols)} features, lightgbm RMSE {results[-2].rmse.item():.2f}, ridge {results[-1].rmse.item():.2f}")
            del fd
        logger.write(s, pd.concat(results, ignore_index=True), pd.concat(preds, ignore_index=True))


def stage_seed_noise(subsets, sets=("F11_+cycle", "R1_F11 -expanding", "R2_F7 +cycle (minimal)"), seeds=(1, 2, 3, 4, 5)):
    """
    Model-seed noise of the fixed ablation LightGBM (column/row subsampling), on identical
    validation samples: ablation differences smaller than this are not attributable to features.
    """
    logger = Logger("seed_noise")
    all_sets = feature_sets()
    for s in subsets:
        if logger.is_done(s):
            log(f"seed_noise {s}: already done, skipping")
            continue
        train = load_train(s)
        folds, cuts = make_cv_plan(train)
        fd = build_fold_data(train, folds, cuts, lambda: FoldPreprocessor(config=FULL_CONFIG))
        meta = fd[0].preprocessor.feature_meta_
        results, preds = [], []
        for name in sets:
            cols = select_columns(meta, all_sets[name])
            for seed in seeds:
                params = ABLATION_MODELS["lightgbm"] | {"random_state": seed}
                r, p = run_eval(fd, "lightgbm_seeded", params, [ABLATION_CAP], cols,
                                {"stage": "seed_noise", "subset": s, "feature_set": name})
                results.append(r.assign(seed=seed))
                preds.append(p.assign(seed=seed))
            log(f"seed_noise {s} {name}: RMSE over seeds {[round(x, 2) for x in pd.concat(results).query('feature_set == @name').rmse]}")
        logger.write(s, pd.concat(results, ignore_index=True), pd.concat(preds, ignore_index=True))


def choose_feature_set() -> str:
    """Pre-specified rule: lowest mean LightGBM RMSE over the four subsets."""
    res = load_stage("ablation")
    lgb = res[res.model == "lightgbm"].pivot(index="feature_set", columns="subset", values="rmse")
    if set(lgb.columns) != set(SUBSETS):
        raise RuntimeError("feature-set choice needs ablation results for all four subsets")
    return lgb.mean(axis=1).idxmin()


def stage_tune(subsets, feature_set: Optional[str] = None, n_random: int = 8):
    feature_set = feature_set or choose_feature_set()
    spec = feature_sets()[feature_set]
    log(f"tune: feature set = {feature_set}")
    logger = Logger("tune")
    for s in subsets:
        if logger.is_done(s):
            log(f"tune {s}: already done, skipping")
            continue
        train = load_train(s)
        folds, cuts = make_cv_plan(train)
        t0 = time.time()
        fd = build_fold_data(train, folds, cuts, lambda: FoldPreprocessor(config=FULL_CONFIG.with_(normalize=not spec["raw"])))
        cols = select_columns(fd[0].preprocessor.feature_meta_, spec)
        log(f"tune {s}: fold data built in {time.time() - t0:.0f}s, {len(cols)} features")
        results, preds, selected = [], [], []
        for model in MODELS:
            cands = search_space(model, n_random=n_random)
            step1 = []
            for params in cands:
                r, p = run_eval(fd, model, params, [ABLATION_CAP], cols, {"stage": "tune_hparams", "subset": s, "feature_set": feature_set})
                step1.append(r)
                preds.append(p)
            step1 = pd.concat(step1, ignore_index=True)
            results.append(step1)
            log(f"tune {s} {model}: {len(cands)} configs @cap {ABLATION_CAP}, best RMSE {step1.rmse.min():.2f}")
            best = []
            for cid in step1.nsmallest(1, "rmse").config_id:
                params = json.loads(step1.loc[step1.config_id == cid, "params"].iloc[0])
                r, p = run_eval(fd, model, params, CAP_GRID, cols, {"stage": "tune_cap", "subset": s, "feature_set": feature_set})
                results.append(r)
                preds.append(p)
                cap = select_cap(r)
                best.append({"subset": s, "model": model, "config_id": cid, "params": json.dumps(params, sort_keys=True),
                             "cap": cap} | r.loc[r.cap == cap].iloc[0][["rmse", "mae", "bias", "phm08_score_per_sample", "rmse_fold_std"]].to_dict())
            best = pd.DataFrame(best).sort_values("rmse").iloc[:1]
            selected.append(best)
            log(f"tune {s} {model}: selected cap {best.cap.item():g}, RMSE {best.rmse.item():.2f} ± {best.rmse_fold_std.item():.2f}")
        sel = pd.concat(selected, ignore_index=True).assign(feature_set=feature_set)
        sel.to_csv(os.path.join(OUT_DIR, f"selected_models_{s}.csv"), index=False)
        logger.write(s, pd.concat(results, ignore_index=True), pd.concat(preds, ignore_index=True))
        del fd


def _permutation_importance(fold_data, models: dict, cap, cols, meta, n_repeats=3, seed=DEFAULT_SEED):
    """Mean increase in validation RMSE when a whole feature group / sensor is permuted (jointly across samples)."""
    rng = np.random.default_rng(seed)
    m = meta.set_index("feature").loc[cols]
    units = {"group": m["group"], "sensor": m["sensor"].where(m["group"] != "health_index", "health_index")}
    rows = []
    for fd in fold_data:
        idx = [fd.feature_names.index(c) for c in cols]
        X = fd.X_val[:, idx]
        y = fd.val_meta.y_true.to_numpy()
        model = models[(fd.fold, cap)]
        base = np.sqrt(np.mean((np.maximum(model.predict(X), 0) - y) ** 2))
        for kind, labels in units.items():
            for lab in labels.unique():
                j = np.where(labels.to_numpy() == lab)[0]
                inc = []
                for _ in range(n_repeats):
                    Xp = X.copy()
                    Xp[:, j] = X[rng.permutation(len(X))][:, j]  # same row permutation for the whole block
                    inc.append(np.sqrt(np.mean((np.maximum(model.predict(Xp), 0) - y) ** 2)) - base)
                rows.append({"fold": fd.fold, "kind": kind, "unit": lab, "n_features": len(j), "rmse_increase": float(np.mean(inc))})
    return pd.DataFrame(rows)


def stage_finalize(subsets):
    logger = Logger("finalize")
    for s in subsets:
        if logger.is_done(s):
            log(f"finalize {s}: already done, skipping")
            continue
        chosen = pd.read_csv(os.path.join(OUT_DIR, f"selected_models_{s}.csv"))
        feature_set = chosen.feature_set.iloc[0]
        spec = feature_sets()[feature_set]
        imp_path = os.path.join(OUT_DIR, f"importance_{s}.csv")
        if os.path.exists(imp_path):
            os.remove(imp_path)  # recomputed in full for this subset
        train = load_train(s)
        plans = {
            "wide": make_cv_plan(train, max_rul=WIDE_MAX_RUL),
            "confirm": make_cv_plan(train, seed=CONFIRM_SEED),
            "primary": make_cv_plan(train),
        }
        results, preds = [], []
        for plan_name, (folds, cuts) in plans.items():
            fd = build_fold_data(train, folds, cuts, lambda: FoldPreprocessor(config=FULL_CONFIG.with_(normalize=not spec["raw"])))
            meta = fd[0].preprocessor.feature_meta_
            cols = select_columns(meta, spec)
            for _, c in chosen.iterrows():
                params = json.loads(c.params)
                r, p = run_eval(fd, c.model, params, [c.cap], cols, {"stage": f"final_{plan_name}", "subset": s, "feature_set": feature_set})
                results.append(r)
                preds.append(p)
                log(f"finalize {s} {plan_name} {c.model}: RMSE {r.rmse.item():.2f}")
            if plan_name == "primary":
                # importance for the best model of this subset and for LightGBM (if different)
                best_model = chosen.sort_values("rmse").iloc[0].model
                for model in dict.fromkeys([best_model, "lightgbm"]):
                    c = chosen[chosen.model == model].iloc[0]
                    _, models = evaluate_on_folds(fd, model_factory(model, json.loads(c.params)), caps=[c.cap],
                                                  features=cols, keep_models=True)
                    imp = _permutation_importance(fd, models, c.cap, cols, meta).assign(subset=s, model=model,
                                                                                         is_best=model == best_model)
                    if model in ("lightgbm", "xgboost", "random_forest"):
                        gain = np.mean([np.asarray(mdl.feature_importances_, float) / np.sum(mdl.feature_importances_)
                                        for mdl in models.values()], axis=0)
                        g = pd.DataFrame({"unit": cols, "rmse_increase": gain}).assign(kind="feature_native", fold=-1,
                                                                                          n_features=1, subset=s, model=model,
                                                                                          is_best=model == best_model)
                        imp = pd.concat([imp, g], ignore_index=True)
                    imp.to_csv(imp_path, mode="a", header=not os.path.exists(imp_path), index=False)
                    log(f"finalize {s}: importance computed for {model}")
            del fd
        logger.write(s, pd.concat(results, ignore_index=True), pd.concat(preds, ignore_index=True))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ablation", "seed_noise", "tune", "finalize", "all"], default="all")
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--feature-set", default=None, help="override the rule-based feature-set choice")
    args = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    env = {"python": platform.python_version(), "argv": sys.argv, "seed": DEFAULT_SEED, "confirm_seed": CONFIRM_SEED,
           "ablation_cap": ABLATION_CAP, "cap_grid": CAP_GRID, "full_config": repr(FULL_CONFIG)}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm"):
        env[mod] = __import__(mod).__version__
    env["n_jobs_per_model"] = os.environ.get("RUL_N_JOBS", "-1")
    with open(os.path.join(OUT_DIR, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=2, default=str)
    if args.stage in ("ablation", "all"):
        stage_ablation(args.subsets)
    if args.stage in ("seed_noise", "all"):
        stage_seed_noise(args.subsets)
    if args.stage in ("tune", "all"):
        stage_tune(args.subsets, args.feature_set)
    if args.stage in ("finalize", "all"):
        stage_finalize(args.subsets)


if __name__ == "__main__":
    main()
