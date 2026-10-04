"""
Classical regression models for RUL and their (fixed, seeded) search spaces.

Every candidate is a plain dict of hyper-parameters, so experiments are fully
described by (model name, params, RUL cap, feature set) and can be logged and
re-run exactly.
"""

import itertools
import json
from typing import Callable, Dict, List

import numpy as np

import os

SEED = 42
# Threads per model; experiments run one process per subset, so this is set via the environment.
N_JOBS = int(os.environ.get("RUL_N_JOBS", "-1"))


def _ridge(alpha=10.0):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha)


def _elasticnet(alpha=0.01, l1_ratio=0.5):
    from sklearn.linear_model import ElasticNet

    # Gram matrix (n_features^2) is precomputed: n_samples >> n_features here.
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, precompute=True, max_iter=100000, tol=1e-4, selection="cyclic")


def _random_forest(n_estimators=200, min_samples_leaf=20, max_features=0.3, max_samples=0.3):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                 max_samples=max_samples, n_jobs=N_JOBS, random_state=SEED)


def _xgboost(n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=10, subsample=0.8,
             colsample_bytree=0.5, reg_lambda=1.0):
    from xgboost import XGBRegressor

    return XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                        min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree,
                        reg_lambda=reg_lambda, tree_method="hist", n_jobs=N_JOBS, random_state=SEED, verbosity=0)


def _lightgbm(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
              colsample_bytree=0.5, reg_lambda=1.0, random_state=SEED):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                         min_child_samples=min_child_samples, subsample=subsample, subsample_freq=1,
                         colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, n_jobs=N_JOBS, random_state=random_state,
                         deterministic=True, force_row_wise=True, verbose=-1)


MODEL_FACTORIES: Dict[str, Callable] = {
    "ridge": _ridge,
    "elasticnet": _elasticnet,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "lightgbm_seeded": _lightgbm,  # same model; separate name for the seed-noise stage
}

# Fixed configuration used for feature-group ablations (a reasonable, untuned LightGBM),
# plus a linear reference.
ABLATION_MODELS = {
    "lightgbm": dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                     colsample_bytree=0.5, reg_lambda=1.0),
    "ridge": dict(alpha=10.0),
}


def make_model(name: str, params: dict):
    return MODEL_FACTORIES[name](**params)


def model_factory(name: str, params: dict) -> Callable[[], object]:
    return lambda: make_model(name, params)


def _n_trees(lr: float) -> int:
    """Fixed tree budget per learning rate (no early stopping on validation folds)."""
    return {0.05: 500, 0.1: 300}[lr]


def search_space(name: str, n_random: int = 8, seed: int = SEED) -> List[dict]:
    """Deterministic candidate list per model family."""
    if name == "ridge":
        return [dict(alpha=a) for a in (0.1, 1.0, 10.0, 100.0, 1000.0)]
    if name == "elasticnet":
        return [dict(alpha=a, l1_ratio=r) for a, r in itertools.product((0.03, 0.1, 0.3, 1.0), (0.2, 0.8))]
    if name == "random_forest":
        # 30% bootstrap per tree: consecutive cycles of an engine are near-duplicates, so
        # subsampling rows costs little information and makes the forest ~4x cheaper.
        return [dict(n_estimators=200, min_samples_leaf=m, max_features=f, max_samples=0.3)
                for m, f in itertools.product((5, 20, 50), (0.1, 0.3))]
    rng = np.random.default_rng(seed + sum(map(ord, name)))  # stable per family (str hash is salted)
    space = {
        "learning_rate": (0.05, 0.1),
        "subsample": (0.7, 0.85, 1.0),
        "colsample_bytree": (0.3, 0.5, 0.8),
        "reg_lambda": (0.0, 1.0, 10.0),
    }
    if name == "xgboost":
        space |= {"max_depth": (3, 4, 6, 8), "min_child_weight": (1, 10, 50)}
    elif name == "lightgbm":
        space |= {"num_leaves": (15, 31, 63), "min_child_samples": (20, 50, 200)}
    else:
        raise KeyError(name)
    out, seen = [ABLATION_MODELS["lightgbm"].copy()] if name == "lightgbm" else [], set()
    while len(out) < n_random:
        cand = {k: v[rng.integers(len(v))] for k, v in space.items()}
        cand = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in cand.items()}
        cand["n_estimators"] = _n_trees(cand["learning_rate"])
        key = json.dumps(cand, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out
