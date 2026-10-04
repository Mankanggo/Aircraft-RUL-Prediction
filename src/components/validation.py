"""
Engine-level, test-like validation for C-MAPSS RUL models.

Protocol (per FD subset):
  * Folds are formed from whole ENGINES (never rows), stratified by lifespan.
  * Each validation engine is truncated at several cut points to mimic the
    partially observed test trajectories. Exactly ONE prediction is made per
    truncated sample, at its last observed cycle, from the truncated data only.
  * The validation target is the uncapped true RUL at the cut
    (lifespan - cut_cycle), the same definition as RUL_FD00x.txt.
  * All preprocessing is fitted on the training-fold engines only.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_schema import RUL_COLUMN
from src.utils import cap_rul, rul_metrics

DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
# Truncation bounds documented in the paper (Saxena et al. 2008, Sec. VI): test-set RULs
# ranged 10-150, validation-set RULs 6-190. Primary protocol: true RUL at the cut in
# [6, 150]; WIDE_MAX_RUL is the secondary robustness setting. Taken from the paper, not
# from RUL_FD00x.txt; 150 also gives the best label-free match of observed lengths to the
# official test trajectories (notebooks/02_target_and_validation.ipynb, Section 3).
DEFAULT_MIN_RUL = 6
DEFAULT_MAX_RUL = 150
WIDE_MAX_RUL = 190
DEFAULT_STRATEGY = "uniform"
DEFAULT_N_CUTS = 10


def engine_lifespans(train_df: pd.DataFrame) -> pd.Series:
    """Lifespan (= last cycle, the failure cycle) of each training engine."""
    return train_df.groupby("unit_id")["cycle"].max().rename("lifespan")


def make_engine_folds(train_df: pd.DataFrame, n_splits: int = DEFAULT_N_SPLITS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Assign every engine to exactly one fold, stratified by lifespan: engines are
    sorted by lifespan, cut into consecutive blocks of `n_splits`, and the fold
    labels are randomly permuted within each block. Deterministic for a seed.
    Returns columns [unit_id, lifespan, fold].
    """
    life = engine_lifespans(train_df).reset_index()
    if len(life) < n_splits:
        raise ValueError(f"{len(life)} engines cannot fill {n_splits} folds")
    life = life.sort_values(["lifespan", "unit_id"], kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(life), dtype=int)
    for start in range(0, len(life), n_splits):
        stop = min(start + n_splits, len(life))
        folds[start:stop] = rng.permutation(n_splits)[: stop - start]
    life["fold"] = folds
    return life.sort_values("unit_id").reset_index(drop=True)


def assert_engine_disjoint(train_df: pd.DataFrame, val_df: pd.DataFrame, key: str = "unit_id") -> None:
    """Raise if any engine appears on both sides of a split."""
    overlap = set(train_df[key].unique()) & set(val_df[key].unique())
    if overlap:
        raise AssertionError(f"{len(overlap)} engine(s) in both train and validation: {sorted(overlap)[:10]}")


def iter_folds(train_df: pd.DataFrame, folds: pd.DataFrame) -> Iterator[Tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, train_part, val_part) with whole engines on each side."""
    fold_of = folds.set_index("unit_id")["fold"]
    unit_fold = train_df["unit_id"].map(fold_of)
    if unit_fold.isna().any():
        raise ValueError("some engines have no fold assignment")
    for k in sorted(folds["fold"].unique()):
        tr, va = train_df[unit_fold != k], train_df[unit_fold == k]
        assert_engine_disjoint(tr, va)
        yield int(k), tr, va


def sample_truncation_cuts(
    engines_df: pd.DataFrame,
    n_cuts: int = DEFAULT_N_CUTS,
    strategy: str = DEFAULT_STRATEGY,
    reference_lengths: Optional[Sequence[int]] = None,
    min_rul: int = DEFAULT_MIN_RUL,
    max_rul: Optional[int] = DEFAULT_MAX_RUL,
    min_history: int = 1,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Choose up to `n_cuts` distinct cut cycles per run-to-failure engine. The
    sample for a cut keeps cycles 1..cut_cycle; its target is the true RUL
    lifespan - cut_cycle, constrained to [min_rul, max_rul] (max_rul=None: no
    upper bound).

    strategy="test_lengths": observed lengths are drawn (without replacement,
        weighted by frequency) from `reference_lengths`, the observed lengths of
        the official test trajectories. These are model inputs available at
        prediction time; no test labels are used.
    strategy="uniform": cut cycles drawn uniformly from the feasible range,
        i.e. the true RUL at the cut is uniform on [min_rul, max_rul].

    Engines with no feasible reference length fall back to uniform.
    Returns [sample_id, unit_id, lifespan, cut_cycle, rul_at_cut].
    """
    if strategy not in ("test_lengths", "uniform"):
        raise ValueError(f"unknown strategy {strategy!r}")
    if strategy == "test_lengths" and reference_lengths is None:
        raise ValueError("strategy='test_lengths' needs reference_lengths")
    rng = np.random.default_rng(seed)
    ref = pd.Series(reference_lengths).value_counts() if reference_lengths is not None else None
    rows = []
    for unit, life in engine_lifespans(engines_df).items():
        lo, hi = min_history, life - min_rul
        if max_rul is not None:
            lo = max(lo, life - max_rul)
        if hi < lo:
            continue
        candidates = None
        if strategy == "test_lengths":
            feasible = ref[(ref.index >= lo) & (ref.index <= hi)]
            if len(feasible):
                k = min(n_cuts, len(feasible))
                candidates = rng.choice(feasible.index.to_numpy(), size=k, replace=False,
                                        p=(feasible / feasible.sum()).to_numpy())
        if candidates is None:
            k = min(n_cuts, hi - lo + 1)
            candidates = rng.choice(np.arange(lo, hi + 1), size=k, replace=False)
        for c in np.sort(candidates):
            rows.append((int(unit), int(life), int(c), int(life - c)))
    cuts = pd.DataFrame(rows, columns=["unit_id", "lifespan", "cut_cycle", "rul_at_cut"])
    cuts.insert(0, "sample_id", np.arange(len(cuts)))
    return cuts


def make_truncated_samples(engines_df: pd.DataFrame, cuts: pd.DataFrame) -> pd.DataFrame:
    """
    Materialise each cut as its own trajectory: rows of the engine with
    cycle <= cut_cycle, tagged with sample_id. Rows after the cut are removed
    (not masked), so no later cycle can reach the features of the sample.
    Any RUL column of the input is dropped; targets live in `cuts`.
    """
    cols = [c for c in engines_df.columns if c != RUL_COLUMN]
    merged = engines_df[cols].merge(cuts[["sample_id", "unit_id", "cut_cycle"]], on="unit_id", how="inner")
    out = merged[merged["cycle"] <= merged["cut_cycle"]].drop(columns="cut_cycle")
    return out.sort_values(["sample_id", "cycle"], kind="mergesort").reset_index(drop=True)


def last_observed_rows(df: pd.DataFrame, group_col: str = "sample_id") -> pd.DataFrame:
    """The row at the last observed cycle of each trajectory (where the prediction is made)."""
    idx = df.groupby(group_col)["cycle"].idxmax()
    return df.loc[idx].sort_values(group_col).reset_index(drop=True)


def make_cv_plan(
    train_df: pd.DataFrame,
    reference_lengths: Optional[Sequence[int]] = None,
    n_splits: int = DEFAULT_N_SPLITS,
    n_cuts: int = DEFAULT_N_CUTS,
    strategy: str = DEFAULT_STRATEGY,
    min_rul: int = DEFAULT_MIN_RUL,
    max_rul: Optional[int] = DEFAULT_MAX_RUL,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fixed CV plan for one subset: engine folds + validation cuts for every
    engine (a cut belongs to the fold of its engine). Generate it once and reuse
    it for every model, so all candidates are compared on identical samples.
    """
    folds = make_engine_folds(train_df, n_splits, seed)
    cuts = []
    for k, _, va in iter_folds(train_df, folds):
        c = sample_truncation_cuts(va, n_cuts, strategy, reference_lengths, min_rul, max_rul, seed=seed + 1000 + k)
        cuts.append(c.assign(fold=k))
    cuts = pd.concat(cuts, ignore_index=True)
    cuts["sample_id"] = np.arange(len(cuts))
    return folds, cuts


@dataclass
class FoldData:
    """Features of one CV fold, built once and reused by every model / cap / feature subset."""

    fold: int
    X_train: np.ndarray  # every cycle of the training-fold engines
    rul_train: np.ndarray  # UNCAPPED training RUL (rebuilt from cycles)
    X_val: np.ndarray  # one row per validation sample, at its last observed cycle
    val_meta: pd.DataFrame  # sample_id, unit_id, cut_cycle, lifespan, y_true (uncapped)
    feature_names: List[str]
    preprocessor: object


def build_fold_data(
    train_df: pd.DataFrame,
    folds: pd.DataFrame,
    cuts: pd.DataFrame,
    make_preprocessor: Callable[[], object],
) -> List[FoldData]:
    """
    For each fold: fit the preprocessor on the training-fold engines only, build
    training rows (every cycle) from them, and build each validation sample's
    features from its truncated history only (rows after the cut do not exist).
    """
    out = []
    for k, tr, va in iter_folds(train_df, folds):
        fold_cuts = cuts[cuts["fold"] == k]
        assert set(fold_cuts["unit_id"]) <= set(va["unit_id"]), "cuts must come from the validation engines"
        pre = make_preprocessor()
        X_tr = pre.fit_transform(tr, group_cols=("unit_id",))
        assert pre.fit_units_.isdisjoint(set(va["unit_id"])), "preprocessor saw validation engines"
        # Always rebuild the uncapped target from the cycles (never trust an input RUL column,
        # which may already be capped); caps are applied per model.
        rul_tr = (tr.groupby("unit_id")["cycle"].transform("max") - tr["cycle"]).reindex(X_tr.index)

        samples = make_truncated_samples(va, fold_cuts)
        X_va = last_observed_rows(pre.transform(samples, group_cols=("sample_id",)))
        X_va = X_va.merge(fold_cuts[["sample_id", "rul_at_cut", "lifespan", "cut_cycle"]], on="sample_id")
        assert (X_va["cycle"] == X_va["cut_cycle"]).all()
        assert len(X_va) == len(fold_cuts)
        out.append(FoldData(
            fold=int(k),
            X_train=X_tr[pre.feature_names_].to_numpy(dtype=np.float32),
            rul_train=rul_tr.to_numpy(dtype=float),
            X_val=X_va[pre.feature_names_].to_numpy(dtype=np.float32),
            val_meta=X_va[["sample_id", "unit_id", "cut_cycle", "lifespan"]].assign(y_true=X_va["rul_at_cut"].to_numpy()),
            feature_names=list(pre.feature_names_),
            preprocessor=pre,
        ))
    return out


def evaluate_on_folds(
    fold_data: List[FoldData],
    make_model: Callable[[], object],
    caps: Iterable[Optional[float]] = (None,),
    features: Optional[Sequence[str]] = None,
    clip_min: float = 0.0,
    keep_models: bool = False,
):
    """
    Fit one model per (fold, cap) on the training rows (target = min(RUL, cap))
    and predict every validation sample. `features` selects a column subset
    (feature-group ablations). Returns predictions (one row per cap x sample),
    plus the fitted models {(fold, cap): model} if keep_models.
    """
    out, models = [], {}
    for fd in fold_data:
        cols = slice(None) if features is None else [fd.feature_names.index(f) for f in features]
        X_tr, X_va = fd.X_train[:, cols], fd.X_val[:, cols]
        for cap in caps:
            model = make_model()
            model.fit(X_tr, cap_rul(fd.rul_train, cap))
            pred = np.maximum(np.asarray(model.predict(X_va), dtype=float), clip_min)
            out.append(fd.val_meta.assign(cap=np.nan if cap is None else cap, fold=fd.fold, y_pred=pred))
            if keep_models:
                models[(fd.fold, cap)] = model
    preds = pd.concat(out, ignore_index=True)[["cap", "fold", "sample_id", "unit_id", "cut_cycle", "lifespan", "y_true", "y_pred"]]
    return (preds, models) if keep_models else preds


def cross_validate(
    train_df: pd.DataFrame,
    folds: pd.DataFrame,
    cuts: pd.DataFrame,
    make_preprocessor: Callable[[], object],
    make_model: Callable[[], object],
    caps: Iterable[Optional[float]] = (None,),
    clip_min: float = 0.0,
) -> pd.DataFrame:
    """
    Engine-level CV with test-like truncated validation samples
    (build_fold_data + evaluate_on_folds). Preprocessing does not depend on the
    cap, so it is fitted once per fold and shared by all caps.
    Returns one row per (cap, validation sample) with the uncapped true RUL.
    """
    return evaluate_on_folds(build_fold_data(train_df, folds, cuts, make_preprocessor), make_model, caps, clip_min=clip_min)


def summarise_cv(preds: pd.DataFrame, by: Sequence[str] = ("cap",)) -> pd.DataFrame:
    """
    Metrics vs the uncapped true RUL, pooled over all validation samples, plus
    the fold-to-fold std of the RMSE. Also RMSE vs the capped truth, for
    comparison with literature that caps test labels (secondary metric only).
    """
    rows = []
    for key, g in preds.groupby(list(by), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        cap = g["cap"].iloc[0]
        m = rul_metrics(g["y_true"], g["y_pred"])
        fold_rmse = g.groupby("fold").apply(lambda f: np.sqrt(np.mean((f.y_pred - f.y_true) ** 2)))
        capped_truth = g["y_true"] if pd.isna(cap) else np.minimum(g["y_true"], cap)
        rows.append(dict(zip(by, key)) | m | {
            "rmse_fold_std": float(fold_rmse.std()),
            "rmse_vs_capped_truth": float(np.sqrt(np.mean((g["y_pred"] - capped_truth) ** 2))),
        })
    return pd.DataFrame(rows)


def select_cap(summary: pd.DataFrame, n_splits: int = DEFAULT_N_SPLITS) -> float:
    """
    Protocol rule: lowest RMSE; tie-break to the LARGEST cap whose RMSE is within
    one standard error (fold std / sqrt(n_splits)) of the best.
    `summary` = summarise_cv output for one model/subset (numeric caps only).
    """
    s = summary.dropna(subset=["cap"])
    best = s.loc[s["rmse"].idxmin()]
    se = best["rmse_fold_std"] / np.sqrt(n_splits)
    return float(s.loc[s["rmse"] <= best["rmse"] + se, "cap"].max())


def paired_rmse_difference(preds_a: pd.DataFrame, preds_b: pd.DataFrame, n_boot: int = 2000, seed: int = DEFAULT_SEED) -> dict:
    """
    RMSE(b) - RMSE(a) on identical validation samples, with a 95% bootstrap CI
    that resamples ENGINES (samples of one engine are correlated).
    Negative = b is better.
    """
    a = preds_a.set_index("sample_id")
    b = preds_b.set_index("sample_id").loc[a.index]
    if not (a["y_true"] == b["y_true"]).all():
        raise ValueError("predictions are not on the same validation samples")
    se_a = (a["y_pred"] - a["y_true"]) ** 2
    se_b = (b["y_pred"] - b["y_true"]) ** 2
    per_engine = pd.DataFrame({"a": se_a, "b": se_b, "n": 1.0, "unit": a["unit_id"]}).groupby("unit").sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_engine), size=(n_boot, len(per_engine)))
    A, B, N = (per_engine[c].to_numpy()[idx].sum(axis=1) for c in ("a", "b", "n"))
    diffs = np.sqrt(B / N) - np.sqrt(A / N)
    point = float(np.sqrt(se_b.mean()) - np.sqrt(se_a.mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_rmse": point, "ci_low": float(lo), "ci_high": float(hi), "significant": bool(hi < 0 or lo > 0)}
