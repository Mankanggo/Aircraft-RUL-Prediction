"""
Leakage-safe preprocessing and feature engineering for C-MAPSS.

Two rules are enforced here:
1. Every statistic (per-condition normalisation, health-index regression,
   feature scaling) is fitted on the TRAINING rows passed to `fit` and only
   applied by `transform`.
2. Every engineered feature at cycle t uses only cycles <= t of the same
   engine/sample (trailing windows, lags, expanding statistics). This is checked
   by the prefix-invariance tests in tests/test_leakage.py.
"""

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_schema import CANDIDATE_SENSORS, RUL_COLUMN
from src.utils import apply_condition_norm, assign_operating_condition, fit_condition_stats

DEFAULT_WINDOWS = (5, 15, 30)
DEFAULT_BASELINE_CYCLES = 10

FEATURE_GROUPS = ("current", "mean", "std", "minmax", "slope", "delta", "baseline", "expanding")

# Physically motivated combinations of Table-2 sensors, computed row-wise on raw values
# (so they are causal) and then condition-normalised like any other sensor.
DERIVED_SENSORS: Dict[str, str] = {
    "Wf_est": "phi * Ps30: fuel flow implied by the definition of phi (fuel flow / Ps30)",
    "dT_HPC": "T30 - T24: temperature rise across the HPC (HPC degradation raises it)",
    "Nc_over_Nf": "Nc / Nf: core-to-fan speed ratio (fan vs core degradation shift it differently)",
    "T50_over_T30": "T50 / T30: LPT-outlet vs HPC-outlet temperature ratio (hot-section balance)",
}


def add_derived_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise derived quantities (no cross-row information)."""
    out = df.copy()
    out["Wf_est"] = df["phi"] * df["Ps30"]
    out["dT_HPC"] = df["T30"] - df["T24"]
    out["Nc_over_Nf"] = df["Nc"] / df["Nf"]
    out["T50_over_T30"] = df["T50"] / df["T30"]
    return out


@dataclass(frozen=True)
class FeatureConfig:
    """Which causal feature groups to build, and with which windows."""

    groups: Tuple[str, ...] = ("current", "mean", "std", "slope", "baseline")
    mean_windows: Tuple[int, ...] = DEFAULT_WINDOWS
    std_windows: Tuple[int, ...] = DEFAULT_WINDOWS
    slope_windows: Tuple[int, ...] = DEFAULT_WINDOWS
    minmax_windows: Tuple[int, ...] = (10, 30)
    delta_lags: Tuple[int, ...] = (10,)
    momentum_long: int = 30
    smooth_window: int = 5  # short trailing mean used by baseline / delta / expanding features
    baseline_cycles: int = DEFAULT_BASELINE_CYCLES
    sensors: Tuple[str, ...] = tuple(CANDIDATE_SENSORS)
    derived: bool = False  # add DERIVED_SENSORS as extra sensors
    normalize: bool = True  # per-operating-condition z-scoring (False: raw values + condition one-hot)
    health_index: bool = False  # train-fold-fitted linear health index + its causal features
    hi_window: int = 10
    hi_cap: int = 125
    hi_crossfit_folds: int = 5  # training rows get out-of-fold HI values (0/1 = in-sample)
    include_cycle: bool = False

    def __post_init__(self):
        unknown = set(self.groups) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"unknown feature groups {sorted(unknown)}")

    @property
    def all_sensors(self) -> List[str]:
        return list(self.sensors) + (list(DERIVED_SENSORS) if self.derived else [])

    def with_(self, **kw) -> "FeatureConfig":
        return replace(self, **kw)


class ConditionNormalizer:
    """Per-operating-condition z-scoring with statistics fitted on training rows only."""

    def __init__(self, sensors: Sequence[str] = CANDIDATE_SENSORS):
        self.sensors = list(sensors)

    def fit(self, train_df: pd.DataFrame) -> "ConditionNormalizer":
        self.stats_ = fit_condition_stats(train_df, self.sensors)
        self.n_fit_rows_ = len(train_df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "stats_"):
            raise RuntimeError("ConditionNormalizer must be fitted before transform")
        out = apply_condition_norm(df, self.stats_, self.sensors)
        out["op_condition"] = assign_operating_condition(df)
        return out


def _grouped_rolling(df: pd.DataFrame, group_cols: List[str], cols: List[str], window: int, how: str = "mean") -> pd.DataFrame:
    """Trailing rolling statistic within each group (rows must be sorted by group, cycle)."""
    r = getattr(df.groupby(group_cols, sort=False)[cols].rolling(window, min_periods=1), how)()
    r.index = r.index.droplevel(list(range(len(group_cols))))
    return r.reindex(df.index)


def _grouped_rolling_mean(df: pd.DataFrame, group_cols: List[str], cols: List[str], window: int) -> pd.DataFrame:
    return _grouped_rolling(df, group_cols, cols, window, "mean")


def add_causal_features(
    df: pd.DataFrame,
    sensors: Sequence[str],
    windows: Iterable[int] = DEFAULT_WINDOWS,
    baseline_cycles: int = DEFAULT_BASELINE_CYCLES,
    group_cols: Sequence[str] = ("unit_id",),
    config: Optional[FeatureConfig] = None,
    return_meta: bool = False,
):
    """
    Causal features per (group, cycle), using only rows with cycle <= current cycle.
    With `config=None` the legacy set is built (current, mean/std/slope over
    `windows`, delta_base). Feature naming, per sensor s:
      current    {s}                     current (normalised) value
      mean       {s}_mean{w}             trailing mean over the last w cycles (fewer at the start)
      std        {s}_std{w}              trailing sample std (0 with one cycle)
      minmax     {s}_min{w}, {s}_max{w}  trailing min / max
      slope      {s}_slope{w}            trailing least-squares slope vs cycle (0 with one cycle)
      delta      {s}_diff{k}             m(t) - m(t-k), m = trailing mean over smooth_window;
                                         before cycle k+1: m(t) - m(first cycle)
                 {s}_momentum            m(t) - trailing mean over momentum_long
      baseline   {s}_delta_base          m(t) - mean of the first baseline_cycles cycles
                                         (fewer if not yet observed)
      expanding  {s}_expmax, {s}_expmin  running max / min of m up to t (degradation envelope)
    Returns identifier columns + features in the original row order
    (and, with return_meta, a DataFrame [feature, group, sensor]).
    """
    if config is None:
        w = tuple(sorted(set(windows)))
        config = FeatureConfig(mean_windows=w, std_windows=w, slope_windows=w, smooth_window=w[0],
                               baseline_cycles=baseline_cycles)
    groups = set(config.groups)
    group_cols = list(group_cols)
    sensors = list(sensors)
    data = df.sort_values(group_cols + ["cycle"], kind="mergesort")
    t = data["cycle"].astype(float)
    x = data[sensors].astype(float)
    n_obs = data.groupby(group_cols, sort=False).cumcount().add(1).astype(float)

    feats: Dict[str, pd.Series] = {}
    meta: List[Tuple[str, str, str]] = []

    def put(name, values, group, sensor):
        feats[name] = values
        meta.append((name, group, sensor))

    if "current" in groups:
        for s in sensors:
            put(s, x[s], "current", s)

    needed = set()
    if "mean" in groups:
        needed |= set(config.mean_windows)
    if "std" in groups:
        needed |= set(config.std_windows)
    if "slope" in groups:
        needed |= set(config.slope_windows)
    if groups & {"delta", "baseline", "expanding"}:
        needed.add(config.smooth_window)
    if "delta" in groups:
        needed.add(config.momentum_long)

    helper = pd.concat([data[group_cols], x, x.mul(t, axis=0).add_suffix("__tx"), x.pow(2).add_suffix("__xx"),
                        t.rename("__t"), (t ** 2).rename("__tt")], axis=1)
    roll_cols = [c for c in helper.columns if c not in group_cols]
    means: Dict[int, pd.DataFrame] = {}
    for w in sorted(needed):
        m = _grouped_rolling_mean(helper, group_cols, roll_cols, w)
        means[w] = m[sensors]
        n = n_obs.clip(upper=w)
        var_t = m["__tt"] - m["__t"] ** 2
        for s in sensors:
            if "mean" in groups and w in config.mean_windows:
                put(f"{s}_mean{w}", m[s], "mean", s)
            if "std" in groups and w in config.std_windows:
                # population variance -> sample std (ddof=1); 0 when n == 1
                var_x = (m[f"{s}__xx"] - m[s] ** 2).clip(lower=0) * n / (n - 1).where(n > 1, np.nan)
                put(f"{s}_std{w}", np.sqrt(var_x).fillna(0.0), "std", s)
            if "slope" in groups and w in config.slope_windows:
                cov_tx = m[f"{s}__tx"] - m["__t"] * m[s]
                put(f"{s}_slope{w}", (cov_tx / var_t.where(var_t > 1e-9, np.nan)).fillna(0.0), "slope", s)
    del helper

    if "minmax" in groups:
        xg = pd.concat([data[group_cols], x], axis=1)
        for w in config.minmax_windows:
            lo = _grouped_rolling(xg, group_cols, sensors, w, "min")
            hi = _grouped_rolling(xg, group_cols, sensors, w, "max")
            for s in sensors:
                put(f"{s}_min{w}", lo[s], "minmax", s)
                put(f"{s}_max{w}", hi[s], "minmax", s)

    if groups & {"delta", "baseline", "expanding"}:
        m0 = means[config.smooth_window]
        keys = [data[c] for c in group_cols]
        if "delta" in groups:
            first = m0.groupby(keys, sort=False).transform("first")  # value at the group's first cycle
            for k in config.delta_lags:
                lagged = m0.groupby(keys, sort=False).shift(k).fillna(first)
                for s in sensors:
                    put(f"{s}_diff{k}", m0[s] - lagged[s], "delta", s)
            for s in sensors:
                put(f"{s}_momentum", m0[s] - means[config.momentum_long][s], "delta", s)
        if "baseline" in groups:
            early = x.where(n_obs <= config.baseline_cycles, 0.0)
            base = early.groupby(keys, sort=False).cumsum().div(n_obs.clip(upper=config.baseline_cycles), axis=0)
            for s in sensors:
                put(f"{s}_delta_base", m0[s] - base[s], "baseline", s)
        if "expanding" in groups:
            cmax = m0.groupby(keys, sort=False).cummax()
            cmin = m0.groupby(keys, sort=False).cummin()
            for s in sensors:
                put(f"{s}_expmax", cmax[s], "expanding", s)
                put(f"{s}_expmin", cmin[s], "expanding", s)

    id_cols = [c for c in data.columns if c in group_cols or c in ("unit_id", "cycle", "op_condition", RUL_COLUMN, "sample_id")]
    out = pd.concat([data[id_cols], pd.DataFrame(feats, index=data.index)], axis=1)
    out = out.loc[:, ~out.columns.duplicated()].reindex(df.index)
    if return_meta:
        return out, pd.DataFrame(meta, columns=["feature", "group", "sensor"])
    return out


def training_rul(train_df: pd.DataFrame) -> np.ndarray:
    """
    The ONLY label access in this module: uncapped RUL of run-to-failure training
    rows, rebuilt from each engine's last cycle (any RUL column in the input is
    ignored). Used solely to fit the health-index regression in FoldPreprocessor.fit.
    """
    return (train_df.groupby("unit_id")["cycle"].transform("max") - train_df["cycle"]).to_numpy()


class FoldPreprocessor:
    """
    Leakage-safe preprocessing + feature engineering for one training fold (or
    for the final train/test):
      fit(train_df):  condition-normalisation stats, the optional health-index
                      regression, and the feature scaler - from training rows only;
      transform(df):  applies the stored statistics; never refits.
      fit_transform:  fit + features of the TRAINING rows. With the health index on,
                      training rows get cross-fitted (out-of-fold) HI values: each
                      training engine's HI comes from a ridge fitted on other training
                      engines only, so no row's HI was fitted on its own label. New data
                      (validation / test) uses the ridge fitted on all training engines.
                      Use fit_transform, not transform, for the training rows.

    `group_cols` identifies one trajectory: ("unit_id",) for full engines, or
    ("sample_id",) for truncated validation/test samples.
    """

    ID_COLS = ("unit_id", "cycle", "sample_id", "op_condition", RUL_COLUMN)

    def __init__(
        self,
        sensors: Sequence[str] = CANDIDATE_SENSORS,
        windows: Iterable[int] = DEFAULT_WINDOWS,
        baseline_cycles: int = DEFAULT_BASELINE_CYCLES,
        include_cycle: bool = False,
        scale: bool = True,
        config: Optional[FeatureConfig] = None,
    ):
        if config is None:
            w = tuple(sorted(set(windows)))
            config = FeatureConfig(mean_windows=w, std_windows=w, slope_windows=w, smooth_window=w[0],
                                   baseline_cycles=baseline_cycles, sensors=tuple(sensors), include_cycle=include_cycle)
        self.config = config
        self.sensors = config.all_sensors
        self.scale = scale

    # -- internal -----------------------------------------------------------
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        base = add_derived_sensors(df) if cfg.derived else df
        if cfg.normalize:
            return self.normalizer_.transform(base)
        out = base.copy()
        out["op_condition"] = assign_operating_condition(base)
        return out

    def _hi_inputs(self, prepared: pd.DataFrame, group_cols) -> pd.DataFrame:
        cfg = self.config
        hi_cfg = FeatureConfig(groups=("mean",), mean_windows=(cfg.hi_window,), sensors=cfg.sensors)
        f = add_causal_features(prepared, list(cfg.sensors), group_cols=group_cols, config=hi_cfg)
        return f[[f"{s}_mean{cfg.hi_window}" for s in cfg.sensors]]

    def _features(self, df: pd.DataFrame, group_cols: Sequence[str], hi_values: Optional[pd.Series] = None) -> pd.DataFrame:
        cfg = self.config
        prepared = self._prepare(df)
        feats, meta = add_causal_features(prepared, self.sensors, group_cols=group_cols, config=cfg, return_meta=True)
        extra = []
        if cfg.health_index:
            hi_frame = prepared[[c for c in prepared.columns if c in list(group_cols) + ["unit_id", "cycle"]]].copy()
            if hi_values is not None:  # cross-fitted values for the training rows
                hi_frame["hi"] = hi_values.reindex(df.index).to_numpy()
            else:
                z = self._hi_inputs(prepared, group_cols)
                hi_frame["hi"] = z.to_numpy() @ self.hi_coef_ + self.hi_intercept_
            hi_cfg = FeatureConfig(groups=("current", "slope", "expanding"), slope_windows=(10, 30), smooth_window=5,
                                   sensors=("hi",))
            hf, hmeta = add_causal_features(hi_frame, ["hi"], group_cols=group_cols, config=hi_cfg, return_meta=True)
            hcols = list(hmeta.feature)
            feats = pd.concat([feats, hf[hcols]], axis=1)
            extra.append(hmeta.assign(group="health_index"))
        if not cfg.normalize:
            for k in range(6):
                name = f"cond_{k}"
                feats[name] = (prepared["op_condition"] == k).astype(float)
                extra.append(pd.DataFrame([(name, "condition", "op_condition")], columns=meta.columns))
        if cfg.include_cycle:
            feats["cycle_feature"] = feats["cycle"].astype(float)
            extra.append(pd.DataFrame([("cycle_feature", "cycle", "cycle")], columns=meta.columns))
        self._last_meta = pd.concat([meta] + extra, ignore_index=True) if extra else meta
        return feats

    # -- public -------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, group_cols: Sequence[str] = ("unit_id",)) -> "FoldPreprocessor":
        cfg = self.config
        base = add_derived_sensors(train_df) if cfg.derived else train_df
        self.normalizer_ = ConditionNormalizer(self.sensors).fit(base)
        if cfg.health_index:
            # Linear health index: smoothed sensors -> min(RUL, hi_cap) / hi_cap, fitted on training engines only.
            from sklearn.linear_model import Ridge

            prepared = self._prepare(train_df)
            z = self._hi_inputs(prepared, group_cols).to_numpy()
            target = np.minimum(training_rul(train_df), cfg.hi_cap) / cfg.hi_cap
            reg = Ridge(alpha=1.0).fit(z, target)
            self.hi_coef_, self.hi_intercept_ = reg.coef_, float(reg.intercept_)
            # Out-of-fold HI for the training rows: engines split into groups; each group's
            # HI comes from a ridge fitted on the other groups' engines and labels.
            units = np.sort(train_df["unit_id"].unique())
            k = min(cfg.hi_crossfit_folds, len(units))
            oof = z @ self.hi_coef_ + self.hi_intercept_
            if k >= 2:
                group_of = dict(zip(np.random.default_rng(0).permutation(units), np.arange(len(units)) % k))
                g = train_df["unit_id"].map(group_of).to_numpy()
                for j in range(k):
                    held = g == j
                    r = Ridge(alpha=1.0).fit(z[~held], target[~held])
                    oof[held] = z[held] @ r.coef_ + r.intercept_
            self.train_hi_ = pd.Series(oof, index=train_df.index)
        feats = self._features(train_df, group_cols, getattr(self, "train_hi_", None))
        self.feature_meta_ = self._last_meta.reset_index(drop=True)
        self.feature_names_ = list(self.feature_meta_.feature)
        values = feats[self.feature_names_]
        self.mean_ = values.mean() if self.scale else pd.Series(0.0, index=self.feature_names_)
        std = values.std() if self.scale else pd.Series(1.0, index=self.feature_names_)
        self.std_ = std.where(std > 1e-8, 1.0)  # constant features stay 0 after centring
        self.fit_units_ = frozenset(train_df["unit_id"].unique())
        return self

    def transform(self, df: pd.DataFrame, group_cols: Sequence[str] = ("unit_id",)) -> pd.DataFrame:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("FoldPreprocessor must be fitted before transform")
        feats = self._features(df, group_cols)
        feats[self.feature_names_] = ((feats[self.feature_names_] - self.mean_) / self.std_).astype(np.float32)
        return feats

    def fit_transform(self, train_df: pd.DataFrame, group_cols: Sequence[str] = ("unit_id",)) -> pd.DataFrame:
        self.fit(train_df, group_cols)
        feats = self._features(train_df, group_cols, getattr(self, "train_hi_", None))
        feats[self.feature_names_] = ((feats[self.feature_names_] - self.mean_) / self.std_).astype(np.float32)
        return feats
