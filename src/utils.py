import os
from typing import Optional

import numpy as np
import pandas as pd

from src.data_schema import (
    ID_COLUMNS,
    OPERATING_CONDITIONS,
    RAW_COLUMNS,
    RUL_COLUMN,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    SETTING_ROUNDING,
    SUBSETS,
)

DEFAULT_RAW_DIR = os.path.join("data", "raw")


def read_cmapss_file(subset: str, split: str, raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Read one raw train/test file (read-only) and apply the verified column names.

    Values are read as strings first and converted with errors="coerce", so any
    non-numeric token surfaces as NaN for the data-quality checks instead of
    raising or silently changing dtype.
    """
    if subset not in SUBSETS or split not in ("train", "test"):
        raise ValueError(f"Unknown subset/split: {subset}/{split}")
    return read_raw_file(os.path.join(raw_dir, f"{split}_{subset}.txt"))


def read_raw_file(path) -> pd.DataFrame:
    """Read any whitespace-separated 26-column C-MAPSS-format file (read-only)."""
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, engine="python")
    if raw.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"{path}: expected {len(RAW_COLUMNS)} columns, found {raw.shape[1]}")

    raw.columns = RAW_COLUMNS
    df = raw.apply(pd.to_numeric, errors="coerce")
    df[ID_COLUMNS] = df[ID_COLUMNS].astype("Int64")
    return df


def read_rul_file(subset: str, raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Read RUL_FDxxx.txt: one true RUL per test unit, in unit order (row i -> unit i+1)."""
    path = os.path.join(raw_dir, f"RUL_{subset}.txt")
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=[RUL_COLUMN], dtype=str, engine="python")
    rul[RUL_COLUMN] = pd.to_numeric(rul[RUL_COLUMN], errors="coerce")
    rul.insert(0, "unit_id", np.arange(1, len(rul) + 1))
    return rul


def cap_rul(rul, cap: Optional[float] = None):
    """
    Piecewise-linear RUL target: min(RUL, cap). cap=None returns RUL unchanged.
    Works on scalars, numpy arrays and pandas Series.
    """
    if cap is None:
        return rul
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    return np.minimum(rul, cap)


def add_train_rul(df: pd.DataFrame, cap: Optional[float] = None) -> pd.DataFrame:
    """
    Training-set RUL = last observed cycle of the unit - current cycle.

    Training trajectories run to failure, so the last row (the failure cycle)
    has RUL 0, matching the test convention "remaining cycles after the last
    observed cycle". Optionally capped (piecewise-linear target, see cap_rul).
    Computed per unit, so it never uses information from other units. The
    target uses the unit's final cycle (future information) by definition:
    it is a label only and must never be used to build features.
    """
    out = df.copy()
    last = out.groupby("unit_id")["cycle"].transform("max")
    out[RUL_COLUMN] = cap_rul((last - out["cycle"]).astype(int), cap)
    return out


def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test-set RUL per row = true RUL at the last observed cycle (from RUL file)
    + cycles remaining until that last observed cycle. For evaluation/EDA only;
    never a model input.
    """
    out = test_df.copy()
    last = out.groupby("unit_id")["cycle"].transform("max")
    end_rul = out["unit_id"].astype(int).map(rul_df.set_index("unit_id")[RUL_COLUMN])
    out[RUL_COLUMN] = (end_rul + last - out["cycle"]).astype(int)
    return out


def assign_operating_condition(df: pd.DataFrame) -> pd.Series:
    """
    Map each row to an operating-condition id (0-5, index into
    OPERATING_CONDITIONS) by rounding the settings. Deterministic and
    stateless, so it is safe to apply identically to train and test.
    Raises if a row does not match a known centre.
    """
    rounded = df[SETTING_COLUMNS].round(SETTING_ROUNDING) + 0.0  # + 0.0 drops -0.0
    lookup = {c: i for i, c in enumerate(OPERATING_CONDITIONS)}
    cond = pd.Series([lookup.get(tuple(r)) for r in rounded.itertuples(index=False)], index=df.index)
    if cond.isna().any():
        raise ValueError(f"{cond.isna().sum()} rows do not match a known operating condition")
    return cond.astype(int).rename("op_condition")


def fit_condition_stats(train_df: pd.DataFrame, columns=SENSOR_COLUMNS) -> pd.DataFrame:
    """Per-condition mean/std of each sensor, fitted on TRAINING rows only."""
    cond = assign_operating_condition(train_df)
    return train_df[columns].groupby(cond).agg(["mean", "std"])


def apply_condition_norm(df: pd.DataFrame, stats: pd.DataFrame, columns=SENSOR_COLUMNS, eps: float = 1e-8) -> pd.DataFrame:
    """
    z-score each sensor within its operating condition using train-fitted stats.
    Sensors with ~zero within-condition std become 0 (they carry no
    within-condition information).
    """
    out = df.copy()
    cond = assign_operating_condition(df)
    unseen = set(cond.unique()) - set(stats.index)
    if unseen:
        raise ValueError(f"operating condition(s) {sorted(unseen)} not present in the fitted statistics")
    for c in columns:
        mean = cond.map(stats[(c, "mean")])
        std = cond.map(stats[(c, "std")]).fillna(0.0)
        out[c] = np.where(std > eps, (df[c] - mean) / std.where(std > eps, 1.0), 0.0)
    return out


def phm08_score(y_true, y_pred) -> float:
    """
    PHM08 asymmetric score (paper eq. 11): d = predicted - true;
    exp(-d/13) - 1 for early (d < 0), exp(d/10) - 1 for late (d >= 0).
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def rul_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, mean error (bias, + = late), PHM08 score (sum and per sample)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    score = phm08_score(y_true, y_pred)
    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "phm08_score": score,
        "phm08_score_per_sample": score / len(y_true),
    }
