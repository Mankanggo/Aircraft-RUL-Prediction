"""
Offline inference with the frozen final models: raw C-MAPSS sensor rows -> RUL.

    predictor = RULPredictor.load("FD001")
    preds = predictor.predict_file("data/raw/test_FD001.txt")   # one RUL per engine
    preds = predictor.predict_engines(df)                       # df with the 26 raw columns

For each engine the prediction is made at its LAST observed cycle, from that engine's
history only (causal features). Artifacts are verified against the SHA-256 hashes in
models/final/manifest.json before they are loaded.
"""

import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd

from src.data_schema import ID_COLUMNS, RAW_COLUMNS
from src.utils import assign_operating_condition, read_raw_file

MODEL_DIR = os.path.join("models", "final")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class InputValidationError(ValueError):
    pass


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Checks raw sensor rows before inference; returns a clean copy."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise InputValidationError(f"missing columns: {missing}")
    out = df[RAW_COLUMNS].copy()
    bad = out.isna().sum()
    if bad.any():
        raise InputValidationError(f"missing or non-numeric values in: {bad[bad > 0].to_dict()}")
    if not np.isfinite(out.drop(columns=ID_COLUMNS).to_numpy(float)).all():
        raise InputValidationError("infinite sensor values")
    if out.duplicated(ID_COLUMNS).any():
        raise InputValidationError("duplicate (unit_id, cycle) rows")
    if (out["cycle"] < 1).any():
        raise InputValidationError("cycles must start at 1")
    gaps = out.sort_values(ID_COLUMNS).groupby("unit_id")["cycle"].agg(lambda c: (c.diff().dropna() != 1).any() or c.min() != 1)
    if gaps.any():
        raise InputValidationError(f"engines with non-contiguous cycles or not starting at cycle 1: {list(gaps[gaps].index[:10])}")
    assign_operating_condition(out)  # raises for settings outside the six known operating conditions
    return out


class RULPredictor:
    def __init__(self, subset: str, preprocessor, model, metadata: dict):
        self.subset, self.preprocessor, self.model, self.metadata = subset, preprocessor, model, metadata

    @classmethod
    def load(cls, subset: str, model_dir: str = MODEL_DIR, verify: bool = True) -> "RULPredictor":
        folder = os.path.join(model_dir, subset)
        if verify:
            with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as fh:
                expected = json.load(fh)["subsets"][subset]["artifact_sha256"]
            for name, digest in expected.items():
                if _sha256(os.path.join(folder, name)) != digest:
                    raise RuntimeError(f"{subset}/{name} does not match the hash recorded at training time")
        with open(os.path.join(folder, "metadata.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(subset, joblib.load(os.path.join(folder, "preprocessor.joblib")),
                   joblib.load(os.path.join(folder, "model.joblib")), meta)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Causal features of every row (validated input)."""
        return self.preprocessor.transform(validate_input(df), group_cols=("unit_id",))

    def predict_engines(self, df: pd.DataFrame) -> pd.DataFrame:
        """One prediction per engine at its last observed cycle."""
        feats = self.features(df)
        last = feats.loc[feats.groupby("unit_id")["cycle"].idxmax()].sort_values("unit_id")
        X = last[self.preprocessor.feature_names_].to_numpy(np.float32)
        pred = np.maximum(np.asarray(self.model.predict(X), dtype=float), 0.0)
        return pd.DataFrame({"unit_id": last["unit_id"].astype(int).to_numpy(), "last_cycle": last["cycle"].astype(int).to_numpy(),
                             "predicted_rul": pred})

    def predict_file(self, path: str) -> pd.DataFrame:
        return self.predict_engines(read_raw_file(path))
