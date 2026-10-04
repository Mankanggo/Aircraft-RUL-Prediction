"""
Generate the API golden fixtures (dev only): for each subset, a few TRUNCATED TRAINING engines
and the predictions of the existing direct RULPredictor for them.

Inference-regression fixtures only: no official test data and no RUL labels of any kind are
stored. Expected values come from RULPredictor.predict_engines on exactly the rows in the request.

    python scripts/make_api_golden_fixture.py
"""

import json
import os
import platform
import sys
from importlib.metadata import version

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_schema import RAW_COLUMNS, SUBSETS  # noqa: E402
from src.pipelines.prediction_pipeline import RULPredictor  # noqa: E402
from src.utils import read_cmapss_file  # noqa: E402

OUT_DIR = os.path.join("tests", "fixtures", "api_golden")
N_ENGINES = 4
SEED = 2026


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = json.load(open(os.path.join("models", "final", "manifest.json"), encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    for subset in SUBSETS:
        train = read_cmapss_file(subset, "train")
        units = rng.choice(np.sort(train.unit_id.unique()), size=N_ENGINES, replace=False)
        parts = []
        for i, unit in enumerate(units):
            engine = train[train.unit_id == unit]
            life = int(engine.cycle.max())
            # first engine: short history (exercises the short-history flag); others: random cut
            cut = 12 if i == 0 else int(rng.integers(31, min(life - 6, 150) + 1))
            parts.append(engine[engine.cycle <= cut])
        rows = np.concatenate([p[RAW_COLUMNS].to_numpy(float) for p in parts])
        import pandas as pd

        df = pd.DataFrame(rows, columns=RAW_COLUMNS)
        df[["unit_id", "cycle"]] = df[["unit_id", "cycle"]].astype(int)
        expected = RULPredictor.load(subset).predict_engines(df)
        request = {"dataset": subset, "engines": [
            {"unit_id": int(u), "cycles": [{c: (int(r[c]) if c == "cycle" else float(r[c])) for c in RAW_COLUMNS[1:]}
                                           for r in g.to_dict("records")]}
            for u, g in df.groupby("unit_id", sort=False)]}
        fixture = {
            "description": "Truncated TRAINING engines (no official test data, no RUL labels) and the direct "
                           "RULPredictor predictions for them; used for API/inference regression tests.",
            "generated_with": {"config_body_sha256": manifest["config_body_sha256"], "python": platform.python_version(),
                               "libraries": {m: version(d) for m, d in [("numpy", "numpy"), ("pandas", "pandas"),
                                                                          ("sklearn", "scikit-learn"), ("xgboost", "xgboost"),
                                                                          ("lightgbm", "lightgbm")]},
                               "model_sha256": manifest["subsets"][subset]["artifact_sha256"]["model.joblib"]},
            "request": request,
            "expected": expected.to_dict(orient="records"),
        }
        path = os.path.join(OUT_DIR, f"{subset}.json")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(fixture, fh, separators=(",", ":"))
        print(f"{path}: {len(units)} engines, {len(df)} rows, last cycles {expected.last_cycle.tolist()}")


if __name__ == "__main__":
    main()
