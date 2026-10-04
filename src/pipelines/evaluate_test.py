"""
ONE-SHOT official test evaluation of the frozen final models.

Order of operations (recorded in reports/final_evaluation/evaluation_log.json):
  1. verify the frozen configuration and the model artifacts (SHA-256)
  2. predict every official test engine from its sensor file only
  3. save + hash the predictions BEFORE any RUL file is opened
  4. read RUL_FD00x.txt and score (RMSE, MAE, bias, PHM08) per subset and overall

Refuses to run if results already exist: official test results must not drive iteration.

Usage: python -m src.pipelines.evaluate_test
"""

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from src.data_schema import SUBSETS
from src.pipelines.prediction_pipeline import MODEL_DIR, RULPredictor
from src.pipelines.training_pipeline import RAW_DIR, sha256_file, verify_config_frozen
from src.utils import read_rul_file, rul_metrics

OUT_DIR = os.path.join("reports", "final_evaluation")


def main():
    log_path = os.path.join(OUT_DIR, "evaluation_log.json")
    if os.path.exists(log_path):
        raise RuntimeError(f"{log_path} exists: the official evaluation has already been run once")
    os.makedirs(OUT_DIR, exist_ok=True)
    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    cfg = verify_config_frozen()
    manifest = json.load(open(os.path.join(MODEL_DIR, "manifest.json"), encoding="utf-8"))
    assert manifest["config_body_sha256"] == cfg["frozen"]["config_body_sha256"]
    log["config_body_sha256"] = cfg["frozen"]["config_body_sha256"]
    log["artifact_sha256"] = {s: v["artifact_sha256"] for s, v in manifest["subsets"].items()}

    # -- 2-3. predictions from sensor data only ---------------------------------------
    preds = []
    for s in SUBSETS:
        p = RULPredictor.load(s)  # hash-verified
        test_path = os.path.join(RAW_DIR, f"test_{s}.txt")
        out = p.predict_file(test_path).assign(subset=s, model=p.metadata["family"], rul_cap=p.metadata["rul_cap"])
        log.setdefault("test_file_sha256", {})[s] = sha256_file(test_path)
        preds.append(out)
    preds = pd.concat(preds, ignore_index=True)[["subset", "unit_id", "last_cycle", "predicted_rul", "model", "rul_cap"]]
    pred_path = os.path.join(OUT_DIR, "test_predictions_before_labels.csv")
    preds.to_csv(pred_path, index=False)
    log["predictions_sha256"] = sha256_file(pred_path)
    log["predictions_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # -- 4. labels, used only for scoring --------------------------------------------
    scored, rows = [], []
    for s in SUBSETS:
        truth = read_rul_file(s, RAW_DIR).rename(columns={"RUL": "true_rul"})
        d = preds[preds.subset == s].merge(truth, on="unit_id", how="left", validate="one_to_one")
        assert d.true_rul.notna().all() and len(d) == len(truth)
        scored.append(d)
        cap = d.rul_cap.iloc[0]
        rows.append({"subset": s, "model": d.model.iloc[0], "rul_cap": cap} | rul_metrics(d.true_rul, d.predicted_rul)
                    | {"rmse_vs_capped_truth (secondary)": float(np.sqrt(np.mean((d.predicted_rul - np.minimum(d.true_rul, cap)) ** 2)))})
    scored = pd.concat(scored, ignore_index=True)
    overall = {"subset": "ALL (pooled, 707 engines)", "model": "per-subset", "rul_cap": np.nan} | rul_metrics(scored.true_rul, scored.predicted_rul)
    overall["rmse_vs_capped_truth (secondary)"] = float(np.sqrt(np.mean((scored.predicted_rul - np.minimum(scored.true_rul, scored.rul_cap)) ** 2)))
    metrics = pd.DataFrame(rows + [overall])
    metrics.loc[len(metrics)] = {"subset": "mean of subset RMSEs", "rmse": float(np.mean([r["rmse"] for r in rows]))}
    scored.to_csv(os.path.join(OUT_DIR, "test_predictions_scored.csv"), index=False)
    metrics.to_csv(os.path.join(OUT_DIR, "test_metrics.csv"), index=False)
    log["labels_read"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log["rul_file_sha256"] = {s: sha256_file(os.path.join(RAW_DIR, f"RUL_{s}.txt")) for s in SUBSETS}
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    pd.set_option("display.width", 200)
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
