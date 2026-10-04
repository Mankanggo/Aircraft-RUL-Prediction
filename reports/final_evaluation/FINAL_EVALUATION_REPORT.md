# Final offline model: official test evaluation (classical-v1.0)

Sources:
- configuration: `FINAL_CONFIG.md`
- evaluation: `src/pipelines/evaluate_test.py`
- analysis: `notebooks/04_final_evaluation.ipynb`
- tables and figures: `reports/final_evaluation/{tables,figures}/`

## 1. Freeze and evaluation integrity
| Step | Time (2026-10-04) | Evidence |
|---|---|---|
| Configuration frozen | 21:16:41 | `configs/final_config.json`, body SHA-256 `949c7370…`; feature list SHA-256 `3d5d1058…` |
| Frozen pipeline reproduces the validated CV RMSE exactly (4/4 subsets) | before training | `models/final/cv_reproduction_check.csv` |
| Final models trained on all training engines | 21:20:55 | `models/final/manifest.json` (artifact SHA-256 hashes) |
| Test predictions from sensor files only, saved and hashed | 21:25:59 | `test_predictions_before_labels.csv` |
| RUL files read, for scoring only | 21:25:59 | `evaluation_log.json` |

- **One-shot.** The evaluation ran **once**, and the script refuses to run again.
- **No feedback.** No configuration, model, feature, cap or preprocessing choice was changed after the test results.
- **Reproducible.** Re-predicting all 707 test engines from the raw files with freshly loaded, hash-verified artifacts reproduces the saved predictions (max difference 3 × 10⁻¹⁴).

**Reproducibility fix found before the test evaluation.** XGBoost's `hist` training is not bit-identical across thread counts. The frozen pipeline reproduced the validated XGBoost CV scores only with the validation thread count (4), so `n_jobs: 4` was added to the frozen config and enforced by `verify_config_frozen()`. Models, features and caps were unchanged. This happened before any test data was read.

## 2. Official test results (one prediction per engine, at its last observed cycle)
| Subset | Model (cap) | Engines | RMSE | MAE | Bias | PHM08 (sum) | PHM08 / engine |
|---|---|---|---|---|---|---|---|
| FD001 | XGBoost (130) | 100 | **12.63** | 9.22 | −0.19 | 227.1 | 2.27 |
| FD002 | XGBoost (130) | 259 | **22.52** | 14.32 | −5.85 | 4342.4 | 16.77 |
| FD003 | LightGBM (140) | 100 | **12.66** | 8.67 | +3.07 | 391.9 | 3.92 |
| FD004 | LightGBM (140) | 248 | **20.70** | 14.47 | −3.67 | 2199.8 | 8.87 |
| **All, pooled** | per subset | 707 | **19.53** | 12.85 | −3.02 | 7161.2 | 10.13 |

- Mean of the four subset RMSEs: 17.13.
- Secondary metric, for literature comparability only (RMSE against the capped truth): 12.09 / 11.77 / 12.47 / 13.71; all 12.62.

### Test vs. validation expectations
| RMSE | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| CV primary (RUL ≤ 150) | 13.58 | 14.37 | 14.00 | 15.85 |
| CV confirmation | 13.38 | 14.22 | 12.69 | 16.17 |
| CV wide (RUL ≤ 190) | 22.70 | 23.11 | 21.41 | 22.71 |
| **Test, all engines** | **12.63** | **22.52** | **12.66** | **20.70** |
| Test, true RUL ≤ 150 | 12.63 | 13.12 | 12.66 | 14.31 |
| Share of test engines with true RUL > cap | 8% | 21% | 4% | 21% |

- **FD001/FD003:** the test results match or beat the CV estimates.
- **FD002/FD004:** within the RUL range the primary plan covers (≤ 150), test RMSE (13.1 / 14.3) is again at or below CV. The full-test RMSE matches the **wide-plan** CV estimate.
- **What drove the gap:** the high-RUL engines in FD002/FD004 test, not model degradation or leakage.

## 3. Final error analysis
- **Cap-limited engines (true RUL > cap):**
  - FD002: 55 engines, **78% of squared error** and 86% of PHM08.
  - FD004: 52 engines, **64% of squared error** and 54% of PHM08.
  - All are predicted early: bias −38 and −33; worst case FD002 unit 166, true 188 vs predicted 98.5.
  - This is structural: a model capped at 130–140 cannot predict above it. Early errors are penalised less per cycle by PHM08, but these are large.
- **Late predictions at true RUL 51–100:**
  - Bias is +5.7 to +9.0 cycles in every subset, and 11–29 engines per subset are more than 10 cycles late.
  - This band holds most of the PHM08 score in FD001/FD003 (49% and 79%).
  - The same pattern was seen in CV.
- **Short histories (≤ 60 observed cycles):** RMSE is 25–32 in FD002/FD004. These engines combine little evidence with large true RULs.
- **Lifespan-dependent bias is confirmed on test:**
  - Engines with short implied lifespans are predicted **+7 to +10 cycles late**, even restricted to RUL ≤ cap. corr(error, lifespan) is −0.31 to −0.55.
  - 13–30 short-lived engines per subset are more than 10 cycles late.
  - Worst late case: FD004 unit 166 (observed 63 cycles, true 72, predicted 127).
- **Accurate near failure:** RMSE is 2.5–6.0 for true RUL ≤ 25.

## 4. Explainability (exact tree SHAP; additivity error ≤ 1.5 × 10⁻⁴ cycles)
- **By feature group:** change from early-life baseline dominates (mean |SHAP| ≈ 40 cycles per engine). It is followed by trailing mean (11.5), min/max (10), cycle (6.1), slope (5.5), delta (3.7) and std (2.2). The raw current value contributes about 0.
- **By sensor:** Ps30 (18.1) and T50 (12.1) lead, followed by cycle (6.1), BPR (5.8; 11.0 in FD002), Nc and NRc (about 4.6), htBleed (4.4) and T24 (4.3). epr and farB are about 0.
- **Top features:** `Ps30_delta_base`, `T50_delta_base`, `cycle_feature`, `BPR_delta_base`, `Nc_delta_base`.
- **Consistency:** SHAP agrees with the CV permutation importance. The directions are physically sensible: a rise in Ps30/T50 since baseline lowers the predicted RUL.
- **Failure explanations:**
  - Late fast-degrader case: little Ps30/T50 change since baseline pushes the prediction up by +14 and +7 cycles.
  - High-RUL early case: the prediction stays below the cap regardless of the SHAP terms.

## 5. Inference path (verified)
`read_raw_file` → `validate_input` → `FoldPreprocessor.transform` (frozen statistics) → features at each engine's last observed cycle → model → RUL ≥ 0.

The path lives in `src/pipelines/prediction_pipeline.py` (`RULPredictor.load(subset).predict_file(path)`), and `tests/test_inference.py` (10 tests) checks it:
- the frozen config and the artifacts are consistent;
- a raw text file and the equivalent DataFrame give identical predictions;
- predictions use only each engine's history;
- reloaded artifacts equal a freshly retrained model;
- tampered artifacts are rejected;
- input validation catches missing columns, NaN or non-numeric values, duplicates, non-contiguous cycles and unknown operating conditions;
- a single-condition model rejects multi-condition data.

Full suite: 80 tests pass.

## 6. Saved artifacts (reproducible)
| Path | Content |
|---|---|
| `FINAL_CONFIG.md`, `configs/final_config.json`, `configs/final_feature_list.json` | frozen configuration and the 290-feature list (tracked) |
| `models/final/<FD00x>/preprocessor.joblib`, `model.joblib`, `metadata.json` | fitted preprocessing and model per subset |
| `models/final/manifest.json` | SHA-256 hashes, config hash, library versions |
| `models/final/cv_reproduction_check.csv` | frozen pipeline = validated CV scores |
| `reports/final_evaluation/` | predictions (before labels, and scored), metrics, log, tables, figures |

- **`models/` is gitignored, as in the project template.** The artifacts are regenerated bit-exactly with `python -m src.pipelines.training_pipeline --step train`.
- **Pickles are version-specific:** they need the library versions in the manifest (numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, joblib 1.6.0).
- **Dependencies:** `requirements.txt` now includes xgboost, lightgbm and joblib, which inference needs. SHAP is in `requirements-dev.txt`.

## 7. Limitations of the final classical model
1. **RUL above the cap is unreachable.** This costs about 8–9 RMSE on FD002/FD004 test. The cap was tuned on a validation range (≤ 150) that under-represents these test sets. This was disclosed before evaluation and must not be "fixed" by tuning on test results.
2. **Systematic late predictions** for fast-degrading engines and at true RUL 51–100. This is safety-relevant.
3. **Weak short-history performance** in multi-condition subsets.
