# FINAL_CONFIG - frozen classical-ML configuration (classical-v1.0)

**Frozen:** 2026-10-04 21:16:41. This is before training the final models and before any use of the official test set.

Integrity hashes:
- `configs/final_config.json`, body SHA-256: `949c7370903dfce901c1ccebe1d850ee0a2c393bc7bbcf0dd08222cbc189e4d4`
- `configs/final_feature_list.json`, SHA-256: `3d5d10589098773934b91e5d14e9ca98576170e1d663622428af12d1104c6524`

Every final step calls `verify_config_frozen()` and refuses to run if either file has changed. The official test set must **not** be used to change anything below.

## 1. Selected feature set
**R2_F7 +cycle (minimal)**, with 290 features (feature-list version `classical-v1.0`, file `configs/final_feature_list.json`).

| Group | Definition (uses cycles ≤ t of the same engine only) | Windows |
|---|---|---|
| current | condition-normalised sensor value | - |
| mean | trailing mean | 5, 10, 20, 30 |
| std | trailing sample std | 10, 30 |
| minmax | trailing min and max | 10, 30 |
| slope | trailing least-squares slope vs cycle | 10, 20, 30 |
| delta | m(t) − m(t−10); m(t) − trailing mean over 30 (m = 5-cycle trailing mean) | - |
| baseline | m(t) − mean of the engine's first 10 cycles | - |
| cycle | cycle number | - |

- **Sensors (17):** T24, T30, T50, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, W31, W32.
- **Not used:** derived sensors, the health index, and the expanding (running max/min) group.
- **Per-sensor features:** 17 × 17 = 289, plus `cycle_feature`, gives 290.

## 2-3. Selected model and RUL cap per subset
| Subset | Model | RUL cap | Hyper-parameters |
|---|---|---|---|
| FD001 | xgboost | 130 | `{"colsample_bytree": 0.3, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 1, "n_estimators": 500, "reg_lambda": 1.0, "subsample": 0.85}` |
| FD002 | xgboost | 130 | `{"colsample_bytree": 0.5, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 10, "n_estimators": 500, "reg_lambda": 1.0, "subsample": 0.7}` |
| FD003 | lightgbm | 140 | `{"colsample_bytree": 0.5, "learning_rate": 0.05, "min_child_samples": 50, "n_estimators": 400, "num_leaves": 31, "reg_lambda": 1.0, "subsample": 0.8}` |
| FD004 | lightgbm | 140 | `{"colsample_bytree": 0.5, "learning_rate": 0.05, "min_child_samples": 50, "n_estimators": 500, "num_leaves": 31, "reg_lambda": 10.0, "subsample": 0.85}` |

The model was selected per subset as the lowest primary-plan CV RMSE among the five tuned families. The cap was selected per model by the 1-SE rule. XGBoost and LightGBM are statistically tied in every subset; the choice follows the pre-set rule.

## 4. Preprocessing
- **input columns**: unit_id, cycle, op_setting_1..3, 21 Table-2 sensors (src/data_schema.py)
- **operating condition**: settings rounded to (0, 2, 0) decimals -> 6 known centres (stateless)
- **normalisation**: per-condition z-score of the 17 candidate sensors, statistics from training engines only
- **excluded sensors**: T2, P2, Nf_dmd, PCNfR_dmd
- **feature scaling**: per-feature standardisation with training statistics (irrelevant for trees, kept for parity with CV)
- **causality**: every feature at cycle t uses only cycles <= t of the same engine

Implementation: `FoldPreprocessor(config=FINAL_FEATURE_CONFIG)` in `src/pipelines/training_pipeline.py`.

## 5. Training protocol
- **Target:** RUL = last cycle of the training engine - current cycle (failure row = 0). Training target: min(RUL, rul_cap). Evaluation target: uncapped true RUL.
- **data**: all rows (every cycle) of ALL training engines of the subset; one model per subset
- **prediction**: one prediction per engine at its last observed cycle, clipped at >= 0
- **seed**: 42
- **n jobs**: 4
- **deterministic**: fixed seeds; LightGBM deterministic=True; XGBoost 'hist' results depend on the thread count, so models are fitted with exactly n_jobs threads (the setting used in validation)
- **Command:** `python -m src.pipelines.training_pipeline --step train`. Artifacts go to `models/final/<subset>/` (`preprocessor.joblib`, `model.joblib`, `metadata.json`), with SHA-256 hashes in `models/final/manifest.json`.

## 6. Validation protocol (used for every selection above)
- **folds**: 5 engine-level folds per subset, lifespan-stratified, seed 42
- **validation samples**: 10 truncated samples per engine, strategy=uniform, true RUL at cut in [6, 150]
- **confirmation plan**: same, seed 2024
- **wide plan**: RUL at cut in [6, 190]
- **cap rule**: lowest CV RMSE; tie-break largest cap within 1 SE
- **documents**: reports/target_and_validation_protocol.md, reports/classical_ml_stage_report.md

**Reproduction check:** `--step verify_cv` re-runs the frozen pipeline through the primary CV plan. It reproduces the validated RMSE exactly in all four subsets (`models/final/cv_reproduction_check.csv`).

| CV RMSE (validated) | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| primary | 13.58 | 14.37 | 14.00 | 15.85 |
| confirm | 13.38 | 14.22 | 12.69 | 16.17 |
| wide | 22.70 | 23.11 | 21.41 | 22.71 |
| PHM08 per sample (primary) | 2.93 | 3.67 | 3.55 | 6.18 |

## 7. Official test set
- **Never used for selection.** It is read only by `src/pipelines/evaluate_test.py`, after the models are trained and their hashes recorded.
- **RUL files are used only to score predictions.**
- **Test results do not feed back** into this configuration.
