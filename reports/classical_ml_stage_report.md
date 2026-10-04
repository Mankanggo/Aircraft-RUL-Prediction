# Classical ML stage report: C-MAPSS FD001–FD004

Sources:
- `notebooks/03_features_and_classical_ml.ipynb` (executed, 0 errors)
- raw results: `reports/experiments/classical/`
- tables and figures: `reports/{tables,figures}/classical/`

Protocol:
- 5 engine-level folds per subset;
- 10 truncated validation samples per engine, with true RUL at the cut uniform in [6, 150];
- training-fold-only preprocessing;
- metrics against the **uncapped** true RUL;
- identical samples for every model.

Leakage suite: 70 tests pass, including the cross-fitted health-index tests.

## 1. Feature set
One global set was chosen by the pre-specified rule (lowest mean LightGBM RMSE): **R2 = current, mean, std, min/max, slope, delta and baseline on the 17 base sensors, plus cycle (290 features).** Five-seed mean RMSE (fixed LightGBM, cap 125):

| | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| **R2** (selected) | 14.25 ± 0.05 | **14.75 ± 0.03** | **14.48 ± 0.16** | **15.88 ± 0.05** |
| R1 (+derived, +HI) | **14.17 ± 0.11** | 14.97 ± 0.04 | 14.79 ± 0.21 | 15.93 ± 0.04 |
| F11 (+expanding) | 15.17 ± 0.10 | 15.77 ± 0.09 | 15.81 ± 0.23 | 17.11 ± 0.08 |

- **Per-subset best:** single-seed, R2 is best in FD001–FD003, and R3 in FD004 (15.72 vs 15.78). Across seeds, R1, R2 and R3 are tied within noise in every subset, so R2, the simplest, is the best feature set for each subset.
- **Feature groups that help:** mean, slope, change from baseline, cycle, and per-condition normalisation (FD002/FD004).
- **Feature groups that hurt:** the expanding envelope, and min/max in the forward sequence.
- **Feature groups that are neutral:** std, delta, derived sensors, the health index (once cycle is present), and the current value.

## 2–5. Models, caps and metrics (primary plan)
The cap was chosen per model and subset by the 1-SE rule. In the table, "PHM" is the PHM08 score per sample and "Fold std" is the fold-to-fold RMSE std.

| Subset | Model | Cap | RMSE | Fold std | MAE | Bias | PHM |
|---|---|---|---|---|---|---|---|
| FD001 | **XGBoost** | 130 | **13.58** | 0.43 | 10.13 | −0.45 | **2.93** |
| | LightGBM | 140 | 13.90 | 0.38 | 10.10 | +1.21 | 3.72 |
| | Random Forest | 140 | 15.19 | 0.36 | 11.38 | +1.79 | 4.33 |
| | ElasticNet / Ridge | 140 | 19.42 / 19.56 | 0.89 / 0.99 | 16.06 | +4.8 / +4.4 | 6.98 / 7.16 |
| FD002 | **XGBoost** | 130 | **14.37** | 0.37 | 10.56 | +0.02 | **3.67** |
| | LightGBM | 130 | 14.52 | 0.36 | 10.58 | +0.13 | 3.72 |
| | Random Forest | 130 | 16.21 | 0.52 | 12.15 | +0.56 | 5.07 |
| | ElasticNet / Ridge | 130 | 19.04 / 19.05 | 0.80 / 0.96 | 15.35 | +2.5 / +2.3 | 6.70 / 6.78 |
| FD003 | **LightGBM** | 140 | **14.00** | 1.38 | 9.93 | +0.72 | **3.55** |
| | XGBoost | 150 | 14.47 | 1.53 | 10.40 | +2.35 | 4.19 |
| | Random Forest | 140 | 15.88 | 1.20 | 11.55 | +1.98 | 5.16 |
| | ElasticNet / Ridge | 130 | 20.38 / 20.39 | 1.18 / 1.23 | 16.74 | +3.4 / +3.5 | 8.41 / 8.35 |
| FD004 | **LightGBM** | 140 | **15.85** | 1.89 | 11.53 | +2.12 | **6.18** |
| | XGBoost | 140 | 15.99 | 1.85 | 11.70 | +2.17 | 6.84 |
| | Random Forest | 140 | 17.58 | 2.37 | 13.09 | +2.99 | 7.48 |
| | ElasticNet / Ridge | 130 | 21.02 | 1.77 / 1.80 | 16.99 | +3.2 | 10.38 |

- **Model comparison:**
  - XGBoost vs LightGBM: not significantly different on the primary plan in any subset (paired engine bootstrap).
  - Boosted trees vs others: both beat Random Forest by 1.5–1.9 RMSE and the linear models by 4.7–6.4 (all significant).
- **Caps:** they settle at 130–150 for every model, slightly above the earlier linear-probe optimum of 125.
- **Fold variation:** FD004 fold 0 is much harder than the others (18.5 vs 13.6–16.9).

## 6. Independent confirmation plan (new folds and cuts, seed 2024, no re-tuning)

| RMSE | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Best model, primary → confirm | 13.58 → 13.38 | 14.37 → 14.22 | 14.00 → 12.69 | 15.85 → 16.17 (XGBoost 16.00) |
| Wide plan (RUL ≤ 190) | 22.70 | 23.11 | 21.41 | 22.71 |

- **Selection bias is negligible.** FD001, FD002 and FD004 move by ≤ 0.35. FD003 drops by about 1.0–1.3 for *every* model, including Ridge, which is plan variance from the small subset.
- **The best model flips in FD004.** On the confirmation plan, XGBoost beats LightGBM.
- **The wide plan costs 6–9 RMSE.** This is the price of capping, since RUL above 150 cannot be predicted.

## 7. Seed noise
LightGBM's random subsampling alone moves RMSE with std 0.03–0.23 (largest in FD003). The F11 → R1/R2 gain (≈1 RMSE) is real; R1, R2 and R3 are within noise. The engine bootstrap does not capture this source of noise.

## 8. Feature importance (permutation on validation samples, best models)
- **By group (RMSE increase when permuted):** baseline ≈ 29.6, cycle 6.2, mean 5.3, slope 1.7, min/max 1.6, delta 0.5, std 0.2, current 0.0.
- **By sensor:** Ps30 8.2, cycle 6.2, T50 3.2, NRc 1.5, Nc 1.4, BPR 1.3 (3.1 in FD002), phi 0.6, htBleed 0.5. epr and farB contribute 0.
- **Split gain vs permutation:** gain is spread over min/max (≈37%), mean (≈15%) and baseline (≈15%) features. Min/max are heavily used but substitutable, so removing them costs nothing.

## 9. Error analysis (best model per subset)
- **Short- vs long-lived engines:** short-lived (fast-degrading) engines are over-predicted by +5.4 to +10.2 cycles, and long-lived engines under-predicted by −3.6 to −6.8. The worst engines are all lifespan extremes, at percentiles 0.0–0.1 or 0.9–1.0.
- **Low vs high true RUL:**

  | True RUL | RMSE | Bias | Pattern |
  |---|---|---|---|
  | ≤ 25 | 2.8–6.5 | small | accurate |
  | 51–100 | 12.6–20.5 | **+5 to +8.5** | **late**; 31–46% of samples > 10 cycles late |
  | 126–150 | — | −12 to −18 | saturates at the cap |

- **Fault modes:**
  - FD003: the second mode (phi rises) is easier (11.8 vs 15.5).
  - FD004: the second mode is slightly harder and more late-biased (bias +3.2; PHM08 7.5 vs 5.3 per sample).
- **Worst engines:** RMSE 22–42. For example, FD004 unit 65 (life 351) has bias −41.5 and FD004 unit 114 (life 161) has bias +38.2.

## 10. Lifespan-dependent bias after final selection
**It remains.** For the final models, corr(per-engine bias, lifespan) is −0.43 to −0.75 on the primary plan and −0.50 to −0.78 on the confirmation plan.
- **28–47% of short-lived engines have mean error above +10 cycles (late).**
- It is *not* caused by the cycle feature. The same LightGBM without cycle shows a *stronger* correlation (−0.52 to −0.85), so cycle reduces it.
- The cause is between-engine variation in degradation rate. At a given sensor state, the model predicts the population-average remaining life.

## 11. Remaining weaknesses and failure modes
1. **Late predictions for fast degraders** (sections 9–10). This is the most safety-relevant failure under PHM08.
2. **Late bias in the mid-RUL band (51–100).**
3. **No information above the cap.** True RULs above 140–150 are under-predicted by design; the wide plan is 6–9 RMSE worse. FD002/FD004 test files contain RULs up to 195.
4. **High fold and plan variance** in FD003/FD004 (fold std 1.4–1.9; FD003 plan swing about 1.3).
5. **Search limits:**
   - Boosted trees used 8 random configs at lr ∈ {0.05, 0.1}, with no early stopping.
   - LightGBM's tuned config equals the untuned default in FD001–FD003, so the search space was likely too narrow or coarse.
   - Random Forest's best config sits at the grid edge.
   - Hyperparameters were tuned at cap 125 and the cap tuned afterwards (coordinate-wise).
6. **Linear models plateau at about 19–21 RMSE** regardless of features.
7. **Single-step windows only.** Features see up to 30 cycles, and sequence shape beyond that is summarised only through baseline and cycle. This is a motivation for sequence models.

## 12. Classical configuration to carry forward
- **Data, target and protocol:** unchanged from `reports/target_and_validation_protocol.md`. Plans: seed 42 (primary), seed 2024 (confirmation), and the wide plan (RUL ≤ 190).
- **Features:** `FoldPreprocessor(config=FULL_CONFIG)`, then select `feature_sets()["R2_F7 +cycle (minimal)"]` (290 columns). Equivalently:
  - groups: current, mean (5/10/20/30), std (10/30), min/max (10/30), slope (10/20/30), delta (lag 10, momentum 5 vs 30), baseline (first 10 cycles);
  - 17 candidate sensors, per-condition z-scored on training-fold statistics, plus `cycle`;
  - no derived sensors, no health index, no expanding group.
- **Models (benchmark for the next stage):**

  | Subset | Model | Cap | Hyperparameters |
  |---|---|---|---|
  | FD001 | XGBoost | 130 | lr 0.05, 500 trees, max_depth 3, min_child_weight 1, subsample 0.85, colsample 0.3, λ 1 |
  | FD002 | XGBoost | 130 | lr 0.05, 500 trees, max_depth 4, min_child_weight 10, subsample 0.7, colsample 0.5, λ 1 |
  | FD003 | LightGBM | 140 | lr 0.05, 400 trees, 31 leaves, min_child_samples 50, subsample 0.8, colsample 0.5, λ 1 |
  | FD004 | LightGBM | 140 | lr 0.05, 500 trees, 31 leaves, min_child_samples 50, subsample 0.85, colsample 0.5, λ 10 |

  XGBoost and LightGBM are statistically interchangeable in every subset.
- **Benchmark to beat (best model):**
  - RMSE on the primary / confirmation plans: 13.58 / 13.38, 14.37 / 14.22, 14.00 / 12.69, 15.85 / 16.17.
  - PHM08 per sample (primary): 2.93, 3.67, 3.55, 6.18.

## Official test set
- **Untouched by this stage.** A static scan confirms that the experiment pipeline reads only `read_cmapss_file(subset, "train", ...)`. None of `classical_experiments.py`, `model_trainer.py`, `data_transformation.py` or notebook 03 reads test trajectories or RUL files. The single match in `validation.py` is a docstring.
- **No model has ever been fitted on, predicted on, or selected using test data.**
- **Earlier, disclosed uses** (not for model selection):
  - EDA (notebook 01) inspected the test files.
  - Notebook 02 used test **observed lengths** (inputs, label-free) to compare truncation designs.
  - Notebook 02 read the RUL files once, to verify the target construction and for a one-time sanity check of the protocol after it was fixed.
