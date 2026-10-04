# RUL target and validation protocol (model development)

Source: `notebooks/02_target_and_validation.ipynb`.
Code:
- `src/utils.py`
- `src/components/validation.py`
- `src/components/data_transformation.py`
- tests: `tests/` (54 tests)

## 1. Target generation (exact logic)
**Training engines** (run to failure, lifespan `T` = last cycle):
`RUL(t) = T − t`, for t = 1..T. The failure row has RUL 0, and RUL decreases by exactly 1 per cycle.
- `src.utils.add_train_rul`

**Official test engines** (observed up to cycle `L`):
`RUL(t) = RUL_file + (L − t)`. The last observed row equals `RUL_FD00x.txt`; row i of the file belongs to unit i.
- `src.utils.add_test_rul`
- Used only for final evaluation.

**Validation sample** (training engine cut at cycle `c`):
`target = T − c`, which equals the training RUL at row c.
- This is the **uncapped** true RUL, with the same convention as the RUL files: cycles remaining after the last observed cycle.

**Training target** (piecewise-linear):
`y = min(RUL, cap)`. The cap is a **hyper-parameter**.
- `src.utils.cap_rul`; `cap=None` means uncapped.
- `cross_validate` always rebuilds the uncapped RUL from the cycles and applies the cap per model. A pre-capped input column can therefore never leak into evaluation.

**Evaluation target**: always the uncapped true RUL.
- RMSE against a capped truth flatters results by 2.5–3 RMSE (probe). It may be reported only as a secondary number, for comparison with the literature.

Verified on all four subsets:
- 100% of rows satisfy these definitions.
- 0 mismatches against the RUL files.
- Sample engines were inspected manually.

## 2. Validation protocol
1. **Engine-level folds, per subset.**
   - 5 folds built from whole engines (`make_engine_folds`, seed 42).
   - Stratified by lifespan: sorted by lifespan, with fold labels permuted within blocks of 5. Fold median lifespans differ by ≤ 2.5%.
   - Never split rows; never mix subsets in a fold.
   - Unit ids restart in every file, so they are only a group key.
2. **Test-like validation samples.**
   - Each validation engine yields up to 10 samples (`make_cv_plan`).
   - True RUL at the cut is drawn uniformly (distinct values) from **[6, 150]**.
   - Each sample keeps only cycles 1..c (later rows are deleted).
   - **Exactly one prediction per sample, at its last observed cycle.**
   - Sample counts: FD001 1000, FD002 2600, FD003 1000, FD004 2490. Every engine is validated, and each engine gets the same number of samples.
   - Why [6, 150]: it is the truncation range documented in the paper (test RULs 10–150), and it gives the best **label-free** match of observed lengths to the official test trajectories. KS = 0.05–0.09, against 0.13–0.25 unbounded and 0.07–0.19 with max 190.
3. **Secondary "wide" plan** (same folds, RUL at cut in [6, 190], the paper's validation-set range).
   - Always report it as well.
   - The one-time check against the test labels showed that 14% (FD002) and 17% (FD004) of official test engines have true RUL > 150 (max 194/195). The primary plan cannot represent these. This check was run after the design was fixed and did not change it.
4. **Fit on training engines only.**
   - `FoldPreprocessor.fit(training-fold engines)` learns the per-condition sensor statistics and the feature scaler. `transform` only applies them.
   - `cross_validate` asserts that no validation engine reached `fit`.
   - Final model: refit on all training engines of the subset, then transform the official test trajectories.
5. **Causal features only.**
   - Feature types: current value, trailing mean/std/slope over 5/15/30 cycles, and change from the engine's own first-≤10-cycle baseline.
   - Each uses only cycles ≤ t of the same trajectory.
   - Verified: truncated vs full-trajectory features match exactly (max |Δ| = 0.0).
6. **Fixed, shared plan.**
   - Folds and cuts are deterministic from the seed.
   - Saved to `data/processed/cv_plan/` (`FD00x_folds.csv`, `FD00x_val_cuts.csv`, `FD00x_val_cuts_wide.csv`).
   - Every model is compared on identical samples.
7. **Metrics** (vs uncapped truth, pooled over all validation samples):
   - primary: RMSE;
   - also: fold std of RMSE, MAE, bias, PHM08 score per sample (`rul_metrics`, `summarise_cv`).
8. **Inner splits for early stopping or tuning** (deep learning): carve them from the *training-fold* engines (engine-level), never from the outer validation fold.
9. **Official test set**: used once, for final reporting only. `RUL_FD00x.txt` is never used for model, feature or hyper-parameter selection.

## 3. RUL cap as a hyper-parameter
- **Grid**: {100, 110, 120, 125, 130, 140, 150, 175}, plus no cap as a reference. Tune it per model family and per subset.
- **Selection rule**: lowest primary-plan CV RMSE. Tie-break: take the **largest cap within 1 SE** (fold std / √5) of the best. This hedges towards the true RULs above 150 that FD002/FD004 contain. PHM08 and the wide plan are reported alongside.

**Diagnostic ridge probe** (not a candidate model; it only shows how the cap behaves inside the protocol):

| | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Best cap, primary plan (1-SE set) | 125 (120–130) | 125 (120–130) | 125 (120–130) | 125 (120–130) |
| RMSE at best / no cap | 19.1 / 29.3 | 19.4 / 30.8 | 21.0 / 46.7 | 21.1 / 44.1 |
| Best cap by PHM08 | 120 | 120 | 110 | 110 |
| Best cap, wide plan | 150 | 150 | 150 | 150 |

- **Capping is essential**: RMSE improves by 10–26 compared with no cap.
- **The best cap depends on the RUL range being evaluated**: 125 for RUL ≤ 150, 150 for RUL ≤ 190. An earlier, unbounded validation design pushed the optimum to 150–200. This is why the cap must be tuned under a realistic, test-like protocol.
- **Lower caps help near failure, higher caps help far from it**: lower caps improve accuracy at RUL ≤ 50, where the PHM08 penalty matters most, and hurt accuracy above 100.
- **Linear-model caveat**: a linear model is unusually sensitive to the cap. Non-linear models must re-tune it.

## 4. Known limitations
- The primary plan has no samples with true RUL > 150. FD002/FD004 tests contain some, which is why the wide plan exists.
- 0.6–2.3% of validation samples have fewer than 19 observed cycles (shorter than any test trajectory). This makes validation slightly harder than the test, not easier.
- With 100 engines, FD001/FD003 folds have only 20 validation engines each. Fold-to-fold RMSE std is reported. For close decisions, repeat the CV with other seeds (`make_cv_plan(seed=...)`).
