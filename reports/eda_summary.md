# EDA summary: NASA C-MAPSS (FD001-FD004)

Source: `notebooks/01_eda.ipynb` (fully executed; figures in `reports/figures/eda/`, tables in `reports/tables/eda/`).
Schema: `src/data_schema.py`. The 21 sensor names, descriptions and units come from Table 2 of Saxena et al. (2008); the positional mapping is checked against physical relationships in the notebook (Section 1.1).

## 1. Data quality
- All 12 raw files are clean. Every line has 26 numeric tokens (1 in the RUL files), and there are no missing values, infinities, non-numeric tokens, duplicate rows, duplicate `(unit_id, cycle)` keys or non-positive sensor readings.
- Unit ids run 1..n and cycles are contiguous from 1 in every unit. Each RUL file has exactly one row per test unit.
- **Documentation discrepancy:** `docs/readme.txt` gives FD004 as 248 train / 249 test units. The files contain **249 train / 248 test**, consistent with `RUL_FD004.txt` (248 rows). The data files are treated as authoritative.
- **Schema caveats:**
  - All 8 physical checks pass. The corrected-speed identities `NRf = Nf·√(518.67/T2)` and `NRc = Nc·√(518.67/T24)` hold to ≈10⁻⁴ across all six conditions, which confirms the Table-2 order.
  - Three magnitudes do not fit their Table-2 units: `Ps30` (~47 psia vs `P30` ~554 psia), `phi` (which depends on Ps30), and `PCNfR_dmd` (100/84.93, which looks like a percentage, not rpm). The names are kept and these units are flagged as unverified.
- **Operating settings:** they are not named in the paper, so they keep the generic names `op_setting_1..3`.
- Test end-RULs are 6-145 (FD001/FD003) and 6-195 (FD002/FD004). This differs from the 10-150 the paper quotes for the PHM08 challenge test set.

## 2. Statistical findings
- **Constant sensors:**
  - FD001: `T2`, `P2`, `epr`, `farB`, `Nf_dmd`, `PCNfR_dmd`.
  - FD003: the same, except `epr`.
- **Condition-determined in FD002/FD004** (constant within each condition): `T2`, `P2`, `Nf_dmd`, `PCNfR_dmd`.
- **Quantised:** `epr`, `farB` and `P15` take a few 0.01-step levels.
- **FD002/FD004 settings:** six discrete operating conditions. Rounding the settings assigns every row unambiguously (deviation < 0.01), with nothing fitted. Condition frequencies are identical in train and test (≈25% for one condition, ≈15% for each of the others). The condition is re-drawn independently every cycle.
- **Raw sensors in FD002/FD004 are ~99% condition** (η² ≈ 0.97-1.00). Raw correlations there are an artefact: the median inter-sensor |ρ| falls from 0.84 (raw) to 0.34 (condition-normalised) in FD004.
- **Within-condition variability is tiny** (CV ≈ 10⁻⁵ to 10⁻²) even for the sensors that carry signal, so variance thresholds are useless for selection.
- **Highly correlated pairs** (condition-normalised): `Nf`↔`NRf` (ρ 0.81-0.95), `Nc`↔`NRc` (ρ 0.86-0.89), and `P30`↔`phi` in FD003 (ρ 0.95).

## 3. Temporal / degradation findings
- **Shape:** degradation is non-linear. It drifts slowly for most of life and accelerates sharply over the last ~100-150 cycles; the HI proxy gains +1.1 z in the final 25 cycles. The first ~50 cycles are near-flat in every subset.
- **Robust core sensors:** `T24`, `T30`, `T50`, `Ps30`, `htBleed` rise monotonically in 100% of engines in all four subsets.
- **FD001 (HPC):** `BPR`, `Nf`, `NRf` also rise in all engines, and `P30`, `phi`, `W31`, `W32` fall in all of them.
- **Two engine groups in FD003/FD004** (44/100 and 101/249 engines). In these engines:
  - `P30`, `phi`, `W31`, `W32` rise instead of fall, and `BPR` stays flat;
  - `epr` (constant in FD001) steps up, perfectly coinciding with the phi-direction grouping;
  - the groups separate with Cohen's |d| ≈ 10-15.

  This is most likely the fan-fault mode (an inferred hypothesis; no labels are given). These sensors have near-zero pooled correlation with RUL (|ρ| ≈ 0.07-0.14) but strong within-unit trends.
- **Raw FD002/FD004 series** jump between six condition levels every cycle. The trend only becomes visible after per-condition normalisation.

## 4. Differences between FD subsets
| | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Conditions / fault modes | 1 / HPC | 6 / HPC | 1 / HPC+Fan | 6 / HPC+Fan |
| Train / test units | 100 / 100 | 260 / 259 | 100 / 100 | 249 / 248 |
| Train life min/median/max | 128/199/362 | 128/199/378 | 145/220/525 | 128/234/543 |
| Test length min/median | 31/134 | 21/132 | 38/148 | 19/154 |
| Consistent-trend sensors | 12 | 8 | 7 | 5 |

- **Conditions** change the sensor *scale*.
- **Fault modes** change sensor *direction*, lengthen lives and widen their spread.
- **FD004** combines both effects and is the hardest subset.

## 5. Modelling implications
- **Normalisation:** normalise each sensor **within its operating condition**, with statistics fitted on training data only, before any rolling or sequence feature.
- **Target:** use a **piecewise-linear (capped) RUL target**, with the cap tuned by validation (try ~100-150). Early-life RUL is not identifiable from the sensors.
- **Windows:** use temporal windows, because single cycles are noisy. The minimum test lengths (19-38 cycles) cap the window size or require padding. 2-11 test units per subset are shorter than 30 cycles, and these short units have the **largest** true RULs (mean ≈ 120-157 vs ≈ 72-80).
- **Fault modes:** FD003/FD004 models need the fault-mode-discriminating sensors (`P30`, `phi`, `W31`, `W32`, `BPR`, `epr`), or explicit fault-mode features. A single global "health direction" does not fit them.
- **Pooling and evaluation:** pooling subsets is feasible after per-condition normalisation, but evaluate **per subset**. Note that the PHM08 score penalises late predictions more (paper eq. 11).
- **Train/test consistency:** at matched life positions the KS statistic falls to 0.01-0.09, below the unit-level critical value, so train and test engines come from the same population. The naive shift (KS 0.10-0.18) is just test truncation.

## 6. Potential leakage risks
- **Random row-level splits (high).** Within-unit autocorrelation is ≈0.99 after smoothing. → Split by engine (GroupKFold on `unit_id` within subset).
- **`unit_id` as a feature or cross-file key (high).** Ids restart at 1 in every file. → Use it only as a group key.
- **Target-derived or non-causal features (high).** Examples are life fraction, or per-unit statistics over the full trajectory. → Use only cycles ≤ t.
- **Tuning on `RUL_FD00x` (high).** → Reserve it for the final test only.
- **Scalers or condition statistics fitted on validation/test data (medium).**
- **Validation on full trajectories instead of truncated ones (medium).**
- **Cycle number as an age shortcut (medium).** ρ(cycle, RUL) is −0.58 to −0.79 in train rows, and observed test length vs true RUL is r −0.41 to −0.68. This is a legitimate signal, but it should be ablated.
- **Checked and clear:** no test row duplicates a training row.

## 7. Recommendations for preprocessing / feature engineering
1. **Loader:** reuse `src.utils.read_cmapss_file`/`read_rul_file` and assert the data-quality invariants on load. Add the operating condition with `assign_operating_condition` (stateless).
2. **Health-feature exclusions:** exclude `T2`, `P2`, `Nf_dmd`, `PCNfR_dmd` and the raw settings from the health features. Keep the settings or condition id as context. Keep `epr`, `farB` and `P15` until ablation decides.
3. **Normalisation:** fit per-condition z-score statistics on training folds only (`fit_condition_stats` / `apply_condition_norm`).
4. **Causal features:** trailing rolling mean/std/slope over windows ≤ 19-30 cycles. Add change-from-early-life-baseline features (these expose step-like signals such as `epr`). Optionally add the cycle count as a separately ablated feature.
5. **Target:** piecewise-linear RUL with the cap as a hyper-parameter.
6. **Validation:**
   - GroupKFold by engine, per subset.
   - Score each validation engine at random truncation points that mimic the test-length distribution.
   - Report RMSE together with the PHM08 score (`src.utils.phm08_score`).
7. **Sensor selection:** treat the sensor tiers (`reports/tables/eda/sensor_hypothesis_tiers.csv`) as hypotheses. Validate the core set (`T24`, `T30`, `T50`, `Ps30`, `htBleed`) + fault-mode sensors + de-duplicated speeds against "all non-constant sensors" with grouped CV. Do not drop sensors on pooled correlation.
8. **Fault-mode feature for FD003/FD004:** consider an explicit fault-mode indicator, derived causally from early-to-current changes in `phi`/`epr`, as a hypothesis to test.
