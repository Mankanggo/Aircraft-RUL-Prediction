# Superseded run (v1) - kept for traceability

Ablation results from the first run (2026-10-04). Superseded because:

1. The health index for TRAINING rows was in-sample (ridge fitted on those rows' own labels).
   Validation/test were unaffected, but the downstream model trained on a slightly optimistic HI
   (R^2 gap 0.003-0.014). v2 uses cross-fitted (out-of-fold) HI for training rows.
2. The ablation showed the expanding group hurts (LOGO) while the rule-selected set (F11_+cycle)
   contained it; v2 adds refinement sets R1-R3 to the comparison.

Tuning was stopped before any subset finished, so this run has no tuning/finalize results.
