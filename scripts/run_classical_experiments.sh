#!/usr/bin/env bash
# Reproduce all classical-ML experiments (notebooks/03_features_and_classical_ml.ipynb reads the results).
# One process per subset, 4 threads per model. Stages are resumable: finished subsets are skipped.
#   bash scripts/run_classical_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-venv/Scripts/python}"
OUT=reports/experiments/classical
mkdir -p "$OUT"
export RUL_N_JOBS="${RUL_N_JOBS:-4}" PYTHONIOENCODING=utf-8

run_stage() {  # $1 = stages to run, in parallel over subsets
  pids=()
  for s in FD001 FD002 FD003 FD004; do
    ( for st in $1; do "$PY" -m src.pipelines.classical_experiments --stage "$st" --subsets "$s" || exit 1; done ) \
      >"$OUT/stdout_${1// /_}_$s.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

run_stage "ablation"           # feature-set choice needs all four subsets
run_stage "seed_noise tune finalize"
echo "all stages finished"
