#!/usr/bin/env bash
# Final production run (1989-2023), corresponding to HPC run_full.slurm.
# The M4 (k,b) table must already exist at
#   $WINDPAPER_RESULTS/optimal_kb/optimal_kb.csv
# and the model/generation inputs must be configured through WINDPAPER_*.
#
#SBATCH --job-name=m4prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=m4prod_%j.out
#SBATCH --error=m4prod_%j.err

set -euo pipefail

ROOT="${WINDPAPER_ROOT:?set WINDPAPER_ROOT to the repository root}"
PY="${PYTHON_BIN:-python}"
MAX_WORKERS="${MAX_WORKERS:-96}"
OUTER_WORKERS="${OUTER_WORKERS:-12}"
INNER_WORKERS="${INNER_WORKERS:-8}"

export WINDPAPER_ROOT="$ROOT"
export WINDPAPER_RESULTS="${WINDPAPER_RESULTS:-$ROOT/results}"
export WINDPAPER_MAX_PARALLEL="$MAX_WORKERS"

cd "$ROOT/pipeline"
echo "JOB START $(date -Is) host=$(hostname)"
"$PY" -u panel.py \
  --steps 9 \
  --max-workers "$MAX_WORKERS" \
  --outer-max-workers "$OUTER_WORKERS" \
  --inner-max-workers "$INNER_WORKERS" \
  --no-strict
rc=$?
echo "JOB END $(date -Is) rc=$rc"
exit "$rc"
