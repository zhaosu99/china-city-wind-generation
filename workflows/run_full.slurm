#!/usr/bin/env bash
# Wind-calibrated generation for 1989-2023 (intermediate stage).
# This runs only panel.py --steps 9.  The final city dataset is produced
# afterwards by drivers/build_final_city.py (see workflows/README.md).
# The M4 (k,b) table must already exist at:
#   $WINDPAPER_RESULTS/optimal_kb/optimal_kb.csv
# Steps 1-3 (farm_raw_w, farm_power_curves) must already exist.
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$ROOT/pipeline"
echo "JOB START $(date -Is) host=$(hostname)"
set +e
"$PY" -u panel.py \
  --steps 9 \
  --max-workers "$MAX_WORKERS" \
  --outer-max-workers "$OUTER_WORKERS" \
  --inner-max-workers "$INNER_WORKERS" \
  --strict
rc=$?
set -e
echo "JOB END $(date -Is) rc=$rc"
exit "$rc"
