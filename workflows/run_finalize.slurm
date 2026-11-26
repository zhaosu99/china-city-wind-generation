#!/usr/bin/env bash
# Final generation-correction stage: calibrated_gen -> city_final.
# Run after workflows/run_full.slurm has completed.
#SBATCH --job-name=m4final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=m4final_%j.out
#SBATCH --error=m4final_%j.err

set -euo pipefail
ROOT="${WINDPAPER_ROOT:?set WINDPAPER_ROOT to the repository root}"
PY="${PYTHON_BIN:-python}"
RES="${WINDPAPER_RESULTS:-$ROOT/results}"

"$PY" "$ROOT/drivers/build_final_city.py" \
  --src "${WINDPAPER_CALIBRATED_CITY_DIR:-$RES/calibrated_gen/gen/city}" \
  --dst "${WINDPAPER_CITY_FINAL_DIR:-$RES/city_final}" \
  --model-monthly "${WINDPAPER_MODEL_MONTHLY:-$ROOT/prov_month_from_city.csv}" \
  --kb "${WINDPAPER_KB:-$ROOT/parameters/gen_correction_M4prod_final.csv}" \
  --pre2012 "${WINDPAPER_PRE2012_OFFICIAL:-$ROOT/parameters/pre2012_official.csv}"
