#!/usr/bin/env bash
# Apply the final M4 parameters to 2024-2025.
#SBATCH --job-name=m4prod2425
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=m4prod2425_%j.out
#SBATCH --error=m4prod2425_%j.err

set -euo pipefail
ROOT="${WINDPAPER_ROOT:?set WINDPAPER_ROOT to the repository root}"
PY="${PYTHON_BIN:-python}"
OUT="${WINDPAPER_RESULTS_2425:-$ROOT/results_2425}/calibrated_gen"
KB="${WINDPAPER_OPTIMAL_KB:-$ROOT/results/optimal_kb/optimal_kb.csv}"
mkdir -p "$OUT"

cd "$ROOT/pipeline"
"$PY" -u farm_aggr_with_kb.py \
  --era5_dir "${WINDPAPER_ERA5_DIR:?set WINDPAPER_ERA5_DIR}" \
  --optimal_kb_csv "$KB" \
  --inventory_csv_path "${WINDPAPER_INVENTORY:?set WINDPAPER_INVENTORY}" \
  --farm_power_curve_dir "${WINDPAPER_CURVE_DIR:?set WINDPAPER_CURVE_DIR}" \
  --farm_adjusted_w_dir "${WINDPAPER_RAW_W_DIR:?set WINDPAPER_RAW_W_DIR}" \
  --farm_adjusted_w_varname raw_w \
  --output_dir "$OUT" \
  --period_mode season \
  --city_capacity_adjust_csv_path "${WINDPAPER_CITY_CAPACITY:?set WINDPAPER_CITY_CAPACITY}" \
  --prov_utility_rate_adjust_csv_path "${WINDPAPER_UTILIZATION:?set WINDPAPER_UTILIZATION}" \
  --aggr_level both \
  --horizon_interp_method loess \
  --outer_max_workers "${OUTER_WORKERS:-12}" \
  --inner_max_workers "${INNER_WORKERS:-8}" \
  --overwrite True
