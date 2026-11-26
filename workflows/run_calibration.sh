#!/usr/bin/env bash
# Run the final M4 calibration stages:
#   1. wind-only (k,b) selection
#   2. 143-period cross-validation + selection validation
#   3. production (K,B) fit
#
# All inputs are supplied through WINDPAPER_* variables.  See
# workflows/README.md for the expected layout.
set -euo pipefail

ROOT="${WINDPAPER_ROOT:?set WINDPAPER_ROOT to the repository root}"
PY="${PYTHON_BIN:-python}"
PYTHONPATH="$ROOT/pipeline:$ROOT/drivers${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

: "${WINDPAPER_WIND_SMAPE_DIR:?set WINDPAPER_WIND_SMAPE_DIR}"
: "${WINDPAPER_GEN_SMAPE_DIR:?set WINDPAPER_GEN_SMAPE_DIR}"
: "${WINDPAPER_PATH_DIR:?set WINDPAPER_PATH_DIR}"
: "${WINDPAPER_ENUM_DIR:?set WINDPAPER_ENUM_DIR}"
: "${WINDPAPER_MODEL_MONTHLY:?set WINDPAPER_MODEL_MONTHLY}"
: "${WINDPAPER_OFFICIAL_MONTHLY:?set WINDPAPER_OFFICIAL_MONTHLY}"

OUT="${WINDPAPER_CALIBRATION_OUT:-$ROOT/calibration_out}"
mkdir -p "$OUT"

echo "[1/3] M4 wind-only selection"
"$PY" "$ROOT/pipeline/m4_selection.py" \
  --input-dir "$WINDPAPER_GEN_SMAPE_DIR" \
  --wind-dir "$WINDPAPER_WIND_SMAPE_DIR" \
  --output-dir "$OUT/selection" \
  --overwrite

echo "[2/3] 143-period M4 cross-validation"
"$PY" "$ROOT/drivers/cv_m4.py" \
  --paths-dir "$WINDPAPER_PATH_DIR" \
  --enum-dir "$WINDPAPER_ENUM_DIR" \
  --out "$OUT/cv" \
  --splits "${WINDPAPER_CV_SPLITS:-50}"

echo "[3/3] production generation calibration"
"$PY" "$ROOT/drivers/fit_generation_calibration.py" \
  --model-monthly "$WINDPAPER_MODEL_MONTHLY" \
  --official-monthly "$WINDPAPER_OFFICIAL_MONTHLY" \
  --out "$OUT/gen_correction.csv"

echo "calibration outputs: $OUT"
