#!/usr/bin/env bash
# Run the §4.4 out-of-sample validation and Figures 11-15.
# Inputs are configured through the WINDPAPER_* variables; see
# validation/README.md.
set -euo pipefail

ROOT="${WINDPAPER_ROOT:?set WINDPAPER_ROOT to the repository root}"
PY="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT/validation:$ROOT/figures/_lib${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$ROOT/validation/compute_validation.py"
"$PY" "$ROOT/validation/fig11_data.py"
"$PY" "$ROOT/validation/make_fig11_12.py"
"$PY" "$ROOT/validation/make_fig13_14.py"
"$PY" "$ROOT/validation/make_fig15.py"

echo "validation outputs: ${WINDPAPER_VALIDATION_OUT:-$ROOT/validation_out}"
