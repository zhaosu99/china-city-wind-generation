# -*- coding: utf-8 -*-
"""Central path configuration for the §4.4 validation and figure scripts.

All paths can be overridden with environment variables; the defaults point
at the repository's ``validation_out/`` and ``results/`` layout.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("WINDPAPER_ROOT", Path(__file__).resolve().parents[1])).expanduser()
M4 = Path(os.environ.get("WINDPAPER_M4PROD_DIR", ROOT / "m4prod")).expanduser()
OUTD = Path(os.environ.get("WINDPAPER_VALIDATION_OUT", ROOT / "validation_out")).expanduser()
OUTDIR = Path(os.environ.get("WINDPAPER_FIG_OUT", OUTD / "figures")).expanduser()
OUT = Path(os.environ.get("WINDPAPER_VALIDATION_OUT", OUTD)).expanduser()

PROD23 = Path(os.environ.get("WINDPAPER_MODEL_MONTHLY", M4 / "prov_month_from_city.csv"))
PROD24 = Path(os.environ.get("WINDPAPER_MODEL_MONTHLY_2425", M4 / "prov_month_2425.csv"))
KB = Path(os.environ.get("WINDPAPER_KB", ROOT / "parameters" / "gen_correction_M4prod_final.csv"))
OFF = Path(os.environ.get("WINDPAPER_OFFICIAL_MONTHLY", M4 / "official_prov_month.csv"))
CAP = Path(os.environ.get("WINDPAPER_CAP_ADJ", ROOT / "results" / "cap_adj_applied_2425.csv"))
UTIL = Path(os.environ.get("WINDPAPER_UTILIZATION", ROOT / "results" / "util_applied_2425.csv"))

CV_WIND = Path(os.environ.get("WINDPAPER_CV_WIND", M4 / "cv_ext" / "wind_cv_2020_slim.csv"))
FULL_WIND = Path(os.environ.get("WINDPAPER_OPTIMAL_KB", ROOT / "parameters" / "optimal_kb_M4.csv"))
CV_GEN = Path(os.environ.get("WINDPAPER_CV_GEN", M4 / "cv_ext" / "cv_per_run_2025pool_febfix.csv"))
KB_VALIDATION_DETAIL_CSV = Path(os.environ.get(
    "WINDPAPER_KB_VALIDATION_DETAIL_CSV",
    ROOT / "results" / "kb_validation" / "kb_validation_detail.csv",
))

