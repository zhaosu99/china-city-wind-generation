# -*- coding: utf-8 -*-
"""Path constants for the final figure scripts (Fig. 11-15, etc.).

All values can be overridden through environment variables.  The module
lives in ``figures/_lib`` so scripts in sibling figure directories can
import it after adding that directory to ``sys.path``.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("WINDPAPER_ROOT", Path(__file__).resolve().parents[2])).expanduser()
M4 = Path(os.environ.get("WINDPAPER_M4PROD_DIR", ROOT / "m4prod")).expanduser()
OUTDIR = Path(os.environ.get("WINDPAPER_FIG_OUT", ROOT / "results" / "figures")).expanduser()

PROD23 = Path(os.environ.get("WINDPAPER_MODEL_MONTHLY", M4 / "prov_month_from_city.csv"))
PROD24 = Path(os.environ.get("WINDPAPER_MODEL_MONTHLY_2425", M4 / "prov_month_2425.csv"))
KB = Path(os.environ.get("WINDPAPER_KB", ROOT / "parameters" / "gen_correction_M4prod_final.csv"))
OFF = Path(os.environ.get("WINDPAPER_OFFICIAL_MONTHLY", M4 / "official_prov_month.csv"))
CAP = Path(os.environ.get("WINDPAPER_CAP_ADJ", ROOT / "results" / "cap_adj_applied_2425.csv"))
UTIL = Path(os.environ.get("WINDPAPER_UTILIZATION", ROOT / "results" / "util_applied_2425.csv"))

CV_WIND = Path(os.environ.get("WINDPAPER_CV_WIND", M4 / "cv_ext" / "wind_cv_2020_slim.csv"))
FULL_WIND = Path(os.environ.get("WINDPAPER_OPTIMAL_KB", ROOT / "parameters" / "optimal_kb_M4.csv"))
CV_GEN = Path(os.environ.get("WINDPAPER_CV_GEN", M4 / "cv_ext" / "cv_per_run_2025pool_febfix.csv"))

GEN_PROD23 = PROD23
GEN_PROD24 = PROD24
GEN_OFF = OFF
GEN_CAP = CAP
GEN_UTIL = UTIL
