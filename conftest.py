# -*- coding: utf-8 -*-
"""Make the flat pipeline/drivers imports available to pytest from repo root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("pipeline", "drivers"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
