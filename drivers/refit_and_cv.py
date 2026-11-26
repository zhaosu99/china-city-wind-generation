#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry point for the final M4 cross-validation.

The old 2013-2023 refit/CV script was replaced by
``drivers/cv_m4.py`` (143 periods, 2013-2025) and
``drivers/fit_generation_calibration.py`` (production 2013-2023 fit).

This shim is kept so that readers following older commit messages can
find the new entry point; it contains no M2 selection logic.
"""
from __future__ import annotations

from cv_m4 import main

if __name__ == "__main__":
    main()
