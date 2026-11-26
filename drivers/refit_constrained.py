#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry point for the constrained production (K, B) fit.

The implementation is in ``drivers/fit_generation_calibration.py``.
The old disk-path script is preserved only in
``legacy/refit_constrained_2013_2023.py``.
"""
from __future__ import annotations

from fit_generation_calibration import main

if __name__ == "__main__":
    main()
