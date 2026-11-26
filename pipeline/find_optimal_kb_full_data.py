#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for M4 wind-only selection.

The final manuscript uses M4: the (k, b) pair is selected on the
wind-speed error surface only.  The implementation lives in
``m4_selection.py``.  This module keeps the historical function name
used by ``panel.py`` and earlier releases.

The superseded M2 code that built the candidate path from the
generation-error surface is archived in
``legacy/m2_find_optimal_kb_full_data.py`` and is not imported here.
"""
from __future__ import annotations

from m4_selection import find_optimal_kb_m4

find_optimal_kb_full_data = find_optimal_kb_m4

__all__ = ["find_optimal_kb_full_data", "find_optimal_kb_m4"]
