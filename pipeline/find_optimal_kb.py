# -*- coding: utf-8 -*-
"""Compatibility import for the wind-surface helpers.

The final workflow imports :mod:`wind_valley` directly.  This module is
kept only so that older scripts that used ``from find_optimal_kb import
...`` keep working.  The M2 generation-surface selector is archived in
``legacy/find_optimal_kb_with_m2.py``.
"""
from __future__ import annotations

from wind_valley import *  # noqa: F401,F403
