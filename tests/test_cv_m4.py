# -*- coding: utf-8 -*-
"""Tests for the 143-period M4 CV and the production (K, B) fit."""
import numpy as np
import pytest

import cv_base as base
import cv_m4


def test_period_grid_is_143():
    assert len(base.GRID) == 143
    counts = {s: len(base.season_codes(s)) for s in base.SEASONS}
    assert counts == {"spring": 39, "summer": 39, "autumn": 39, "winter": 26}


def test_cv_splits_have_29_test_periods():
    splits = base.make_splits(n_splits=3)
    assert len(splits) == 3
    for split in splits:
        assert len(split["test_codes"]) == 29
        assert len(split["train_codes"]) == 114
        assert set(split["test_codes"]).isdisjoint(split["train_codes"])


def test_non_negative_intercept_constraint():
    x = np.linspace(1, 50, 50)
    y = 2.0 * x - 1.0  # unconstrained B is negative
    K, B, n, mode = cv_m4.fit_kb_constrained(x, y)
    assert n == 50
    assert B == 0.0
    assert mode == "B_constrained_to_0"
    assert K > 0


def test_unconstrained_fit_recovers_line():
    x = np.linspace(1, 10, 50)
    y = 3.0 * x + 4.0
    K, B, n, mode = cv_m4.fit_kb_constrained(x, y)
    assert K == pytest.approx(3.0, abs=1e-9)
    assert B == pytest.approx(4.0, abs=1e-9)
    assert mode == "unconstrained"
