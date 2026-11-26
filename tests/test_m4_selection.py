# -*- coding: utf-8 -*-
"""Unit tests for the M4 wind-only selection."""
import numpy as np
import pytest

import m4_selection
from find_optimal_kb_full_data import find_optimal_kb_full_data


def test_select_m4_uses_only_wind_path():
    k = np.array([0.5, 0.8, 1.1, 1.4])
    b = np.array([0.1, 0.2, -0.1, 0.0])
    wind = np.array([9.0, 5.0, 3.0, 7.0])
    out = m4_selection.select_m4_from_path(k, b, wind)
    assert out["selection_mode"] == "M4_wind_only"
    assert out["k"] == pytest.approx(1.1)
    assert out["b"] == pytest.approx(-0.1)
    assert out["w_smape"] == pytest.approx(3.0)
    assert out["candidate_point_count"] == 4
    assert not out["on_boundary"]


def test_select_m4_marks_boundary():
    out = m4_selection.select_m4_from_path(
        np.array([0.5, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 2.0])
    )
    assert out["k"] == pytest.approx(0.5)
    assert out["on_boundary"] is True


def test_m4_rejects_length_mismatch():
    with pytest.raises(ValueError):
        m4_selection.select_m4_from_path([1.0], [1.0, 2.0], [1.0])


def test_m4_ignores_generation_values():
    """The M4 selector must be unaffected by any generation array."""
    k = np.array([0.5, 0.9, 1.2])
    b = np.array([0.0, 0.1, -0.1])
    wind = np.array([8.0, 4.0, 6.0])
    out1 = m4_selection.select_m4_from_path(k, b, wind)
    # There is no generation argument; this is a structural guarantee.
    assert "gen_smape" not in out1


def test_compatibility_wrapper():
    assert find_optimal_kb_full_data is m4_selection.find_optimal_kb_m4
