# -*- coding: utf-8 -*-
"""Synthetic end-to-end M4 CV test: paths -> selection -> CV -> (K,B)."""
from pathlib import Path

import numpy as np
import pandas as pd

import cv_base as base
import cv_m4


def _write_enum(path: Path, official: float = 1000.0):
    rows = []
    for year in range(2013, 2026):
        labels = ["1-2月"] + [f"{m}月" for m in range(3, 13)]
        for month, label in zip([0] + list(range(3, 13)), labels):
            n = (year - 2013) * 12 + (month or 1)
            rows.append(
                {
                    "time": f"{year}年{label}",
                    "official": official + 10.0 * n,
                    "k0.6b0.2": (official + 10.0 * n) * 1.0,
                    "k0.6b0.3": (official + 10.0 * n) * 1.01,
                    "k0.7b0.2": (official + 10.0 * n) * 1.02,
                    "k0.7b0.3": (official + 10.0 * n) * 1.03,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_m4_cv_end_to_end(tmp_path):
    paths = tmp_path / "paths"
    enums = tmp_path / "enums"
    out = tmp_path / "out"
    paths.mkdir()
    enums.mkdir()
    for season in base.SEASONS:
        np.savez(
            paths / f"test_{season}.npz",
            k=np.array([0.6, 0.6]),
            b=np.array([0.2, 0.3]),
            wind_at_path=np.array([4.0, 5.0]),
        )
        _write_enum(enums / f"test_{season}_monthly_gen_enums.csv")

    selection = cv_m4.build_selection(paths)
    assert len(selection) == 4
    assert set(selection["season"]) == set(base.SEASONS)

    result = cv_m4.run_cv(selection, enums, out, n_splits=1)
    assert len(result["selection"]) == 4
    assert len(result["gen_correction"]) == 1
    assert (result["gen_correction"]["B"] >= 0).all()
    for name in ("selection.csv", "per_run.csv", "per_prov.csv", "summary.csv",
                 "gen_correction.csv"):
        assert (out / name).exists(), name
