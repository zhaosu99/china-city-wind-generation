# -*- coding: utf-8 -*-
"""Test the production generation-calibration fitter."""
import pandas as pd
import pytest

from fit_generation_calibration import fit_generation_calibration


def test_fit_generation_calibration_constrained(tmp_path):
    # Deliberately negative intercept in the synthetic data: the fit must
    # return B = 0 rather than negative generation for small months.
    model_rows, official_rows = [], []
    for i, x in enumerate([10.0, 20.0, 30.0, 40.0], start=1):
        model_rows.append({"prov": "anhui", "ym": f"2023{ i:02d}", "gen_MWh": x})
        official_rows.append({"cn": "安徽", "year": 2023, "month": i, "value_MWh": 2.0 * x - 15.0})
    model_csv = tmp_path / "model.csv"
    official_csv = tmp_path / "official.csv"
    out_csv = tmp_path / "gen_correction.csv"
    pd.DataFrame(model_rows).to_csv(model_csv, index=False)
    pd.DataFrame(official_rows).to_csv(official_csv, index=False)

    result = fit_generation_calibration(model_csv, official_csv, out_csv)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["prov"] == "安徽"
    assert row["B"] == 0.0
    assert row["n"] == 3  # January and February are merged
    assert out_csv.exists()
