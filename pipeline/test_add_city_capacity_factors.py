from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from add_city_capacity_factors import add_city_capacity_factors_to_dataset


def _city_dataset(include_offshore: bool) -> xr.Dataset:
    data_vars = {
        "total_gen": ("time", [40.0]),
        "onshore_gen": ("time", [30.0]),
        "total_monthly_gen": ("date", [11160.0]),
        "onshore_monthly_gen": ("date", [7440.0]),
    }
    if include_offshore:
        data_vars.update(
            {
                "offshore_gen": ("time", [10.0]),
                "offshore_monthly_gen": ("date", [3720.0]),
            }
        )
    else:
        data_vars["total_gen"] = ("time", [30.0])
        data_vars["total_monthly_gen"] = ("date", [7440.0])
    return xr.Dataset(
        data_vars,
        coords={
            "time": pd.DatetimeIndex(["2020-01-01"]),
            "date": pd.DatetimeIndex(["2020-01-31"]),
        },
        attrs={"province_short_name": "测试省", "city_short_name": "测试市"},
    )


class CityCapacityFactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capacity_lookup = {
            ("测试省", "测试市", 2020, 1, "onshore"): 100.0,
            ("测试省", "测试市", 2020, 1, "offshore"): 50.0,
        }

    def test_city_with_offshore_gets_all_capacity_factors(self) -> None:
        result = add_city_capacity_factors_to_dataset(_city_dataset(True), self.capacity_lookup)

        self.assertAlmostEqual(float(result["total_cf"].item()), 40 / 150)
        self.assertAlmostEqual(float(result["onshore_cf"].item()), 0.3)
        self.assertAlmostEqual(float(result["offshore_cf"].item()), 0.2)
        self.assertAlmostEqual(float(result["offshore_monthly_cf"].item()), 0.1)

    def test_city_without_offshore_does_not_get_offshore_variables(self) -> None:
        result = add_city_capacity_factors_to_dataset(_city_dataset(False), self.capacity_lookup)

        self.assertIn("total_cf", result)
        self.assertIn("onshore_cf", result)
        self.assertNotIn("offshore_cf", result)
        self.assertNotIn("offshore_monthly_cf", result)

    def test_offshore_nan_is_derived_without_changing_generation(self) -> None:
        dataset = _city_dataset(True)
        dataset["total_gen"][:] = 30.0
        dataset["offshore_gen"][:] = np.nan
        dataset["total_monthly_gen"][:] = 7440.0
        dataset["offshore_monthly_gen"][:] = np.nan

        result = add_city_capacity_factors_to_dataset(dataset, self.capacity_lookup)

        self.assertTrue(np.isnan(result["offshore_gen"].item()))
        self.assertTrue(np.isnan(result["offshore_monthly_gen"].item()))
        self.assertEqual(float(result["offshore_cf"].item()), 0.0)
        self.assertEqual(float(result["offshore_monthly_cf"].item()), 0.0)


if __name__ == "__main__":
    unittest.main()