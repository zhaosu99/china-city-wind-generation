from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from add_province_capacity_factors import (
    add_capacity_factors_to_dataset,
    load_province_monthly_capacity,
)


class ProvinceCapacityFactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capacity_lookup = {
            ("测试省", 2020, 1, "onshore"): 100.0,
            ("测试省", 2020, 1, "offshore"): 50.0,
        }
        time = pd.date_range("2020-01-01", periods=2, freq="h")
        date = pd.DatetimeIndex(["2020-01-31"])
        self.dataset = xr.Dataset(
            {
                "total_gen": ("time", [40.0, 80.0]),
                "onshore_gen": ("time", [30.0, 60.0]),
                "offshore_gen": ("time", [10.0, 20.0]),
                "total_monthly_gen": ("date", [11160.0]),
                "onshore_monthly_gen": ("date", [7440.0]),
                "offshore_monthly_gen": ("date", [3720.0]),
            },
            coords={"time": time, "date": date},
            attrs={"province_short_name": "测试省"},
        )

    def test_adds_hourly_and_monthly_capacity_factors(self) -> None:
        result = add_capacity_factors_to_dataset(self.dataset, self.capacity_lookup)

        np.testing.assert_allclose(result["total_cf"].values, [40 / 150, 80 / 150])
        np.testing.assert_allclose(result["onshore_cf"].values, [0.3, 0.6])
        np.testing.assert_allclose(result["offshore_cf"].values, [0.2, 0.4])
        np.testing.assert_allclose(result["total_monthly_cf"].values, [0.1])
        np.testing.assert_allclose(result["onshore_monthly_cf"].values, [0.1])
        np.testing.assert_allclose(result["offshore_monthly_cf"].values, [0.1])
        self.assertEqual(result["total_cf"].attrs["units"], "1")

    def test_rejects_positive_generation_with_zero_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive generation with zero capacity"):
            add_capacity_factors_to_dataset(self.dataset, {})

    def test_drops_exact_capacity_rows_but_rejects_conflicts(self) -> None:
        columns = ["年", "月", "省简称", "市简称", "能源类型", "累积装机(MW)"]
        exact_rows = [[2020, 1, "测试省", "测试市", "陆上风电", 100.0]] * 2
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "capacity.csv"
            pd.DataFrame(exact_rows, columns=columns).to_csv(csv_path, index=False, encoding="gbk")
            lookup = load_province_monthly_capacity(csv_path)
            self.assertEqual(lookup[("测试省", 2020, 1, "onshore")], 100.0)

            conflicting_rows = exact_rows[:1] + [[2020, 1, "测试省", "测试市", "陆上风电", 120.0]]
            pd.DataFrame(conflicting_rows, columns=columns).to_csv(csv_path, index=False, encoding="gbk")
            with self.assertRaisesRegex(ValueError, "conflicting duplicate business keys"):
                load_province_monthly_capacity(csv_path)


if __name__ == "__main__":
    unittest.main()