from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from add_province_capacity_factors import (
    BUSINESS_KEY_COLUMNS,
    CAPACITY_COLUMN,
    ENERGY_TYPE_MAP,
    _capacity_factor,
)
from utils import normalize_text, safe_read_csv, sanitize_netcdf_attrs, write_netcdf_with_retry


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAPACITY_CSV = PROJECT_ROOT / "city_monthly_wind_capacity.csv"
DEFAULT_CITY_DIR = PROJECT_ROOT / "calibrated_gen" / "city"
BASE_KINDS = ("total", "onshore")
OFFSHORE_GENERATION_VARIABLES = ("offshore_gen", "offshore_monthly_gen")


def load_city_monthly_capacity(
    capacity_csv: Union[str, Path],
) -> dict[tuple[str, str, int, int, str], float]:
    capacity_df = safe_read_csv(capacity_csv)
    required_columns = set(BUSINESS_KEY_COLUMNS + [CAPACITY_COLUMN])
    missing_columns = sorted(required_columns - set(capacity_df.columns))
    if missing_columns:
        raise KeyError(f"capacity CSV is missing columns: {missing_columns}")

    prepared = capacity_df[list(required_columns)].copy()
    for column in ("省简称", "市简称", "能源类型"):
        prepared[column] = prepared[column].map(normalize_text)
    prepared["年"] = pd.to_numeric(prepared["年"], errors="raise").astype(int)
    prepared["月"] = pd.to_numeric(prepared["月"], errors="raise").astype(int)
    prepared[CAPACITY_COLUMN] = pd.to_numeric(prepared[CAPACITY_COLUMN], errors="raise").astype(float)

    invalid_months = prepared.loc[~prepared["月"].between(1, 12), ["年", "月"]]
    if not invalid_months.empty:
        raise ValueError(f"capacity CSV contains invalid months: {invalid_months.head().to_dict('records')}")
    if prepared[["省简称", "市简称", "能源类型"]].isna().any().any():
        raise ValueError("capacity CSV contains empty province, city, or energy type")
    if (~np.isfinite(prepared[CAPACITY_COLUMN])).any() or (prepared[CAPACITY_COLUMN] < 0).any():
        raise ValueError(f"{CAPACITY_COLUMN} must contain finite non-negative values")

    exact_duplicate_count = int(prepared.duplicated(keep="first").sum())
    if exact_duplicate_count:
        logger.warning("dropping %s exact duplicate capacity rows", exact_duplicate_count)
        prepared = prepared.drop_duplicates(keep="first")

    conflicting_duplicates = prepared.duplicated(BUSINESS_KEY_COLUMNS, keep=False)
    if conflicting_duplicates.any():
        duplicate_keys = prepared.loc[conflicting_duplicates, BUSINESS_KEY_COLUMNS]
        raise ValueError(
            "capacity CSV contains conflicting duplicate business keys: "
            f"{duplicate_keys.head().to_dict('records')}"
        )

    prepared["kind"] = prepared["能源类型"].map(ENERGY_TYPE_MAP)
    unsupported_types = sorted(prepared.loc[prepared["kind"].isna(), "能源类型"].unique())
    if unsupported_types:
        raise ValueError(f"unsupported energy types: {unsupported_types}")

    return {
        (
            str(row["省简称"]),
            str(row["市简称"]),
            int(row["年"]),
            int(row["月"]),
            str(row["kind"]),
        ): float(row[CAPACITY_COLUMN])
        for _, row in prepared.iterrows()
    }


def _city_capacity_values(
    province_short_name: str,
    city_short_name: str,
    timestamps: pd.DatetimeIndex,
    capacity_lookup: dict[tuple[str, str, int, int, str], float],
) -> dict[str, np.ndarray]:
    onshore = np.asarray(
        [
            capacity_lookup.get(
                (province_short_name, city_short_name, timestamp.year, timestamp.month, "onshore"),
                0.0,
            )
            for timestamp in timestamps
        ],
        dtype=float,
    )
    offshore = np.asarray(
        [
            capacity_lookup.get(
                (province_short_name, city_short_name, timestamp.year, timestamp.month, "offshore"),
                0.0,
            )
            for timestamp in timestamps
        ],
        dtype=float,
    )
    return {"total": onshore + offshore, "onshore": onshore, "offshore": offshore}


def add_city_capacity_factors_to_dataset(
    ds: xr.Dataset,
    capacity_lookup: dict[tuple[str, str, int, int, str], float],
    overwrite_existing: bool = False,
) -> xr.Dataset:
    province_short_name = normalize_text(ds.attrs.get("province_short_name"))
    city_short_name = normalize_text(ds.attrs.get("city_short_name"))
    if province_short_name is None or city_short_name is None:
        raise KeyError("city dataset must contain province_short_name and city_short_name attrs")
    if "time" not in ds.coords or "date" not in ds.coords:
        raise KeyError("city dataset must contain time and date coordinates")

    required_base_generation = {
        "total_gen",
        "total_monthly_gen",
        "onshore_gen",
        "onshore_monthly_gen",
    }
    missing_generation = sorted(required_base_generation - set(ds.data_vars))
    if missing_generation:
        raise KeyError(f"city dataset is missing generation variables: {missing_generation}")

    offshore_present = [name in ds.data_vars for name in OFFSHORE_GENERATION_VARIABLES]
    if any(offshore_present) and not all(offshore_present):
        raise KeyError("city dataset must contain both offshore_gen and offshore_monthly_gen or neither")
    kinds = BASE_KINDS + (("offshore",) if all(offshore_present) else ())
    target_variables = {
        variable_name
        for kind in kinds
        for variable_name in (f"{kind}_cf", f"{kind}_monthly_cf")
    }
    existing_variables = sorted(target_variables & set(ds.data_vars))
    if existing_variables and not overwrite_existing:
        raise ValueError(f"city dataset already contains capacity factor variables: {existing_variables}")

    time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    date_index = pd.DatetimeIndex(pd.to_datetime(ds["date"].values))
    hourly_capacity = _city_capacity_values(
        province_short_name,
        city_short_name,
        time_index,
        capacity_lookup,
    )
    monthly_capacity = _city_capacity_values(
        province_short_name,
        city_short_name,
        date_index,
        capacity_lookup,
    )
    monthly_hours = np.asarray([timestamp.days_in_month * 24 for timestamp in date_index], dtype=float)

    result = ds.copy(deep=True)
    for kind in kinds:
        hourly_generation = result[f"{kind}_gen"]
        monthly_generation = result[f"{kind}_monthly_gen"]
        if kind == "offshore":
            hourly_generation = hourly_generation.where(
                np.isfinite(hourly_generation),
                result["total_gen"] - result["onshore_gen"],
            )
            monthly_generation = monthly_generation.where(
                np.isfinite(monthly_generation),
                result["total_monthly_gen"] - result["onshore_monthly_gen"],
            )
        result[f"{kind}_cf"] = _capacity_factor(
            hourly_generation,
            hourly_capacity[kind],
            np.ones(len(time_index), dtype=float),
            f"{kind}_cf",
        )
        result[f"{kind}_monthly_cf"] = _capacity_factor(
            monthly_generation,
            monthly_capacity[kind],
            monthly_hours,
            f"{kind}_monthly_cf",
        )

    result.attrs.update(
        {
            "capacity_factor_source": "city_monthly_wind_capacity.csv",
            "capacity_factor_definition": "generation_mwh / (cumulative_capacity_mw * period_hours)",
        }
    )
    return result


def update_city_file(
    nc_path: Union[str, Path],
    capacity_lookup: dict[tuple[str, str, int, int, str], float],
    overwrite_existing: bool = False,
) -> None:
    path = Path(nc_path)
    temporary_path = path.with_name(f"{path.name}.capacity-factor.tmp")
    if temporary_path.exists():
        raise FileExistsError(f"temporary output already exists: {temporary_path}")

    try:
        with xr.open_dataset(path) as source_ds:
            original = source_ds.load()
        updated = add_city_capacity_factors_to_dataset(
            original,
            capacity_lookup,
            overwrite_existing=overwrite_existing,
        )
        expected_variables = {"total_cf", "total_monthly_cf", "onshore_cf", "onshore_monthly_cf"}
        if "offshore_gen" in original.data_vars:
            expected_variables.update({"offshore_cf", "offshore_monthly_cf"})
        updated = sanitize_netcdf_attrs(updated, "city_capacity_factor_ds")
        write_netcdf_with_retry(updated, temporary_path)
        with xr.open_dataset(temporary_path) as check_ds:
            if not expected_variables.issubset(check_ds.data_vars):
                raise RuntimeError(f"temporary output is missing capacity factor variables: {temporary_path}")
            for variable_name in original.data_vars:
                if not original[variable_name].identical(check_ds[variable_name]):
                    raise RuntimeError(f"original variable changed while adding capacity factors: {variable_name}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def add_city_capacity_factors(
    city_dir: Union[str, Path] = DEFAULT_CITY_DIR,
    capacity_csv: Union[str, Path] = DEFAULT_CAPACITY_CSV,
    overwrite_existing: bool = False,
) -> list[Path]:
    city_root = Path(city_dir)
    city_files = sorted(city_root.glob("*/full_year/gen.nc"))
    if not city_files:
        raise FileNotFoundError(f"no city full-year NetCDF files found under: {city_root}")

    capacity_lookup = load_city_monthly_capacity(capacity_csv)
    for index, city_file in enumerate(city_files, start=1):
        update_city_file(city_file, capacity_lookup, overwrite_existing=overwrite_existing)
        print(f"[{index}/{len(city_files)}] updated {city_file}")
    return city_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Add hourly and monthly capacity factors to city NetCDF files.")
    parser.add_argument("--city-dir", type=Path, default=DEFAULT_CITY_DIR)
    parser.add_argument("--capacity-csv", type=Path, default=DEFAULT_CAPACITY_CSV)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    updated_files = add_city_capacity_factors(
        city_dir=args.city_dir,
        capacity_csv=args.capacity_csv,
        overwrite_existing=args.overwrite_existing,
    )
    print(f"complete: updated {len(updated_files)} city files")


if __name__ == "__main__":
    main()