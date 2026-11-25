from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import xarray as xr

from utils import normalize_text, safe_read_csv, sanitize_netcdf_attrs, write_netcdf_with_retry


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAPACITY_CSV = PROJECT_ROOT / "city_monthly_wind_capacity.csv"
DEFAULT_PROVINCE_DIR = PROJECT_ROOT / "calibrated_gen" / "prov"
ENERGY_TYPE_MAP = {"陆上风电": "onshore", "海上风电": "offshore"}
CAPACITY_COLUMN = "累积装机(MW)"
BUSINESS_KEY_COLUMNS = ["年", "月", "省简称", "市简称", "能源类型"]
GENERATION_VARIABLES = {
    "total": ("total_gen", "total_monthly_gen"),
    "onshore": ("onshore_gen", "onshore_monthly_gen"),
    "offshore": ("offshore_gen", "offshore_monthly_gen"),
}
CAPACITY_FACTOR_VARIABLES = {
    "total_cf",
    "onshore_cf",
    "offshore_cf",
    "total_monthly_cf",
    "onshore_monthly_cf",
    "offshore_monthly_cf",
}


def load_province_monthly_capacity(
    capacity_csv: Union[str, Path],
) -> dict[tuple[str, int, int, str], float]:
    capacity_path = Path(capacity_csv)
    capacity_df = safe_read_csv(capacity_path)
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

    province_capacity = (
        prepared.groupby(["省简称", "年", "月", "kind"], as_index=False)[CAPACITY_COLUMN]
        .sum()
    )
    return {
        (str(row["省简称"]), int(row["年"]), int(row["月"]), str(row["kind"])): float(row[CAPACITY_COLUMN])
        for _, row in province_capacity.iterrows()
    }


def _capacity_values(
    province_short_name: str,
    timestamps: pd.DatetimeIndex,
    capacity_lookup: dict[tuple[str, int, int, str], float],
) -> dict[str, np.ndarray]:
    onshore = np.asarray(
        [capacity_lookup.get((province_short_name, timestamp.year, timestamp.month, "onshore"), 0.0) for timestamp in timestamps],
        dtype=float,
    )
    offshore = np.asarray(
        [capacity_lookup.get((province_short_name, timestamp.year, timestamp.month, "offshore"), 0.0) for timestamp in timestamps],
        dtype=float,
    )
    return {"total": onshore + offshore, "onshore": onshore, "offshore": offshore}


def _capacity_factor(
    generation: xr.DataArray,
    capacity_mw: np.ndarray,
    denominator_hours: np.ndarray,
    variable_name: str,
) -> xr.DataArray:
    generation_values = np.asarray(generation.values, dtype=float)
    denominator = capacity_mw * denominator_hours
    invalid = (denominator <= 0) & (generation_values > 1e-9)
    if invalid.any():
        invalid_index = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"{variable_name} has positive generation with zero capacity at "
            f"{generation.dims[0]} index {invalid_index}: generation={generation_values[invalid_index]}"
        )

    values = np.divide(
        generation_values,
        denominator,
        out=np.zeros_like(generation_values, dtype=float),
        where=denominator > 0,
    )
    if (~np.isfinite(values)).any() or (values < -1e-12).any() or (values > 1 + 1e-9).any():
        raise ValueError(
            f"{variable_name} is outside [0, 1]: min={float(values.min())}, max={float(values.max())}"
        )

    result = xr.DataArray(values, coords=generation.coords, dims=generation.dims, name=variable_name)
    result.attrs.update(
        {
            "long_name": variable_name.replace("_", " "),
            "units": "1",
            "capacity_source": "city_monthly_wind_capacity.csv cumulative installed capacity (MW)",
            "zero_capacity_policy": "0 when both capacity and generation are 0; error when generation is positive",
        }
    )
    return result


def add_capacity_factors_to_dataset(
    ds: xr.Dataset,
    capacity_lookup: dict[tuple[str, int, int, str], float],
    overwrite_existing: bool = False,
) -> xr.Dataset:
    province_short_name = normalize_text(ds.attrs.get("province_short_name"))
    if province_short_name is None:
        raise KeyError("province dataset is missing attrs['province_short_name']")
    if "time" not in ds.coords or "date" not in ds.coords:
        raise KeyError("province dataset must contain time and date coordinates")

    required_generation = {
        variable_name
        for variable_pair in GENERATION_VARIABLES.values()
        for variable_name in variable_pair
    }
    missing_generation = sorted(required_generation - set(ds.data_vars))
    if missing_generation:
        raise KeyError(f"province dataset is missing generation variables: {missing_generation}")

    existing_cf = sorted(CAPACITY_FACTOR_VARIABLES & set(ds.data_vars))
    if existing_cf and not overwrite_existing:
        raise ValueError(f"province dataset already contains capacity factor variables: {existing_cf}")

    time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    date_index = pd.DatetimeIndex(pd.to_datetime(ds["date"].values))
    hourly_capacity = _capacity_values(province_short_name, time_index, capacity_lookup)
    monthly_capacity = _capacity_values(province_short_name, date_index, capacity_lookup)
    monthly_hours = np.asarray([timestamp.days_in_month * 24 for timestamp in date_index], dtype=float)

    result = ds.copy(deep=True)
    for kind, (hourly_generation_name, monthly_generation_name) in GENERATION_VARIABLES.items():
        result[f"{kind}_cf"] = _capacity_factor(
            result[hourly_generation_name],
            hourly_capacity[kind],
            np.ones(len(time_index), dtype=float),
            f"{kind}_cf",
        )
        result[f"{kind}_monthly_cf"] = _capacity_factor(
            result[monthly_generation_name],
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


def update_province_file(
    nc_path: Union[str, Path],
    capacity_lookup: dict[tuple[str, int, int, str], float],
    overwrite_existing: bool = False,
) -> None:
    path = Path(nc_path)
    temporary_path = path.with_name(f"{path.name}.capacity-factor.tmp")
    if temporary_path.exists():
        raise FileExistsError(f"temporary output already exists: {temporary_path}")

    try:
        with xr.open_dataset(path) as source_ds:
            original = source_ds.load()
        updated = add_capacity_factors_to_dataset(original, capacity_lookup, overwrite_existing=overwrite_existing)
        updated = sanitize_netcdf_attrs(updated, "province_capacity_factor_ds")
        write_netcdf_with_retry(updated, temporary_path)
        with xr.open_dataset(temporary_path) as check_ds:
            if not CAPACITY_FACTOR_VARIABLES.issubset(check_ds.data_vars):
                raise RuntimeError(f"temporary output is missing capacity factor variables: {temporary_path}")
            for variable_name in original.data_vars:
                if not original[variable_name].identical(check_ds[variable_name]):
                    raise RuntimeError(f"original variable changed while adding capacity factors: {variable_name}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def add_province_capacity_factors(
    province_dir: Union[str, Path] = DEFAULT_PROVINCE_DIR,
    capacity_csv: Union[str, Path] = DEFAULT_CAPACITY_CSV,
    overwrite_existing: bool = False,
) -> list[Path]:
    province_root = Path(province_dir)
    province_files = sorted(province_root.glob("*/full_year/gen.nc"))
    if not province_files:
        raise FileNotFoundError(f"no province full-year NetCDF files found under: {province_root}")

    capacity_lookup = load_province_monthly_capacity(capacity_csv)
    for index, province_file in enumerate(province_files, start=1):
        update_province_file(province_file, capacity_lookup, overwrite_existing=overwrite_existing)
        print(f"[{index}/{len(province_files)}] updated {province_file}")
    return province_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Add hourly and monthly capacity factors to province NetCDF files.")
    parser.add_argument("--province-dir", type=Path, default=DEFAULT_PROVINCE_DIR)
    parser.add_argument("--capacity-csv", type=Path, default=DEFAULT_CAPACITY_CSV)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    updated_files = add_province_capacity_factors(
        province_dir=args.province_dir,
        capacity_csv=args.capacity_csv,
        overwrite_existing=args.overwrite_existing,
    )
    print(f"complete: updated {len(updated_files)} province files")


if __name__ == "__main__":
    main()