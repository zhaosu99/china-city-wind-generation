from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import xarray as xr

from aggr_farm_aux import DEFAULT_INVENTORY_CSV_PATH
from prov_gen_enums import _build_official_details
from utils import (
    adjust_da_by_time,
    hourly_to_monthly,
    normalize_text,
    open_dataset_with_fallback,
    resolve_existing_column,
    safe_read_csv,
    write_netcdf_with_retry,
)

logger = logging.getLogger(__name__)


DEFAULT_DATA_DIR = Path("/geogfs1/groups/geog_liulb/hongyu/data")
DEFAULT_MONTHLY_CUMULATIVE_CSV = DEFAULT_DATA_DIR / "分省月度数据-风电-当年累积值(亿千瓦时).csv"
DEFAULT_MONTHLY_CURRENT_CSV = DEFAULT_DATA_DIR / "分省月度数据-风电-当期值(亿千瓦时).csv"
DEFAULT_PRE2012_YEARLY_CSV = DEFAULT_DATA_DIR / "2012年以前风电发电量(亿千瓦时).csv"
DEFAULT_HPC_INVENTORY_CSV = DEFAULT_DATA_DIR / "wind_farms_inventory_till_2023.csv"
KWH_100M_TO_MWH = 100_000.0
PROVINCE_GEN_START_MONTHS: dict[str, tuple[int, int]] = {
    "新疆": (1989, 10),
    "内蒙古": (1990, 1),
}


def _default_inventory_csv() -> Path:
    if DEFAULT_HPC_INVENTORY_CSV.exists():
        return DEFAULT_HPC_INVENTORY_CSV
    return Path(DEFAULT_INVENTORY_CSV_PATH)


def _parse_year_month_header(text: Any) -> Optional[tuple[int, int]]:
    normalized = normalize_text(text)
    if normalized is None:
        return None

    match = re.fullmatch(r"(\d{4})年(\d{1,2})月", normalized)
    if match is None:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return year, month


def _year_label(year: int) -> str:
    return f"{year}年"


def _month_label(year: int, month: int) -> str:
    return f"{year}年{month}月"


def _is_valid_official_target(value: Any) -> bool:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return False
    return float(numeric_value) > 0


def _load_inventory_maps(inventory_csv_path: Union[str, Path]) -> tuple[dict[int, str], dict[str, str], set[str]]:
    inventory_df = safe_read_csv(inventory_csv_path)
    if inventory_df is None or inventory_df.empty:
        raise ValueError(f"inventory CSV is empty: {inventory_csv_path}")

    id_col = resolve_existing_column(inventory_df.columns, ["风电场ID", "风电场Id", "farm_id", "ID"], "风电场ID")
    prov_short_col = resolve_existing_column(inventory_df.columns, ["省简称", "prov_short"], "省简称")
    prov_long_col = resolve_existing_column(inventory_df.columns, ["省全称", "prov_long"], "省全称")

    farm_to_prov_short: dict[int, str] = {}
    full_to_short: dict[str, str] = {}
    short_set: set[str] = set()

    for _, row in inventory_df.iterrows():
        farm_id_raw = row.get(id_col)
        prov_short = normalize_text(row.get(prov_short_col))
        prov_long = normalize_text(row.get(prov_long_col))

        if prov_short is not None:
            short_set.add(prov_short)
        if prov_short is not None and prov_long is not None:
            full_to_short[prov_long] = prov_short

        if prov_short is None or pd.isna(farm_id_raw):
            continue

        try:
            farm_id = int(farm_id_raw)
        except (TypeError, ValueError):
            continue
        farm_to_prov_short[farm_id] = prov_short

    if not farm_to_prov_short:
        raise ValueError(f"failed to build farm province mapping from inventory: {inventory_csv_path}")

    return farm_to_prov_short, full_to_short, short_set


def _resolve_prov_short_name(raw_name: Any, full_to_short: dict[str, str], short_set: set[str]) -> Optional[str]:
    normalized = normalize_text(raw_name)
    if normalized is None:
        return None

    if normalized in full_to_short:
        return full_to_short[normalized]
    if normalized in short_set:
        return normalized

    suffixes = [
        "维吾尔自治区",
        "壮族自治区",
        "回族自治区",
        "特别行政区",
        "自治区",
        "省",
        "市",
    ]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            trimmed = normalized[: -len(suffix)]
            if trimmed in short_set:
                return trimmed

    return None


def _read_official_monthly_targets(
    monthly_current_csv_path: Union[str, Path],
    monthly_cumulative_csv_path: Union[str, Path],
    full_to_short: dict[str, str],
    short_set: set[str],
    unit_to_mwh_factor: float,
) -> dict[str, dict[str, dict[Any, float]]]:
    current_df = safe_read_csv(monthly_current_csv_path)
    cumulative_df = safe_read_csv(monthly_cumulative_csv_path)

    if current_df is None or current_df.empty:
        raise ValueError(f"monthly current CSV is empty: {monthly_current_csv_path}")
    if cumulative_df is None or cumulative_df.empty:
        raise ValueError(f"monthly cumulative CSV is empty: {monthly_cumulative_csv_path}")

    current_region_col = resolve_existing_column(current_df.columns, ["地区", "region", "省", "province"], "地区")
    cumulative_region_col = resolve_existing_column(cumulative_df.columns, ["地区", "region", "省", "province"], "地区")

    official_details = _build_official_details(
        current_df=current_df,
        cumulative_df=cumulative_df,
        current_region_col=current_region_col,
        cumulative_region_col=cumulative_region_col,
        full_to_short=full_to_short,
        short_set=short_set,
        unit_to_mwh_factor=unit_to_mwh_factor,
    )

    official_targets: dict[str, dict[str, dict[Any, float]]] = {}
    for prov_short, detail in official_details.items():
        province_targets = {"monthly": {}, "jan_feb": {}}
        for label, value in detail.get("series", {}).items():
            jan_feb_match = re.fullmatch(r"(\d{4})年1-2月", label)
            if jan_feb_match is not None:
                province_targets["jan_feb"][int(jan_feb_match.group(1))] = float(value)
                continue

            parsed = _parse_year_month_header(label)
            if parsed is None:
                continue
            province_targets["monthly"][parsed] = float(value)

        official_targets[prov_short] = province_targets

    return official_targets


def _read_official_pre2012_yearly_targets(
    yearly_csv_path: Union[str, Path],
    unit_to_mwh_factor: float,
) -> dict[int, float]:
    yearly_df = safe_read_csv(yearly_csv_path)
    if yearly_df is None or yearly_df.empty:
        raise ValueError(f"pre-2012 yearly CSV is empty: {yearly_csv_path}")

    region_col = resolve_existing_column(yearly_df.columns, ["地区", "region", "省", "province"], "地区")
    region_series = yearly_df[region_col].map(normalize_text)
    national_mask = region_series == "全国"
    if national_mask.any():
        row = yearly_df.loc[national_mask].iloc[0]
    elif len(yearly_df) == 1:
        row = yearly_df.iloc[0]
    else:
        raise ValueError(f"unable to locate national row in yearly CSV: {yearly_csv_path}")

    official_yearly: dict[int, float] = {}
    for col in yearly_df.columns:
        col_text = normalize_text(col)
        if col_text is None or re.fullmatch(r"\d{4}", col_text) is None:
            continue
        year = int(col_text)
        value = pd.to_numeric(row.get(col), errors="coerce")
        if not _is_valid_official_target(value):
            continue
        official_yearly[year] = float(value) * unit_to_mwh_factor

    return official_yearly


def _copy_input_tree(input_dir: Union[str, Path], output_dir: Union[str, Path], overwrite: bool) -> tuple[Path, Path]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"input_dir does not exist or is not a directory: {input_path}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"output_dir already exists, set overwrite=True to replace it: {output_path}")
        shutil.rmtree(output_path)

    shutil.copytree(input_path, output_path)
    return input_path, output_path


def _dataset_period_is_full_year(ds: xr.Dataset, nc_path: Optional[Path] = None) -> bool:
    if normalize_text(ds.attrs.get("period")) == "full_year":
        return True
    if nc_path is not None:
        return any(normalize_text(part) == "full_year" for part in nc_path.parts)
    return False


def _extract_total_monthly_series(ds: xr.Dataset) -> dict[tuple[int, int], float]:
    if "total_gen" in ds.data_vars and "time" in ds["total_gen"].dims:
        monthly_da = hourly_to_monthly(ds["total_gen"], time_name="time", agg_method="sum")
        return {
            (int(timestamp.year), int(timestamp.month)): float(value)
            for timestamp, value in monthly_da.to_series().items()
            if not pd.isna(value)
        }

    if "total_monthly_gen" in ds.data_vars and "date" in ds["total_monthly_gen"].dims:
        return {
            (int(timestamp.year), int(timestamp.month)): float(value)
            for timestamp, value in ds["total_monthly_gen"].to_series().items()
            if not pd.isna(value)
        }

    raise KeyError("dataset does not contain total_gen or total_monthly_gen")


def _extract_total_yearly_series(ds: xr.Dataset) -> dict[int, float]:
    monthly_series = _extract_total_monthly_series(ds)
    yearly_totals: dict[int, float] = {}
    for (year, _month), value in monthly_series.items():
        yearly_totals[year] = yearly_totals.get(year, 0.0) + float(value)
    return yearly_totals


def _resolve_dataset_province_short(
    ds: xr.Dataset,
    nc_path: Path,
    farm_to_prov_short: dict[int, str],
    full_to_short: dict[str, str],
    short_set: set[str],
) -> str:
    for attr_name in ("province_short_name", "prov_short", "province", "省简称"):
        prov_short = _resolve_prov_short_name(ds.attrs.get(attr_name), full_to_short, short_set)
        if prov_short is not None:
            return prov_short

    for attr_name in ("province_long_name", "prov_long", "省全称"):
        prov_short = _resolve_prov_short_name(ds.attrs.get(attr_name), full_to_short, short_set)
        if prov_short is not None:
            return prov_short

    for part in nc_path.parts:
        match = re.fullmatch(r"farm_(\d+)", part)
        if match is None:
            match = re.fullmatch(r"farm_(\d+).*", part)
        if match is None:
            continue
        farm_id = int(match.group(1))
        if farm_id in farm_to_prov_short:
            return farm_to_prov_short[farm_id]

    stem_match = re.search(r"farm_(\d+)", nc_path.stem)
    if stem_match is not None:
        farm_id = int(stem_match.group(1))
        if farm_id in farm_to_prov_short:
            return farm_to_prov_short[farm_id]

    raise KeyError(f"unable to resolve province for dataset: {nc_path}")


def _discover_province_full_year_files(
    root_dir: Union[str, Path],
    farm_to_prov_short: dict[int, str],
    full_to_short: dict[str, str],
    short_set: set[str],
) -> dict[str, Path]:
    prov_root = Path(root_dir) / "prov"
    if not prov_root.exists():
        raise FileNotFoundError(f"province directory does not exist under gen root: {prov_root}")

    province_files: dict[str, Path] = {}
    for nc_path in prov_root.rglob("*.nc"):
        ds = open_dataset_with_fallback(nc_path)
        if not _dataset_period_is_full_year(ds, nc_path=nc_path):
            continue
        prov_short = _resolve_dataset_province_short(ds, nc_path, farm_to_prov_short, full_to_short, short_set)
        if prov_short in province_files and province_files[prov_short] != nc_path:
            raise ValueError(
                f"multiple full_year province files found for {prov_short}: "
                f"{province_files[prov_short]} and {nc_path}"
            )
        province_files[prov_short] = nc_path

    if not province_files:
        raise ValueError(f"no full_year province gen files found under: {prov_root}")

    return province_files


def _build_national_year_factors(
    province_files: dict[str, Path],
    official_yearly_targets: dict[int, float],
) -> dict[int, float]:
    national_model_yearly: dict[int, float] = {}
    for prov_short, nc_path in province_files.items():
        ds = open_dataset_with_fallback(nc_path)
        ds = _apply_generation_start_cutoff(ds, prov_short)
        model_yearly = _extract_total_yearly_series(ds)
        for year, value in model_yearly.items():
            national_model_yearly[year] = national_model_yearly.get(year, 0.0) + float(value)

    national_factors: dict[int, float] = {}
    for year, official_value in sorted(official_yearly_targets.items()):
        model_value = national_model_yearly.get(year)
        if model_value is None:
            logger.warning("skip yearly adjustment for %s because model total is missing", year)
            continue
        if model_value == 0:
            logger.warning("skip yearly adjustment for %s because model total is zero", year)
            continue
        national_factors[year] = float(official_value) / float(model_value)

    return national_factors


def _build_province_adjustment_info(
    province_files: dict[str, Path],
    official_monthly_targets: dict[str, dict[str, dict[Any, float]]],
    national_year_factors: dict[int, float],
) -> dict[str, dict[Union[str, tuple[str, str]], float]]:
    adjustment_info_map: dict[str, dict[Union[str, tuple[str, str]], float]] = {}

    for prov_short, nc_path in province_files.items():
        ds = open_dataset_with_fallback(nc_path)
        ds = _apply_generation_start_cutoff(ds, prov_short)
        model_monthly = _extract_total_monthly_series(ds)
        adjust_info: dict[Union[str, tuple[str, str]], float] = {
            _year_label(year): factor for year, factor in sorted(national_year_factors.items())
        }

        province_targets = official_monthly_targets.get(prov_short, {"monthly": {}, "jan_feb": {}})
        for (year, month), official_value in sorted(province_targets.get("monthly", {}).items()):
            if not _is_valid_official_target(official_value):
                logger.info(
                    "skip monthly adjustment for %s %s-%02d because official value is missing or non-positive",
                    prov_short,
                    year,
                    month,
                )
                continue
            model_value = model_monthly.get((year, month))
            if model_value is None:
                logger.warning(
                    "skip monthly adjustment for %s %s-%02d because model value is missing",
                    prov_short,
                    year,
                    month,
                )
                continue
            if model_value == 0:
                logger.warning(
                    "skip monthly adjustment for %s %s-%02d because model value is zero",
                    prov_short,
                    year,
                    month,
                )
                continue
            adjust_info[_month_label(year, month)] = float(official_value) / float(model_value)

        for year, official_value in sorted(province_targets.get("jan_feb", {}).items()):
            if not _is_valid_official_target(official_value):
                logger.info(
                    "skip Jan-Feb adjustment for %s %s because official value is missing or non-positive",
                    prov_short,
                    year,
                )
                continue
            jan_value = model_monthly.get((year, 1), 0.0)
            feb_value = model_monthly.get((year, 2), 0.0)
            model_value = float(jan_value) + float(feb_value)
            if model_value == 0:
                logger.warning("skip Jan-Feb adjustment for %s %s because model value is zero", prov_short, year)
                continue
            adjust_info[(f"{year}年1月", f"{year}年2月")] = float(official_value) / model_value

        adjustment_info_map[prov_short] = adjust_info

    return adjustment_info_map


def _monthly_source_var(var_name: str) -> Optional[str]:
    mapping = {
        "monthly_gen": "gen",
        "total_monthly_gen": "total_gen",
        "onshore_monthly_gen": "onshore_gen",
        "offshore_monthly_gen": "offshore_gen",
    }
    return mapping.get(var_name)


def _apply_generation_start_cutoff(ds: xr.Dataset, prov_short: str) -> xr.Dataset:
    start_year_month = PROVINCE_GEN_START_MONTHS.get(prov_short)
    if start_year_month is None:
        return ds

    start_year, start_month = start_year_month
    cutoff_time = pd.Timestamp(year=start_year, month=start_month, day=1)
    updated_ds = ds.copy(deep=True)

    for var_name, da in ds.data_vars.items():
        if "time" in da.dims:
            time_index = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
            keep_mask = xr.DataArray(time_index >= cutoff_time, coords={"time": da["time"]}, dims=("time",))
            updated_ds[var_name] = da.where(keep_mask, 0)
            continue

        if "date" in da.dims:
            date_index = pd.DatetimeIndex(pd.to_datetime(da["date"].values))
            keep_mask = xr.DataArray(date_index >= cutoff_time, coords={"date": da["date"]}, dims=("date",))
            updated_ds[var_name] = da.where(keep_mask, 0)

    updated_ds.attrs.update(ds.attrs)
    updated_ds.attrs["final_gen_start_cutoff_applied"] = 1
    updated_ds.attrs["final_gen_start_cutoff_province"] = prov_short
    updated_ds.attrs["final_gen_start_cutoff_month"] = f"{start_year:04d}-{start_month:02d}"
    return updated_ds


def _apply_adjustments_to_dataset(
    ds: xr.Dataset,
    adjust_info: dict[Union[str, tuple[str, str]], float],
    prov_short: str,
) -> xr.Dataset:
    base_ds = _apply_generation_start_cutoff(ds, prov_short)
    updated_ds = base_ds.copy(deep=True)
    adjusted_hourly_vars: dict[str, xr.DataArray] = {}

    for var_name, da in base_ds.data_vars.items():
        if "time" not in da.dims:
            continue
        adjusted_da = adjust_da_by_time(da, adjust_info, time_name="time")
        adjusted_da = adjusted_da.rename(var_name)
        adjusted_hourly_vars[var_name] = adjusted_da
        updated_ds[var_name] = adjusted_da

    for var_name, da in base_ds.data_vars.items():
        if "date" not in da.dims:
            continue

        source_var = _monthly_source_var(var_name)
        if source_var is not None and source_var in adjusted_hourly_vars:
            monthly_da = hourly_to_monthly(adjusted_hourly_vars[source_var], time_name="time", agg_method="sum")
            if "time" in monthly_da.dims:
                monthly_da = monthly_da.rename({"time": "date"})
            monthly_da = monthly_da.rename(var_name)
            monthly_da.attrs.update(da.attrs)
            updated_ds[var_name] = monthly_da
            continue

        adjusted_monthly_da = adjust_da_by_time(da, adjust_info, time_name="date")
        adjusted_monthly_da = adjusted_monthly_da.rename(var_name)
        updated_ds[var_name] = adjusted_monthly_da

    updated_ds.attrs.update(base_ds.attrs)
    updated_ds.attrs["final_gen_adjustment_applied"] = 1
    updated_ds.attrs["final_gen_adjustment_province"] = prov_short
    updated_ds.attrs["final_gen_adjustment_rule_count"] = len(adjust_info)
    return updated_ds


def _apply_adjustments_to_tree(
    output_dir: Union[str, Path],
    adjustment_info_map: dict[str, dict[Union[str, tuple[str, str]], float]],
    farm_to_prov_short: dict[int, str],
    full_to_short: dict[str, str],
    short_set: set[str],
) -> int:
    adjusted_count = 0
    for nc_path in sorted(Path(output_dir).rglob("*.nc")):
        ds = open_dataset_with_fallback(nc_path)
        prov_short = _resolve_dataset_province_short(ds, nc_path, farm_to_prov_short, full_to_short, short_set)
        adjust_info = adjustment_info_map.get(prov_short)
        if not adjust_info:
            cutoff_ds = _apply_generation_start_cutoff(ds, prov_short)
            if cutoff_ds is ds:
                logger.info("skip %s because no adjustment info is available for %s", nc_path, prov_short)
                continue
            write_netcdf_with_retry(cutoff_ds, nc_path)
            adjusted_count += 1
            continue

        updated_ds = _apply_adjustments_to_dataset(ds, adjust_info, prov_short)
        write_netcdf_with_retry(updated_ds, nc_path)
        adjusted_count += 1

    return adjusted_count


def final_gen(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    monthly_current_csv_path: Union[str, Path] = DEFAULT_MONTHLY_CURRENT_CSV,
    monthly_cumulative_csv_path: Union[str, Path] = DEFAULT_MONTHLY_CUMULATIVE_CSV,
    pre2012_yearly_csv_path: Union[str, Path] = DEFAULT_PRE2012_YEARLY_CSV,
    inventory_csv_path: Union[str, Path] = _default_inventory_csv(),
    unit_to_mwh_factor: float = KWH_100M_TO_MWH,
    overwrite: bool = False,
) -> dict[str, Any]:
    _, output_path = _copy_input_tree(input_dir=input_dir, output_dir=output_dir, overwrite=overwrite)

    farm_to_prov_short, full_to_short, short_set = _load_inventory_maps(inventory_csv_path)
    province_files = _discover_province_full_year_files(
        root_dir=output_path,
        farm_to_prov_short=farm_to_prov_short,
        full_to_short=full_to_short,
        short_set=short_set,
    )

    official_monthly_targets = _read_official_monthly_targets(
        monthly_current_csv_path=monthly_current_csv_path,
        monthly_cumulative_csv_path=monthly_cumulative_csv_path,
        full_to_short=full_to_short,
        short_set=short_set,
        unit_to_mwh_factor=unit_to_mwh_factor,
    )
    official_yearly_targets = _read_official_pre2012_yearly_targets(
        yearly_csv_path=pre2012_yearly_csv_path,
        unit_to_mwh_factor=unit_to_mwh_factor,
    )
    national_year_factors = _build_national_year_factors(
        province_files=province_files,
        official_yearly_targets=official_yearly_targets,
    )
    adjustment_info_map = _build_province_adjustment_info(
        province_files=province_files,
        official_monthly_targets=official_monthly_targets,
        national_year_factors=national_year_factors,
    )

    adjusted_file_count = _apply_adjustments_to_tree(
        output_dir=output_path,
        adjustment_info_map=adjustment_info_map,
        farm_to_prov_short=farm_to_prov_short,
        full_to_short=full_to_short,
        short_set=short_set,
    )

    return {
        "input_dir": str(Path(input_dir)),
        "output_dir": str(output_path),
        "province_count": len(province_files),
        "adjusted_file_count": adjusted_file_count,
        "national_year_factor_count": len(national_year_factors),
        "province_adjustment_count": sum(len(item) for item in adjustment_info_map.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a gen directory and align farm/city/prov generation outputs to official monthly and yearly statistics."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input gen directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output gen directory")
    parser.add_argument(
        "--monthly_current_csv_path",
        type=str,
        default=str(DEFAULT_MONTHLY_CURRENT_CSV),
        help="Provincial monthly current-value CSV in 100 million kWh",
    )
    parser.add_argument(
        "--monthly_cumulative_csv_path",
        type=str,
        default=str(DEFAULT_MONTHLY_CUMULATIVE_CSV),
        help="Provincial cumulative CSV in 100 million kWh; February holds Jan-Feb totals",
    )
    parser.add_argument(
        "--pre2012_yearly_csv_path",
        type=str,
        default=str(DEFAULT_PRE2012_YEARLY_CSV),
        help="National yearly CSV for 2005-2012 in 100 million kWh",
    )
    parser.add_argument(
        "--inventory_csv_path",
        type=str,
        default=str(_default_inventory_csv()),
        help="Wind farm inventory CSV used to map farm files back to provinces",
    )
    parser.add_argument(
        "--unit_to_mwh_factor",
        type=float,
        default=KWH_100M_TO_MWH,
        help="Conversion factor from CSV units to MWh",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output_dir if it already exists")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    result = final_gen(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        monthly_current_csv_path=args.monthly_current_csv_path,
        monthly_cumulative_csv_path=args.monthly_cumulative_csv_path,
        pre2012_yearly_csv_path=args.pre2012_yearly_csv_path,
        inventory_csv_path=args.inventory_csv_path,
        unit_to_mwh_factor=args.unit_to_mwh_factor,
        overwrite=args.overwrite,
    )
    logger.info("final_gen done: %s", result)


if __name__ == "__main__":
    main()