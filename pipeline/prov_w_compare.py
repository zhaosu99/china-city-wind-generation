from __future__ import annotations

import argparse
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from utils import filter_dataarray_by_period, safe_read_csv

SEASONS = ("spring", "summer", "autumn", "winter")


def _build_province_farm_map(
    farm_csv_path: Path,
) -> dict[str, dict[str, Any]]:
    farm_df = safe_read_csv(farm_csv_path)
    required_cols = ["风电场ID", "省全称", "省简称"]
    missing_cols = [col for col in required_cols if col not in farm_df.columns]
    if missing_cols:
        raise ValueError(f"farm_csv 缺少必要列: {missing_cols}")

    farm_df = farm_df[required_cols].copy().dropna(subset=required_cols)
    farm_df["风电场ID"] = pd.to_numeric(farm_df["风电场ID"], errors="coerce")
    farm_df = farm_df.dropna(subset=["风电场ID"])
    farm_df["风电场ID"] = farm_df["风电场ID"].astype(int)
    if farm_df.empty:
        raise ValueError("farm_csv 中未找到有效的风电场ID/省份信息")

    province_map: dict[str, dict[str, Any]] = {}
    for province_full, group in farm_df.groupby("省全称", sort=True):
        province_short = str(group["省简称"].iloc[0]).strip()
        farm_ids = sorted(group["风电场ID"].astype(int).tolist())
        province_map[str(province_full)] = {
            "province_short": province_short,
            "farm_ids": farm_ids,
        }
    return province_map


def _to_time_1d(da: xr.DataArray, farm_id: int, var_name: str) -> np.ndarray:
    squeezed = da.squeeze(drop=True)
    if "time" not in squeezed.dims:
        raise ValueError(f"farm_id={farm_id} var={var_name} 缺少time维度")

    if len(squeezed.dims) != 1:
        remain_dims = [d for d in squeezed.dims if d != "time"]
        raise ValueError(
            f"farm_id={farm_id} var={var_name} 不是单点格网，仍有额外维度: {remain_dims}"
        )

    return np.asarray(squeezed.values).reshape(-1)


def _process_province_season(task: dict[str, Any]) -> dict[str, Any]:
    province_full = str(task["province_full"])
    province_short = str(task["province_short"])
    farm_ids = [int(fid) for fid in task["farm_ids"]]
    season = str(task["season"])
    input_dir = Path(task["input_dir"])
    output_dir = Path(task["output_dir"])
    era5_col_name = str(task["era5_col_name"])
    cdmet_col_name = str(task["cdmet_col_name"])
    filename_template = str(task["filename_template"])
    overwrite = bool(task["overwrite"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{province_short}_{season}_w_compare.csv"
    if out_path.exists() and not overwrite:
        return {
            "province_full": province_full,
            "province_short": province_short,
            "season": season,
            "farm_count": len(farm_ids),
            "written": False,
            "warned": False,
            "skipped_existing": True,
        }

    season_era5_parts: list[np.ndarray] = []
    season_cdmet_parts: list[np.ndarray] = []

    for farm_id in farm_ids:
        nc_path = input_dir / filename_template.format(farm_id=farm_id)
        if not nc_path.exists():
            raise FileNotFoundError(f"缺少风电场文件: {nc_path}")

        with xr.open_dataset(nc_path) as ds:
            if "farm_era5_w" not in ds.data_vars:
                raise KeyError(f"文件缺少变量 farm_era5_w: {nc_path}")
            if "farm_era5_cdmet_w" not in ds.data_vars:
                raise KeyError(f"文件缺少变量 farm_era5_cdmet_w: {nc_path}")

            da_era5 = ds["farm_era5_w"]
            da_cdmet = ds["farm_era5_cdmet_w"]

            da_era5_s = filter_dataarray_by_period(da_era5, period=season, time_name="time", drop=True)
            da_cdmet_s = filter_dataarray_by_period(da_cdmet, period=season, time_name="time", drop=True)

            arr_era5 = _to_time_1d(da_era5_s, farm_id=farm_id, var_name="farm_era5_w")
            arr_cdmet = _to_time_1d(da_cdmet_s, farm_id=farm_id, var_name="farm_era5_cdmet_w")

            if len(arr_era5) != len(arr_cdmet):
                raise ValueError(
                    "同一风电场同一季节的ERA5/CDMET长度不一致: "
                    f"farm_id={farm_id}, season={season}, era5={len(arr_era5)}, cdmet={len(arr_cdmet)}"
                )

            season_era5_parts.append(arr_era5)
            season_cdmet_parts.append(arr_cdmet)
    if len(season_era5_parts) == 0:
        warnings.warn(
            f"[{province_full}] [{season}] 没有可写入数据，跳过输出",
            stacklevel=2,
        )
        return {
            "province_full": province_full,
            "province_short": province_short,
            "season": season,
            "farm_count": len(farm_ids),
            "written": False,
            "warned": True,
            "skipped_existing": False,
        }

    merged_era5 = np.concatenate(season_era5_parts)
    merged_cdmet = np.concatenate(season_cdmet_parts)

    if len(merged_era5) == 0:
        warnings.warn(
            f"[{province_full}] [{season}] 合并后长度为0，跳过输出",
            stacklevel=2,
        )
        return {
            "province_full": province_full,
            "province_short": province_short,
            "season": season,
            "farm_count": len(farm_ids),
            "written": False,
            "warned": True,
            "skipped_existing": False,
        }

    if len(merged_era5) != len(merged_cdmet):
        raise ValueError(
            f"[{province_full}] [{season}] 省级拼接后长度不一致: "
            f"era5={len(merged_era5)}, cdmet={len(merged_cdmet)}"
        )

    out_df = pd.DataFrame(
        {
            era5_col_name: merged_era5,
            cdmet_col_name: merged_cdmet,
        }
    )

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    return {
        "province_full": province_full,
        "province_short": province_short,
        "season": season,
        "farm_count": len(farm_ids),
        "written": True,
        "warned": False,
        "skipped_existing": False,
    }


def prov_w_compare(
    farm_csv_path: Path,
    input_dir: Path,
    output_dir: Path,
    era5_col_name: str = "era5_w",
    cdmet_col_name: str = "cdmet_w",
    filename_template: str = "farm_{farm_id}_period_full_year_k_1_b_0_w_compare.nc",
    max_workers: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    从w_compare目录读取风电场ERA5/CDMET风速，按省-季节拼接并输出CSV。

    规则:
    1. 省内风电场按风电场ID升序拼接，且四个季节均复用相同顺序。
    2. 缺失文件、坏文件、缺变量、长度不一致时直接报错终止。
    3. 某省某季无数据时不输出文件，并发出warning。
    4. 当输出 CSV 已存在且 overwrite=False 时，跳过该省季任务。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    province_map = _build_province_farm_map(farm_csv_path=farm_csv_path)
    if not province_map:
        raise ValueError("未构建出任何省级风电场映射，请检查farm_csv和输入目录")

    tasks = []
    for province_full, info in province_map.items():
        for season in SEASONS:
            tasks.append(
                {
                    "province_full": province_full,
                    "province_short": info["province_short"],
                    "farm_ids": info["farm_ids"],
                    "season": season,
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "era5_col_name": era5_col_name,
                    "cdmet_col_name": cdmet_col_name,
                    "filename_template": filename_template,
                    "overwrite": overwrite,
                }
            )

    max_workers_final = len(tasks) if max_workers is None else max(1, min(int(max_workers), len(tasks)))
    print(
        f"Found {len(province_map)} provinces and {len(tasks)} (province, season) tasks. "
        f"Using ProcessPoolExecutor with {max_workers_final} workers."
    )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers_final) as executor:
        future_map = {executor.submit(_process_province_season, task): task for task in tasks}
        for idx, future in enumerate(as_completed(future_map), 1):
            task = future_map[future]
            province_full = task["province_full"]
            season = task["season"]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{idx}/{len(tasks)}] done {province_full}/{season}: "
                    f"farms={result['farm_count']}, written={result['written']}, warned={result['warned']}, "
                    f"skipped_existing={result.get('skipped_existing', False)}"
                )
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    total_written = int(sum(1 for item in results if item["written"]))
    total_warned = int(sum(1 for item in results if item["warned"]))
    total_skipped_existing = int(sum(1 for item in results if item.get("skipped_existing", False)))
    return {
        "province_count": len(province_map),
        "task_count": len(results),
        "total_written_files": total_written,
        "total_warning_skips": total_warned,
        "total_existing_skips": total_skipped_existing,
        "results": results,
    }


def parse_arguments() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Province-season w_compare concatenation tool (ERA5 vs CDMET)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--farm-csv-path",
        type=Path,
        default=Path("/geogfs1/groups/geog_liulb/hongyu/data/wind_farms_inventory_till_2023.csv"),
        help="Path to farm inventory CSV.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/geogfs1/groups/geog_liulb/hongyu/results/farm_raw_w/w_compare"),
        help="Directory containing farm-level w_compare nc files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/geogfs1/groups/geog_liulb/hongyu/results/prov_w_compare"),
        help="Directory for province-season output CSV files.",
    )
    parser.add_argument(
        "--era5-col-name",
        type=str,
        default="era5_w",
        help="Output CSV column name for ERA5 wind speed.",
    )
    parser.add_argument(
        "--cdmet-col-name",
        type=str,
        default="cdmet_w",
        help="Output CSV column name for CDMET wind speed.",
    )
    parser.add_argument(
        "--filename-template",
        type=str,
        default="farm_{farm_id}_period_full_year_k_1_b_0_w_compare.nc",
        help="Filename template used when opening each farm file.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Process count for province-level parallelism. Default: number of provinces.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing province-season CSV files.",
    )

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    try:
        print("Initializing argument parser...")
        run_kwargs = parse_arguments()
        print("Starting province-season wind speed concatenation with config:")
        for key, value in run_kwargs.items():
            print(f"  {key}: {value}")
        print()

        summary = prov_w_compare(**run_kwargs)
        print("\n--- Execution Complete ---")
        print(
            "Summary: "
            f"provinces={summary['province_count']}, "
            f"written_files={summary['total_written_files']}, "
            f"warning_skips={summary['total_warning_skips']}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[CRITICAL] Script failed: {str(e)}")
        traceback.print_exc()
