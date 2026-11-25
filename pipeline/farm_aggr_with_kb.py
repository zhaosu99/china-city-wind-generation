from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import pandas as pd

from aggr_farm_aux import DEFAULT_INVENTORY_CSV_PATH, SeasonName, _filter_inventory_farms, _load_inventory_table
from aggr_farm import aggr_farm_output
from parse_aux import parse_csv_list, parse_gen_adjust_info, parse_none, parse_select_period, parse_tf
from utils import normalize_text

from config import clamp_two_layer_workers  # 并行预算护栏

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


PROVINCE_KB_FALLBACKS: dict[str, str] = {
    "香港": "广东",
    "西藏": "青海",
}


def _write_batch_inventory_csv(batch_df: pd.DataFrame, batch_index: int, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    batch_path = tmp_dir / f"inventory_batch_{batch_index:03d}.csv"
    batch_df.to_csv(batch_path, index=False, encoding="utf-8-sig")
    return batch_path


def _build_province_batch_specs(
    filtered_inventory_path: Union[str, Path],
    tmp_dir: Path,
) -> list[dict[str, Any]]:
    filtered_df, column_map = _load_inventory_table(filtered_inventory_path)
    province_groups = []
    for province_short, group_df in filtered_df.groupby(column_map["prov_short"], dropna=False):
        if province_short is None or pd.isna(province_short):
            continue
        province_groups.append((str(province_short), group_df.copy()))

    province_groups.sort(key=lambda item: len(item[1]), reverse=True)

    batch_specs: list[dict[str, Any]] = []
    for batch_index, (province_short, group_df) in enumerate(province_groups, start=1):
        batch_inventory_path = _write_batch_inventory_csv(group_df, batch_index=batch_index, tmp_dir=tmp_dir)
        batch_specs.append(
            {
                "batch_index": batch_index,
                "batch_name": province_short,
                "province_short": province_short,
                "farm_count": int(len(group_df)),
                "inventory_csv_path": str(batch_inventory_path),
            }
        )

    return batch_specs


def _run_single_farm_aggr_batch_task(task: dict[str, Any]) -> dict[str, Any]:
    result = aggr_farm_output(**task["params"])
    return {
        "batch_index": task["batch_index"],
        "batch_name": task["batch_name"],
        "province_short": task["province_short"],
        "farm_count": task["farm_count"],
        "result": result,
    }


def _merge_period_output_maps(batch_results: list[dict[str, Any]], key: str) -> dict[str, dict[Any, str]]:
    merged: dict[str, dict[Any, str]] = {}
    for batch_item in batch_results:
        result_map = batch_item["result"].get(key, {}) or {}
        for period_key, item_map in result_map.items():
            merged.setdefault(period_key, {})
            merged[period_key].update(item_map)
    return merged


def _merge_batch_results(
    *,
    batch_results: list[dict[str, Any]],
    skip_info: dict[str, Any],
    output_root: Path,
    inventory_csv_path: Union[str, Path],
    optimal_kb_csv: Union[str, Path],
    effective_outer_max_workers: int,
    effective_inner_max_workers: Optional[int],
    failed_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    if not batch_results:
        return {
            "inventory_csv_path": str(Path(inventory_csv_path)),
            "output_dir": str(output_root),
            "optimal_kb_csv": str(Path(optimal_kb_csv)),
            "selected_farm_ids": [],
            "farm_generation_files": {},
            "farm_wind_files": {},
            "farm_raw_w_files": {},
            "city_generation_files": {},
            "province_generation_files": {},
            "province_seasonal_wind_csvs": {},
            "batch_count": 0,
            "success_count": 0,
            "failure_count": len(failed_batches),
            "outer_max_workers": effective_outer_max_workers,
            "inner_max_workers": effective_inner_max_workers,
            "failed_batches": failed_batches,
            **skip_info,
        }

    first_result = batch_results[0]["result"]
    selected_farm_ids = sorted(
        {
            int(farm_id)
            for batch_item in batch_results
            for farm_id in batch_item["result"].get("selected_farm_ids", [])
        }
    )

    merged = dict(first_result)
    merged.update(
        {
            "inventory_csv_path": str(Path(inventory_csv_path)),
            "output_dir": str(output_root),
            "optimal_kb_csv": str(Path(optimal_kb_csv)),
            "selected_farm_ids": selected_farm_ids,
            "farm_generation_files": _merge_period_output_maps(batch_results, "farm_generation_files"),
            "farm_wind_files": _merge_period_output_maps(batch_results, "farm_wind_files"),
            "farm_raw_w_files": _merge_period_output_maps(batch_results, "farm_raw_w_files"),
            "city_generation_files": _merge_period_output_maps(batch_results, "city_generation_files"),
            "province_generation_files": _merge_period_output_maps(batch_results, "province_generation_files"),
            "province_seasonal_wind_csvs": _merge_period_output_maps(batch_results, "province_seasonal_wind_csvs"),
            "batch_count": len(batch_results) + len(failed_batches),
            "success_count": len(batch_results),
            "failure_count": len(failed_batches),
            "outer_max_workers": effective_outer_max_workers,
            "inner_max_workers": effective_inner_max_workers,
            "batch_summaries": [
                {
                    "batch_index": batch_item["batch_index"],
                    "batch_name": batch_item["batch_name"],
                    "province_short": batch_item["province_short"],
                    "farm_count": batch_item["farm_count"],
                    "selected_farm_ids": batch_item["result"].get("selected_farm_ids", []),
                }
                for batch_item in batch_results
            ],
            "failed_batches": failed_batches,
        }
    )
    merged.update(skip_info)
    return merged


def _prepare_kb_csv(optimal_kb_csv_path: Union[str, Path], tmp_dir: Optional[Path] = None) -> Path:
    """
    从 optimal_kb.csv 中读取唯一的 (prov, season, k, b) 结果，写入临时 CSV 供 kb_info 使用。

    optimal_kb.csv 格式: prov, season, k, b, ...
    generate_farm_seasonal_kb_dict 需要: prov/省简称, season/period, k, b 列，列名已兼容。
    """
    df = pd.read_csv(optimal_kb_csv_path, encoding="utf-8-sig")

    required_columns = ["prov", "season", "k", "b"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"optimal_kb_csv 缺少必要列: {missing_columns}，实际列为: {list(df.columns)}"
        )

    df_filtered = df.copy()
    if df_filtered.empty:
        raise ValueError("optimal_kb_csv 为空")

    duplicate_mask = df_filtered.duplicated(subset=["prov", "season"], keep=False)
    if duplicate_mask.any():
        duplicate_pairs = (
            df_filtered.loc[duplicate_mask, ["prov", "season"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            "optimal_kb_csv 在同一 (prov, season) 下出现重复: "
            f"{duplicate_pairs}"
        )


    df_filtered = df_filtered.copy()
    df_filtered["prov"] = df_filtered["prov"].map(normalize_text)
    df_filtered["season"] = df_filtered["season"].map(
        lambda value: normalized.lower() if (normalized := normalize_text(value)) is not None else None
    )

    alias_rows: list[pd.DataFrame] = []
    alias_notes: list[str] = []
    for target_prov, source_prov in PROVINCE_KB_FALLBACKS.items():
        normalized_target = normalize_text(target_prov)
        normalized_source = normalize_text(source_prov)
        if normalized_target is None or normalized_source is None:
            continue

        source_rows = df_filtered.loc[df_filtered["prov"] == normalized_source].copy()
        if source_rows.empty:
            raise ValueError(
                f"无法为 {normalized_target} 继承 kb：optimal_kb_csv 中缺少来源省份 {normalized_source}"
            )

        missing_seasons = sorted({"spring", "summer", "autumn", "winter"} - set(source_rows["season"].dropna()))
        if missing_seasons:
            raise ValueError(
                f"无法为 {normalized_target} 继承 kb：来源省份 {normalized_source} 缺少季节 {missing_seasons}"
            )

        target_rows = df_filtered.loc[df_filtered["prov"] == normalized_target].copy()
        target_season_set = set(target_rows["season"].dropna().tolist())
        missing_target_seasons = [
            season for season in ("spring", "summer", "autumn", "winter") if season not in target_season_set
        ]
        if not missing_target_seasons:
            continue

        alias_source_rows = source_rows.loc[source_rows["season"].isin(missing_target_seasons)].copy()
        alias_source_rows.loc[:, "prov"] = normalized_target
        alias_rows.append(alias_source_rows)
        alias_notes.append(
            f"{normalized_target}<-{normalized_source}({','.join(missing_target_seasons)})"
        )

    if alias_rows:
        df_filtered = pd.concat([df_filtered, *alias_rows], ignore_index=True)

    final_duplicate_mask = df_filtered.duplicated(subset=["prov", "season"], keep=False)
    if final_duplicate_mask.any():
        duplicate_pairs = (
            df_filtered.loc[final_duplicate_mask, ["prov", "season"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            "补充别名省份后 k/b CSV 在同一 (prov, season) 下出现重复: "
            f"{duplicate_pairs}"
        )

    # 只保留 generate_farm_seasonal_kb_dict 需要的列
    df_filtered = df_filtered[["prov", "season", "k", "b"]].copy()

    if tmp_dir is None:
        raise ValueError("_prepare_kb_csv 需要显式提供 tmp_dir")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / "kb_for_aggr.csv"

    df_filtered.to_csv(out_path, index=False, encoding="utf-8-sig")

    if alias_notes:
        logger.info("已为缺失省份补充继承 kb: %s", "; ".join(alias_notes))
    logger.info("已将 kb CSV 写入: %s  (共 %s 行)", out_path, len(df_filtered))
    return out_path


def _prepare_inventory_with_available_kb(
    inventory_csv_path: Union[str, Path],
    kb_csv_path: Union[str, Path],
    tmp_dir: Path,
    report_dir: Path,
    prov_short: Optional[Union[str, Iterable[str]]] = None,
    prov_long: Optional[Union[str, Iterable[str]]] = None,
    city_short: Optional[Union[str, Iterable[str]]] = None,
    city_long: Optional[Union[str, Iterable[str]]] = None,
) -> tuple[Optional[Path], dict[str, Any]]:
    """
    生成仅保留“有 kb 记录的省份”风电场清单，并将被跳过对象写入留档文件。

    返回过滤后的临时 inventory CSV 路径；若没有任何可运行风电场，则返回 None。
    """
    farm_df, column_map = _load_inventory_table(inventory_csv_path)

    effective_prov_short = prov_short
    if prov_short is None and prov_long is None and city_short is None and city_long is None:
        effective_prov_short = farm_df[column_map["prov_short"]].dropna().unique().tolist()

    selected_df, filter_info = _filter_inventory_farms(
        farm_df=farm_df,
        column_map=column_map,
        prov_short=effective_prov_short,
        prov_long=prov_long,
        city_short=city_short,
        city_long=city_long,
    )

    kb_df = pd.read_csv(kb_csv_path, encoding="utf-8-sig")
    kb_prov_set = {
        normalized
        for normalized in kb_df["prov"].map(normalize_text)
        if normalized is not None
    }

    selected_df = selected_df.copy()
    selected_df["__normalized_prov_short"] = selected_df[column_map["prov_short"]].map(normalize_text)

    selected_with_prov = selected_df[selected_df[column_map["prov_short"]].notna()].copy()
    missing_kb_df = selected_with_prov[~selected_with_prov["__normalized_prov_short"].isin(kb_prov_set)].copy()
    runnable_df = selected_df[selected_df["__normalized_prov_short"].isin(kb_prov_set)].copy()

    selected_df = selected_df.drop(columns=["__normalized_prov_short"], errors="ignore")
    selected_with_prov = selected_with_prov.drop(columns=["__normalized_prov_short"], errors="ignore")
    missing_kb_df = missing_kb_df.drop(columns=["__normalized_prov_short"], errors="ignore")
    runnable_df = runnable_df.drop(columns=["__normalized_prov_short"], errors="ignore")

    skip_info: dict[str, Any] = {
        "filter_mode": filter_info["kind"],
        "filter_value": sorted(filter_info["value"]) if isinstance(filter_info["value"], set) else filter_info["value"],
        "missing_kb_provinces": [],
        "missing_kb_farm_count": 0,
        "missing_kb_province_report": None,
        "missing_kb_farm_report": None,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    if not missing_kb_df.empty:
        province_report_df = (
            missing_kb_df[[column_map["prov_short"]]]
            .drop_duplicates()
            .rename(columns={column_map["prov_short"]: "prov_short"})
            .sort_values("prov_short")
            .reset_index(drop=True)
        )
        farm_report_columns = {
            column_map["id"]: "farm_id",
            column_map["prov_short"]: "prov_short",
            column_map["city_short"]: "city_short",
        }
        farm_report_df = (
            missing_kb_df[[column_map["id"], column_map["prov_short"], column_map["city_short"]]]
            .rename(columns=farm_report_columns)
            .sort_values(["prov_short", "city_short", "farm_id"])
            .reset_index(drop=True)
        )

        province_report_path = report_dir / "missing_kb_provinces.csv"
        farm_report_path = report_dir / "missing_kb_farms.csv"
        province_report_df.to_csv(province_report_path, index=False, encoding="utf-8-sig")
        farm_report_df.to_csv(farm_report_path, index=False, encoding="utf-8-sig")

        skip_info.update(
            {
                "missing_kb_provinces": province_report_df["prov_short"].tolist(),
                "missing_kb_farm_count": int(len(farm_report_df)),
                "missing_kb_province_report": str(province_report_path),
                "missing_kb_farm_report": str(farm_report_path),
            }
        )
        logger.warning(
            "以下省份缺少 kb 记录，已跳过并留档: %s | farm_count=%s",
            skip_info["missing_kb_provinces"],
            skip_info["missing_kb_farm_count"],
        )

    if runnable_df.empty:
        logger.warning("筛选后的风电场全部因缺少 kb 记录而被跳过")
        return None, skip_info

    tmp_dir.mkdir(parents=True, exist_ok=True)
    filtered_inventory_path = tmp_dir / "inventory_with_kb.csv"
    runnable_df.to_csv(filtered_inventory_path, index=False, encoding="utf-8-sig")
    return filtered_inventory_path, skip_info


def farm_aggr_with_kb(
    era5_dir: str,
    optimal_kb_csv: Union[str, Path],
    cdmet_dir: Optional[str] = None,
    inventory_csv_path: Union[str, Path] = DEFAULT_INVENTORY_CSV_PATH,
    farm_power_curve_dir: Union[str, Path, None] = None,
    farm_adjusted_w_dir: Optional[Union[str, Path]] = (
        "/geogfs1/groups/geog_liulb/hongyu/results/farm_raw_w/raw_w"
    ),
    farm_adjusted_w_varname: str = "raw_w",
    output_dir: Optional[Union[str, Path]] = "farm_aggr_with_kb_outputs",
    select_period: Optional[
        Union[
            SeasonName,
            int,
            List[Union[SeasonName, int]],
            Tuple[Union[SeasonName, int], ...],
            Literal["all"],
        ]
    ] = None,
    period_mode: Literal["season", "month"] = "season",
    prov_short: Optional[Union[str, Iterable[str]]] = None,
    prov_long: Optional[Union[str, Iterable[str]]] = None,
    city_short: Optional[Union[str, Iterable[str]]] = None,
    city_long: Optional[Union[str, Iterable[str]]] = None,
    city_capacity_adjust_csv_path: Optional[Union[str, Path]] = (
        r"D:\HKU PhD\wind data paper\city_capacity_adjust_info.csv"
    ),
    prov_utility_rate_adjust_csv_path: Optional[Union[str, Path]] = (
        r"D:\HKU PhD\wind data paper\prov_wind_utility_rates.csv"
    ),
    gen_adjust_info: Optional[Union[Dict, List[Dict], Tuple[Dict, ...]]] = None,
    overwrite: bool = False,
    aggr_level: Literal["city", "prov", "both"] = "both",
    include_kb_in_gen_filenames: bool = False,
    horizon_interp_method: Literal["closest", "idw", "loess"] = "loess",
    max_workers: Optional[int] = 4,
    verbose: bool = True,
    kb_tmp_dir: Optional[Union[str, Path]] = None,
    outer_max_workers: Optional[int] = None,
    inner_max_workers: Optional[int] = None,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    """
    读取 optimal_kb.csv 中的最优 (k, b) 参数，对所有风电场运行 aggr_farm_output。

    optimal_kb.csv 由 find_optimal_kb.py 生成，格式至少包含：
        prov, season, k, b

    此函数会：
    1. 读取每个（省, 季节）唯一的 kb 记录，生成 kb 映射 CSV；
    2. 按省份拆分为多个 batch，并以该 CSV 作为 kb_info 调用 aggr_farm_output；
    3. 输出 output_hourly_gen=True、output_monthly_gen=True，其余输出关闭；
    4. 使用 farm_adjusted_w_dir 复用已计算的原始风速，跳过重算。

    Args:
        era5_dir: ERA5 风速数据目录。
        optimal_kb_csv: find_optimal_kb.py 输出的最终 optimal kb CSV 路径。
        farm_adjusted_w_dir: 已计算 full_year 原始风速目录，默认为服务器路径。
        kb_tmp_dir: 兼容保留参数；当前临时文件固定写入 output_dir/_tmp_farm_aggr_with_kb。
        outer_max_workers: 外层 batch 并行数；None 表示并发跑全部省份 batch。
        inner_max_workers: 传给单个 aggr_farm_output 的 max_workers；None 时回退到 max_workers。
        其余参数同 aggr_farm_output。

    Returns:
        aggr_farm_output 的返回 dict。
    """
    output_root = Path(output_dir) if output_dir is not None else Path(".")
    output_root.mkdir(parents=True, exist_ok=True)
    effective_inner_max_workers = max_workers if inner_max_workers is None else inner_max_workers

    tmp_work_dir = output_root / "_tmp_farm_aggr_with_kb"
    if kb_tmp_dir is not None and Path(kb_tmp_dir) != tmp_work_dir:
        logger.info(
            "kb_tmp_dir=%s 已忽略；临时文件统一写入 output_dir 下: %s",
            kb_tmp_dir,
            tmp_work_dir,
        )

    try:
        kb_csv_path = _prepare_kb_csv(optimal_kb_csv, tmp_dir=tmp_work_dir)

        filtered_inventory_path, skip_info = _prepare_inventory_with_available_kb(
            inventory_csv_path=inventory_csv_path,
            kb_csv_path=kb_csv_path,
            tmp_dir=tmp_work_dir,
            report_dir=output_root,
            prov_short=prov_short,
            prov_long=prov_long,
            city_short=city_short,
            city_long=city_long,
        )

        logger.info(
            "启动 farm_aggr_with_kb | kb_csv=%s | farm_adjusted_w_dir=%s",
            kb_csv_path,
            farm_adjusted_w_dir,
        )

        if filtered_inventory_path is None:
            return {
                "inventory_csv_path": str(Path(inventory_csv_path)),
                "output_dir": str(output_root),
                "optimal_kb_csv": str(Path(optimal_kb_csv)),
                "skipped_all_due_to_missing_kb": True,
                "outer_max_workers": 0,
                "inner_max_workers": effective_inner_max_workers,
                **skip_info,
            }

        batch_specs = _build_province_batch_specs(filtered_inventory_path, tmp_dir=tmp_work_dir)
        if not batch_specs:
            return {
                "inventory_csv_path": str(Path(inventory_csv_path)),
                "output_dir": str(output_root),
                "optimal_kb_csv": str(Path(optimal_kb_csv)),
                "selected_farm_ids": [],
                "skipped_all_due_to_missing_kb": True,
                "outer_max_workers": 0,
                "inner_max_workers": effective_inner_max_workers,
                **skip_info,
            }

        effective_outer_max_workers = (
            len(batch_specs)
            if outer_max_workers is None
            else max(1, min(int(outer_max_workers), len(batch_specs)))
        )

        # 并行预算：两层是**相乘**（outer × inner），必须保证乘积 ≤ 上限
        # （防 HPC 超订惩罚；inner=None 时由 aggr_farm 侧 resolve_worker_count 兜底）
        effective_outer_max_workers, effective_inner_max_workers = clamp_two_layer_workers(
            effective_outer_max_workers, effective_inner_max_workers, context="farm_aggr_with_kb"
        )

        logger.info(
            "farm_aggr_with_kb batch plan | provinces=%s | outer_workers=%s | inner_workers=%s",
            len(batch_specs),
            effective_outer_max_workers,
            effective_inner_max_workers,
        )
        for batch_spec in batch_specs:
            logger.info(
                "prepared batch %s | province=%s | farms=%s",
                batch_spec["batch_index"],
                batch_spec["province_short"],
                batch_spec["farm_count"],
            )

        batch_tasks = []
        for batch_spec in batch_specs:
            batch_tasks.append(
                {
                    "batch_index": batch_spec["batch_index"],
                    "batch_name": batch_spec["batch_name"],
                    "province_short": batch_spec["province_short"],
                    "farm_count": batch_spec["farm_count"],
                    "params": {
                        "era5_dir": era5_dir,
                        "cdmet_dir": cdmet_dir,
                        "inventory_csv_path": batch_spec["inventory_csv_path"],
                        "farm_power_curve_dir": farm_power_curve_dir,
                        "farm_adjusted_w_dir": farm_adjusted_w_dir,
                        "farm_adjusted_w_varname": farm_adjusted_w_varname,
                        "output_dir": output_dir,
                        "output_hourly_gen": True,
                        "output_monthly_gen": True,
                        "output_hourly_cf": False,
                        "output_w_compare": False,
                        "output_raw_w": False,
                        "select_period": select_period,
                        "period_mode": period_mode,
                        "prov_short": None,
                        "prov_long": None,
                        "city_short": None,
                        "city_long": None,
                        "kb_info": str(kb_csv_path),
                        "gen_adjust_info": gen_adjust_info,
                        "city_capacity_adjust_csv_path": city_capacity_adjust_csv_path,
                        "prov_utility_rate_adjust_csv_path": prov_utility_rate_adjust_csv_path,
                        "overwrite": overwrite,
                        "aggr_level": aggr_level,
                        "include_kb_in_gen_filenames": include_kb_in_gen_filenames,
                        "horizon_interp_method": horizon_interp_method,
                        "max_workers": effective_inner_max_workers,
                        "verbose": verbose,
                    },
                }
            )

        successful_batches: list[dict[str, Any]] = []
        failed_batches: list[dict[str, Any]] = []

        if len(batch_tasks) == 1 or effective_outer_max_workers == 1:
            for batch_task in batch_tasks:
                logger.info(
                    "starting batch %s/%s | province=%s | farms=%s",
                    batch_task["batch_index"],
                    len(batch_tasks),
                    batch_task["province_short"],
                    batch_task["farm_count"],
                )
                try:
                    batch_result = _run_single_farm_aggr_batch_task(batch_task)
                    successful_batches.append(batch_result)
                    logger.info(
                        "finished batch %s/%s | province=%s | farms=%s",
                        batch_task["batch_index"],
                        len(batch_tasks),
                        batch_task["province_short"],
                        batch_task["farm_count"],
                    )
                except Exception as exc:  # noqa: BLE001
                    failed_item = {
                        "batch_index": batch_task["batch_index"],
                        "province_short": batch_task["province_short"],
                        "farm_count": batch_task["farm_count"],
                        "error": str(exc),
                    }
                    failed_batches.append(failed_item)
                    logger.error(
                        "batch failed | index=%s | province=%s | err=%s",
                        batch_task["batch_index"],
                        batch_task["province_short"],
                        exc,
                    )
                    if stop_on_error:
                        raise
        else:
            with ProcessPoolExecutor(max_workers=effective_outer_max_workers) as executor:
                future_map = {
                    executor.submit(_run_single_farm_aggr_batch_task, batch_task): batch_task
                    for batch_task in batch_tasks
                }
                pending_futures = set(future_map.keys())

                while pending_futures:
                    done_futures, pending_futures = wait(
                        pending_futures,
                        timeout=60.0,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done_futures:
                        logger.info(
                            "farm_aggr_with_kb batches running | done=%s/%s | pending=%s",
                            len(successful_batches) + len(failed_batches),
                            len(batch_tasks),
                            len(pending_futures),
                        )
                        continue

                    for future in done_futures:
                        batch_task = future_map[future]
                        try:
                            batch_result = future.result()
                            successful_batches.append(batch_result)
                            logger.info(
                                "finished batch %s/%s | province=%s | farms=%s",
                                batch_task["batch_index"],
                                len(batch_tasks),
                                batch_task["province_short"],
                                batch_task["farm_count"],
                            )
                        except Exception as exc:  # noqa: BLE001
                            failed_item = {
                                "batch_index": batch_task["batch_index"],
                                "province_short": batch_task["province_short"],
                                "farm_count": batch_task["farm_count"],
                                "error": str(exc),
                            }
                            failed_batches.append(failed_item)
                            logger.error(
                                "batch failed | index=%s | province=%s | err=%s",
                                batch_task["batch_index"],
                                batch_task["province_short"],
                                exc,
                            )
                            if stop_on_error:
                                raise

        return _merge_batch_results(
            batch_results=successful_batches,
            skip_info=skip_info,
            output_root=output_root,
            inventory_csv_path=inventory_csv_path,
            optimal_kb_csv=optimal_kb_csv,
            effective_outer_max_workers=effective_outer_max_workers,
            effective_inner_max_workers=effective_inner_max_workers,
            failed_batches=failed_batches,
        )
    finally:
        if tmp_work_dir.exists():
            shutil.rmtree(tmp_work_dir, ignore_errors=True)
            logger.info("已删除临时目录: %s", tmp_work_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "读取 optimal_kb.csv 中的最优 (k,b)，"
            "以 farm_adjusted_w_dir 复用风速，重新聚合发电量输出。"
        )
    )

    parser.add_argument("--era5_dir", type=str, help="ERA5 数据目录")
    parser.add_argument(
        "--optimal_kb_csv",
        type=Path,
        help="find_optimal_kb.py 输出的 optimal_kb.csv 路径",
    )
    parser.add_argument("--cdmet_dir", type=parse_none, default=None)
    parser.add_argument(
        "--inventory_csv_path",
        type=parse_none,
        default=str(DEFAULT_INVENTORY_CSV_PATH),
    )
    parser.add_argument("--farm_power_curve_dir", type=parse_none, default=None)
    parser.add_argument(
        "--farm_adjusted_w_dir",
        type=parse_none,
        default="/geogfs1/groups/geog_liulb/hongyu/results/farm_raw_w/raw_w",
        help="已计算 full_year 原始风速目录",
    )
    parser.add_argument("--farm_adjusted_w_varname", type=str, default="raw_w")
    parser.add_argument(
        "--output_dir",
        type=parse_none,
        default="farm_aggr_with_kb_outputs",
        help="输出根目录",
    )
    parser.add_argument("--select_period", type=parse_select_period, default=None)
    parser.add_argument("--period_mode", choices=["season", "month"], default="season")
    parser.add_argument("--prov_short", type=parse_none, default=None)
    parser.add_argument("--prov_long", type=parse_none, default=None)
    parser.add_argument("--city_short", type=parse_none, default=None)
    parser.add_argument("--city_long", type=parse_none, default=None)
    parser.add_argument(
        "--city_capacity_adjust_csv_path",
        type=parse_none,
        default=r"D:\HKU PhD\wind data paper\city_capacity_adjust_info.csv",
    )
    parser.add_argument(
        "--prov_utility_rate_adjust_csv_path",
        type=parse_none,
        default=r"D:\HKU PhD\wind data paper\prov_wind_utility_rates.csv",
    )
    parser.add_argument("--gen_adjust_info", type=parse_gen_adjust_info, default=None)
    parser.add_argument("--overwrite", type=parse_tf, default=False)
    parser.add_argument("--aggr_level", choices=["city", "prov", "both"], default="both")
    parser.add_argument("--include_kb_in_gen_filenames", type=parse_tf, default=False)
    parser.add_argument(
        "--horizon_interp_method", choices=["closest", "idw", "loess"], default="loess"
    )
    parser.add_argument("--max_workers", type=parse_none, default=4, help="兼容参数；未设置 inner_max_workers 时作为其默认值")
    parser.add_argument("--outer_max_workers", type=parse_none, default=1)
    parser.add_argument("--inner_max_workers", type=parse_none, default=None)
    parser.add_argument("--stop_on_error", type=parse_tf, default=False)
    parser.add_argument("--verbose", type=parse_tf, default=True)
    parser.add_argument(
        "--kb_tmp_dir",
        type=parse_none,
        default=None,
        help="兼容保留参数；当前临时文件固定写入 output_dir/_tmp_farm_aggr_with_kb",
    )

    args = parser.parse_args()

    prov_short = parse_csv_list(args.prov_short)
    prov_long = parse_csv_list(args.prov_long)
    city_short = parse_csv_list(args.city_short)
    city_long = parse_csv_list(args.city_long)
    max_workers = None if args.max_workers is None else int(args.max_workers)
    outer_max_workers = None if args.outer_max_workers is None else int(args.outer_max_workers)
    inner_max_workers = None if args.inner_max_workers is None else int(args.inner_max_workers)

    farm_aggr_with_kb(
        era5_dir=args.era5_dir,
        optimal_kb_csv=args.optimal_kb_csv,
        cdmet_dir=args.cdmet_dir,
        inventory_csv_path=args.inventory_csv_path,
        farm_power_curve_dir=args.farm_power_curve_dir,
        farm_adjusted_w_dir=args.farm_adjusted_w_dir,
        farm_adjusted_w_varname=args.farm_adjusted_w_varname,
        output_dir=args.output_dir,
        select_period=args.select_period,
        period_mode=args.period_mode,
        prov_short=prov_short,
        prov_long=prov_long,
        city_short=city_short,
        city_long=city_long,
        city_capacity_adjust_csv_path=args.city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=args.prov_utility_rate_adjust_csv_path,
        gen_adjust_info=args.gen_adjust_info,
        overwrite=args.overwrite,
        aggr_level=args.aggr_level,
        include_kb_in_gen_filenames=args.include_kb_in_gen_filenames,
        horizon_interp_method=args.horizon_interp_method,
        max_workers=max_workers,
        verbose=args.verbose,
        kb_tmp_dir=args.kb_tmp_dir,
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        stop_on_error=args.stop_on_error,
    )


if __name__ == "__main__":
    main()
