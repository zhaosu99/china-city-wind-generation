from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Union, List, Tuple
import logging
import os

from config import resolve_worker_count  # 并行预算护栏（防 HPC 超订惩罚）

from tqdm import tqdm

from aggr_farm_aux import (
    AggrLevel,
    DEFAULT_INVENTORY_CSV_PATH,
    PeriodItem,
    PeriodMode,
    SeasonName,
    _aggregate_generation_paths,
    _build_output_layout,
    _build_farm_gen_adjust_info_map,
    _build_farm_generation_adjustment_attr_map,
    _filter_inventory_farms,
    _group_kb_file_tag,
    _group_kb_pair_tokens,
    _kb_suffix_for_filename,
    _kb_pair_tokens,
    _load_inventory_table,
    _load_city_capacity_adjust_map,
    _load_prov_utility_rate_map,
    _period_key,
    _progress,
    _resolve_periods,
    _run_single_farm_task,
)
from parse_aux import (
    parse_csv_list as _parse_csv_list,
    parse_gen_adjust_info as _parse_gen_adjust_info,
    parse_kb_info as _parse_kb_info,
    parse_none as _parse_none,
    parse_select_period as _parse_select_period,
    parse_tf as _parse_tf,
)
from utils import (
    chinese_to_pinyi,
    concat_w_ds,
    normalize_text,
    resolve_existing_column,
    safe_read_csv,
    sanitize_filename_part,
    sanitize_netcdf_attrs,
    write_netcdf_with_retry,
)

logger = logging.getLogger(__name__)


def _log_verbose_info(verbose: bool, message: str, *args: Any) -> None:
    if verbose:
        logger.info(message, *args)
        return
    logger.debug(message, *args)


def _city_output_slug(prov_short_name: Any, city_short_name: Any) -> str:
    prov_slug = sanitize_filename_part(chinese_to_pinyi(str(prov_short_name)))
    city_slug = sanitize_filename_part(chinese_to_pinyi(str(city_short_name)))

    if prov_slug and city_slug:
        return f"{prov_slug}__{city_slug}"
    if city_slug:
        return city_slug
    return prov_slug


def _legacy_city_output_slug(city_short_name: Any) -> str:
    return sanitize_filename_part(chinese_to_pinyi(str(city_short_name)))


def _build_farm_kb_adjust_info_map(
    selected_df,
    column_map: dict[str, str],
    kb_info: Optional[Union[str, Path, tuple[float, float]]],
    verbose: bool = True,
) -> dict[int, Optional[Union[Dict[str, tuple[float, float]], tuple[float, float]]]]:
    selected_farm_ids = selected_df[column_map["id"]].astype(int).tolist()
    farm_kb_adjust_info_map: dict[int, Optional[Union[Dict[str, tuple[float, float]], tuple[float, float]]]] = {
        int(farm_id): None for farm_id in selected_farm_ids
    }

    if kb_info is None:
        return farm_kb_adjust_info_map

    if isinstance(kb_info, tuple):
        if len(kb_info) != 2:
            raise ValueError("kb_info 传入 tuple 时必须是长度为2的 (k, b)")

        try:
            kb_tuple = (float(kb_info[0]), float(kb_info[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError("kb_info 传入 tuple 时，k 和 b 必须是可转换为 float 的数值") from exc

        for farm_id in selected_farm_ids:
            farm_kb_adjust_info_map[int(farm_id)] = kb_tuple
        return farm_kb_adjust_info_map

    kb_csv_path_obj = Path(kb_info)
    kb_df = safe_read_csv(kb_csv_path_obj)
    kb_province_col = resolve_existing_column(
        kb_df.columns,
        ["prov", "省简称", "省份简称", "province_short", "province"],
        "prov",
    )
    period_col = resolve_existing_column(kb_df.columns, ["period", "season", "季节"], "season")
    k_col = resolve_existing_column(kb_df.columns, ["min_w_smape_k", "k"], "k")
    b_col = resolve_existing_column(kb_df.columns, ["min_w_smape_b", "b"], "b")

    kb_df = kb_df.copy()
    kb_df[kb_province_col] = kb_df[kb_province_col].map(normalize_text)
    kb_df[period_col] = kb_df[period_col].map(
        lambda value: normalized.lower() if (normalized := normalize_text(value)) is not None else None
    )

    selected_provinces = [
        province
        for province in selected_df[column_map["prov_short"]].dropna().unique().tolist()
        if province is not None
    ]
    _log_verbose_info(
        verbose,
        "Building seasonal kb map once for %s farms across %s provinces from %s",
        len(selected_farm_ids),
        len(selected_provinces),
        kb_csv_path_obj,
    )

    province_kb_map: dict[str, dict[str, tuple[float, float]]] = {}
    for province_short in selected_provinces:
        province_rows = kb_df.loc[kb_df[kb_province_col] == province_short]
        if province_rows.empty:
            raise ValueError(f"在 k/b CSV 中未找到省简称={province_short} 的记录")

        seasonal_kb: dict[str, tuple[float, float]] = {}
        for season in ("spring", "summer", "autumn", "winter"):
            season_rows = province_rows.loc[province_rows[period_col] == season]
            if season_rows.empty:
                raise ValueError(f"省简称={province_short} 在 k/b CSV 中缺少季节 {season} 的记录")
            if len(season_rows) > 1:
                raise ValueError(f"省简称={province_short} 在 k/b CSV 中季节 {season} 存在多条记录")

            row = season_rows.iloc[0]
            try:
                seasonal_kb[season] = (float(row[k_col]), float(row[b_col]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"省简称={province_short} 季节={season} 的 k/b 无法转换为浮点数: "
                    f"k={row[k_col]}, b={row[b_col]}"
                ) from exc

        province_kb_map[str(province_short)] = seasonal_kb

    for _, row in selected_df[[column_map["id"], column_map["prov_short"]]].iterrows():
        farm_id = int(row[column_map["id"]])
        province_short = row[column_map["prov_short"]]
        if province_short is None:
            raise ValueError(f"farm_id={farm_id} 对应的省简称为空，无法匹配 seasonal kb")
        farm_kb_adjust_info_map[farm_id] = province_kb_map[str(province_short)]

    _log_verbose_info(verbose, "Finished building seasonal kb map for %s farms", len(selected_farm_ids))
    return farm_kb_adjust_info_map


def main() -> None:
    parser = argparse.ArgumentParser(description="批量计算风电场并聚合到 city/prov 层级")
    parser.add_argument("--era5_dir", type=str, required=True)
    parser.add_argument("--cdmet_dir", type=_parse_none, default=None)
    parser.add_argument("--inventory_csv_path", type=_parse_none, default=str(DEFAULT_INVENTORY_CSV_PATH))
    parser.add_argument("--farm_power_curve_dir", type=_parse_none, default=None)
    parser.add_argument("--farm_adjusted_w_dir", type=_parse_none, default=None)
    parser.add_argument("--farm_adjusted_w_varname", type=str, default="raw_w")

    parser.add_argument("--output_dir", type=_parse_none, default=None)

    parser.add_argument("--output_hourly_gen", type=_parse_tf, default=True)
    parser.add_argument("--output_monthly_gen", type=_parse_tf, default=False)
    parser.add_argument("--output_hourly_cf", type=_parse_tf, default=False)
    parser.add_argument("--output_w_compare", type=_parse_tf, default=False)
    parser.add_argument("--output_raw_w", type=_parse_tf, default=False)

    parser.add_argument("--select_period", type=_parse_select_period, default="all")
    parser.add_argument("--period_mode", choices=["season", "month"], default="season")
    parser.add_argument("--prov_short", type=_parse_none, default=None)
    parser.add_argument("--prov_long", type=_parse_none, default=None)
    parser.add_argument("--city_short", type=_parse_none, default=None)
    parser.add_argument("--city_long", type=_parse_none, default=None)

    parser.add_argument("--kb_info", type=_parse_kb_info, default=None)
    parser.add_argument("--gen_adjust_info", type=_parse_gen_adjust_info, default=None)
    parser.add_argument(
        "--city_capacity_adjust_csv_path",
        type=_parse_none,
        default=r"D:\HKU PhD\wind data paper\city_capacity_adjust_info.csv",
    )
    parser.add_argument(
        "--prov_utility_rate_adjust_csv_path",
        type=_parse_none,
        default=r"D:\HKU PhD\wind data paper\prov_wind_utility_rates.csv",
    )

    parser.add_argument("--overwrite", type=_parse_tf, default=False)
    parser.add_argument("--aggr_level", choices=["city", "prov", "both"], default="both")
    parser.add_argument("--horizon_interp_method", choices=["closest", "idw", "loess"], default="loess")
    parser.add_argument("--max_workers", type=_parse_none, default=4)
    parser.add_argument("--verbose", type=_parse_tf, default=True)
    parser.add_argument("--show_progress", type=_parse_tf, default=True)

    args = parser.parse_args()

    max_workers = None if args.max_workers is None else int(args.max_workers)
    prov_short = _parse_csv_list(args.prov_short)
    prov_long = _parse_csv_list(args.prov_long)
    city_short = _parse_csv_list(args.city_short)
    city_long = _parse_csv_list(args.city_long)

    result = aggr_farm_output(
        era5_dir=args.era5_dir,
        cdmet_dir=args.cdmet_dir,
        inventory_csv_path=args.inventory_csv_path,
        farm_power_curve_dir=args.farm_power_curve_dir,
        farm_adjusted_w_dir=args.farm_adjusted_w_dir,
        farm_adjusted_w_varname=args.farm_adjusted_w_varname,
        output_dir=args.output_dir,
        output_hourly_gen=args.output_hourly_gen,
        output_monthly_gen=args.output_monthly_gen,
        output_hourly_cf=args.output_hourly_cf,
        output_w_compare=args.output_w_compare,
        output_raw_w=args.output_raw_w,
        select_period=args.select_period,
        period_mode=args.period_mode,
        prov_short=prov_short,
        prov_long=prov_long,
        city_short=city_short,
        city_long=city_long,
        kb_info=args.kb_info,
        gen_adjust_info=args.gen_adjust_info,
        city_capacity_adjust_csv_path=args.city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=args.prov_utility_rate_adjust_csv_path,
        overwrite=args.overwrite,
        aggr_level=args.aggr_level,
        horizon_interp_method=args.horizon_interp_method,
        max_workers=max_workers,
        verbose=args.verbose,
        show_progress=args.show_progress,
    )

    logger.info(
        "CLI done | farms=%s | periods=%s | aggr_level=%s",
        len(result.get("selected_farm_ids", [])),
        result.get("period"),
        result.get("aggr_level"),
    )


def aggr_farm_output(
    era5_dir: str,
    cdmet_dir: Optional[str] = None,
    inventory_csv_path: Union[str, Path] = DEFAULT_INVENTORY_CSV_PATH,
    farm_power_curve_dir: Union[str, Path, None] = None,
    farm_adjusted_w_dir: Optional[Union[str, Path]] = None,
    farm_adjusted_w_varname: str = "raw_w",

    output_dir: Optional[Union[str, Path]] = None,

    output_hourly_gen: bool = True,
    output_monthly_gen: bool = False,
    output_hourly_cf: bool = False,
    output_w_compare: Optional[bool] = False,
    output_raw_w: Optional[bool] = False,

    select_period: Optional[Union[SeasonName, int, list[Union[SeasonName, int]], tuple[Union[SeasonName, int], ...], Literal["all"]]] = "all",
    period_mode: PeriodMode = "season",
    prov_short: Optional[Union[str, Iterable[str]]] = None,
    prov_long: Optional[Union[str, Iterable[str]]] = None,
    city_short: Optional[Union[str, Iterable[str]]] = None,
    city_long: Optional[Union[str, Iterable[str]]] = None,
    
    kb_info: Optional[Union[str, Path, tuple[float, float]]] = None,
    gen_adjust_info: Optional[Union[Dict, List[Dict], Tuple[Dict, ...]]] = None,
    city_capacity_adjust_csv_path: Optional[Union[str, Path]] = r"D:\HKU PhD\wind data paper\city_capacity_adjust_info.csv",
    prov_utility_rate_adjust_csv_path: Optional[Union[str, Path]] = r"D:\HKU PhD\wind data paper\prov_wind_utility_rates.csv",
    
    overwrite: bool = False,
    aggr_level: AggrLevel = "both",
    include_kb_in_gen_filenames: bool = True,
    horizon_interp_method: Literal["closest", "idw", "loess"] = "loess",
    max_workers: Optional[int] = 4,
    verbose: bool = True,
    show_progress: bool = True,
    **legacy_output_kwargs: Any,
) -> dict[str, Any]:
    """
    Batch 计算风电场输出，并按 city/prov 层级聚合发电量。

    关键语义：
    - ``output_dir`` 是 batch 层主输出根目录；函数会按需派生 ``gen/``、``w_compare/``、``raw_w/``。
    - ``select_period='all'`` 会按 ``period_mode`` 展开全部季节或月份。
    - ``select_period=None`` 表示不做时间筛选，只执行一轮 ``full_year`` 任务。
        - ``city_capacity_adjust_csv_path`` / ``prov_utility_rate_adjust_csv_path`` 的 generation 修正会在 farm 层先完成，
            后续 city/prov 聚合仅做累加，不再二次调整。
        - ``include_kb_in_gen_filenames=False`` 时，generation nc 文件名不再包含 k/b token。
    - ``output_hourly_gen``、``output_monthly_gen``、``output_hourly_cf`` 三者任一为 True 时才生成发电量目录与结果。
    - ``output_w_compare`` / ``output_raw_w`` 为 True 时才生成对应目录与结果。

    常用参数：
    - ``era5_dir`` / ``cdmet_dir``: 输入风速数据目录。
    - ``inventory_csv_path``: 风电场清单。
    - ``farm_power_curve_dir``: 逐场功率曲线 CSV 目录。**只要非 None 就会使用**
      （读取其下 ``farm_{farm_id}.csv``）；为 None 时才用盘点表机型参数现场生成曲线。
    - ``farm_adjusted_w_dir``: 复用已存在的 full_year 原始风速文件目录。
    - ``prov_short`` / ``prov_long`` / ``city_short`` / ``city_long``: 风电场筛选条件，四者一次只能使用一种。
    - ``kb_info``: 可为 ``None``、``(k, b)`` 或 kb CSV 路径。
    - ``aggr_level``: ``'city'``、``'prov'``、``'both'``。
    - ``show_progress``: 是否显示内部 tqdm 进度条。

    返回值：
    返回摘要 dict，核心字段包括 ``period``、``selected_farm_ids``、
    ``farm_generation_files``、``farm_wind_files``、``farm_raw_w_files``、
    ``city_generation_files``、``province_generation_files``，以及实际使用的
    ``output_gen_dir``、``output_w_compare_dir``、``output_raw_w_dir``。

    约束：
    - 使用 ``city_short`` 或 ``city_long`` 时，``aggr_level`` 不能为 ``'prov'``；若传 ``'both'`` 会自动降为 ``'city'``。
    - 为兼容旧代码，仍接受 ``output_gen_nc_path`` / ``output_w_compare_nc_path`` / ``output_raw_w_nc_path`` 作为历史 kwargs，但在本层都按目录处理。
    """
    target_periods = _resolve_periods(select_period, period_mode)

    output_gen_dir = legacy_output_kwargs.pop("output_gen_nc_path", None)
    output_w_compare_dir = legacy_output_kwargs.pop("output_w_compare_nc_path", None)
    output_raw_w_dir = legacy_output_kwargs.pop("output_raw_w_nc_path", None)
    if legacy_output_kwargs:
        unknown_keys = ", ".join(sorted(legacy_output_kwargs.keys()))
        raise TypeError(f"aggr_farm_output() got unexpected keyword argument(s): {unknown_keys}")

    should_output_gen_ds = output_hourly_gen or output_monthly_gen or output_hourly_cf

    if output_w_compare is None:
        output_w_compare = output_w_compare_dir is not None
    if output_raw_w is None:
        output_raw_w = output_raw_w_dir is not None

    if not (should_output_gen_ds or output_w_compare or output_raw_w):
        raise ValueError(
            "至少启用一种输出：发电量(output_hourly_gen/output_monthly_gen/output_hourly_cf)、"
            "风速对比(output_w_compare)或原始风速(output_raw_w)"
        )

    if aggr_level not in ("city", "prov", "both"):
        raise ValueError(f"aggr_level 仅支持 city/prov/both，收到: {aggr_level}")

    if (city_short is not None or city_long is not None):
        if aggr_level == "prov":
            raise ValueError("当使用 city_short/city_long 筛选时，aggr_level 不能为 'prov'")
        if aggr_level == "both":
            logger.info("使用 city_short/city_long 筛选且 aggr_level='both'，已自动降级为 aggr_level='city'")
            aggr_level = "city"

    def _as_dir_or_none(path_like: Optional[Union[str, Path]], name: str) -> Optional[Path]:
        if path_like is None:
            return None
        path_obj = Path(path_like)
        if path_obj.suffix.lower() == ".nc":
            raise ValueError(f"{name} 在 batch 模式下必须是目录，不应是单个 .nc 文件路径: {path_obj}")
        return path_obj

    output_base_dir = Path(output_dir) if output_dir is not None else Path(".")
    if should_output_gen_ds and output_gen_dir is None:
        output_gen_dir = output_base_dir / "gen"
    if output_w_compare and output_w_compare_dir is None:
        output_w_compare_dir = output_base_dir / "w_compare"
    if output_raw_w and output_raw_w_dir is None:
        output_raw_w_dir = output_base_dir / "raw_w"

    if not output_w_compare:
        output_w_compare_dir = None
    if not output_raw_w:
        output_raw_w_dir = None

    output_gen_dir = _as_dir_or_none(output_gen_dir, "output_gen_dir")
    output_w_compare_dir = _as_dir_or_none(output_w_compare_dir, "output_w_compare_dir")
    output_raw_w_dir = _as_dir_or_none(output_raw_w_dir, "output_raw_w_dir")

    farm_df, column_map = _load_inventory_table(inventory_csv_path)

    if prov_short is None and prov_long is None and city_short is None and city_long is None:
        prov_short = farm_df[column_map["prov_short"]].dropna().unique().tolist()

    selected_df, filter_info = _filter_inventory_farms(
        farm_df=farm_df,
        column_map=column_map,
        prov_short=prov_short,
        prov_long=prov_long,
        city_short=city_short,
        city_long=city_long,
    )

    selected_farm_ids = selected_df[column_map["id"]].astype(int).tolist()
    farm_kb_adjust_info_map = _build_farm_kb_adjust_info_map(
        selected_df=selected_df,
        column_map=column_map,
        kb_info=kb_info,
        verbose=verbose,
    )

    single_farm = len(selected_farm_ids) == 1
    gen_layout = (
        _build_output_layout(output_gen_dir, single_farm, "farm_generation_outputs")
        if should_output_gen_ds
        else None
    )
    wind_layout = (
        _build_output_layout(output_w_compare_dir, single_farm, "farm_wind_outputs", with_city_prov_dirs=False)
        if output_w_compare
        else None
    )
    raw_layout = (
        _build_output_layout(output_raw_w_dir, single_farm, "farm_raw_w_outputs", with_city_prov_dirs=False)
        if output_raw_w
        else None
    )

    _log_verbose_info(
        verbose,
        "Selected %s farms for computation, period_mode=%s, periods=%s",
        len(selected_farm_ids),
        period_mode,
        ["full_year" if period_item is None else period_item for period_item in target_periods],
    )

    city_capacity_map = _load_city_capacity_adjust_map(city_capacity_adjust_csv_path)
    prov_utility_map = _load_prov_utility_rate_map(prov_utility_rate_adjust_csv_path)
    farm_gen_adjust_info_map = _build_farm_gen_adjust_info_map(
        selected_df=selected_df,
        column_map=column_map,
        base_gen_adjust_info=gen_adjust_info,
        city_capacity_map=city_capacity_map,
        prov_utility_map=prov_utility_map,
    )
    farm_generation_adjustment_attr_map = _build_farm_generation_adjustment_attr_map(
        selected_df=selected_df,
        column_map=column_map,
        city_capacity_map=city_capacity_map,
        prov_utility_map=prov_utility_map,
    )

    def _resolve_adjusted_w_input_path(farm_id: int) -> Optional[str]:
        if farm_adjusted_w_dir is None:
            return None

        base_dir = Path(farm_adjusted_w_dir)
        candidate_with_kb = base_dir / f"farm_{int(farm_id)}_period_full_year_k_1_b_0_raw_w.nc"
        candidate_plain = base_dir / f"farm_{int(farm_id)}_period_full_year_raw_w.nc"

        if candidate_with_kb.exists():
            return str(candidate_with_kb)
        if candidate_plain.exists():
            return str(candidate_plain)

        raise FileNotFoundError(
            "farm_adjusted_w_dir 下未找到可用的 full_year 风速文件，"
            f"farm_id={farm_id}, candidates={[str(candidate_with_kb), str(candidate_plain)]}"
        )

    task_list: list[dict[str, Any]] = []
    skipped_results: list[dict[str, Any]] = []
    for period_item in target_periods:
        period_key = _period_key(period_item, period_mode)
        for farm_id in selected_farm_ids:
            farm_k_token, farm_b_token = _kb_pair_tokens(farm_kb_adjust_info_map[int(farm_id)])
            gen_output_path = (
                str(
                    gen_layout["farm_dir"]
                    / f"farm_{int(farm_id)}"
                    / f"{period_key}"
                    / (
                        f"k{farm_k_token}b{farm_b_token}_gen.nc"
                        if include_kb_in_gen_filenames
                        else "gen.nc"
                    )
                )
                if gen_layout is not None
                else None
            )
            if gen_output_path is not None:
                Path(gen_output_path).parent.mkdir(parents=True, exist_ok=True)
            wind_output_path = (
                str(
                    wind_layout["farm_dir"]
                    / f"farm_{int(farm_id)}_period_{period_key}{_kb_suffix_for_filename(farm_kb_adjust_info_map[int(farm_id)])}_w_compare.nc"
                )
                if wind_layout is not None
                else None
            )
            raw_output_path = (
                str(
                    raw_layout["farm_dir"]
                    / f"farm_{int(farm_id)}_period_{period_key}{_kb_suffix_for_filename(farm_kb_adjust_info_map[int(farm_id)])}_raw_w.nc"
                )
                if raw_layout is not None
                else None
            )

            if not overwrite:
                gen_ready = (
                    (gen_output_path is None)
                    or Path(gen_output_path).exists()
                )
                wind_ready = (wind_output_path is None) or Path(wind_output_path).exists()
                raw_ready = (raw_output_path is None) or Path(raw_output_path).exists()
                if gen_ready and wind_ready and raw_ready:
                    skipped_results.append(
                        {
                            "farm_id": int(farm_id),
                            "period_item": period_item,
                            "period_mode": period_mode,
                            "output_gen_nc_path": gen_output_path,
                            "output_w_compare_nc_path": wind_output_path,
                            "output_raw_w_nc_path": raw_output_path,
                        }
                    )
                    continue

            task_list.append(
                {
                    "farm_id": int(farm_id),
                    "period_item": period_item,
                    "period_mode": period_mode,
                    "inventory_csv_path": str(Path(inventory_csv_path)),
                    "farm_power_curve_dir": str(Path(farm_power_curve_dir)) if farm_power_curve_dir is not None else None,
                    "era5_dir": era5_dir,
                    "cdmet_dir": cdmet_dir,
                    "output_gen_nc_path": gen_output_path,
                    "output_w_compare_nc_path": wind_output_path,
                    "output_raw_w_nc_path": raw_output_path,
                    "farm_adjusted_w_nc_path": _resolve_adjusted_w_input_path(int(farm_id)),
                    "farm_adjusted_w_varname": farm_adjusted_w_varname,
                    "kb_adjust_info": farm_kb_adjust_info_map[int(farm_id)],
                    "gen_adjust_info": farm_gen_adjust_info_map[int(farm_id)],
                    "generation_adjustment_attrs": farm_generation_adjustment_attr_map[int(farm_id)],
                    "output_hourly_gen": output_hourly_gen,
                    "output_monthly_gen": output_monthly_gen,
                    "output_hourly_cf": output_hourly_cf,
                    "output_w_compare": output_w_compare,
                    "output_raw_w": output_raw_w,
                    "horizon_interp_method": horizon_interp_method,
                    "verbose": verbose,
                }
            )

    farm_generation_files: dict[str, dict[int, str]] = {
        _period_key(period_item, period_mode): {} for period_item in target_periods
    }
    farm_wind_files: dict[str, dict[int, str]] = {
        _period_key(period_item, period_mode): {} for period_item in target_periods
    }
    farm_raw_w_files: dict[str, dict[int, str]] = {
        _period_key(period_item, period_mode): {} for period_item in target_periods
    }
    
    for result in skipped_results:
        farm_id = int(result["farm_id"])
        period_key = _period_key(result["period_item"], result["period_mode"])
        if result["output_gen_nc_path"] is not None:
            farm_generation_files[period_key][farm_id] = str(result["output_gen_nc_path"])
        if result["output_w_compare_nc_path"] is not None:
            farm_wind_files[period_key][farm_id] = str(result["output_w_compare_nc_path"])
        if result["output_raw_w_nc_path"] is not None:
            farm_raw_w_files[period_key][farm_id] = str(result["output_raw_w_nc_path"])

    if skipped_results:
        _log_verbose_info(verbose, "Skipped %s existing farm-period task(s) because overwrite=False", len(skipped_results))

    if not task_list:
        _log_verbose_info(verbose, "No farm-period tasks need computation")
    else:
        # 并行预算：max_workers=None 时**不能**按 cpu_count-1 无界回落
        # （HPC 节点 384 核 ⇒ 383，远超上限 96 会被惩罚）；见 config.resolve_worker_count
        worker_count = resolve_worker_count(max_workers, len(task_list), context="aggr_farm_output")

        with tqdm(
            total=len(task_list),
            desc="Computing farm-period",
            unit="task",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as pbar:
            if worker_count == 1:
                for task in task_list:
                    result = _run_single_farm_task(task)
                    farm_id = int(result["farm_id"])
                    result_period_item: Optional[PeriodItem] = result["period_item"]
                    result_period_mode = result["period_mode"]
                    period_key = _period_key(result_period_item, result_period_mode)
                    if result["output_gen_nc_path"] is not None:
                        farm_generation_files[period_key][farm_id] = str(result["output_gen_nc_path"])
                    if result["output_w_compare_nc_path"] is not None:
                        farm_wind_files[period_key][farm_id] = str(result["output_w_compare_nc_path"])
                    if result["output_raw_w_nc_path"] is not None:
                        farm_raw_w_files[period_key][farm_id] = str(result["output_raw_w_nc_path"])
                    _log_verbose_info(
                        verbose,
                        "Finished farm_id=%s period_mode=%s period=%s",
                        farm_id,
                        result_period_mode,
                        result_period_item,
                    )
                    pbar.update(1)
                pbar.set_postfix_str(f"done={pbar.n}/{len(task_list)}, pending=0")
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    pending_futures = {executor.submit(_run_single_farm_task, task) for task in task_list}
                    while pending_futures:
                        done_futures, pending_futures = wait(
                            pending_futures,
                            timeout=1.0,
                            return_when=FIRST_COMPLETED,
                        )

                        if not done_futures:
                            pbar.set_postfix_str(f"running~{min(worker_count, len(pending_futures))}, pending={len(pending_futures)}")
                            pbar.refresh()
                            continue

                        for future in done_futures:
                            result = future.result()
                            farm_id = int(result["farm_id"])
                            result_period_item: Optional[PeriodItem] = result["period_item"]
                            result_period_mode = result["period_mode"]
                            period_key = _period_key(result_period_item, result_period_mode)
                            if result["output_gen_nc_path"] is not None:
                                farm_generation_files[period_key][farm_id] = str(result["output_gen_nc_path"])
                            if result["output_w_compare_nc_path"] is not None:
                                farm_wind_files[period_key][farm_id] = str(result["output_w_compare_nc_path"])
                            if result["output_raw_w_nc_path"] is not None:
                                farm_raw_w_files[period_key][farm_id] = str(result["output_raw_w_nc_path"])
                            _log_verbose_info(
                                verbose,
                                "Finished farm_id=%s period_mode=%s period=%s",
                                farm_id,
                                result_period_mode,
                                result_period_item,
                            )
                            pbar.update(1)

                        pbar.set_postfix_str(f"done={pbar.n}/{len(task_list)}, pending={len(pending_futures)}")

    city_generation_files: dict[str, dict[str, str]] = {
        _period_key(period_item, period_mode): {} for period_item in target_periods
    }
    prov_generation_files: dict[str, dict[str, str]] = {
        _period_key(period_item, period_mode): {} for period_item in target_periods
    }
    prov_seasonal_wind_csvs: dict[str, dict[str, str]] = {}

    need_city_level = aggr_level in ("city", "both")
    need_prov_level = aggr_level in ("prov", "both")

    if gen_layout is not None and (need_city_level or need_prov_level):
        for period_item in _progress(
            target_periods,
            total=len(target_periods),
            desc="Aggregating generation",
            disable=not show_progress,
        ):
            period_key = _period_key(period_item, period_mode)

            city_groups = selected_df.groupby([column_map["prov_short"], column_map["city_short"]], dropna=False)
            for (prov_short_name, city_short_name), group_df in city_groups:
                city_slug = _city_output_slug(prov_short_name, city_short_name)
                city_farm_ids = group_df[column_map["id"]].astype(int).tolist()
                city_kb_tag = _group_kb_file_tag(city_farm_ids, farm_kb_adjust_info_map)
                city_output_path = gen_layout["city_dir"] / city_slug / period_key / (
                    f"{city_kb_tag}.nc" if include_kb_in_gen_filenames else "gen.nc"
                )
                city_output_path.parent.mkdir(parents=True, exist_ok=True)
                city_key = f"{prov_short_name}/{city_short_name}"

                # prov/both 支持直接复用现有市级结果
                if city_output_path.exists() and not overwrite:
                    city_generation_files[period_key][city_key] = str(city_output_path)
                    continue

                farm_paths: list[str] = []
                missing_farm_items: list[str] = []
                for farm_id in city_farm_ids:
                    farm_path = farm_generation_files[period_key].get(int(farm_id))
                    if farm_path is None:
                        missing_farm_items.append(f"farm_id={int(farm_id)} (not produced)")
                        continue
                    if not Path(farm_path).exists():
                        missing_farm_items.append(f"farm_id={int(farm_id)} (missing file: {farm_path})")
                        continue
                    farm_paths.append(farm_path)

                if missing_farm_items:
                    raise FileNotFoundError(
                        "City aggregation requires all farm generation files. "
                        f"period={period_key}, prov={prov_short_name}, city={city_short_name}, "
                        f"missing={missing_farm_items}"
                    )

                if not farm_paths:
                    raise FileNotFoundError(
                        "City aggregation has no usable farm generation files. "
                        f"period={period_key}, prov={prov_short_name}, city={city_short_name}"
                    )

                city_ds = _aggregate_generation_paths(
                    ds_paths=farm_paths,
                    mode="farm2city",
                    output_hourly_gen=output_hourly_gen,
                    output_monthly_gen=output_monthly_gen,
                    output_hourly_cf=output_hourly_cf,
                    period=None,
                )
                city_ds.attrs.update(
                    {
                        "province_short_name": prov_short_name,
                        "city_short_name": city_short_name,
                        "farm_count": int(len(farm_paths)),
                        "period": "full_year" if period_item is None else period_item,
                        "period_mode": period_mode,
                        "generation_adjustment_stage": "farm",
                        "city_capacity_adjust_applied_upstream": int(bool(city_capacity_map)),
                        "prov_utility_rate_adjust_applied_upstream": int(bool(prov_utility_map)),
                    }
                )
                city_ds = sanitize_netcdf_attrs(city_ds, "city_ds")
                write_netcdf_with_retry(city_ds, city_output_path)
                city_generation_files[period_key][city_key] = str(city_output_path)

            if need_prov_level:
                prov_groups = selected_df.groupby(column_map["prov_short"], dropna=False)
                for prov_short_name, group_df in prov_groups:
                    city_paths = []
                    missing_city_items: list[str] = []
                    for city_short_name in group_df[column_map["city_short"]].dropna().unique().tolist():
                        city_key = f"{prov_short_name}/{city_short_name}"
                        city_path = city_generation_files[period_key].get(city_key)
                        if city_path is None:
                            # 若前面未写入映射，尝试直接读取既有市级文件
                            city_group_df = city_groups.get_group((prov_short_name, city_short_name))
                            city_farm_ids = city_group_df[column_map["id"]].astype(int).tolist()
                            city_slug = _city_output_slug(prov_short_name, city_short_name)
                            legacy_city_slug = _legacy_city_output_slug(city_short_name)
                            city_kb_tag = _group_kb_file_tag(city_farm_ids, farm_kb_adjust_info_map)
                            candidate_city_paths = [
                                gen_layout["city_dir"] / city_slug / period_key / (
                                    f"{city_kb_tag}.nc" if include_kb_in_gen_filenames else "gen.nc"
                                )
                            ]
                            if legacy_city_slug != city_slug:
                                candidate_city_paths.append(
                                    gen_layout["city_dir"] / legacy_city_slug / period_key / (
                                        f"{city_kb_tag}.nc" if include_kb_in_gen_filenames else "gen.nc"
                                    )
                                )

                            for fallback_city_path in candidate_city_paths:
                                if fallback_city_path.exists():
                                    city_path = str(fallback_city_path)
                                    city_generation_files[period_key][city_key] = city_path
                                    break

                        if city_path is not None and Path(city_path).exists():
                            city_paths.append(city_path)
                        elif city_path is not None:
                            missing_city_items.append(f"city={city_short_name} (missing file: {city_path})")
                        else:
                            missing_city_items.append(f"city={city_short_name} (not produced)")

                    if missing_city_items:
                        raise FileNotFoundError(
                            "Province aggregation requires all city generation files. "
                            f"period={period_key}, prov={prov_short_name}, missing={missing_city_items}"
                        )

                    if not city_paths:
                        raise FileNotFoundError(
                            "Province aggregation has no usable city generation files. "
                            f"period={period_key}, prov={prov_short_name}"
                        )

                    prov_ds = _aggregate_generation_paths(
                        ds_paths=city_paths,
                        mode="city2prov",
                        output_hourly_gen=output_hourly_gen,
                        output_monthly_gen=output_monthly_gen,
                        output_hourly_cf=output_hourly_cf,
                        period=None,
                    )
                    prov_long_name = group_df[column_map["prov_long"]].dropna().iloc[0] if not group_df[column_map["prov_long"]].dropna().empty else None
                    prov_ds.attrs.update(
                        {
                            "province_short_name": prov_short_name,
                            "province_long_name": prov_long_name,
                            "city_count": int(len(city_paths)),
                            "farm_count": int(len(group_df)),
                            "period": "full_year" if period_item is None else period_item,
                            "period_mode": period_mode,
                            "generation_adjustment_stage": "farm",
                            "city_capacity_adjust_applied_upstream": int(bool(city_capacity_map)),
                            "prov_utility_rate_adjust_applied_upstream": int(bool(prov_utility_map)),
                        }
                    )
                    prov_ds = sanitize_netcdf_attrs(prov_ds, "prov_ds")
                    prov_slug = sanitize_filename_part(chinese_to_pinyi(str(prov_short_name)))
                    prov_farm_ids = group_df[column_map["id"]].astype(int).tolist()
                    prov_k_token, prov_b_token = _group_kb_pair_tokens(prov_farm_ids, farm_kb_adjust_info_map)
                    prov_output_path = gen_layout["prov_dir"] / prov_slug / period_key / (
                        f"k{prov_k_token}b{prov_b_token}_gen.nc"
                        if include_kb_in_gen_filenames
                        else "gen.nc"
                    )
                    prov_output_path.parent.mkdir(parents=True, exist_ok=True)
                    if (not overwrite) and prov_output_path.exists():
                        prov_generation_files[period_key][str(prov_short_name)] = str(prov_output_path)
                        continue
                    write_netcdf_with_retry(prov_ds, prov_output_path)
                    prov_generation_files[period_key][str(prov_short_name)] = str(prov_output_path)

    if wind_layout is not None:
        prov_groups = selected_df.groupby(column_map["prov_short"], dropna=False)
        for prov_short_name, group_df in prov_groups:
            season_paths: dict[str, str] = {}
            for period_item in _progress(
                target_periods,
                total=len(target_periods),
                desc=f"Concatenating wind {prov_short_name}",
                disable=not show_progress,
            ):
                period_key = _period_key(period_item, period_mode)
                wind_paths = [
                    farm_wind_files[period_key][int(farm_id)]
                    for farm_id in group_df[column_map["id"]].astype(int).tolist()
                    if int(farm_id) in farm_wind_files[period_key]
                ]
                if not wind_paths:
                    continue

                season_df = concat_w_ds(
                    ds_list=wind_paths,
                    concat_vars=["farm_era5_w", "farm_era5_cdmet_w"],
                    time_name="time",
                )
                output_csv = wind_layout["root_dir"] / "province_seasonal_csv" / f"{sanitize_filename_part(chinese_to_pinyi(str(prov_short_name)))}_{period_key}_w.csv"
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                if (not overwrite) and output_csv.exists():
                    season_paths[period_key] = str(output_csv)
                    continue
                season_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
                season_paths[period_key] = str(output_csv)

            if season_paths:
                prov_seasonal_wind_csvs[str(prov_short_name)] = season_paths

    return {
        "inventory_csv_path": str(Path(inventory_csv_path)),
        "output_dir": str(output_base_dir),
        "period_mode": period_mode,
        "period": ["full_year" if period_item is None else period_item for period_item in target_periods],
        "aggr_level": aggr_level,
        "include_kb_in_gen_filenames": include_kb_in_gen_filenames,
        "overwrite": overwrite,
        "output_hourly_gen": output_hourly_gen,
        "output_monthly_gen": output_monthly_gen,
        "output_hourly_cf": output_hourly_cf,
        "output_w_compare": output_w_compare,
        "output_raw_w": output_raw_w,
        "city_capacity_adjust_csv_path": str(city_capacity_adjust_csv_path) if city_capacity_adjust_csv_path is not None else None,
        "prov_utility_rate_adjust_csv_path": str(prov_utility_rate_adjust_csv_path) if prov_utility_rate_adjust_csv_path is not None else None,
        "generation_adjustment_stage": "farm",
        "output_gen_dir": str(output_gen_dir) if output_gen_dir is not None else None,
        "output_w_compare_dir": str(output_w_compare_dir) if output_w_compare_dir is not None else None,
        "output_raw_w_dir": str(output_raw_w_dir) if output_raw_w_dir is not None else None,
        "output_gen_nc_path": str(output_gen_dir) if output_gen_dir is not None else None,
        "output_w_compare_nc_path": str(output_w_compare_dir) if output_w_compare_dir is not None else None,
        "output_raw_w_nc_path": str(output_raw_w_dir) if output_raw_w_dir is not None else None,
        "farm_adjusted_w_dir": str(farm_adjusted_w_dir) if farm_adjusted_w_dir is not None else None,
        "farm_adjusted_w_varname": farm_adjusted_w_varname,
        "filter_mode": filter_info["kind"],
        "filter_value": sorted(filter_info["value"]) if isinstance(filter_info["value"], set) else filter_info["value"],
        "selected_farm_ids": selected_farm_ids,
        "farm_generation_files": farm_generation_files,
        "farm_wind_files": farm_wind_files,
        "farm_raw_w_files": farm_raw_w_files,
        "city_generation_files": city_generation_files,
        "province_generation_files": prov_generation_files,
        "province_seasonal_wind_csvs": prov_seasonal_wind_csvs,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()

