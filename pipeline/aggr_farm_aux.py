from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Union, List, Tuple
import logging
import re

import xarray as xr
import pandas as pd
from tqdm import tqdm

from unit_farm_workflow import unit_farm_output_from_files

from utils import (
    aggr_gen_ds,
    adjust_da_by_time,
    filter_dataarray_by_period,
    format_for_filename,
    normalize_text,
    normalize_text_set,
    open_dataset_with_fallback,
    resolve_existing_column,
    safe_read_csv,
    sanitize_filename_part,
    sanitize_netcdf_attrs,
    write_netcdf_with_retry,
)

logger = logging.getLogger(__name__)


def check_cf_csv_requirement(task: Dict[str, Any], farm_id: int) -> Optional[Path]:
    """按「**cf_csv_dir 是否为 None**」这一唯一判据校验逐场 CF csv 是否可用（fail-loud）。

    返回解析后的 ``farm_power_curve_dir``（可能为 None）。规则：
      * ``farm_power_curve_dir`` 非 None -> 要求 ``farm_<id>.csv`` 存在，否则 FileNotFoundError；
      * ``farm_power_curve_dir`` 为 None -> 返回 None，表示用盘点表机型参数现场生成曲线
        （历史遗留的 ``use_cf_csv`` 字段已彻底移除，不再参与判断）。

    单独抽成函数是为了能在不跑批处理的前提下做单元测试。
    """
    farm_power_curve_dir_raw = task.get("farm_power_curve_dir")
    farm_power_curve_dir = Path(farm_power_curve_dir_raw) if farm_power_curve_dir_raw is not None else None

    if farm_power_curve_dir is not None:
        cf_csv_path = farm_power_curve_dir / f"farm_{farm_id}.csv"
        if not cf_csv_path.exists():
            raise FileNotFoundError(f"未找到风电场功率曲线文件: {cf_csv_path}")
    return farm_power_curve_dir

DEFAULT_INVENTORY_CSV_PATH = Path(r"D:\HKU PhD\wind data paper\wind_farms_inventory_till_2023.csv")
SEASON_ORDER = ("spring", "summer", "autumn", "winter")
SeasonName = Literal["spring", "summer", "autumn", "winter"]
PeriodMode = Literal["season", "month"]
PeriodItem = Union[SeasonName, int]
AggrLevel = Literal["city", "prov", "both"]


def _normalize_onshore_flag(value: Any) -> Optional[int]:
    """
    将风场陆海标识统一为 ``1=onshore``、``0=offshore``。

    在 aggr_farm 中用于把 inventory 里的 onshore_flag/陆海1-2 转成统一口径，
    以便在 farm 层正确匹配 city 级的 onshore/offshore 调整表。
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, str):
        text = normalize_text(value)
        if text is None:
            return None
        lowered = text.lower()
        if lowered in {"1", "onshore", "on_shore", "land", "陆上"}:
            return 1
        if lowered in {"0", "2", "offshore", "off_shore", "sea", "海上"}:
            return 0

    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"无法识别风场陆海标识: {value}") from None

    if numeric == 1:
        return 1
    if numeric in {0, 2}:
        return 0
    raise ValueError(f"风场陆海标识只能为 1(onshore) 或 0/2(offshore)，实际为: {value}")


def _normalize_gen_adjust_sequence(
    adjust_info: Optional[Union[Dict[str, float], List[Dict[str, float]], Tuple[Dict[str, float], ...]]]
) -> list[dict[str, float]]:
    """
    将 batch 层传入的 gen_adjust_info 规范化为 dict 列表。

    在 aggr_farm 中用于把用户原始 gen_adjust_info 与 city/prov 调整拼接成
    每个 farm 的专属调整序列，再下发给单 farm 计算。
    """
    if adjust_info is None:
        return []

    if isinstance(adjust_info, dict):
        return [dict(adjust_info)]

    if isinstance(adjust_info, (list, tuple)):
        normalized: list[dict[str, float]] = []
        for idx, item in enumerate(adjust_info, start=1):
            if item is None:
                continue
            if not isinstance(item, dict):
                raise ValueError(
                    f"gen_adjust_info 第 {idx} 项不是 dict，实际类型: {type(item).__name__}"
                )
            normalized.append(dict(item))
        return normalized

    raise ValueError("gen_adjust_info 必须为 dict、list[dict]、tuple[dict, ...] 或 None")


def _load_inventory_table(inventory_csv_path: Union[str, Path]):
    """
    读取并标准化风电场清单 CSV。

    将 CSV 读入 DataFrame，兼容多种列名写法（中文/英文），并对省市名列执行
    normalize_text 文本标准化，确保后续筛选时大小写、空格、全半角等差异不影响匹配。

    主函数中的作用：
        aggr_farm_output 在启动时调用此函数，获取完整的风电场元数据表
        以及列名映射字典（column_map），后续所有按省/市维度筛选都依赖于此。

    Returns:
        tuple[DataFrame, dict]: (标准化后的风电场 DataFrame, 列名映射字典)
    """
    inventory_path = Path(inventory_csv_path)
    farm_df = safe_read_csv(inventory_path)
    if farm_df is None or farm_df.empty:
        raise ValueError(f"inventory CSV 为空: {inventory_path}")

    id_col = resolve_existing_column(farm_df.columns, ["风电场ID", "风电场Id", "farm_id", "ID"], "风电场ID")
    prov_short_col = resolve_existing_column(farm_df.columns, ["省简称", "prov_short"], "省简称")
    prov_long_col = resolve_existing_column(farm_df.columns, ["省全称", "prov_long"], "省全称")
    city_short_col = resolve_existing_column(farm_df.columns, ["市简称", "city_short"], "市简称")
    city_long_col = resolve_existing_column(farm_df.columns, ["市全称", "city_long"], "市全称")
    try:
        onshore_flag_col = resolve_existing_column(
            farm_df.columns,
            ["onshore_flag", "陆海1-2", "陆海", "land_sea_flag"],
            "onshore_flag",
        )
    except KeyError:
        onshore_flag_col = None

    prepared_df = farm_df.copy()
    prepared_df[id_col] = prepared_df[id_col].astype(int)

    for col in (prov_short_col, prov_long_col, city_short_col, city_long_col):
        prepared_df[col] = prepared_df[col].map(normalize_text)
    if onshore_flag_col is not None:
        prepared_df[onshore_flag_col] = prepared_df[onshore_flag_col].map(_normalize_onshore_flag)

    column_map = {
        "id": id_col,
        "prov_short": prov_short_col,
        "prov_long": prov_long_col,
        "city_short": city_short_col,
        "city_long": city_long_col,
    }
    if onshore_flag_col is not None:
        column_map["onshore_flag"] = onshore_flag_col

    return prepared_df, column_map


def _filter_inventory_farms(
    farm_df,
    column_map: dict[str, str],
    prov_short: Optional[Union[str, Iterable[str]]] = None,
    prov_long: Optional[Union[str, Iterable[str]]] = None,
    city_short: Optional[Union[str, Iterable[str]]] = None,
    city_long: Optional[Union[str, Iterable[str]]] = None,
):
    """
    从完整清单中筛选出目标风电场子集，并返回筛选信息。

    每次调用只允许使用一种筛选维度（prov_short、prov_long、city_short、city_long 之一），
    传入多种会抛出 ValueError，以避免混用造成结果歧义。
    筛选时使用已标准化的文本进行集合匹配（isin）。

    主函数中的作用：
        aggr_farm_output 根据调用方传入的省/市参数调用此函数，
        得到 selected_df（决定要计算哪些风电场）和 filter_info（写入返回摘要）。

    Returns:
        tuple[DataFrame, dict]: (筛选后的子集 DataFrame（按 farm_id 排序）, 筛选信息)
    """
    # 仅允许一种筛选维度，避免省市条件混用造成歧义。
    text_filters = {
        "prov_short": normalize_text_set(prov_short),
        "prov_long": normalize_text_set(prov_long),
        "city_short": normalize_text_set(city_short),
        "city_long": normalize_text_set(city_long),
    }

    provided_filters: dict[str, Any] = {}
    for key, values in text_filters.items():
        if values is not None:
            provided_filters[key] = values

    if len(provided_filters) != 1:
        raise ValueError(
            "prov_short/prov_long/city_short/city_long 一次只能使用一种筛选方式"
        )

    selected = farm_df.copy()
    filter_kind = next(iter(provided_filters.keys()))
    filter_value = provided_filters[filter_kind]

    selected = selected[selected[column_map[filter_kind]].isin(filter_value)]

    if selected.empty:
        raise ValueError("筛选后没有匹配到任何风电场")

    return selected.sort_values(column_map["id"]).reset_index(drop=True), {
        "kind": filter_kind,
        "value": filter_value,
    }


def _period_key(period_item: Optional[PeriodItem], period_mode: PeriodMode) -> str:
    """
    将时间段标识符转换为用于文件命名的字符串键。

    规则如下：
    - None: 返回 "full_year"，表示不做时间筛选；
    - 季节模式: 直接返回季节名（如 "spring"）；
    - 月份模式: 返回零填充的两位数字符串（如月份 3 -> "m03"）。

    这样可以让全年输出与分季/分月输出共用同一套文件命名与字典键机制。

    主函数中的作用：
        在遍历 target_periods 时，为每个（farm_id, period）组合构造输出文件名，
        以及用作 farm_generation_files / farm_wind_files 等字典的键。
    """
    if period_item is None:
        return "full_year"
    if period_mode == "season":
        return str(period_item)
    return f"m{int(period_item):02d}"


def _kb_suffix_for_filename(kb_info: Optional[Union[Dict, tuple[float, float], str, Path]]) -> str:
    """
    根据 kb_info 生成 farm 风速类文件名里的 ``_k_x_b_y`` 后缀。

    在 aggr_farm 中主要用于 raw_w 和 w_compare 的 farm 级输出命名，
    让不同 k,b 方案的文件可以直接从文件名区分。
    """
    include_kb_in_name = False
    k_token = "1"
    b_token = "0"

    if kb_info is None:
        include_kb_in_name = True
    elif isinstance(kb_info, tuple):
        include_kb_in_name = True
        k_token = format_for_filename(float(kb_info[0]))
        b_token = format_for_filename(float(kb_info[1]))
    elif isinstance(kb_info, dict) and len(kb_info) == 1:
        only_value = next(iter(kb_info.values()))
        if not isinstance(only_value, (list, tuple)) or len(only_value) != 2:
            raise ValueError("kb_info 为单项 dict 时，其值必须是长度为2的(k, b)序列")
        include_kb_in_name = True
        k_token = format_for_filename(float(only_value[0]))
        b_token = format_for_filename(float(only_value[1]))

    if include_kb_in_name:
        return f"_k_{k_token}_b_{b_token}"
    return ""


def _kb_file_tag(kb_info: Optional[Union[Dict, tuple[float, float], str, Path]]) -> str:
    """
    将 kb_info 转成聚合发电量文件使用的简短 tag。

    在 aggr_farm 的 city/prov 聚合输出中，这个 tag 会进入最终文件名，
    用来标识当前聚合结果对应的 k,b 方案。
    """
    if kb_info is None:
        return "k1b0_gen"

    if isinstance(kb_info, tuple):
        return f"k{format_for_filename(float(kb_info[0]))}b{format_for_filename(float(kb_info[1]))}_gen"

    if isinstance(kb_info, dict) and len(kb_info) == 1:
        only_value = next(iter(kb_info.values()))
        if isinstance(only_value, (list, tuple)) and len(only_value) == 2:
            return f"k{format_for_filename(float(only_value[0]))}b{format_for_filename(float(only_value[1]))}_gen"

    return "mixed_kb"


def _group_kb_file_tag(
    farm_ids: list[int],
    farm_kb_adjust_info_map: dict[int, Optional[Union[Dict[str, tuple[float, float]], tuple[float, float]]]],
) -> str:
    """
    为一组 farm 计算统一的 kb 文件 tag。

    在 aggr_farm 的 city 聚合阶段使用；若同组 farm 的 kb 设置不一致，
    则返回 ``mixed_kb``，避免生成误导性的文件名。
    """
    tags = {
        _kb_file_tag(farm_kb_adjust_info_map.get(int(farm_id)))
        for farm_id in farm_ids
    }
    if len(tags) == 1:
        return next(iter(tags))
    return "mixed_kb"


def _kb_pair_tokens(kb_info: Optional[Union[Dict, tuple[float, float], str, Path]]) -> tuple[str, str]:
    """
    将 kb_info 解析为 ``(k_token, b_token)`` 二元字符串。

    在 aggr_farm 的 farm/prov 发电量文件命名中使用，便于把 k,b 直接拼到文件名里。
    """
    if kb_info is None:
        return ("1", "0")

    if isinstance(kb_info, tuple):
        return (format_for_filename(float(kb_info[0])), format_for_filename(float(kb_info[1])))

    if isinstance(kb_info, dict) and len(kb_info) == 1:
        only_value = next(iter(kb_info.values()))
        if isinstance(only_value, (list, tuple)) and len(only_value) == 2:
            return (format_for_filename(float(only_value[0])), format_for_filename(float(only_value[1])))

    return ("mixed", "mixed")


def _group_kb_pair_tokens(
    farm_ids: list[int],
    farm_kb_adjust_info_map: dict[int, Optional[Union[Dict[str, tuple[float, float]], tuple[float, float]]]],
) -> tuple[str, str]:
    """
    为一组 farm 生成统一的 ``(k_token, b_token)``。

    在 aggr_farm 的 province 聚合命名里使用；若组内 farm 的 kb 不一致，
    则返回 ``('mixed', 'mixed')``。
    """
    pair_set = {
        _kb_pair_tokens(farm_kb_adjust_info_map.get(int(farm_id)))
        for farm_id in farm_ids
    }
    if len(pair_set) == 1:
        return next(iter(pair_set))
    return ("mixed", "mixed")


def _normalize_type_label(value: Any) -> str:
    """
    统一风场类型标签到 onshore/offshore 两类标准写法。

    在 aggr_farm 读取 city 容量调整表时使用，保证不同中英文写法都能落到统一 key。
    """
    text = normalize_text(value)
    if text is None:
        return ""

    lowered = text.lower()
    if lowered in {"onshore", "on_shore", "land", "陆上"}:
        return "onshore"
    if lowered in {"offshore", "off_shore", "sea", "海上"}:
        return "offshore"
    return lowered


def _is_supported_adjust_key(key_text: str) -> bool:
    """
    判断列名是否可作为时间调整 key 使用。

    在 aggr_farm 加载 city/prov 调整 CSV 时，用于识别像 ``2000年``、``1月``、
    ``2012年12月`` 这类可交给 adjust_da_by_time 处理的列。
    """
    return (
        re.fullmatch(r"\d{4}年", key_text) is not None
        or re.fullmatch(r"\d{1,2}月", key_text) is not None
        or re.fullmatch(r"\d{4}年\d{1,2}月", key_text) is not None
    )


def _load_city_capacity_adjust_map(
    city_capacity_adjust_csv_path: Optional[Union[str, Path]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    """
    读取市级容量修正表，并整理成 ``(prov, city, type) -> adjust_info`` 映射。

    在新的 aggr_farm 流程中，这个映射不再用于 city 聚合后修正，
    而是先下沉到 farm 层，按每个 farm 的 ``prov_short/city_short/onshore_flag``
    组装进其专属 gen_adjust_info。
    """
    if city_capacity_adjust_csv_path is None:
        return {}

    csv_path = Path(city_capacity_adjust_csv_path)
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"city_capacity_adjust_csv_path 不存在或不是文件: {csv_path}")

    df = safe_read_csv(csv_path)
    if df is None or df.empty:
        raise ValueError(f"city_capacity_adjust_csv_path 对应 CSV 为空: {csv_path}")

    prov_col = resolve_existing_column(df.columns, ["省简称", "prov_short", "province", "省"], "省简称")
    city_col = resolve_existing_column(df.columns, ["市简称", "city_short", "city", "城市"], "市简称")
    type_col = resolve_existing_column(df.columns, ["type", "类型", "风场类型"], "type")

    adjust_cols: list[tuple[str, str]] = []
    for col in df.columns:
        normalized_col = normalize_text(col)
        if normalized_col is not None and _is_supported_adjust_key(normalized_col):
            adjust_cols.append((str(col), normalized_col))

    if not adjust_cols:
        raise ValueError(
            "city_capacity_adjust_csv_path 缺少可识别的调整列（如 '2000年'、'1月'、'2012年12月'）"
        )

    result: dict[tuple[str, str, str], dict[str, float]] = {}
    for _, row in df.iterrows():
        prov_name = normalize_text(row.get(prov_col))
        city_name = normalize_text(row.get(city_col))
        row_type = _normalize_type_label(row.get(type_col))
        if prov_name is None or city_name is None or row_type not in {"onshore", "offshore"}:
            continue

        key = (prov_name, city_name, row_type)
        result.setdefault(key, {})
        for source_col, adjust_key in adjust_cols:
            factor = pd.to_numeric(row.get(source_col), errors="coerce")
            if not pd.isna(factor):
                result[key][adjust_key] = float(factor)

    return result


def _build_farm_gen_adjust_info_map(
    selected_df: pd.DataFrame,
    column_map: dict[str, str],
    base_gen_adjust_info: Optional[Union[Dict[str, float], List[Dict[str, float]], Tuple[Dict[str, float], ...]]],
    city_capacity_map: dict[tuple[str, str, str], dict[str, float]],
    prov_utility_map: dict[str, dict[str, float]],
) -> dict[int, Optional[list[dict[str, float]]]]:
    """
    为每个 farm 构建最终下发给 unit_farm_output 的 gen_adjust_info 序列。

    组装顺序为：用户原始 gen_adjust_info -> city 调整 -> prov 调整。
    这样单 farm 在输出 generation 时就已经是最终口径，后续 city/prov 聚合只需累加。
    """
    base_sequence = _normalize_gen_adjust_sequence(base_gen_adjust_info)

    if city_capacity_map and "onshore_flag" not in column_map:
        raise ValueError(
            "启用 city_capacity_adjust_csv_path 时，inventory_csv_path 必须包含 onshore_flag 或等价的陆海标识列"
        )

    result: dict[int, Optional[list[dict[str, float]]]] = {}
    for _, row in selected_df.iterrows():
        farm_id = int(row[column_map["id"]])
        prov_short_name = normalize_text(row.get(column_map["prov_short"]))
        city_short_name = normalize_text(row.get(column_map["city_short"]))

        farm_sequence = [dict(item) for item in base_sequence]

        if city_capacity_map:
            onshore_flag = _normalize_onshore_flag(row.get(column_map["onshore_flag"]))
            if onshore_flag is None:
                raise ValueError(f"farm_id={farm_id} 缺少有效 onshore_flag，无法匹配 city 级调整")
            farm_type = "onshore" if onshore_flag == 1 else "offshore"
            city_adjust = city_capacity_map.get((str(prov_short_name), str(city_short_name), farm_type))
            if city_adjust:
                farm_sequence.append(dict(city_adjust))

        if prov_short_name is not None:
            prov_adjust = prov_utility_map.get(str(prov_short_name))
            if prov_adjust:
                farm_sequence.append(dict(prov_adjust))

        result[farm_id] = farm_sequence or None

    return result


def _build_farm_generation_adjustment_attr_map(
    selected_df: pd.DataFrame,
    column_map: dict[str, str],
    city_capacity_map: dict[tuple[str, str, str], dict[str, float]],
    prov_utility_map: dict[str, dict[str, float]],
) -> dict[int, dict[str, Any]]:
    """
    为每个 farm 生成需要写入 generation nc attrs 的校正记录。

    这些 attrs 只描述 aggr_farm 在 farm 层额外附加的市级/省级修正是否命中，
    便于后续直接从单 farm generation 文件判断结果口径。
    """
    if city_capacity_map and "onshore_flag" not in column_map:
        raise ValueError(
            "启用 city_capacity_adjust_csv_path 时，inventory_csv_path 必须包含 onshore_flag 或等价的陆海标识列"
        )

    result: dict[int, dict[str, Any]] = {}
    for _, row in selected_df.iterrows():
        farm_id = int(row[column_map["id"]])
        prov_short_name = normalize_text(row.get(column_map["prov_short"]))
        city_short_name = normalize_text(row.get(column_map["city_short"]))

        city_adjust = None
        city_adjust_key = None
        if city_capacity_map:
            onshore_flag = _normalize_onshore_flag(row.get(column_map["onshore_flag"]))
            if onshore_flag is None:
                raise ValueError(f"farm_id={farm_id} 缺少有效 onshore_flag，无法匹配 city 级调整")
            farm_type = "onshore" if onshore_flag == 1 else "offshore"
            city_adjust_key = (str(prov_short_name), str(city_short_name), farm_type)
            city_adjust = city_capacity_map.get(city_adjust_key)

        prov_adjust = None if prov_short_name is None else prov_utility_map.get(str(prov_short_name))

        result[farm_id] = {
            "generation_adjustment_stage": "farm",
            "city_capacity_adjust_applied": int(bool(city_adjust)),
            "prov_utility_rate_adjust_applied": int(bool(prov_adjust)),
            "city_capacity_adjust_key": (
                f"{city_adjust_key[0]}/{city_adjust_key[1]}/{city_adjust_key[2]}"
                if city_adjust_key is not None and city_adjust
                else None
            ),
            "prov_utility_rate_adjust_key": str(prov_short_name) if prov_adjust else None,
        }

    return result


def _write_farm_generation_adjustment_attrs(
    output_gen_nc_path: Optional[Union[str, Path]],
    adjustment_attrs: Optional[dict[str, Any]],
) -> None:
    """将 farm 层 generation 校正记录写回 generation nc attrs。"""
    if output_gen_nc_path is None or not adjustment_attrs:
        return

    path = Path(output_gen_nc_path)
    if not path.exists():
        return

    ds = open_dataset_with_fallback(path).load()
    ds.attrs.update(adjustment_attrs)
    ds = sanitize_netcdf_attrs(ds, "farm_generation_ds")
    write_netcdf_with_retry(ds, path)


def _load_prov_utility_rate_map(
    prov_utility_rate_adjust_csv_path: Optional[Union[str, Path]],
) -> dict[str, dict[str, float]]:
    """
    读取省级利用率修正表，并整理成 ``prov -> adjust_info`` 映射。

    在新的 aggr_farm 流程中，这个映射会先下沉到 farm 层，
    按 farm 所属省份追加到其专属 gen_adjust_info 序列中。
    """
    if prov_utility_rate_adjust_csv_path is None:
        return {}

    csv_path = Path(prov_utility_rate_adjust_csv_path)
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"prov_utility_rate_adjust_csv_path 不存在或不是文件: {csv_path}")

    df = safe_read_csv(csv_path)
    if df is None or df.empty:
        raise ValueError(f"prov_utility_rate_adjust_csv_path 对应 CSV 为空: {csv_path}")

    prov_col = resolve_existing_column(df.columns, ["省简称", "prov_short", "province", "省"], "省简称")

    adjust_cols: list[tuple[str, str]] = []
    for col in df.columns:
        normalized_col = normalize_text(col)
        if normalized_col is not None and _is_supported_adjust_key(normalized_col):
            adjust_cols.append((str(col), normalized_col))

    if not adjust_cols:
        raise ValueError(
            "prov_utility_rate_adjust_csv_path 缺少可识别的调整列（如 '2000年'、'1月'、'2012年12月'）"
        )

    result: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        prov_name = normalize_text(row.get(prov_col))
        if prov_name is None:
            continue

        result.setdefault(prov_name, {})
        for source_col, adjust_key in adjust_cols:
            factor = pd.to_numeric(row.get(source_col), errors="coerce")
            if not pd.isna(factor):
                result[prov_name][adjust_key] = float(factor)

    return result


def _apply_adjust_info_to_da(da: xr.DataArray, adjust_info: Optional[dict[str, float]]) -> xr.DataArray:
    """
    按 DataArray 的时间维度把 adjust_info 应用到单个变量上。

    这是 aggr_farm 调整逻辑里的最小执行单元；当前主流程已将 generation 调整
    下沉到 farm 层，但该函数仍保留给 dataset 级补充处理或回退逻辑使用。
    """
    if not adjust_info:
        return da
    if "time" in da.dims:
        return adjust_da_by_time(da, adjust_info, time_name="time")
    if "date" in da.dims:
        return adjust_da_by_time(da, adjust_info, time_name="date")
    return da


def _apply_city_capacity_adjustment(
    city_ds: xr.Dataset,
    prov_short_name: Any,
    city_short_name: Any,
    city_capacity_map: dict[tuple[str, str, str], dict[str, float]],
) -> xr.Dataset:
    """
    对 city 聚合结果应用市级容量修正，并回算 total_* 变量。

    当前主流程已不再在 city 聚合后调用该函数；保留它是为了提供 dataset 级
    回退能力，避免 city 调整逻辑完全散落在主流程外。
    """
    prov_name = normalize_text(prov_short_name)
    city_name = normalize_text(city_short_name)
    if prov_name is None or city_name is None or not city_capacity_map:
        return city_ds

    adjusted_ds = city_ds.copy(deep=True)
    base_names: set[str] = set()
    for var_name in list(adjusted_ds.data_vars):
        if var_name.startswith("onshore_"):
            base_names.add(var_name[len("onshore_"):])
            adjusted_ds[var_name] = _apply_adjust_info_to_da(
                adjusted_ds[var_name],
                city_capacity_map.get((prov_name, city_name, "onshore")),
            )
        elif var_name.startswith("offshore_"):
            base_names.add(var_name[len("offshore_"):])
            adjusted_ds[var_name] = _apply_adjust_info_to_da(
                adjusted_ds[var_name],
                city_capacity_map.get((prov_name, city_name, "offshore")),
            )
        elif var_name.startswith("total_"):
            base_names.add(var_name[len("total_"):])

    for base_name in base_names:
        onshore_name = f"onshore_{base_name}"
        offshore_name = f"offshore_{base_name}"
        total_name = f"total_{base_name}"
        has_onshore = onshore_name in adjusted_ds.data_vars
        has_offshore = offshore_name in adjusted_ds.data_vars
        if has_onshore and has_offshore:
            adjusted_ds[total_name] = adjusted_ds[onshore_name] + adjusted_ds[offshore_name]
        elif has_onshore:
            adjusted_ds[total_name] = adjusted_ds[onshore_name].copy(deep=True)
        elif has_offshore:
            adjusted_ds[total_name] = adjusted_ds[offshore_name].copy(deep=True)

    return adjusted_ds


def _apply_prov_utility_rate_adjustment(
    prov_ds: xr.Dataset,
    prov_short_name: Any,
    prov_utility_map: dict[str, dict[str, float]],
) -> xr.Dataset:
    """
    对 province 聚合结果应用省级利用率修正。

    当前主流程已不再在 prov 聚合后调用该函数；保留它是为了提供 dataset 级
    回退能力，便于单独处理已有 province Dataset。
    """
    prov_name = normalize_text(prov_short_name)
    info = None if prov_name is None else prov_utility_map.get(prov_name)
    if not info:
        return prov_ds

    adjusted_ds = prov_ds.copy(deep=True)
    for var_name in list(adjusted_ds.data_vars):
        adjusted_ds[var_name] = _apply_adjust_info_to_da(adjusted_ds[var_name], info)
    return adjusted_ds


def _resolve_periods(
    period: Optional[Union[SeasonName, int, list[Union[SeasonName, int]], tuple[Union[SeasonName, int], ...], Literal["all"]]],
    period_mode: PeriodMode,
) -> list[Optional[PeriodItem]]:
    """
    将用户传入的 period 参数规范化为有序、去重的时间段列表。

    支持以下输入形式：
    - None: 返回 [None]，表示全年输出且不做时间筛选。
    - "all": 展开为当前模式的全部时间段（4 个季节或 12 个月）。
    - 单值：返回包含该值的单元素列表。
    - 列表/元组：去重后按输入顺序返回。
    同时对类型合法性做严格校验，避免如季节模式下传入月份整数等错误。

    主函数中的作用：
        aggr_farm_output 在开始时调用此函数，得到 target_periods，
        后续所有循环（farm 计算、发电量聚合、风速拼接）都以此列表为迭代对象。
    """
    if period_mode not in ("season", "month"):
        raise ValueError("period_mode 仅支持 'season' 或 'month'")

    if period is None:
        return [None]

    if period == "all":
        if period_mode == "season":
            return list(SEASON_ORDER)
        return list(range(1, 13))

    if period_mode == "season":
        if isinstance(period, str):
            if period not in SEASON_ORDER:
                raise ValueError(f"period_mode='season' 时，period 仅支持 {SEASON_ORDER}、'all' 或 None")
            return [period]

        if isinstance(period, int):
            raise ValueError("period_mode='season' 时，period 不能是月份整数")

        periods: list[PeriodItem] = []
        for item in period:
            if not isinstance(item, str) or item not in SEASON_ORDER:
                raise ValueError(f"period_mode='season' 时，period 列表元素必须属于 {SEASON_ORDER}，实际: {item}")
            periods.append(item)
    else:
        if isinstance(period, int):
            if 1 <= period <= 12:
                return [period]
            raise ValueError("period_mode='month' 时，period 整数必须在 1-12")

        if isinstance(period, str):
            raise ValueError("period_mode='month' 时，period 不能是季节字符串")

        periods = []
        for item in period:
            if not isinstance(item, int) or not (1 <= item <= 12):
                raise ValueError(f"period_mode='month' 时，period 列表元素必须是 1-12 的整数，实际: {item}")
            periods.append(item)

    if len(periods) == 0:
        raise ValueError("period 列表不能为空")

    unique_periods: list[PeriodItem] = []
    for item in periods:
        if item not in unique_periods:
            unique_periods.append(item)
    return unique_periods


def _build_output_layout(
    output_path: Optional[Union[str, Path]],
    single_farm: bool,
    default_root_name: str,
    with_city_prov_dirs: bool = True,
) -> Optional[dict[str, Path]]:
    """
    根据用户指定的输出路径，构建并创建输出目录结构，返回各子目录路径的字典。

    路径解析规则：
    - 若 output_path 以 .nc 结尾且为单风电场，则直接写入该文件（single_file 模式）。
    - 若 output_path 以 .nc 结尾但有多风电场，去掉后缀作为根目录。
    - 否则直接将 output_path 作为根目录。
    根目录下始终创建 farm/ 子目录；当 with_city_prov_dirs=True 时，额外创建
    city/ 和 prov/ 子目录（仅发电量输出需要三级聚合时使用）。
    注：aggr_farm_output 在 batch 模式下会先校验输出参数必须为目录，
    因此主流程通常不会走 single_file 模式。

    主函数中的作用：
        分别为 gen_layout（发电量）、wind_layout（风速对比）、raw_layout（原始风速）
        调用此函数，得到各输出类型的目录结构，后续文件写入路径全部由此派生。
    """
    if output_path is None:
        return None

    raw_path = Path(output_path)
    if raw_path.suffix.lower() == ".nc":
        if single_farm:
            root_dir = raw_path.parent
            single_file = raw_path
        else:
            root_dir = raw_path.with_suffix("")
            single_file = None
    else:
        root_dir = raw_path
        single_file = None

    if str(root_dir) == "":
        root_dir = Path(default_root_name)

    farm_dir = root_dir / "farm"
    farm_dir.mkdir(parents=True, exist_ok=True)

    layout: dict[str, Path] = {
        "root_dir": root_dir,
        "farm_dir": farm_dir,
        "single_file": single_file,
    }

    if with_city_prov_dirs:
        city_dir = root_dir / "city"
        prov_dir = root_dir / "prov"
        city_dir.mkdir(parents=True, exist_ok=True)
        prov_dir.mkdir(parents=True, exist_ok=True)
        layout["city_dir"] = city_dir
        layout["prov_dir"] = prov_dir

    return layout


def _progress(iterable, total: Optional[int], desc: str, disable: bool = False):
    """
    对可迭代对象包装 tqdm 进度条，统一进度条样式。

    主函数中的作用：
        用于 farm 计算结果收集循环、发电量聚合循环、风速拼接循环等耗时操作，
        向终端实时显示进度。
    """
    return tqdm(iterable, total=total, desc=desc, unit="task", dynamic_ncols=True, disable=disable)


def _aggregate_generation_paths(
    ds_paths: list[Union[str, Path]],
    mode: Literal["farm2city", "city2prov"],
    output_hourly_gen: bool,
    output_monthly_gen: bool,
    output_hourly_cf: bool,
    period: Optional[str],
) -> xr.Dataset:
    """
    从一组 NetCDF 输入中聚合发电量相关变量，返回单个 Dataset。

    在 aggr_farm 中，这个函数承担两类聚合：farm->city 和 city->prov。
    它只负责“读文件、按变量聚合、合并结果”，不负责分组和命名。
    """
    filtered_inputs: list[xr.Dataset] = []
    missing_paths: list[str] = []
    for ds_path in ds_paths:
        path = Path(ds_path).resolve()
        if not path.exists():
            missing_paths.append(str(path))
            continue

        ds = open_dataset_with_fallback(path)
        if period is None:
            filtered_inputs.append(ds)
            continue

        filtered_vars: dict[str, xr.DataArray] = {}
        for var_name, da in ds.data_vars.items():
            if "time" in da.dims:
                filtered_vars[var_name] = filter_dataarray_by_period(da, period=period, time_name="time", drop=True)
            elif "date" in da.dims:
                filtered_vars[var_name] = filter_dataarray_by_period(da, period=period, time_name="date", drop=True)
            else:
                filtered_vars[var_name] = da

        filtered_ds = xr.Dataset(filtered_vars)
        filtered_ds.attrs.update(ds.attrs)
        filtered_inputs.append(filtered_ds.load())

    if not filtered_inputs:
        raise FileNotFoundError(
            f"聚合输入文件全部缺失或不可用，mode={mode}, missing_paths={missing_paths}"
        )

    pieces: list[xr.Dataset] = []
    if output_hourly_gen:
        pieces.append(aggr_gen_ds(ds_list=filtered_inputs, aggr_vars=["gen"], time_name="time", mode=mode))
    if output_monthly_gen:
        pieces.append(aggr_gen_ds(ds_list=filtered_inputs, aggr_vars=["monthly_gen"], time_name="date", mode=mode))
    if output_hourly_cf:
        pieces.append(aggr_gen_ds(ds_list=filtered_inputs, aggr_vars=["cf"], time_name="time", mode=mode))
    if not pieces:
        raise ValueError("至少需要输出一个发电量相关变量：output_hourly_gen/output_monthly_gen/output_hourly_cf")

    return _merge_datasets(pieces)


def _run_single_farm_task(task: dict[str, Any]) -> dict[str, Any]:
    """
    执行单个 ``(farm_id, period)`` 任务并返回产物路径摘要。

    在 aggr_farm 中，这个函数被进程池并行调用：先生成单场结果文件，
    再在需要时把结果文件按 period 过滤到对应时间段。
    """
    farm_id = int(task["farm_id"])
    # 「是否使用逐场 CF csv」的**唯一判据 = cf_csv_dir（= farm_power_curve_dir）是否为 None**；
    # 校验逻辑抽到 check_cf_csv_requirement 以便单测（见 tests/test_cf_csv_requirement.py）。
    farm_power_curve_dir = check_cf_csv_requirement(task, farm_id)

    period_item: Optional[PeriodItem] = task["period_item"]
    period_mode = task["period_mode"]

    unit_farm_output_from_files(
        farm_csv=task["inventory_csv_path"],
        farm_id=farm_id,
        era5_dir=task["era5_dir"],
        cdmet_dir=task["cdmet_dir"],
        output_gen_nc_path=task["output_gen_nc_path"],
        output_w_compare_nc_path=task["output_w_compare_nc_path"],
        output_raw_w_nc_path=task["output_raw_w_nc_path"],
        farm_adjusted_w_nc_path=task.get("farm_adjusted_w_nc_path"),
        farm_adjusted_w_varname=str(task.get("farm_adjusted_w_varname", "raw_w")),
        output_w_compare=task["output_w_compare"],
        output_raw_w=task["output_raw_w"],
        # batch worker 绝不隐式回填"默认功率曲线目录"：是否用 CF csv 只看 cf_csv_dir
        allow_default_cf_csv_dir=False,
        cf_csv_dir=str(farm_power_curve_dir) if farm_power_curve_dir is not None else None,
        kb_adjust_info=task["kb_adjust_info"],
        gen_adjust_info=task["gen_adjust_info"],
        output_hourly_gen=task["output_hourly_gen"],
        output_monthly_gen=task["output_monthly_gen"],
        output_hourly_cf=task["output_hourly_cf"],
        horizon_interp_method=str(task["horizon_interp_method"]),
        verbose=task["verbose"],
    )

    _write_farm_generation_adjustment_attrs(
        output_gen_nc_path=task.get("output_gen_nc_path"),
        adjustment_attrs=task.get("generation_adjustment_attrs"),
    )

    for key in ("output_gen_nc_path", "output_w_compare_nc_path", "output_raw_w_nc_path"):
        nc_path = task[key]
        if nc_path is None or period_item is None:
            continue

        path = Path(nc_path)
        if not path.exists():
            continue

        with xr.open_dataset(path) as ds:
            filtered_vars: dict[str, xr.DataArray] = {}
            for var_name, da in ds.data_vars.items():
                if "time" in da.dims:
                    filtered_vars[var_name] = filter_dataarray_by_period(da, period=period_item, time_name="time", drop=True)
                elif "date" in da.dims:
                    filtered_vars[var_name] = filter_dataarray_by_period(da, period=period_item, time_name="date", drop=True)
                else:
                    filtered_vars[var_name] = da

            filtered_ds = xr.Dataset(filtered_vars)
            filtered_ds.attrs.update(ds.attrs)
            filtered_ds.attrs["period"] = period_item
            filtered_ds.attrs["period_mode"] = period_mode
            filtered_ds = sanitize_netcdf_attrs(filtered_ds, "filtered_ds").load()

        write_netcdf_with_retry(filtered_ds, path)

    return {
        "farm_id": farm_id,
        "period_item": period_item,
        "period_mode": period_mode,
        "output_gen_nc_path": task["output_gen_nc_path"],
        "output_w_compare_nc_path": task["output_w_compare_nc_path"],
        "output_raw_w_nc_path": task["output_raw_w_nc_path"],
    }


def _merge_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
    """
    将多个 xr.Dataset 的变量合并为一个 Dataset。

    按变量名逐一复制，全局属性取自第一个 Dataset。适用于把分别计算的
    hourly_gen、monthly_gen、cf 等片段拼合成完整输出。

    主函数中的作用：
        _aggregate_generation_paths 在分别执行各变量的聚合后，调用此函数
        将结果合并为单个 Dataset，再写入 NetCDF 文件。
    """
    merged = xr.Dataset()
    for ds in datasets:
        for var_name, da in ds.data_vars.items():
            merged[var_name] = da
    merged.attrs.update(datasets[0].attrs if datasets else {})
    return merged


