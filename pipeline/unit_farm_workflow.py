

from typing import Union, Literal, Optional, List, Dict, Any, Tuple
from pathlib import Path
from time import perf_counter
from dataclasses import dataclass
import logging
import json
import ast
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from vars_computation import (
    cdmet_10m_w,
    era5_10m_w,
    era5_100m_w,
    era5_alpha,
    era5_rou,
    farm_cf,
    farm_generation,
    vertical_extrap_w,
    rou_adjustment_w,
    hub_w_kb_adjust,
)
from horizontal_interp import horizontal_interp
from parse_aux import parse_list, parse_none, parse_tf

from config import curve_dir

from utils import (
    SEASON_ORDER,
    normalize_text,
    resolve_existing_column,
    load_and_merge_subdatasets,
    adjust_hourly_with_daily,
    align_da_to_reference_year_range,
    concat_das,
    format_for_filename,
    hourly_to_monthly,
    sanitize_netcdf_attrs,
    safe_read_csv,
    adjust_da_by_time,
    filter_dataarray_by_period,
    filter_xr_by_period,
)

logger = logging.getLogger(__name__)


def _normalize_inventory_onshore_flag(value: Any) -> int:
    """Normalize inventory land/sea flag to 1=onshore, 0=offshore."""
    if value is None or pd.isna(value):
        raise ValueError("风电场陆海标识为空")

    if isinstance(value, str):
        text = normalize_text(value)
        if text is None:
            raise ValueError("风电场陆海标识为空")

        lowered = text.lower()
        if lowered in {"1", "onshore", "on_shore", "land", "陆上"}:
            return 1
        if lowered in {"0", "2", "offshore", "off_shore", "sea", "海上"}:
            return 0

    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"无法识别风电场陆海标识: {value}") from None

    if numeric == 1:
        return 1
    if numeric in {0, 2}:
        return 0
    raise ValueError(f"风电场陆海标识只能为 1(onshore) 或 0/2(offshore)，实际为: {value}")


# ---------------------------------------------------------------------------
# 盘点表列名解析（修复：别名与实际列名不一致 ⇒ 静默兜底成默认 3.0/12.0/25.0）
# ---------------------------------------------------------------------------
# 背景：原实现用 `resolve_existing_column` 式的**精确**列名匹配，候选别名写成
#   "Cut-in w" / "Cut-in wind" / "cut_in_v"
# 而盘点表真实列名是
#   "Cut-in wind speed" / "Rated wind speed" / "Cut-out wind speed"
# 三列全部匹配失败，于是每个风电场都被静默地套上 unit_farm_output 的默认
# 3.0/12.0/25.0（而 farm_power_curves.py 用全名读取，生成的是逐场真实参数曲线），
# 造成"读 CF csv"与"现场生成曲线"两条路相差约 2 倍。
#
# 修法分三层：
#   1) 别名表补上真实全名（首选精确匹配）；
#   2) 匹配前对**列名与别名都做规范化**（大小写/空格/标点/单位后缀/噪声词），
#      使旧别名 "Cut-in wind"、"cut_in_v" 也能命中，同类笔误不再致命；
#   3) **禁止静默兜底**：确实取不到时，strict=False 发 warnings.warn（含缺哪些参数、
#      回退到什么默认值、可用列名），strict=True 直接 raise。
_COLUMN_NOISE_TOKENS = frozenset({
    "speed", "speeds", "velocity", "vel", "value", "values", "val",
    "w", "wind", "v", "ms", "mps", "kmh",
})

TURBINE_CURVE_PARAM_SPECS: Tuple[Tuple[str, Tuple[str, ...], float], ...] = (
    ("cut_in_v", ("Cut-in wind speed", "Cut-in w", "Cut-in wind", "cut_in_v"), 3.0),
    ("rated_v", ("Rated wind speed", "Rated wind", "rated_v"), 12.0),
    ("cut_out_v", ("Cut-out wind speed", "Cut-out w", "cut_out_v"), 25.0),
)

# 调用方未给 cf_csv_dir 且 allow_default_cf_csv_dir=True 时回填的默认目录。
# 注意：它**不是**"是否使用 CF csv"的判据——唯一判据是 cf_csv_dir 是否为 None。
def _default_cf_csv_dir() -> Path:
    """默认功率曲线目录（惰性求值：环境变量在调用时生效，而不是 import 时被冻结）。

    用 config.curve_dir()：与 panel.py 的 farm_power_curve_dir 同源，
    因此"step1 写曲线的目录"与"下游读曲线兜底目录"永远是同一个，
    不会在设了 WINDPAPER_CURVE_DIR（读写分离）时静默回落到只读输入目录的旧曲线。
    """
    return curve_dir()


def normalize_column_key(name: Any) -> str:
    """把列名/别名规范化为可比较的 key：小写、去括号备注、去标点、去单位与噪声词。

    "Cut-in wind speed" / "Cut-in wind" / "Cut-in w" / "cut_in_v" -> "cutin"
    "Rated wind speed" / "Rated wind" / "rated_v"               -> "rated"
    "Cut-out wind speed" / "Cut-out w" / "cut_out_v"            -> "cutout"
    中文列名（如 "Hub height"、"轮毂高度"）同样适用。
    """
    text = normalize_text(name)
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"（[^）]*）", " ", text)
    tokens = re.split(r"[^0-9a-z\u4e00-\u9fff]+", text)
    return "".join(t for t in tokens if t and t not in _COLUMN_NOISE_TOKENS)


def match_column(columns: Any, alias: str) -> Optional[str]:
    """返回命中的真实列名：先精确匹配，再规范化精确匹配。

    规范化匹配必须**唯一**命中，命中 0 个或 ≥2 个都返回 None（不猜，避免张冠李戴）。
    """
    columns_list = [str(c) for c in columns]
    if alias in columns_list:
        return alias
    key = normalize_column_key(alias)
    if not key:
        return None
    hits = [c for c in columns_list if normalize_column_key(c) == key]
    return hits[0] if len(hits) == 1 else None


def resolve_turbine_curve_params(
    row: Any,
    columns: Any,
    *,
    strict: bool = False,
    context: str = "",
) -> Dict[str, float]:
    """从一行里解析 cut_in_v / rated_v / cut_out_v。

    取不到时**绝不静默**：strict=True 抛 ValueError；否则发 UserWarning，
    消息里含缺哪些参数、回退到什么默认值、以及可用列名列表。
    """
    values: Dict[str, float] = {}
    defaults_used: Dict[str, float] = {}
    missing: List[str] = []

    for name, aliases, default in TURBINE_CURVE_PARAM_SPECS:
        found: Optional[float] = None
        for alias in aliases:
            col = match_column(columns, alias)
            if col is None:
                continue
            try:
                raw = row[col]
            except (KeyError, IndexError):
                continue
            if raw is None or pd.isna(raw):
                continue
            try:
                found = float(raw)
            except (TypeError, ValueError):
                continue
            break

        if found is None:
            values[name] = float(default)
            defaults_used[name] = float(default)
            missing.append(name)
        else:
            values[name] = found

    if missing:
        message = (
            "盘点表缺少风机风速参数列%s：已回退为默认值 %s；可用列名: %s"
            % (("（%s）" % context) if context else "", defaults_used, [str(c) for c in columns])
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    return values


# ---------------------------------------------------------------------------
# 场址海拔（BUG-1：dem 恒为 0 ⇒ 密度修正被架空）
# ---------------------------------------------------------------------------
DEM_ALIASES: Tuple[str, ...] = ("Elevation", "elevation", "海拔", "高程", "dem", "dem_m")


def resolve_dem_elevation(
    row: Any,
    columns: Any,
    *,
    allow_orography_fallback: bool = False,
    context: str = "",
) -> Optional[float]:
    """解析风电场场址海拔（m，AMSL）。

    **不允许静默取 0**：0 m 是合法的海平面值，静默填 0 会让高原场被当成海平面场
    （`z2 = hub + dem = hub` ⇒ 把轮毂高度当海拔，密度修正方向反向）。

    规则：
      * 命中海拔列且值有效 -> 返回该值；
      * 列缺失或值非法：
          - ``allow_orography_fallback=False``（默认）-> raise ValueError（提示如何启用回退）；
          - ``allow_orography_fallback=True`` -> warnings.warn 并返回 None，下游改用
            ERA5 格点地形位势高度 H1 作为地面（见 ``era5_rou`` 的 orography 模式）。
    """
    for alias in DEM_ALIASES:
        col = match_column(columns, alias)
        if col is None:
            continue
        # 列存在：区分"整列缺失"（配置错误 ⇒ raise）与"本场值不可用"（⇒ warn + 回退）
        try:
            raw = row[col]
        except (KeyError, IndexError):
            raw = None
        value: Optional[float] = None
        if raw is not None and not pd.isna(raw):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = None
            if value is not None and not np.isfinite(value):
                value = None
        if value is not None:
            return value
        warnings.warn(
            "场址海拔列 %s 存在但取值不可用%s（值=%r）：已回退为 ERA5 格点地形位势高度 H1 作地面、"
            "向上外推到轮毂（H2−H1≈hub）。请核实该场 DEM。"
            % (col, ("（%s）" % context) if context else "", raw),
            UserWarning,
            stacklevel=2,
        )
        return None

    # 一个海拔列都没命中 = 配置/数据错误
    message = (
        "盘点表缺少可用的场址海拔列%s（候选列名 %s；可用列名: %s）。"
        "缺省会静默取 0，而 0 m 是合法海平面值，会让高原场密度偏高（方向相反）。"
        % (("（%s）" % context) if context else "", list(DEM_ALIASES), [str(c) for c in columns])
    )
    if allow_orography_fallback:
        warnings.warn(
            message + " 已按 allow_era5_orography_fallback=True 回退：以 ERA5 格点地形位势高度 H1 "
            "作为地面并向上外推到轮毂（H2−H1≈hub）。生产重跑请优先使用带海拔的盘点表。",
            UserWarning,
            stacklevel=2,
        )
        return None

    raise ValueError(
        message + " 请使用带海拔列的盘点表（如 wind_farms_inventory_till_2023_with_elevation.csv），"
        "或显式设置 allow_era5_orography_fallback=True 以用 ERA5 地形位势高度代替 DEM。"
    )



def generate_farm_seasonal_kb_dict(
    farm_id: Union[int, str],
    farm_info_csv_path: Union[str, Path],
    kb_csv_path: Union[str, Path],
    farm_id_col_aliases: Optional[List[str]] = None,
    province_col_aliases: Optional[List[str]] = None,
    kb_province_col_aliases: Optional[List[str]] = None,
    period_col_aliases: Optional[List[str]] = None,
    k_col_aliases: Optional[List[str]] = None,
    b_col_aliases: Optional[List[str]] = None,
    season_order: Tuple[str, ...] = SEASON_ORDER,
) -> Dict[str, Tuple[float, float]]:
    """
    根据风电场 ID 获取其所在省份在四季对应的 k、b 参数。

    流程：
    1. 在风电场信息 CSV 中根据 farm_id 找到该风电场对应的省简称；
    2. 在 k/b CSV 中根据省简称筛选四季记录；
    3. 返回 {'spring': (k, b), ...} 格式字典。

    Args:
        farm_id: 风电场 ID。
        farm_info_csv_path: 风电场信息 CSV 路径，需包含风电场 ID 和省简称列。
        kb_csv_path: 季节性 k/b CSV 路径，需包含省简称、季节、k、b 列。
        *_aliases: 对应列名候选，未传入时使用默认候选列表。
        season_order: 期望输出和校验的季节顺序。

    Returns:
        Dict[str, Tuple[float, float]]: 例如
            {'spring': (k1, b1), 'summer': (k2, b2), ...}
    """
    farm_id_aliases = farm_id_col_aliases or ['风电场ID', 'farm_id', 'farm id', 'id']
    province_aliases = province_col_aliases or ['省简称', '省份简称', 'province_short', 'prov', 'province']
    kb_province_aliases = kb_province_col_aliases or ['prov', '省简称', '省份简称', 'province_short', 'province']
    period_aliases = period_col_aliases or ['period', 'season', '季节']
    k_aliases = k_col_aliases or ['min_w_smape_k', 'k']
    b_aliases = b_col_aliases or ['min_w_smape_b', 'b']

    farm_df = safe_read_csv(farm_info_csv_path)
    farm_id_col = resolve_existing_column(farm_df.columns, farm_id_aliases, '风电场ID')
    farm_province_col = resolve_existing_column(farm_df.columns, province_aliases, '省简称')

    farm_id_text = normalize_text(farm_id)
    if farm_id_text is None:
        raise ValueError('farm_id 不能为空')

    farm_id_series = farm_df[farm_id_col].map(normalize_text)
    farm_rows = farm_df.loc[farm_id_series == farm_id_text]
    if farm_rows.empty:
        raise ValueError(f"在风电场信息 CSV 中未找到 farm_id={farm_id}")
    if len(farm_rows) > 1:
        raise ValueError(f"在风电场信息 CSV 中 farm_id={farm_id} 对应多条记录，无法唯一确定省简称")

    province_short = normalize_text(farm_rows.iloc[0][farm_province_col])
    if province_short is None:
        raise ValueError(f"farm_id={farm_id} 对应的省简称为空")

    kb_df = safe_read_csv(kb_csv_path)
    kb_province_col = resolve_existing_column(kb_df.columns, kb_province_aliases, '省简称')
    period_col = resolve_existing_column(kb_df.columns, period_aliases, 'period')
    k_col = resolve_existing_column(kb_df.columns, k_aliases, 'k')
    b_col = resolve_existing_column(kb_df.columns, b_aliases, 'b')

    kb_df = kb_df.copy()
    kb_df[kb_province_col] = kb_df[kb_province_col].map(normalize_text)
    kb_df[period_col] = kb_df[period_col].map(lambda x: normalize_text(x).lower() if normalize_text(x) is not None else None)

    province_rows = kb_df.loc[kb_df[kb_province_col] == province_short]
    if province_rows.empty:
        raise ValueError(f"在 k/b CSV 中未找到省简称={province_short} 的记录")

    seasonal_kb: Dict[str, Tuple[float, float]] = {}
    for season in season_order:
        season_rows = province_rows.loc[province_rows[period_col] == season]
        if season_rows.empty:
            raise ValueError(f"省简称={province_short} 在 k/b CSV 中缺少季节 {season} 的记录")
        if len(season_rows) > 1:
            raise ValueError(f"省简称={province_short} 在 k/b CSV 中季节 {season} 存在多条记录")

        row = season_rows.iloc[0]
        try:
            k_value = float(row[k_col])
            b_value = float(row[b_col])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"省简称={province_short} 季节={season} 的 k/b 无法转换为浮点数: "
                f"k={row[k_col]}, b={row[b_col]}"
            ) from exc

        seasonal_kb[season] = (k_value, b_value)

    return seasonal_kb


@dataclass(frozen=True)
class _BatchFarmTask:
    """Single process task payload for batch farm generation."""

    # 以下为风电场标识参数（仅 from_files/batch 层需要）
    farm_csv: Union[str, Path, pd.DataFrame]
    farm_id: Union[int, str]

    # 以下为输入数据路径参数
    era5_dir: str
    era5_varnames: Optional[List[str]]
    time_name: str
    cdmet_dir: Optional[str]

    # 以下为输出控制参数
    select_period: Optional[Union[str, int]]
    output_hourly_gen: bool
    output_monthly_gen: bool
    output_hourly_cf: bool
    output_w_compare: bool
    output_raw_w: bool
    overwrite: bool
    raw_w_varname: str
    output_dir: Optional[Union[str, Path]]
    output_gen_nc_path: Optional[str]
    output_w_compare_nc_path: Optional[str]
    output_raw_w_nc_path: Optional[str]

    # 以下为调整输入参数（from_files 传入）
    farm_adjusted_w_nc_path: Optional[Union[str, Path]]
    farm_adjusted_w_varname: str
    w_grid: Optional[np.ndarray]
    cf_grid: Optional[np.ndarray]
    allow_default_cf_csv_dir: bool
    cf_csv_dir: Optional[Union[str, Path]]
    kb_adjust_info: Optional[Union[Dict, Tuple[float, float], str, Path]]
    gen_adjust_info: Optional[Union[Dict, List[Dict], Tuple[Dict, ...]]]

    # 以下为风场默认参数（优先从表中读取）
    tech_life: int
    end_year: int

    # 以下为模拟参数
    power_order: int
    smooth: bool
    fixed_sigma: Union[bool, float]
    sigma_slope: float
    sigma_intercept: float
    sigma_truncate: float
    w_step: float
    w_max: float
    n_nearest_points: int
    horizon_interp_method: Literal["closest", "idw", "loess"]
    nearest_method: str
    alpha_method: str
    alpha_window: int
    unit_step_hours: float

    # 以下为日志控制参数
    verbose: bool

    @property
    def period_key(self) -> str:
        return "full_year" if self.select_period is None else str(self.select_period)

    @property
    def task_key(self) -> str:
        return f"{self.farm_id}|{self.period_key}"


def unit_farm_output(
        
    # 以下为输入数据路径参数
    era5_dir: str,
    era5_varnames: Optional[List[str]] = ["u10", "v10", "u100", "v100", "t2m", "sp", "d2m", "z"],
    time_name: str = "time",
    cdmet_dir: Optional[str] = None,

    # 以下为输出控制参数
    select_period: Optional[Union[str, int]] = None,
    output_hourly_gen: bool = True,
    output_monthly_gen: bool = True,
    output_hourly_cf: bool = False,
    output_w_compare: bool = True,
    output_raw_w: bool = True,
    overwrite: bool = False,
    raw_w_varname: str = "raw_w",
    output_dir: Optional[Union[str, Path]] = None,
    output_gen_nc_path: Optional[str] = None,
    output_w_compare_nc_path: Optional[str] = None,
    output_raw_w_nc_path: Optional[str] = None,
    
    # 以下为调整输入参数（直接传入调整配置或已调整的风速，跳过前序计算）
    farm_adjusted_w_da: Optional[xr.DataArray] = None,
    w_grid: Optional[np.ndarray] = None,
    cf_grid: Optional[np.ndarray] = None,
    kb_adjust_info: Optional[Union[Dict, Tuple[float, float]]] = None,
    gen_adjust_info: Optional[Union[Dict, List[Dict], Tuple[Dict, ...], str, Path]] = None,
    
    # 以下为风机与风场参数
    start_year: int = 2015,
    farm_id: Optional[Union[int, str]] = None,
    windfarm_lon: float = 114.0,
    windfarm_lat: float = 22.0,
    hub_height: float = 100,
    dem: Optional[float] = None,
    tech_life: int = 25,
    onshore_flag: Optional[bool] = True,
    end_year: int = 2023,
    farm_capacity_mw: Optional[float] = 3,
    cut_in_v: float = 3.0,
    rated_v: float = 12.0,
    cut_out_v: float = 25.0,

    # 以下为模拟参数
    power_order: int = 3,
    smooth: bool = True,
    fixed_sigma: Union[bool, float] = False,
    sigma_slope: float = 0.2,
    sigma_intercept: float = 0.6,
    sigma_truncate: float = 4.0,
    w_step: float = 0.05,
    w_max: float = 40.0,
    n_nearest_points: int = 16,
    horizon_interp_method: Literal["closest", "idw", "loess"] = "loess",
    nearest_method: str = "binary",
    alpha_method: str = "moving_average_hourly",
    alpha_window: int = 3,
    unit_step_hours: float = 1.0,
    
    # 以下为日志控制参数
    verbose: bool = True,
):
    """
    计算单个风电场的风速/发电量时间序列并写入 NetCDF。

    流程概览：
    1. 若直接传入 farm_adjusted_w_da，则跳过 ERA5/CDMet 与风速修正计算。
    2. 否则逐年读取 ERA5（可选读取 CDMet）并拼接，计算 10m/100m 风速、风切变指数、轮毂高度风速与空气密度修正。
     3. 若 output_w_compare=True，输出风速对比数据：
       - era5_w: 基于 ERA5 的轮毂高度修正风速
       - era5_cdmet_w: ERA5 10m 先做 CDMet 日尺度校正后推算的轮毂高度修正风速
     4. 当 output_hourly_gen/output_monthly_gen/output_hourly_cf 任一为 True 时，进行 k-b 调整与发电量计算并输出 gen_ds。
    5. 两个输出可以同时启用，相互独立。

    Args:
        era5_dir: ERA5 数据根目录。
        start_year: 起始年份（含）。
        windfarm_lon: 风场经度。
        windfarm_lat: 风场纬度。
        output_dir: 当未显式提供 output_gen_nc_path/output_w_compare_nc_path 时，
            自动生成文件将保存到该目录下。
        output_gen_nc_path: 发电量输出 NetCDF 文件路径。
        output_w_compare_nc_path: 风速对比输出 NetCDF 文件路径。
        output_raw_w_nc_path: 原始修正风速输出 NetCDF 文件路径。
        cdmet_dir: CDMet 数据目录（年份落入 2000-2020 时尝试读取）。
        hub_height: 轮毂高度（m）。
        dem: 场址高程（m, AMSL）。
        tech_life: 技术寿命年限，用于限制终止年份。
        end_year: 计算终止年份（含）。
        n_nearest_points: 空间插值使用的近邻点数。
        horizon_interp_method: 水平插值方法，支持 closest/idw/loess。
        era5_varnames: 读取 ERA5 时要求的变量名列表。
        time_name: 时间维度名称。
        nearest_method: 最近邻搜索方法。
        alpha_method: 风切变指数计算方法。
        alpha_window: moving_average_hourly 模式下的滑窗长度。
        select_period: 时间筛选条件（None/季节字符串/月整数）。不为 None 时，会在每次读取 nc 后进行筛选。
        kb_adjust_info: 轮毂风速的 k-b 调整配置。
            可传 dict（分时段配置）或 (k, b) tuple（全时段统一配置）。
        gen_adjust_info: 发电量分段乘法调整配置。
            支持单个 dict，或由多个 dict 组成的 list/tuple（按顺序依次应用）。
        output_hourly_cf: 是否在输出中额外包含容量因子变量 `cf`（并应用 gen_adjust_info）。
        farm_adjusted_w_da: 可直接输入的空气密度修正后轮毂风速（DataArray）。
            传入后会跳过前序气象计算。
        farm_capacity_mw: 风场总装机容量（MW）。
        w_grid: 自定义功率曲线风速网格。
        cf_grid: 自定义功率曲线容量因子网格。
        unit_step_hours: 时间步长对应小时数。
        cut_in_v/rated_v/cut_out_v/...: 自动生成功率曲线参数。
        output_hourly_gen: 是否在输出中包含小时发电量变量 `gen`。
        output_monthly_gen: 是否在输出中包含月度发电量变量 `monthly_gen`。
        output_w_compare: 是否输出风速对比文件。
            当 farm_adjusted_w_da 不为 None 时，该选项会被强制视为 False。
        output_raw_w: 是否单独输出原始修正风速文件。
            当 farm_adjusted_w_da 不为 None 时，该选项会被强制视为 False。
        overwrite: 当目标输出文件已存在时，是否覆盖并重新计算。
            - True: 允许覆盖（保持既有行为）；
            - False: 若全部目标输出都已存在则直接跳过；若仅部分存在则报错，避免部分覆盖。
        verbose: 是否输出详细日志（步骤说明、分区耗时、DataArray 前10个值）。
        raw_w_varname: 原始修正风速输出时使用的变量名。

    Returns:
        Optional[xr.Dataset]:
            - 当启用发电量输出（hourly/monthly/cf）时，返回写出的发电量 Dataset。
            - 当仅输出风速对比或原始风速时，返回 None。
            - 当 overwrite=False 且目标输出已完整存在时，若存在发电量文件则返回其加载结果，否则返回 None。
    """
  
    total_start = perf_counter()
    section_starts = {}

    def log_info(msg: str, *args) -> None:
        if verbose:
            logger.info(msg, *args)

    def log_warning(msg: str, *args) -> None:
        if verbose:
            logger.warning(msg, *args)

    def start_section(name: str) -> None:
        section_starts[name] = perf_counter()
        log_info("[START] %s", name)

    def end_section(name: str) -> None:
        start_t = section_starts.get(name)
        if start_t is None:
            log_info("[END] %s", name)
            return
        log_info("[END] %s | elapsed: %.2fs", name, perf_counter() - start_t)

    def log_da_head10(name: str, da: xr.DataArray, verbose=verbose) -> None:
        if not verbose:
            return

        values = np.asarray(da.values).reshape(-1)
        preview = values[:10]
        nan_count = int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.number) else 0
        finite_values = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.number) else np.array([])
        finite_min = float(finite_values.min()) if finite_values.size > 0 else None
        finite_max = float(finite_values.max()) if finite_values.size > 0 else None

        time_info = "no-time-dim"
        if time_name in da.coords:
            try:
                time_values = da.coords[time_name].values
                if len(time_values) > 0:
                    time_info = f"{time_values[0]} -> {time_values[-1]} (n={len(time_values)})"
                else:
                    time_info = "empty-time"
            except Exception:
                time_info = "time-unavailable"

        logger.info(
            "[DA] %s | shape=%s | time=%s | nan=%s/%s | finite_min=%s | finite_max=%s | first10=%s",
            name,
            da.shape,
            time_info,
            nan_count,
            values.size,
            finite_min,
            finite_max,
            preview.tolist(),
        )

    def log_path_decision(label: str, path: Optional[str], verbose=verbose) -> None:
        if not verbose:
            return
        logger.info("[PATH] %s -> %s", label, path)

    log_info(
        "[START] unit_farm_output | years=%s-%s, farm=(%.4f, %.4f), interp=%s",
        start_year,
        end_year,
        windfarm_lat,
        windfarm_lon,
        horizon_interp_method,
    )

    start_section("prepare_output_and_year_range")
    should_output_gen_ds = output_hourly_gen or output_monthly_gen or output_hourly_cf
    direct_w_input = farm_adjusted_w_da is not None
    # BUG-1：记录海拔来源（inventory / era5_orography_fallback / direct_w_input 未用密度链）
    dem_source: Optional[str] = None
    if direct_w_input and (output_w_compare or output_raw_w):
        log_warning(
            "farm_adjusted_w_da 已提供，强制关闭 output_w_compare/output_raw_w（两者将按 False 处理）"
        )
        output_w_compare = False
        output_raw_w = False

    period_token = "full_year" if select_period is None else format_for_filename(select_period)
    include_kb_in_name = False
    k_token = "1"
    b_token = "0"

    if kb_adjust_info is None:
        include_kb_in_name = True
    elif isinstance(kb_adjust_info, tuple):
        include_kb_in_name = True
        k_token = format_for_filename(float(kb_adjust_info[0]))
        b_token = format_for_filename(float(kb_adjust_info[1]))
    elif isinstance(kb_adjust_info, dict):
        if len(kb_adjust_info) == 1:
            only_value = next(iter(kb_adjust_info.values()))
            if not isinstance(only_value, (list, tuple)) or len(only_value) != 2:
                raise ValueError("kb_adjust_info 为单项 dict 时，其值必须是长度为2的(k, b)序列")
            include_kb_in_name = True
            k_token = format_for_filename(float(only_value[0]))
            b_token = format_for_filename(float(only_value[1]))

    output_base_dir = Path(output_dir) if output_dir is not None else Path('.')
    farm_token = format_for_filename(farm_id) if farm_id is not None else "unknown"

    if output_gen_nc_path is None and should_output_gen_ds:
        if include_kb_in_name:
            output_gen_nc_path = str(
                output_base_dir / f"gen/farm_{farm_token}_period_{period_token}_k_{k_token}_b_{b_token}_gen.nc"
            )
        else:
            output_gen_nc_path = str(output_base_dir / f"gen/farm_{farm_token}_period_{period_token}_gen.nc")

    if output_w_compare_nc_path is None and output_w_compare:
        if include_kb_in_name:
            output_w_compare_nc_path = str(
                output_base_dir / f"w_compare/farm_{farm_token}_period_{period_token}_k_{k_token}_b_{b_token}_w_compare.nc"
            )
        else:
            output_w_compare_nc_path = str(output_base_dir / f"w_compare/farm_{farm_token}_period_{period_token}_w_compare.nc")

    if output_raw_w_nc_path is None and output_raw_w:
        if include_kb_in_name:
            output_raw_w_nc_path = str(
                output_base_dir / f"raw_w/farm_{farm_token}_period_{period_token}_k_{k_token}_b_{b_token}_raw_w.nc"
            )
        else:
            output_raw_w_nc_path = str(output_base_dir / f"raw_w/farm_{farm_token}_period_{period_token}_raw_w.nc")

    log_info(
        "[CONFIG] select_period=%s | output_gen=%s | output_w_compare=%s | output_hourly_cf=%s | output_raw_w=%s",
        select_period,
        should_output_gen_ds,
        output_w_compare,
        output_hourly_cf,
        output_raw_w,
    )
    log_info(
        "[CONFIG] kb_adjust_info=%s | gen_adjust_info=%s | output_dir=%s",
        kb_adjust_info,
        gen_adjust_info,
        output_dir,
    )
    log_path_decision("output_gen_nc_path", output_gen_nc_path)
    log_path_decision("output_w_compare_nc_path", output_w_compare_nc_path)
    log_path_decision("output_raw_w_nc_path", output_raw_w_nc_path)

    if should_output_gen_ds and output_gen_nc_path is None:
        raise ValueError("当 output_hourly_gen/output_monthly_gen/output_hourly_cf 任一为 True 时，必须提供 output_gen_nc_path")
    if output_w_compare and output_w_compare_nc_path is None:
        raise ValueError("output_w_compare=True 时，必须提供 output_w_compare_nc_path")
    if output_raw_w and output_raw_w_nc_path is None:
        raise ValueError("output_raw_w=True 时，必须提供 output_raw_w_nc_path")

    if should_output_gen_ds and output_gen_nc_path:
        Path(output_gen_nc_path).parent.mkdir(parents=True, exist_ok=True)
    if output_w_compare and output_w_compare_nc_path:
        Path(output_w_compare_nc_path).parent.mkdir(parents=True, exist_ok=True)
    if output_raw_w and output_raw_w_nc_path:
        Path(output_raw_w_nc_path).parent.mkdir(parents=True, exist_ok=True)

    target_outputs: Dict[str, str] = {}
    if should_output_gen_ds and output_gen_nc_path is not None:
        target_outputs["gen"] = output_gen_nc_path
    if output_w_compare and output_w_compare_nc_path is not None:
        target_outputs["w_compare"] = output_w_compare_nc_path
    if output_raw_w and output_raw_w_nc_path is not None:
        target_outputs["raw_w"] = output_raw_w_nc_path

    existing_outputs = {
        key: path for key, path in target_outputs.items() if Path(path).exists()
    }

    if not overwrite and len(target_outputs) > 0:
        if len(existing_outputs) == len(target_outputs):
            log_info(
                "Skip computation: all target outputs already exist and overwrite=False | outputs=%s",
                existing_outputs,
            )
            if should_output_gen_ds and output_gen_nc_path is not None:
                with xr.open_dataset(output_gen_nc_path) as existing_gen_ds:
                    return existing_gen_ds.load()
            return None

        if len(existing_outputs) > 0:
            raise FileExistsError(
                "overwrite=False 且检测到部分目标输出已存在，已中止以避免覆盖。"
                f" 已存在: {existing_outputs}; 目标: {target_outputs}"
            )

    if direct_w_input and time_name not in farm_adjusted_w_da.dims:
        raise ValueError(f"直接输入的 farm_adjusted_w_da 必须包含时间维度: {time_name}")

    end_year = min(start_year + tech_life, end_year)
    log_info("Adjusted year range to %s-%s (tech_life=%s)", start_year, end_year, tech_life)
    end_section("prepare_output_and_year_range")

    cdmet_dss = []
    if not direct_w_input:
        start_section("load_and_concat_era5_cdmet")
        era5_dss = []

        def _load_one_year(year: int):
            year_start = perf_counter()
            log_info("Processing year=%s: loading ERA5 subdatasets", year)

            wind_path = f"{era5_dir}/wind/era5_wind_{year}.nc"
            solar_path = f"{era5_dir}/solar/era5_solar_{year}.nc"
            building_path = f"{era5_dir}/building/era5_building_{year}.nc"
            z_path = f"{era5_dir}/geopotential.nc"

            era5_start = perf_counter()
            single_era5_ds = load_and_merge_subdatasets(
                nc_paths=[wind_path, solar_path, building_path, z_path],
                target_point=(windfarm_lat, windfarm_lon),
                n_points=n_nearest_points,
                nearest_method=nearest_method,
                required_varnames=era5_varnames,
                time_name_candidates=("time", "valid_time", "date"),
                lat_name_candidates=("lat", "latitude", "y"),
                lon_name_candidates=("lon", "longitude", "x"),
            )
            single_era5_ds = filter_xr_by_period(single_era5_ds, select_period, time_name=time_name)
            era5_elapsed = perf_counter() - era5_start
           
            cdmet_10m_w_da = None
            cdmet_error = None
            cdmet_elapsed = 0.0
            if cdmet_dir is not None and 2000 <= year <= 2020:
                log_info("Processing year=%s: loading CDMet", year)
                try:
                    cdmet_start = perf_counter()
                    cdmet_path = f"{cdmet_dir}/CDMet_win_{year}.nc"
                    single_cdmet_ds = load_and_merge_subdatasets(
                        nc_paths=[cdmet_path],
                        target_point=(windfarm_lat, windfarm_lon),
                        n_points=100,  # CDMet没有海上数据，原文件中海上格点值均为空值，故取大一点
                        nearest_method='binary',
                        required_varnames=None,
                        time_name_candidates=("time", "valid_time", "date"),
                        lat_name_candidates=("lat", "latitude", "y"),
                        lon_name_candidates=("lon", "longitude", "x"),
                    )
                    cdmet_10m_w_da = cdmet_10m_w(single_cdmet_ds, year, "win", time_name)
                    cdmet_10m_w_da = filter_xr_by_period(cdmet_10m_w_da, select_period, time_name=time_name)
                    cdmet_elapsed = perf_counter() - cdmet_start
                except Exception as exc:
                    cdmet_error = str(exc)
                    cdmet_elapsed = perf_counter() - cdmet_start
          
            return {
                "year": year,
                "era5_ds": single_era5_ds,
                "cdmet_da": cdmet_10m_w_da,
                "cdmet_error": cdmet_error,
                "era5_elapsed": era5_elapsed,
                "cdmet_elapsed": cdmet_elapsed,
                "year_elapsed": perf_counter() - year_start,
            }
        
        # 逐年读取并抽取风场周边网格，再在时间维上拼接。
        years = list(range(start_year, end_year + 1))
        load_all_start = perf_counter()
        year_results = {}
        for year in years:
            year_results[year] = _load_one_year(year)
         
        load_all_elapsed = perf_counter() - load_all_start
        log_info(
            "[TIMER] yearly loading finished | mode=sequential | years=%s | total_elapsed=%.2fs",
            len(years),
            load_all_elapsed,
        )
  
        for year in years:
           
            result = year_results[year]
            single_era5_ds = result["era5_ds"]
            cdmet_10m_w_da = result["cdmet_da"]
            cdmet_error = result["cdmet_error"]

            log_info(
                "[TIMER] year=%s | era5=%.2fs | cdmet=%.2fs | total=%.2fs",
                year,
                result["era5_elapsed"],
                result["cdmet_elapsed"],
                result["year_elapsed"],
            )

            era5_dss.append(single_era5_ds)

            if cdmet_10m_w_da is not None:
                log_info("Processing year=%s: CDMet loaded", year)
                log_da_head10(f"cdmet_10m_w_da_{year}", cdmet_10m_w_da)
                cdmet_dss.append(cdmet_10m_w_da)
            elif cdmet_error is not None:
                log_warning("Processing year=%s: CDMet load failed (%s)", year, cdmet_error)
      
        whole_era5_ds = concat_das(era5_dss, time_name=time_name)
    
        if cdmet_dss:
            whole_cdmet_ds = concat_das(cdmet_dss, time_name=time_name)
            cdmet_years = whole_cdmet_ds[time_name].dt.year
            cdmet_start_year = int(cdmet_years.min().item())
            cdmet_end_year = int(cdmet_years.max().item())
            log_info(
                "CDMet available for %s year(s), year range=%s-%s",
                len(cdmet_dss),
                cdmet_start_year,
                cdmet_end_year,
            )
        else:
            whole_cdmet_ds = None
            log_info("CDMet unavailable in selected years; continue with ERA5 only")
        end_section("load_and_concat_era5_cdmet")
        
        start_section("compute_farm_wind_and_density")
        era5_10m_w_da = era5_10m_w(whole_era5_ds, u10_name="u10", v10_name="v10")
        log_da_head10("era5_10m_w_da", era5_10m_w_da)

        farm_10m_w_da = horizontal_interp(
            era5_10m_w_da,
            (windfarm_lat, windfarm_lon),
            horizon_interp_method,
            "lon",
            "lat",
            time_name,
            verbose=verbose,
        )
        log_da_head10("farm_10m_w_da", farm_10m_w_da)
        
        era5_100m_w_da = era5_100m_w(whole_era5_ds, u100_name="u100", v100_name="v100")
        log_da_head10("era5_100m_w_da", era5_100m_w_da)

        farm_100m_w_da = horizontal_interp(
            era5_100m_w_da,
            (windfarm_lat, windfarm_lon),
            horizon_interp_method,
            "lon",
            "lat",
            time_name,
            verbose=verbose,
        )
        log_da_head10("farm_100m_w_da", farm_100m_w_da)

        farm_alpha_da = era5_alpha(farm_10m_w_da, farm_100m_w_da, alpha_method, alpha_window)
        log_da_head10("farm_alpha_da", farm_alpha_da)

        farm_hub_height_w_da = vertical_extrap_w(
            farm_10m_w_da,
            farm_100m_w_da,
            farm_alpha_da,
            hub_height,
        )
        log_da_head10("farm_hub_height_w_da", farm_hub_height_w_da)

        # BUG-1：密度目标高度必须是「场址海拔 + 轮毂高」。
        # dem=None（盘点表无海拔列，且调用方显式允许回退）时，用 ERA5 格点地形位势高度
        # H1 当"地面"，向上外推 hub —— 这样 H2−H1 ≈ hub 恰好正确，而不是把轮毂高度当海拔。
        if dem is None:
            dem_source = "era5_orography_fallback"
            era5_rou_da = era5_rou(
                whole_era5_ds,
                None,
                t2m_name="t2m",
                d2m_name="d2m",
                sp_name="sp",
                z_name="z",
                z2_above_orography=float(hub_height),
            )
        else:
            dem_source = "inventory"
            z2 = hub_height + float(dem)
            era5_rou_da = era5_rou(
                whole_era5_ds,
                z2,
                t2m_name="t2m",
                d2m_name="d2m",
                sp_name="sp",
                z_name="z",
            )
        log_da_head10("era5_rou_da", era5_rou_da)

        farm_hub_rou_da = horizontal_interp(
            era5_rou_da,
            (windfarm_lat, windfarm_lon),
            horizon_interp_method,
            "lon",
            "lat",
            time_name,
            verbose=verbose,
        )
        log_da_head10("farm_hub_rou_da", farm_hub_rou_da)

        farm_adjusted_w_da = rou_adjustment_w(farm_hub_height_w_da, farm_hub_rou_da)
        log_da_head10("farm_adjusted_w_da", farm_adjusted_w_da)
        end_section("compute_farm_wind_and_density")
    else:
        whole_cdmet_ds = None
        farm_adjusted_w_da = filter_xr_by_period(farm_adjusted_w_da, select_period, time_name=time_name)
        log_info("Input farm_adjusted_w_da provided; skip ERA5/CDMet loading and wind-density computation")
        log_da_head10("farm_adjusted_w_da_input", farm_adjusted_w_da)

    if output_w_compare:
        start_section("export_w_compare")
        if whole_cdmet_ds is not None:
            farm_adjusted_w_compare_da = align_da_to_reference_year_range(
                farm_adjusted_w_da,
                whole_cdmet_ds,
                time_name=time_name,
            )
            farm_10m_w_compare_da = align_da_to_reference_year_range(
                farm_10m_w_da,
                whole_cdmet_ds,
                time_name=time_name,
            )
            farm_alpha_compare_da = align_da_to_reference_year_range(
                farm_alpha_da,
                whole_cdmet_ds,
                time_name=time_name,
            )
            farm_hub_rou_compare_da = align_da_to_reference_year_range(
                farm_hub_rou_da,
                whole_cdmet_ds,
                time_name=time_name,
            )

            if farm_adjusted_w_compare_da.sizes[time_name] == 0:
                raise ValueError("按 CDMet 年份范围裁剪后，ERA5 序列为空，无法输出风速对比结果")
           
            cdmet_farm_10m_w_da = horizontal_interp(
                whole_cdmet_ds,
                (windfarm_lat, windfarm_lon),
                horizon_interp_method,
                "lon",
                "lat",
                time_name,
                verbose=verbose,
            )
            
            log_da_head10("cdmet_farm_10m_w_da", cdmet_farm_10m_w_da)

            era5_farm_10m_w_cdmet_da = adjust_hourly_with_daily(
                farm_10m_w_compare_da,
                cdmet_farm_10m_w_da,
                time_name,
                "1D",
            )
            log_da_head10("era5_farm_10m_w_cdmet_da", era5_farm_10m_w_cdmet_da)

            farm_hub_height_w_cdmet_da = vertical_extrap_w(
                era5_farm_10m_w_cdmet_da,
                None,
                farm_alpha_compare_da,
                hub_height,
            )
            log_da_head10("farm_hub_height_w_cdmet_da", farm_hub_height_w_cdmet_da)

            farm_adjusted_w_cdmet_da = rou_adjustment_w(
                farm_hub_height_w_cdmet_da,
                farm_hub_rou_compare_da,
            )
            log_da_head10("farm_adjusted_w_cdmet_da", farm_adjusted_w_cdmet_da)
        elif not direct_w_input:
            farm_adjusted_w_compare_da = farm_adjusted_w_da
            farm_adjusted_w_cdmet_da = farm_adjusted_w_da
            log_da_head10("farm_adjusted_w_cdmet_da", farm_adjusted_w_cdmet_da)
        else:
            # 防御性分支：当 direct_w_input=True 时，前面已强制 output_w_compare=False，
            # 因此不应进入该路径。
            raise RuntimeError(
                "不应在 farm_adjusted_w_da 直传模式下进入 output_w_compare 分支，请检查上游参数处理"
            )

        output_ds = xr.Dataset(
            {
                "farm_era5_w": farm_adjusted_w_compare_da,
                "farm_era5_cdmet_w": farm_adjusted_w_cdmet_da,
            }
        )
        output_ds.attrs["cdmet_data_available"] = int(len(cdmet_dss) > 0)
        if len(cdmet_dss) == 0:
            output_ds.attrs["note"] = (
                "CDMet data not available for specified year range (2000-2020); both variables use ERA5"
            )
        output_ds = sanitize_netcdf_attrs(output_ds, "output_ds")
        output_ds.to_netcdf(output_w_compare_nc_path)
        log_info("CDMet comparison file exported: %s", output_w_compare_nc_path)
        for var_name in output_ds.data_vars:
            log_da_head10(f"output_ds[{var_name}]", output_ds[var_name])
        end_section("export_w_compare")

    if output_raw_w:
        start_section("export_raw_w")
        raw_w_da = farm_adjusted_w_da.rename(raw_w_varname)
        log_da_head10("raw_w_da", raw_w_da)

        raw_w_ds = xr.Dataset({raw_w_varname: raw_w_da})
        raw_w_ds.attrs["hub_height"] = hub_height
        raw_w_ds.attrs["dem"] = float(dem) if dem is not None else float("nan")
        raw_w_ds.attrs["dem_source"] = dem_source or "not_used_direct_w_input"
        raw_w_ds.attrs["start_year"] = start_year
        raw_w_ds.attrs["end_year"] = end_year
        raw_w_ds.attrs["cdmet_data_available"] = int(len(cdmet_dss) > 0)
        if onshore_flag is not None:
            raw_w_ds.attrs["onshore_flag"] = int(onshore_flag)
        raw_w_ds = sanitize_netcdf_attrs(raw_w_ds, "raw_w_ds")
        raw_w_ds.to_netcdf(output_raw_w_nc_path)
        log_info("Raw wind file exported: %s", output_raw_w_nc_path)
        log_da_head10(f"raw_w_ds[{raw_w_varname}]", raw_w_ds[raw_w_varname])
        end_section("export_raw_w")

    def _normalize_gen_adjust_sequence(
        adjust_info: Optional[Union[Dict, List[Dict], Tuple[Dict, ...], str, Path]]
    ) -> List[Dict]:
        """Normalize generation adjust input into an ordered list of dicts."""
        if adjust_info is None:
            return []

        # Keep backward compatibility: if caller passes a raw JSON/dict string, parse here.
        if isinstance(adjust_info, (str, Path)):
            text = str(adjust_info)
            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception as exc:
                    raise ValueError(
                        "gen_adjust_info 字符串必须是可解析的 JSON/字典/列表文本"
                    ) from exc
            adjust_info = parsed

        if isinstance(adjust_info, dict):
            return [adjust_info]

        if isinstance(adjust_info, (list, tuple)):
            normalized: List[Dict] = []
            for idx, item in enumerate(adjust_info):
                if item is None:
                    continue
                if not isinstance(item, dict):
                    raise ValueError(
                        f"gen_adjust_info 第 {idx + 1} 项不是 dict，实际类型: {type(item).__name__}"
                    )
                normalized.append(item)
            return normalized

        raise ValueError(
            "gen_adjust_info 必须为 dict、list[dict]、tuple[dict, ...]、JSON 文本或 None"
        )

    gen_adjust_sequence = _normalize_gen_adjust_sequence(gen_adjust_info)

    gen_ds = None
    if should_output_gen_ds:
        start_section("compute_and_export_generation")
        kb_adjust_w_da = hub_w_kb_adjust(farm_adjusted_w_da, kb_adjust_info, time_name)
        log_da_head10("kb_adjust_w_da", kb_adjust_w_da)

        cf_da = farm_cf(
            w_hub=kb_adjust_w_da,
            w_grid=w_grid,
            cf_grid=cf_grid,
            time_name=time_name,
            cut_in_v=cut_in_v,
            rated_v=rated_v,
            cut_out_v=cut_out_v,
            v_slope=1,
            v_intercept=0,
            power_order=power_order,
            smooth=smooth,
            fixed_sigma=fixed_sigma,
            sigma_slope=sigma_slope,
            sigma_intercept=sigma_intercept,
            sigma_truncate=sigma_truncate,
            w_step=w_step,
            w_max=w_max,
        )
        log_da_head10("cf_da", cf_da)

        needs_gen = output_hourly_gen or output_monthly_gen
        gen_da = None
        if needs_gen:
            if farm_capacity_mw is None:
                raise ValueError("输出小时或月度发电量时，farm_capacity_mw 不能为空")

            gen_da = farm_generation(
                cf_da=cf_da,
                farm_capacity_mw=farm_capacity_mw,
                time_name=time_name,
                unit_step_hours=unit_step_hours,
            )
            log_da_head10("gen_da_raw", gen_da)

            gen_da_adjusted = gen_da
            for idx, one_adjust in enumerate(gen_adjust_sequence, start=1):
                log_info(
                    "[ADJUST] apply gen adjustment %s/%s",
                    idx,
                    len(gen_adjust_sequence),
                )
                gen_da_adjusted = adjust_da_by_time(gen_da_adjusted, one_adjust, time_name)
            gen_da = gen_da_adjusted
            log_da_head10("gen_da_adjusted", gen_da)

        gen_ds = xr.Dataset()
        if output_hourly_gen and gen_da is not None:
            gen_ds["gen"] = gen_da

        if output_hourly_cf:
            cf_da_adjusted = cf_da
            for one_adjust in gen_adjust_sequence:
                cf_da_adjusted = adjust_da_by_time(cf_da_adjusted, one_adjust, time_name)
            cf_da_adjusted = cf_da_adjusted.rename("cf")
            log_da_head10("cf_da_adjusted", cf_da_adjusted)
            gen_ds["cf"] = cf_da_adjusted

        if output_monthly_gen and gen_da is not None:
            monthly_gen_da = hourly_to_monthly(
                hourly_da=gen_da,
                time_name=time_name,
                agg_method="sum",
            )
            if time_name in monthly_gen_da.dims:
                monthly_gen_da = monthly_gen_da.rename({time_name: "date"})
            log_da_head10("monthly_gen_da", monthly_gen_da)
            gen_ds["monthly_gen"] = monthly_gen_da

        gen_ds.attrs["hub_height"] = hub_height
        gen_ds.attrs["dem"] = float(dem) if dem is not None else float("nan")
        gen_ds.attrs["dem_source"] = dem_source or "not_used_direct_w_input"
        gen_ds.attrs["start_year"] = start_year
        gen_ds.attrs["end_year"] = end_year
        gen_ds.attrs["farm_capacity_mw"] = farm_capacity_mw
        gen_ds.attrs["cdmet_data_available"] = int(len(cdmet_dss) > 0)
        if onshore_flag is not None:
            gen_ds.attrs["onshore_flag"] = int(onshore_flag)
        if len(cdmet_dss) == 0 and not direct_w_input:
            gen_ds.attrs["note"] = (
                "CDMet data not available for specified year range (2000-2020); wind speeds not adjusted"
            )
        gen_ds = sanitize_netcdf_attrs(gen_ds, "gen_ds")
        gen_ds.to_netcdf(output_gen_nc_path)
        log_info("Generation file exported: %s", output_gen_nc_path)
        for var_name in gen_ds.data_vars:
            log_da_head10(f"gen_ds[{var_name}]", gen_ds[var_name])
        end_section("compute_and_export_generation")

    log_info("[END] unit_farm_output | total elapsed: %.2fs", perf_counter() - total_start)

    return gen_ds


def unit_farm_output_from_files(

    # 以下为风电场标识参数
    farm_csv: Union[str, Path, pd.DataFrame],
    farm_id: Union[int, str],

    # 以下为输入数据路径参数
    era5_dir: str,
    era5_varnames: Optional[List[str]] = ["u10", "v10", "u100", "v100", "t2m", "sp", "d2m", "z"],
    time_name: str = "time",
    cdmet_dir: Optional[str] = None,

    # 以下为输出控制参数
    select_period: Optional[Union[str, int]] = None,
    output_hourly_gen: bool = True,
    output_monthly_gen: bool = True,
    output_hourly_cf: bool = False,
    output_w_compare: bool = True,
    output_raw_w: bool = True,
    overwrite: bool = True,
    raw_w_varname: str = "raw_w",
    output_dir: Optional[Union[str, Path]] = None,
    output_gen_nc_path: Optional[str] = None,
    output_w_compare_nc_path: Optional[str] = None,
    output_raw_w_nc_path: Optional[str] = None,

    # 以下为调整输入参数（直接传入调整配置或已调整的风速，跳过前序计算）
    farm_adjusted_w_nc_path: Optional[Union[str, Path]] = None,
    farm_adjusted_w_varname: str = "raw_w",
    w_grid: Optional[np.ndarray] = None,
    cf_grid: Optional[np.ndarray] = None,
    allow_default_cf_csv_dir: bool = True,
    cf_csv_dir: Optional[Union[str, Path]] = None,
    kb_adjust_info: Optional[Union[Dict, Tuple[float, float], str, Path]] = None,
    gen_adjust_info: Optional[Union[Dict, List[Dict], Tuple[Dict, ...]]] = None,
    
    # 以下为风场默认参数（优先从表中读取）
    tech_life: int = 25,
    end_year: int = 2023,

    # 以下为模拟参数
    power_order: int = 3,
    smooth: bool = True,
    fixed_sigma: Union[bool, float] = False,
    sigma_slope: float = 0.2,
    sigma_intercept: float = 0.6,
    sigma_truncate: float = 4.0,
    w_step: float = 0.05,
    w_max: float = 40.0,
    n_nearest_points: int = 16,
    horizon_interp_method: Literal["closest", "idw", "loess"] = "loess",
    nearest_method: str = "binary",
    alpha_method: str = "moving_average_hourly",
    alpha_window: int = 3,
    unit_step_hours: float = 1.0,

    # 以下为日志控制参数
    verbose: bool = True,

    # 以下为参数校验控制
    strict_curve_params: bool = False,
    allow_era5_orography_fallback: bool = False,
):
    """
    从 DataFrame 中按 farm_id 读取 10 个参数（含陆海标识），其他参数直接转发给 unit_farm_output。

    表中期望的列名（**首选精确匹配**；匹配前会做列名规范化，故下列"旧别名"写法同样能命中）：
        - onshore_flag: onshore_flag, 陆海1-2, 陆海, land_sea_flag
        - windfarm_lon: 经度, lon, longitude, lng
        - windfarm_lat: 纬度, lat, latitude
        - hub_height: Hub heigh, Hub height, hub_height, 轮毂高度, 轮毂高度m
        - dem: Elevation, elevation, 海拔, 高程
        - start_year: 年, year, start_year, 投产年, 并网年
        - cut_in_v: Cut-in wind speed（真实列名）, Cut-in w, Cut-in wind, cut_in_v
        - rated_v: Rated wind speed（真实列名）, Rated wind, rated_v
        - cut_out_v: Cut-out wind speed（真实列名）, Cut-out w, cut_out_v
        - farm_capacity_mw: 风电场装机MW, 风电场装机 MW, farm_capacity_mw, capacity_mw, 装机容量MW

    Args:
        farm_csv: 风电场表 (DataFrame) 或 CSV 文件路径 (str/Path)。
        farm_id: 风电场 ID（从 "风电场ID" 列中匹配）。
        strict_curve_params: True 时，若 cut_in_v/rated_v/cut_out_v 在表中取不到，
            **直接抛 ValueError**（而不是回退默认值）；默认 False 时仍会回退，但**必定发
            UserWarning**（含缺哪些参数、回退到什么默认值、可用列名），不再静默。
        allow_default_cf_csv_dir: 是否在**未给 cf_csv_dir** 时回填默认功率曲线目录
            （`config.curve_dir()` = `WINDPAPER_CURVE_DIR`，未设时为 `<DATA>/farm_power_curves`）。
            注意：它**不决定**是否使用逐场 CF csv——那个判据只有一个：`cf_csv_dir is not None`。
            新代码请直接传 cf_csv_dir。
        cf_csv_dir: 功率曲线 CSV 目录。**只要非 None 就会使用**：读取其下
            `farm_{farm_id}.csv` 的前两列并覆盖 w_grid/cf_grid（这就是"是否使用逐场 CF csv"
            的唯一判据）。为 None 时才回落到"用盘点表机型参数现场生成曲线"。
        kb_adjust_info: 仅支持三类输入：
            1. dict；
            2. (k, b) tuple；
            3. CSV 文件路径（会自动解析为季节性字典）。
        gen_adjust_info: 可直接传字典，或由多个字典组成的 list/tuple；
            也可传 JSON 文件路径/JSON 字符串（会自动读取为字典或字典列表）。
        farm_adjusted_w_nc_path: 已计算好的风速 DataArray 所在 NetCDF 路径。若不为 None，则读取并跳过前序风速计算。
        farm_adjusted_w_varname: 在 farm_adjusted_w_nc_path 中读取的变量名。
        select_period: 时间筛选条件（None/季节字符串/月整数）。不为 None 时会在读取 nc 后筛选。
        output_w_compare: 是否输出风速对比文件（为 True 时需提供 output_w_compare_nc_path）。
        output_dir: 当未显式提供 output_gen_nc_path/output_w_compare_nc_path 时，
            自动生成文件将保存到该目录下。
        output_raw_w_nc_path: 原始修正风速输出 NetCDF 文件路径。
        overwrite: 当目标输出文件已存在时，是否覆盖并重新计算。
        raw_w_varname: 原始修正风速输出时使用的变量名。

    """
    # 如果 farm_csv 是文件路径，读取；否则直接使用
    if isinstance(farm_csv, (str, Path)):
        farm_df = safe_read_csv(farm_csv)
    else:
        farm_df = farm_csv

    if farm_df is None or len(farm_df) == 0:
        raise ValueError("farm_df 不能为空")

    # 硬编码的 id 列名
    id_col = "风电场ID"
    
    def _as_scalar(value: Any) -> Any:
        """numpy 标量转 Python 标量"""
        if isinstance(value, np.generic):
            return value.item()
        return value

    # 按 farm_id 选行
    if id_col not in farm_df.columns:
        raise KeyError(f"表中不存在 ID 列: {id_col}")

    try:
        onshore_flag_col = resolve_existing_column(
            farm_df.columns,
            ["onshore_flag", "陆海1-2", "陆海", "land_sea_flag"],
            "onshore_flag",
        )
    except KeyError as exc:
        raise KeyError(
            "表中缺少陆海标识列，必须包含 onshore_flag、陆海1-2、陆海 或 land_sea_flag 之一"
        ) from exc

    matched = farm_df[farm_df[id_col].astype(str) == str(farm_id)]
    if len(matched) == 0:
        raise KeyError(f"未找到 farm_id={farm_id}")
    row = matched.iloc[0]

    def _extract(aliases: List[str], cast=None) -> Any:
        """从行中按别名列表提取值，支持类型转换"""
        for name in aliases:
            if name in farm_df.columns:
                val = _as_scalar(row[name])
                if pd.isna(val):
                    continue
                if cast is not None:
                    try:
                        return cast(val)
                    except Exception:
                        continue
                return val
        return None

    # 从表中读取 10 个参数（含陆海标识）
    lon_df = _extract(["经度", "lon", "longitude", "lng"], cast=float)
    lat_df = _extract(["纬度", "lat", "latitude"], cast=float)
    hub_df = _extract(["Hub heigh", "Hub height", "hub_height", "轮毂高度", "轮毂高度m"], cast=float)
    # 海拔不再用 _extract + 静默 0：见下方 resolve_dem_elevation（缺列即报错）
    year_df = _extract(["年", "year", "start_year", "投产年", "并网年"], cast=int)
    # 风机风速参数（cut_in/rated/cut_out）：真实列名优先 + 列名规范化兼容旧别名；
    # 取不到时**不静默**——strict_curve_params=True 抛错，否则发 UserWarning。
    curve_params = resolve_turbine_curve_params(
        row,
        farm_df.columns,
        strict=strict_curve_params,
        context="farm_id=%s" % farm_id,
    )
    cap_df = _extract(["风电场装机MW", "风电场装机 MW", "farm_capacity_mw", "capacity_mw", "装机容量MW"], cast=float)

    try:
        onshore_flag_final = _normalize_inventory_onshore_flag(_as_scalar(row[onshore_flag_col]))
    except ValueError as exc:
        raise ValueError(f"farm_id={farm_id} 的陆海标识无效: {exc}") from exc

    # 必需字段 (无默认值)
    if lon_df is None or lat_df is None or year_df is None:
        raise ValueError("表中必须提供：windfarm_lon、windfarm_lat、start_year")

    # 有默认值的字段 (表中没有则用 unit_farm_output 的默认值)
    hub_height_final = hub_df if hub_df is not None else 100.0
    # BUG-1：海拔缺列/非法 -> 默认 raise（allow_era5_orography_fallback=True 时才 warn+回退）
    dem_final = resolve_dem_elevation(
        row,
        farm_df.columns,
        allow_orography_fallback=allow_era5_orography_fallback,
        context="farm_id=%s" % farm_id,
    )
    cut_in_v_final = curve_params["cut_in_v"]
    rated_v_final = curve_params["rated_v"]
    cut_out_v_final = curve_params["cut_out_v"]
    farm_capacity_mw_final = cap_df

    # ── 是否使用逐场功率曲线 CSV：**唯一判据 = cf_csv_dir 是否为 None** ────────────────
    # `allow_default_cf_csv_dir` **不参与上面的判断**；它只在调用方没给 cf_csv_dir 时
    # 决定是否回填默认功率曲线目录（config.curve_dir()）。新代码请直接传 cf_csv_dir。
    if allow_default_cf_csv_dir and cf_csv_dir is None:
        cf_csv_dir = _default_cf_csv_dir()

    use_farm_cf_csv = cf_csv_dir is not None

    # 若提供功率曲线 CSV，则以其前两列覆盖 w_grid/cf_grid
    if use_farm_cf_csv:
        cf_csv_path = Path(cf_csv_dir) / f"farm_{farm_id}.csv"
        cf_df = safe_read_csv(cf_csv_path)
        if cf_df is None or cf_df.empty:
            raise ValueError(f"cf_csv_path 对应文件为空: {cf_csv_path}")
        if cf_df.shape[1] < 2:
            raise ValueError("cf_csv_path 对应 CSV 至少需要两列（第一列 w_grid，第二列 cf_grid）")

        w_series = pd.to_numeric(cf_df.iloc[:, 0], errors="coerce")
        cf_series = pd.to_numeric(cf_df.iloc[:, 1], errors="coerce")
        valid = ~(w_series.isna() | cf_series.isna())
        if valid.sum() == 0:
            raise ValueError("cf_csv_path 前两列无法解析出有效的 w_grid/cf_grid 数值")

        w_grid = w_series[valid].to_numpy(dtype=float)
        cf_grid = cf_series[valid].to_numpy(dtype=float)

    # kb_adjust_info 仅支持 dict / (k, b) tuple / CSV 文件路径
    if isinstance(kb_adjust_info, (str, Path)):
        if Path(kb_adjust_info).suffix.lower() == ".csv":
            
            try:
                kb_adjust_info = generate_farm_seasonal_kb_dict(
                    farm_id=farm_id,
                    farm_info_csv_path=farm_csv,
                    kb_csv_path=kb_adjust_info,
                )
            except ValueError as exc:
                raise ValueError(f"kb_adjust_info CSV 文件解析失败: {exc}") from exc
        else:
            raise ValueError(
                "kb_adjust_info 仅支持 dict、(k, b) tuple 或 .csv 文件路径；"
                f"收到字符串/路径时必须是 .csv 文件，实际为: {kb_adjust_info}"
            )
    
    # gen_adjust_info 支持 dict / 文件路径 / JSON 字符串
    if isinstance(gen_adjust_info, (str, Path)):
        gen_text = None
        gen_path = Path(gen_adjust_info)

        # 若是现有文件路径，优先按文件读取
        if gen_path.exists() and gen_path.is_file():
            for enc in ("utf-8", "gkb", "latin-1"):
                try:
                    gen_text = gen_path.read_text(encoding=enc)
                    break
                except Exception:
                    continue
            if gen_text is None:
                raise ValueError(f"无法读取 gen_adjust_info 文件: {gen_path}")
        else:
            # 否则按原始字符串处理（如 JSON 文本）
            gen_text = str(gen_adjust_info)

        parsed = None
        try:
            parsed = json.loads(gen_text)
        except Exception:
            try:
                parsed = ast.literal_eval(gen_text)
            except Exception:
                raise ValueError("gen_adjust_info 不是可解析的 JSON/字典文本，也不是可读文件")

        if isinstance(parsed, dict):
            gen_adjust_info = parsed
        elif isinstance(parsed, list):
            if any(not isinstance(item, dict) for item in parsed):
                raise ValueError("gen_adjust_info 解析为列表时，所有元素都必须是字典")
            gen_adjust_info = parsed
        elif isinstance(parsed, tuple):
            if any(not isinstance(item, dict) for item in parsed):
                raise ValueError("gen_adjust_info 解析为元组时，所有元素都必须是字典")
            gen_adjust_info = list(parsed)
        else:
            raise ValueError("gen_adjust_info 解析后必须是字典或字典列表")
    
    loaded_farm_adjusted_w_da = None
    if farm_adjusted_w_nc_path is not None:
        nc_path = Path(farm_adjusted_w_nc_path)
        if not nc_path.exists() or not nc_path.is_file():
            raise ValueError(f"farm_adjusted_w_nc_path 不存在或不是文件: {farm_adjusted_w_nc_path}")
        with xr.open_dataset(nc_path) as w_ds:
            if farm_adjusted_w_varname not in w_ds.data_vars:
                raise KeyError(
                    f"NetCDF 中不存在变量 {farm_adjusted_w_varname}，可用变量: {list(w_ds.data_vars)}"
                )
            # load+copy to detach from file-backed dataset before context manager closes
            loaded_farm_adjusted_w_da = w_ds[farm_adjusted_w_varname].load().copy(deep=True)
            if select_period is not None:
                loaded_farm_adjusted_w_da = filter_dataarray_by_period(
                    loaded_farm_adjusted_w_da,
                    select_period,
                    time_name=time_name,
                )
   
    # 调用 unit_farm_output
    return unit_farm_output(
        era5_dir=era5_dir,
        start_year=int(year_df),
        farm_id=farm_id,
        windfarm_lon=float(lon_df),
        windfarm_lat=float(lat_df),
        output_dir=output_dir,
        output_gen_nc_path=output_gen_nc_path,
        output_w_compare_nc_path=output_w_compare_nc_path,
        output_raw_w_nc_path=output_raw_w_nc_path,
        cdmet_dir=cdmet_dir,
        hub_height=float(hub_height_final),
        dem=dem_final,
        tech_life=tech_life,
        end_year=end_year,
        n_nearest_points=n_nearest_points,
        horizon_interp_method=horizon_interp_method,
        era5_varnames=era5_varnames,
        time_name=time_name,
        nearest_method=nearest_method,
        alpha_method=alpha_method,
        alpha_window=alpha_window,
        select_period=select_period,
        kb_adjust_info=kb_adjust_info,
        gen_adjust_info=gen_adjust_info,
        farm_adjusted_w_da=loaded_farm_adjusted_w_da,
        onshore_flag=bool(onshore_flag_final),
        farm_capacity_mw=farm_capacity_mw_final,
        w_grid=w_grid,
        cf_grid=cf_grid,
        unit_step_hours=unit_step_hours,
        cut_in_v=float(cut_in_v_final),
        rated_v=float(rated_v_final),
        cut_out_v=float(cut_out_v_final),
        power_order=power_order,
        smooth=smooth,
        fixed_sigma=fixed_sigma,
        sigma_slope=sigma_slope,
        sigma_intercept=sigma_intercept,
        sigma_truncate=sigma_truncate,
        w_step=w_step,
        w_max=w_max,
        output_hourly_gen=output_hourly_gen,
        output_monthly_gen=output_monthly_gen,
        output_hourly_cf=output_hourly_cf,
        output_w_compare=output_w_compare,
        output_raw_w=output_raw_w,
        overwrite=overwrite,
        verbose=verbose,
        raw_w_varname=raw_w_varname,
    )


def _batch_unit_farm_output_worker(task: _BatchFarmTask) -> Tuple[str, str, bool, Optional[str]]:
    """
    ProcessPool worker: execute one (farm_id, select_period) task.
    Returns (farm_id, period, success, error_message).
    """
    try:
        unit_farm_output_from_files(
            # 风电场标识参数
            farm_csv=task.farm_csv,
            farm_id=task.farm_id,
            # 输入数据路径参数
            era5_dir=task.era5_dir,
            era5_varnames=task.era5_varnames,
            time_name=task.time_name,
            cdmet_dir=task.cdmet_dir,
            # 输出控制参数
            select_period=task.select_period,
            output_hourly_gen=task.output_hourly_gen,
            output_monthly_gen=task.output_monthly_gen,
            output_hourly_cf=task.output_hourly_cf,
            output_w_compare=task.output_w_compare,
            output_raw_w=task.output_raw_w,
            overwrite=task.overwrite,
            raw_w_varname=task.raw_w_varname,
            output_dir=task.output_dir,
            output_gen_nc_path=task.output_gen_nc_path,
            output_w_compare_nc_path=task.output_w_compare_nc_path,
            output_raw_w_nc_path=task.output_raw_w_nc_path,
            # 调整输入参数
            farm_adjusted_w_nc_path=task.farm_adjusted_w_nc_path,
            farm_adjusted_w_varname=task.farm_adjusted_w_varname,
            w_grid=task.w_grid,
            cf_grid=task.cf_grid,
            allow_default_cf_csv_dir=task.allow_default_cf_csv_dir,
            cf_csv_dir=task.cf_csv_dir,
            kb_adjust_info=task.kb_adjust_info,
            gen_adjust_info=task.gen_adjust_info,
            # 风场默认参数
            tech_life=task.tech_life,
            end_year=task.end_year,
            # 模拟参数
            power_order=task.power_order,
            smooth=task.smooth,
            fixed_sigma=task.fixed_sigma,
            sigma_slope=task.sigma_slope,
            sigma_intercept=task.sigma_intercept,
            sigma_truncate=task.sigma_truncate,
            w_step=task.w_step,
            w_max=task.w_max,
            n_nearest_points=task.n_nearest_points,
            horizon_interp_method=task.horizon_interp_method,
            nearest_method=task.nearest_method,
            alpha_method=task.alpha_method,
            alpha_window=task.alpha_window,
            unit_step_hours=task.unit_step_hours,
            # 日志参数
            verbose=task.verbose,
        )
        return str(task.farm_id), str(task.select_period), True, None
    except Exception as exc:
        return str(task.farm_id), str(task.select_period), False, str(exc)


def batch_farm_output_from_files(
    farm_csv: Union[str, Path, pd.DataFrame],
    era5_dir: str,
    era5_varnames: Optional[List[str]] = ["u10", "v10", "u100", "v100", "t2m", "sp", "d2m", "z"],
    time_name: str = "time",
    cdmet_dir: Optional[str] = None,

    farm_ids: Optional[List[Union[int, str]]] = None,
    select_periods: Optional[Union[Union[str, int, None], List[Union[str, int, None]], Tuple[Union[str, int, None], ...]]] = None,
    periods: Optional[Union[Union[str, int, None], List[Union[str, int, None]], Tuple[Union[str, int, None], ...]]] = None,
    
    output_hourly_gen: bool = True,
    output_monthly_gen: bool = True,
    output_hourly_cf: bool = False,
    output_w_compare: bool = True,
    output_raw_w: bool = True,
    overwrite: bool = True,
    allow_default_cf_csv_dir: bool = True,
    process_workers: int = 4,

    output_dir: Optional[Union[str, Path]] = None,
    output_gen_subdir: str = "gen",
    output_w_compare_subdir: str = "w_compare",
    output_raw_w_subdir: str = "raw_w",
    farm_adjusted_w_dir: Optional[Union[str, Path]] = None,
    farm_adjusted_w_varname: str = "raw_w",
    cf_csv_dir: Optional[Union[str, Path]] = None,
    kb_adjust_info: Optional[Union[Dict, Tuple[float, float], str, Path]] = None,

    w_grid: Optional[np.ndarray] = None,
    cf_grid: Optional[np.ndarray] = None,
    
    gen_adjust_info: Optional[Union[Dict, List[Dict], Tuple[Dict, ...]]] = None,
    tech_life: int = 25,
    end_year: int = 2023,
    power_order: int = 3,
    smooth: bool = True,
    fixed_sigma: Union[bool, float] = False,
    sigma_slope: float = 0.2,
    sigma_intercept: float = 0.6,
    sigma_truncate: float = 4.0,
    w_step: float = 0.05,
    w_max: float = 40.0,
    n_nearest_points: int = 16,
    horizon_interp_method: Literal["closest", "idw", "loess"] = "loess",
    nearest_method: str = "binary",
    alpha_method: str = "moving_average_hourly",
    alpha_window: int = 3,
    unit_step_hours: float = 1.0,
    raw_w_varname: str = "raw_w",
    return_failures_only: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    批量调用 unit_farm_output_from_files：
    1. 读取 farm_csv；
    2. 提取“风电场ID”列；
    3. 使用 concurrent.futures 多进程并发处理每个 (farm_id, select_period) 任务。

    Args:
        farm_csv: 风电场表（DataFrame）或 CSV 路径。
        era5_dir: ERA5 数据根目录。
        process_workers: 进程池并发数，必须 >= 1。
        farm_ids: 可选，仅处理指定 farm_id 列表；None 表示处理表中全部 ID。
        select_periods: 时间段输入。
            - None: 不拆分，按全年计算（select_period=None）；
            - "all_seasons": 展开为四季 ["spring", "summer", "autumn", "winter"]；
            - "spring"/"summer"/"autumn"/"winter": 仅计算对应单季；
            - 列表/元组: 按元素逐项计算（支持混合 None/季节/月份）。
        periods: 兼容旧参数名，含义与 select_periods 相同；若同时传入两者会报错。
        overwrite: 当目标输出文件已存在时，是否覆盖并重新计算。
        output_dir: batch 层输出根目录，不能为空。
        output_gen_subdir/output_w_compare_subdir/output_raw_w_subdir:
            output_dir 下对应输出的次级目录名。
        farm_adjusted_w_dir: 已计算 raw_w 文件所在目录；batch 层会按
            (farm_id, select_period) 自动拼接输入文件路径。
        kb_adjust_info: 当传入 CSV 路径时，会按 farm_id 调用 generate_farm_seasonal_kb_dict
            生成每个风电场专属的四季(k,b)字典并传入单场计算。
        return_failures_only: True 时仅返回失败信息；False 时返回完整汇总。
        overwrite: 是否覆盖已存在的输出文件。为 False 时会跳过已存在文件的计算并记录为成功。

    Notes:
        batch 层不再接受 output_gen_nc_path/output_w_compare_nc_path/
        output_raw_w_nc_path 这类文件路径参数。
        三类输出文件统一写入 output_dir 下的次级目录；
        farm_adjusted_w_dir 为独立输入目录，不从 output_dir 派生。
        函数会为每个 (farm_id, select_period) 自动生成独立文件名，避免多任务写入同一路径。

    Returns:
        dict，包含任务总数、成功/失败数量、成功/失败任务键列表与错误明细。
    """
    if process_workers < 1:
        raise ValueError(f"process_workers 必须 >= 1，得到: {process_workers}")
    if output_dir is None:
        raise ValueError("batch_farm_output_from_files 要求显式提供 output_dir，不能为空")

    if isinstance(farm_csv, (str, Path)):
        farm_df = safe_read_csv(farm_csv)
        farm_csv_for_worker: Union[str, Path, pd.DataFrame] = str(farm_csv)
    else:
        farm_df = farm_csv
        farm_csv_for_worker = farm_df

    if farm_df is None or len(farm_df) == 0:
        raise ValueError("farm_df 不能为空")

    id_col = "风电场ID"
    if id_col not in farm_df.columns:
        raise KeyError(f"表中不存在 ID 列: {id_col}")

    all_ids = farm_df[id_col].dropna().astype(str).tolist()
    if farm_ids is None:
        run_ids = all_ids
    else:
        requested_ids = [str(fid).strip() for fid in farm_ids if str(fid).strip() != ""]
        if len(requested_ids) == 0:
            raise ValueError("farm_ids 提供后不能为空")

        available_id_set = set(all_ids)
        run_ids = [fid for fid in requested_ids if fid in available_id_set]
        missing_ids = [fid for fid in requested_ids if fid not in available_id_set]
        if len(run_ids) == 0:
            raise ValueError(f"farm_ids 均不在风电场表中，缺失ID: {missing_ids}")
        if len(missing_ids) > 0:
            logger.warning("以下 farm_ids 不在风电场表中，已跳过: %s", missing_ids)

    run_ids = list(dict.fromkeys(run_ids))

    if len(run_ids) == 0:
        raise ValueError("没有可处理的 farm_id（风电场ID）")

    if select_periods is not None and periods is not None:
        raise ValueError("select_periods 与 periods 不能同时传入，请只保留一个")
    periods_input = select_periods if select_periods is not None else periods

    if periods_input is None:
        run_periods_raw: List[Union[str, int, None]] = [None]
    elif isinstance(periods_input, (list, tuple)):
        run_periods_raw = list(periods_input)
    else:
        run_periods_raw = [periods_input]

    if len(run_periods_raw) == 0:
        raise ValueError("periods 不能为空")
   
    normalized_periods: List[Optional[Union[str, int]]] = []
    for p in run_periods_raw:
        if p is None:
            normalized_periods.append(None)
            continue

        if isinstance(p, str):
            period_text = p.strip().lower()
            if period_text == "all_seasons":
                normalized_periods.extend(["spring", "summer", "autumn", "winter"])
                continue
            normalized_periods.append(period_text)
            continue

        normalized_periods.append(p)
    
    run_periods = []
    for p in normalized_periods:
        if p not in run_periods:
            run_periods.append(p)

    def _period_key_for_filename(period: Optional[Union[str, int]]) -> str:
        return "full_year" if period is None else format_for_filename(period)

    def _kb_suffix_for_filename(
        kb_info: Optional[Union[Dict, Tuple[float, float], str, Path]]
    ) -> str:
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
                raise ValueError("kb_adjust_info 为单项 dict 时，其值必须是长度为2的(k, b)序列")
            include_kb_in_name = True
            k_token = format_for_filename(float(only_value[0]))
            b_token = format_for_filename(float(only_value[1]))

        if include_kb_in_name:
            return f"_k_{k_token}_b_{b_token}"
        return ""

    def _resolve_output_subdir(base_dir: Path, subdir_name: str, name: str) -> Path:
        if not isinstance(subdir_name, str) or subdir_name.strip() == "":
            raise ValueError(f"{name} 不能为空字符串")
        subdir_path = Path(subdir_name.strip())
        if subdir_path.is_absolute():
            raise ValueError(f"{name} 必须是 output_dir 下的相对目录名，不能是绝对路径: {subdir_path}")
        if subdir_path.suffix.lower() == ".nc":
            raise ValueError(f"{name} 必须是目录名，不能是 .nc 文件路径: {subdir_path}")
        return base_dir / subdir_path

    def _as_existing_dir_or_none(path_like: Optional[Union[str, Path]], name: str) -> Optional[Path]:
        if path_like is None:
            return None
        dir_path = Path(path_like)
        if dir_path.suffix.lower() == ".nc":
            raise ValueError(f"{name} 必须是目录，不能是 .nc 文件路径: {dir_path}")
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"{name} 不存在或不是目录: {dir_path}")
        return dir_path

    output_base_dir = Path(output_dir)
    should_output_gen_ds = output_hourly_gen or output_monthly_gen or output_hourly_cf
    output_gen_base_dir = (
        _resolve_output_subdir(output_base_dir, output_gen_subdir, "output_gen_subdir")
        if should_output_gen_ds
        else None
    )
    output_w_compare_base_dir = (
        _resolve_output_subdir(output_base_dir, output_w_compare_subdir, "output_w_compare_subdir")
        if output_w_compare
        else None
    )
    output_raw_w_base_dir = (
        _resolve_output_subdir(output_base_dir, output_raw_w_subdir, "output_raw_w_subdir")
        if output_raw_w
        else None
    )
    farm_adjusted_w_base_dir = _as_existing_dir_or_none(farm_adjusted_w_dir, "farm_adjusted_w_dir")
    kb_suffix = _kb_suffix_for_filename(kb_adjust_info)

    logger.info(
        "[START] batch_farm_output_from_files | farms=%s | periods=%s | workers=%s",
        len(run_ids),
        len(run_periods),
        process_workers,
    )

    success_tasks: List[str] = []
    failed_tasks: List[str] = []
    error_map: Dict[str, str] = {}

    task_base_kwargs = {
        "farm_csv": farm_csv_for_worker,
        "era5_dir": era5_dir,
        "era5_varnames": era5_varnames,
        "time_name": time_name,
        "cdmet_dir": cdmet_dir,
        "output_hourly_gen": output_hourly_gen,
        "output_monthly_gen": output_monthly_gen,
        "output_hourly_cf": output_hourly_cf,
        "output_w_compare": output_w_compare,
        "output_raw_w": output_raw_w,
        "overwrite": overwrite,
        "raw_w_varname": raw_w_varname,
        "output_dir": None,
        "farm_adjusted_w_varname": farm_adjusted_w_varname,
        "w_grid": w_grid,
        "cf_grid": cf_grid,
        "allow_default_cf_csv_dir": allow_default_cf_csv_dir,
        "cf_csv_dir": cf_csv_dir,
        "kb_adjust_info": kb_adjust_info,
        "gen_adjust_info": gen_adjust_info,
        "tech_life": tech_life,
        "end_year": end_year,
        "power_order": power_order,
        "smooth": smooth,
        "fixed_sigma": fixed_sigma,
        "sigma_slope": sigma_slope,
        "sigma_intercept": sigma_intercept,
        "sigma_truncate": sigma_truncate,
        "w_step": w_step,
        "w_max": w_max,
        "n_nearest_points": n_nearest_points,
        "horizon_interp_method": horizon_interp_method,
        "nearest_method": nearest_method,
        "alpha_method": alpha_method,
        "alpha_window": alpha_window,
        "unit_step_hours": unit_step_hours,
        "verbose": verbose,
    }
    
    with ProcessPoolExecutor(max_workers=process_workers) as executor:
        future_to_task = {}
        for farm_id in run_ids:
            for period in run_periods:
                period_key = _period_key_for_filename(period)
                farm_key = str(farm_id)

                task_output_gen_nc_path = None
                if output_gen_base_dir is not None:
                    task_output_gen_nc_path = str(
                        output_gen_base_dir / f"farm_{farm_key}_period_{period_key}{kb_suffix}_gen.nc"
                    )

                task_output_w_compare_nc_path = None
                if output_w_compare_base_dir is not None:
                    task_output_w_compare_nc_path = str(
                        output_w_compare_base_dir / f"farm_{farm_key}_period_{period_key}{kb_suffix}_w_compare.nc"
                    )

                task_output_raw_w_nc_path = None
                if output_raw_w_base_dir is not None:
                    task_output_raw_w_nc_path = str(
                        output_raw_w_base_dir / f"farm_{farm_key}_period_{period_key}{kb_suffix}_raw_w.nc"
                    )

                task_farm_adjusted_w_nc_path = None
                if farm_adjusted_w_base_dir is not None:
                    task_farm_adjusted_w_nc_path = str(
                        farm_adjusted_w_base_dir / f"farm_{farm_key}_period_{period_key}{kb_suffix}_raw_w.nc"
                    )

                task = _BatchFarmTask(
                    farm_id=farm_id,
                    select_period=period,
                    output_gen_nc_path=task_output_gen_nc_path,
                    output_w_compare_nc_path=task_output_w_compare_nc_path,
                    output_raw_w_nc_path=task_output_raw_w_nc_path,
                    farm_adjusted_w_nc_path=task_farm_adjusted_w_nc_path,
                    **task_base_kwargs,
                )
                future = executor.submit(_batch_unit_farm_output_worker, task)
                future_to_task[future] = task.task_key

        for future in as_completed(future_to_task):
            task_key = future_to_task[future]
            try:
                farm_id_str, period_str, success, err_msg = future.result()
                period_key = "full_year" if period_str == "None" else period_str
                task_key = f"{farm_id_str}|{period_key}"
                if success:
                    success_tasks.append(task_key)
                    logger.info("[BATCH] task=%s done", task_key)
                else:
                    failed_tasks.append(task_key)
                    error_map[task_key] = err_msg or "unknown error"
                    logger.error("[BATCH] task=%s failed: %s", task_key, error_map[task_key])
            except Exception as exc:
                failed_tasks.append(task_key)
                error_map[task_key] = str(exc)
                logger.error("[BATCH] task=%s failed with executor error: %s", task_key, exc)

    summary = {
        "total": len(run_ids) * len(run_periods),
        "success_count": len(success_tasks),
        "failure_count": len(failed_tasks),
        "success_tasks": success_tasks,
        "failed_tasks": failed_tasks,
        "errors": error_map,
    }

    logger.info(
        "[END] batch_farm_output_from_files | success=%s | failure=%s",
        summary["success_count"],
        summary["failure_count"],
    )

    if return_failures_only:
        return {
            "failure_count": summary["failure_count"],
            "failed_tasks": summary["failed_tasks"],
            "errors": summary["errors"],
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="批量风电场发电量计算")
    parser.add_argument("--farm_csv", type=parse_none, default=r"D:\HKU PhD\wind data paper\wind_farms_inventory_till_2023.csv")
    parser.add_argument("--era5_dir", type=parse_none, default=r'D:\Data\ERA5')
    parser.add_argument("--era5_varnames", type=parse_none, default="u10,v10,u100,v100,t2m,sp,d2m,z")
    parser.add_argument("--time_name", default="time")
    parser.add_argument("--cdmet_dir", type=parse_none, default=None)
    parser.add_argument("--farm_ids", type=parse_none, default=None)
    parser.add_argument("--select_periods", "--periods", dest="select_periods", type=parse_none, default=None)
    parser.add_argument("--output_hourly_gen", type=parse_tf, default=True)
    parser.add_argument("--output_monthly_gen", type=parse_tf, default=True)
    parser.add_argument("--output_hourly_cf", type=parse_tf, default=False)
    parser.add_argument("--output_w_compare", type=parse_tf, default=True)
    parser.add_argument("--output_raw_w", type=parse_tf, default=True)
    parser.add_argument("--overwrite", type=parse_tf, default=True)
    parser.add_argument(
        "--allow_default_cf_csv_dir",
        type=parse_tf,
        default=True,
        help="未给 cf_csv_dir 时是否回填默认功率曲线目录（默认 True）；"
             "是否使用逐场 CF csv 只看 cf_csv_dir 是否为 None。",
    )
    parser.add_argument(
        "--use_cf_csv",
        dest="allow_default_cf_csv_dir",
        type=parse_tf,
        help="[已弃用] 等价于 --allow_default_cf_csv_dir，仅为兼容旧命令行保留。",
    )
    parser.add_argument("--process_workers", type=int, default=1)
    parser.add_argument("--output_dir", type=parse_none, required=True)
    parser.add_argument("--output_gen_subdir", default="gen")
    parser.add_argument("--output_w_compare_subdir", default="w_compare")
    parser.add_argument("--output_raw_w_subdir", default="raw_w")
    parser.add_argument("--farm_adjusted_w_dir", type=parse_none, default=None)
    parser.add_argument("--farm_adjusted_w_varname", default="raw_w")
    parser.add_argument("--cf_csv_dir", type=parse_none, default=r'D:\HKU PhD\wind data paper\farm_power_curves')
    parser.add_argument("--kb_adjust_info", type=parse_none, default=None)
    parser.add_argument("--w_grid", type=parse_none, default=None)
    parser.add_argument("--cf_grid", type=parse_none, default=None)
    parser.add_argument("--gen_adjust_info", type=parse_none, default=None)
    parser.add_argument("--tech_life", type=int, default=25)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--power_order", type=int, default=3)
    parser.add_argument("--smooth", type=parse_tf, default=True)
    parser.add_argument("--fixed_sigma", type=parse_tf, default=False)
    parser.add_argument("--sigma_slope", type=float, default=0.2)
    parser.add_argument("--sigma_intercept", type=float, default=0.6)
    parser.add_argument("--sigma_truncate", type=float, default=4.0)
    parser.add_argument("--w_step", type=float, default=0.05)
    parser.add_argument("--w_max", type=float, default=40.0)
    parser.add_argument("--n_nearest_points", type=int, default=16)
    parser.add_argument("--horizon_interp_method", default="loess")
    parser.add_argument("--nearest_method", default="binary")
    parser.add_argument("--alpha_method", default="moving_average_hourly")
    parser.add_argument("--alpha_window", type=int, default=3)
    parser.add_argument("--unit_step_hours", type=float, default=1.0)
    parser.add_argument("--raw_w_varname", default="raw_w")
    parser.add_argument("--return_failures_only", type=parse_tf, default=False)
    parser.add_argument("--verbose", type=parse_tf, default=True)
    args = parser.parse_args()

    # 处理列表参数
    era5_varnames = parse_list(args.era5_varnames)
    farm_ids = parse_list(args.farm_ids)
    select_periods = parse_list(args.select_periods)
    w_grid = ast.literal_eval(args.w_grid) if args.w_grid else None
    cf_grid = ast.literal_eval(args.cf_grid) if args.cf_grid else None
    gen_adjust_info = ast.literal_eval(args.gen_adjust_info) if args.gen_adjust_info else None
    fixed_sigma = ast.literal_eval(args.fixed_sigma) if args.fixed_sigma else False

    batch_farm_output_from_files(
        farm_csv=args.farm_csv,
        era5_dir=args.era5_dir,
        era5_varnames=era5_varnames,
        time_name=args.time_name,
        cdmet_dir=args.cdmet_dir,
        farm_ids=farm_ids,
        select_periods=select_periods,
        output_hourly_gen=args.output_hourly_gen,
        output_monthly_gen=args.output_monthly_gen,
        output_hourly_cf=args.output_hourly_cf,
        output_w_compare=args.output_w_compare,
        output_raw_w=args.output_raw_w,
        overwrite=args.overwrite,
        allow_default_cf_csv_dir=args.allow_default_cf_csv_dir,
        process_workers=args.process_workers,
        output_dir=args.output_dir,
        output_gen_subdir=args.output_gen_subdir,
        output_w_compare_subdir=args.output_w_compare_subdir,
        output_raw_w_subdir=args.output_raw_w_subdir,
        farm_adjusted_w_dir=args.farm_adjusted_w_dir,
        farm_adjusted_w_varname=args.farm_adjusted_w_varname,
        cf_csv_dir=args.cf_csv_dir,
        kb_adjust_info=args.kb_adjust_info,
        w_grid=w_grid,
        cf_grid=cf_grid,
        gen_adjust_info=gen_adjust_info,
        tech_life=args.tech_life,
        end_year=args.end_year,
        power_order=args.power_order,
        smooth=args.smooth,
        fixed_sigma=fixed_sigma,
        sigma_slope=args.sigma_slope,
        sigma_intercept=args.sigma_intercept,
        sigma_truncate=args.sigma_truncate,
        w_step=args.w_step,
        w_max=args.w_max,
        n_nearest_points=args.n_nearest_points,
        horizon_interp_method=args.horizon_interp_method,
        nearest_method=args.nearest_method,
        alpha_method=args.alpha_method,
        alpha_window=args.alpha_window,
        unit_step_hours=args.unit_step_hours,
        raw_w_varname=args.raw_w_varname,
        return_failures_only=args.return_failures_only,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()



'''

他奶奶的！目前忘记了给它做这三种generation 矫正

utility_rate_adjust_info
capacity_adjust_info
gen_adjust_info

'''