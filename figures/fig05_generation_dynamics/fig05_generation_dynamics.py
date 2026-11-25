from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from matplotlib.ticker import FixedLocator, FuncFormatter
from plotly.subplots import make_subplots
from pypinyin import Style, lazy_pinyin


_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import config_paths as _cfg  # noqa: E402

# Single source of truth for the province -> grid mapping and the 蒙东/蒙西 split.
# Do NOT re-declare these constants here (2026-10 audit: 47 hand-copied variants,
# 36 of them wrong).  ``_lib/grid_map.py`` asserts at import time that 内蒙古's 12
# 盟市 are partitioned by 蒙东/蒙西 with no overlap and no gap.
_LIB = Path(__file__).resolve().parents[1] / "_lib"
if str(_LIB) not in sys.path:
	sys.path.insert(0, str(_LIB))
import grid_map as _grid_map  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Original defaults:
#   DEFAULT_CITY_ROOT   = <paper_root>/final_gen/city
#   DEFAULT_OUTPUT_ROOT = <paper_root>/plot/piling
DEFAULT_CITY_ROOT = _cfg.FINAL_GEN_DIR
DEFAULT_OUTPUT_ROOT = _HERE / "output"
DAILY_FREQ = "D"
DEFAULT_PLOT_GRANULARITY = "month"
DEFAULT_START = "1989-01-01"
DEFAULT_END = "2023-12-31"
RENDER_TARGET_ALL = "all"
RENDER_TARGET_BIPLOT_PROVINCE_IN_GRID = "biplot/province_in_grid"
RENDER_TARGET_BIPLOT_GRID_IN_NATIONAL = "biplot/grid_in_national"
RENDER_TARGET_BIPLOT_CITY_IN_PROVINCE_REPRESENTATIVES = "biplot/city_in_province_representatives"
RENDER_TARGET_CHOICES = [
	RENDER_TARGET_ALL,
	RENDER_TARGET_BIPLOT_PROVINCE_IN_GRID,
	RENDER_TARGET_BIPLOT_GRID_IN_NATIONAL,
	RENDER_TARGET_BIPLOT_CITY_IN_PROVINCE_REPRESENTATIVES,
]
PROVINCE_EARLIEST_NONZERO_START = {
	"新疆": pd.Timestamp("1989-10-01"),
	"内蒙古": pd.Timestamp("1990-01-01"),
}
# 蒙东电网口径（2026-10 用户修正）: only 赤峰市 / 通辽市 / 呼伦贝尔市 / 兴安盟.
# 锡林郭勒盟 (xilinguolei) is NOT part of 蒙东电网; it falls through to 蒙西
# (-> 华北电网) via ``_split_neimenggu``.  Do not add it back.
# Imported from the single source of truth ``_lib/grid_map.py``.
NEIMENGGU_EAST_CITY_KEYS = set(_grid_map.INNER_MONGOLIA_EAST_CITY_KEYS)  # chifeng/hulunbeier/tongliao/xingan

# 研究范围排除地区（2026-10 用户要求）: 香港 is not studied at all — it is not
# counted inside 南方电网, has no province-level series/ranking and appears in no
# legend or panel.  The exclusion is applied once, at load time, in
# ``_load_city_daily_long``; keys are *normalised* province labels
# (``_normalize_label`` strips 特别行政区, so "香港特别行政区" -> "香港").
# Imported from the single source of truth; remove it there to bring 香港 back.
EXCLUDED_REGIONS = set(_grid_map.EXCLUDED_REGIONS)

# Imported from ``_lib/grid_map.py`` (was a hand-written literal until 2026-10-14).
PROVINCE_GRID_MAP = dict(_grid_map.PROVINCE_TO_GRID)
GRID_ORDER = list(_grid_map.GRID_ORDER)
GRID_EN_MAP = dict(_grid_map.GRID_NAME_EN)
PROVINCE_EN_MAP = {
	"北京": "Beijing",
	"天津": "Tianjin",
	"河北": "Hebei",
	"山西": "Shanxi",
	"山东": "Shandong",
	"辽宁": "Liaoning",
	"吉林": "Jilin",
	"黑龙江": "Heilongjiang",
	"蒙东": "East Inner Mongolia",
	"上海": "Shanghai",
	"江苏": "Jiangsu",
	"浙江": "Zhejiang",
	"安徽": "Anhui",
	"福建": "Fujian",
	"河南": "Henan",
	"湖北": "Hubei",
	"湖南": "Hunan",
	"江西": "Jiangxi",
	"四川": "Sichuan",
	"重庆": "Chongqing",
	"西藏": "Xizang",
	"陕西": "Shaanxi",
	"甘肃": "Gansu",
	"青海": "Qinghai",
	"宁夏": "Ningxia",
	"新疆": "Xinjiang",
	"广东": "Guangdong",
	"广西": "Guangxi",
	"云南": "Yunnan",
	"贵州": "Guizhou",
	"海南": "Hainan",
	# "香港": dropped from the study via EXCLUDED_REGIONS — no English label needed.
	"蒙西": "West Inner Mongolia",
	"内蒙古": "Inner Mongolia",
}
MUTED_COLORS = [
	"#5C789B",
	"#6A9D60",
	"#CD6664",
	"#E4954F",
	"#85B4B0",
	"#A77F9B",
	"#E8C96D",
	"#957767",
	"#608A97",
	"#C47B94",
	"#829B5B",
	"#BA744E",
	"#778DB2",
	"#89689C",
	"#CEA852",
	"#6B9B77",
	"#B35B5B",
	"#976D4F",
	"#657B9C",
	"#AA657A",
]

PROVINCE_GRID_MATPLOTLIB_RGB = [
	(92 / 255.0, 120 / 255.0, 155 / 255.0),
	(106 / 255.0, 157 / 255.0, 96 / 255.0),
	(205 / 255.0, 102 / 255.0, 100 / 255.0),
	(228 / 255.0, 149 / 255.0, 79 / 255.0),
	(133 / 255.0, 180 / 255.0, 176 / 255.0),
	(167 / 255.0, 127 / 255.0, 155 / 255.0),
	(232 / 255.0, 201 / 255.0, 109 / 255.0),
]

PLOT_FONT_SIZE = 8
LEGEND_X = 1.02
BIPLOT_PROVINCE_GRID_COMPOSITE_SIZE = (20, 26.2)  # 28 → 26.2：压到 239.7 mm，能进 A4 正文
BIPLOT_CITY_PROVINCE_REPRESENTATIVE_COMPOSITE_SIZE = (20, 20)
CITY_PROVINCE_REPRESENTATIVES = [
	("Hebei", "Concentrated"),
	("Hunan", "Multi-centred"),
	("Henan", "Distributed"),
]

# 全局字体兜底：原版只给手写文字设了 Arial，格式化器自动生成的刻度文字（如占比面板的 %）
# 会退回 DejaVu。统一成 Arial，保证全图只有一种字体。除此以外一切不动。
plt.rcParams.update(
	{
		"font.family": "sans-serif",
		"font.sans-serif": ["Arial"],
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
	}
)

# 分区电网总图的两个子图小标题（用户 23:20 要求加上；未加冒号，沿用 "X Dynamics of Y" 形式）
GRID_NATIONAL_PANEL_TITLES = (
	"(a) Generation Dynamics of China's Regional Grids",
	"(b) Share Dynamics of China's Regional Grids",
)

PRINT_WIDTH_MM = 183.0
PRINT_DPI = 300
AXES_LEFT_MM = 19.0
# 单面板堆叠图：图例在右侧，右留白要留出图例列；双面板图例在左侧，右留白只留末刻度溢出。
AXES_RIGHT_MM = 146.0
AXES_RIGHT_NOLEGEND_MM = 178.0
PANEL_TOP_MM = 8.0
PANEL_BOTTOM_MM = 17.0
PANEL_TITLE_OFFSET_MM = 4.6
YLABEL_OFFSET_MM = 13.0
YEAR_LABEL_WIDTH_MM = 6.4
YEAR_LABEL_GAP_MM = 1.6
COLUMN_GAP = 0.32
ROW_GAP = 0.52
ROW_GAP_MM = 14.0  # 复合图的行距固定 12 mm（放刻度行 + 下一行标题）
MM_PER_INCH = 25.4


def _legend_height_mm(entry_count: int) -> float:
	"""单列图例的高度估算（8 pt、默认 labelspacing 0.5 / borderpad 0.4），用来判断画布要不要加高。"""
	entry_mm = PLOT_FONT_SIZE * 1.5 / 72.0 * MM_PER_INCH
	pad_mm = 2 * 0.4 * PLOT_FONT_SIZE / 72.0 * MM_PER_INCH
	return max(entry_count, 1) * entry_mm + pad_mm


def _print_figsize(figsize: tuple[float, float], width_mm: float = PRINT_WIDTH_MM) -> tuple[float, float]:
	"""按指定印刷宽度作画（默认 183 mm）：宽度锁死，高度按原比例推，字号才不会被缩印吃掉。"""
	width_inch = width_mm / MM_PER_INCH
	return (width_inch, width_inch * figsize[1] / figsize[0])


def _panel_height_mm(figure_height_mm: float, rows: int) -> float:
	return (figure_height_mm - PANEL_TOP_MM - PANEL_BOTTOM_MM) / (rows + (rows - 1) * ROW_GAP)


def _apply_print_margins(figure: plt.Figure, *, rows: int = 1, columns: int = 1, right_mm: float = AXES_RIGHT_MM, row_gap_mm: float | None = None, width_mm: float = PRINT_WIDTH_MM) -> None:
	"""留白按毫米反解，不用「看着差不多」的比例。row_gap_mm 给了就按固定毫米排行距。"""
	height_mm = figure.get_size_inches()[1] * MM_PER_INCH
	layout: dict[str, float] = {
		"left": AXES_LEFT_MM / width_mm,
		"right": right_mm / width_mm,
		"top": 1 - PANEL_TOP_MM / height_mm,
		"bottom": PANEL_BOTTOM_MM / height_mm,
	}
	if columns > 1:
		layout["wspace"] = COLUMN_GAP
	if rows > 1:
		if row_gap_mm:
			panel_h_mm = (height_mm - PANEL_TOP_MM - PANEL_BOTTOM_MM - (rows - 1) * row_gap_mm) / rows
			layout["hspace"] = row_gap_mm / panel_h_mm
		else:
			layout["hspace"] = ROW_GAP
	figure.subplots_adjust(**layout)


def _year_label_every(index: pd.Index, axis_width_mm: float) -> int:
	"""年份刻度隔几年标一次：判据是「相邻刻度物理间距 ≥ 文字宽 + 间隙」，不是写死 2 或 3。"""
	if len(index) == 0:
		return 2
	year_count = int(pd.Timestamp(index.max()).year) - int(pd.Timestamp(index.min()).year) + 1
	step_mm = axis_width_mm / max(year_count - 1, 1)
	return max(2, math.ceil((YEAR_LABEL_WIDTH_MM + YEAR_LABEL_GAP_MM) / step_mm))


def _open_dataset_with_fallback(path: str | Path, retries: int = 2, sleep_seconds: float = 0.6) -> xr.Dataset:
	resolved_path = Path(path)
	last_error: Exception | None = None
	for attempt in range(retries + 1):
		try:
			with xr.open_dataset(str(resolved_path)) as ds:
				return ds.load()
		except Exception as exc:  # noqa: BLE001
			last_error = exc
			if attempt < retries:
				time.sleep(sleep_seconds * (attempt + 1))

	try:
		with xr.open_dataset(str(resolved_path), engine="scipy") as ds:
			return ds.load()
	except Exception as exc:  # noqa: BLE001
		raise OSError(f"Failed to open NetCDF file: {resolved_path}; last_error={last_error}; scipy_error={exc}") from exc


def _normalize_label(value: Any) -> str:
	text = str(value).strip()
	for suffix in ("维吾尔自治区", "壮族自治区", "回族自治区", "自治区", "特别行政区", "自治州", "地区", "盟", "省", "市"):
		if text.endswith(suffix) and len(text) > len(suffix):
			text = text[: -len(suffix)]
			break
	return text.strip()


def _safe_slug(text: str) -> str:
	return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _to_english_name(text: str, fallback: str | None = None) -> str:
	normalized = _normalize_label(text)
	if normalized in PROVINCE_EN_MAP:
		return PROVINCE_EN_MAP[normalized]

	if fallback is not None:
		words = fallback.replace("_", " ").replace("-", " ").split()
		if words:
			return " ".join(word.capitalize() for word in words)

	parts = lazy_pinyin(normalized, style=Style.NORMAL)
	if parts:
		return " ".join(part.capitalize() for part in parts)
	return normalized


def _parse_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
	if isinstance(value, pd.Timestamp):
		return value.normalize()
	return pd.Timestamp(value).normalize()


def _granularity_to_rule(granularity: Literal["day", "week", "month"]) -> tuple[str, dict[str, Any], str]:
	if granularity == "day":
		return "D", {}, "%Y-%m"
	if granularity == "week":
		return "W-MON", {"label": "left", "closed": "left"}, "%Y-%m"
	if granularity == "month":
		return "MS", {}, "%Y"
	raise ValueError(f"Unsupported granularity: {granularity}")


def _resample_generation_long(
	df: pd.DataFrame,
	group_cols: list[str],
	granularity: Literal["day", "week", "month"],
) -> pd.DataFrame:
	rule, kwargs, _ = _granularity_to_rule(granularity)
	resampled = (
		df.set_index("time")
		.groupby(group_cols)["generation_mwh"]
		.resample(rule, **kwargs)
		.sum()
		.reset_index()
	)
	return resampled.sort_values(group_cols + ["time"]).reset_index(drop=True)


def _filter_time_range(df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
	start_ts = _parse_timestamp(start)
	end_ts = _parse_timestamp(end)
	if start_ts > end_ts:
		raise ValueError("start must be earlier than or equal to end")
	mask = (df["time"] >= start_ts) & (df["time"] <= end_ts)
	return df.loc[mask].copy()


def _province_earliest_nonzero_start(province_label: str) -> pd.Timestamp | None:
	normalized = _normalize_label(province_label)
	return PROVINCE_EARLIEST_NONZERO_START.get(normalized)


def _enforce_province_earliest_nonzero_start(
	df: pd.DataFrame,
	*,
	province_col: str,
	time_col: str = "time",
	value_col: str = "generation_mwh",
) -> pd.DataFrame:
	result = df.copy()
	province_label = str(result[province_col].iloc[0]) if not result.empty else ""
	earliest_ts = _province_earliest_nonzero_start(province_label)
	if earliest_ts is None:
		return result
	result.loc[result[time_col] < earliest_ts, value_col] = 0.0
	return result


def _build_palette(n_colors: int) -> list[str]:
	if n_colors <= len(MUTED_COLORS):
		return MUTED_COLORS[:n_colors]
	return [MUTED_COLORS[index % len(MUTED_COLORS)] for index in range(n_colors)]


def _build_province_grid_matplotlib_palette(n_colors: int) -> list[tuple[float, float, float]]:
	if n_colors <= len(PROVINCE_GRID_MATPLOTLIB_RGB):
		return PROVINCE_GRID_MATPLOTLIB_RGB[:n_colors]
	return [PROVINCE_GRID_MATPLOTLIB_RGB[index % len(PROVINCE_GRID_MATPLOTLIB_RGB)] for index in range(n_colors)]


def _discover_city_nc_files(city_root: str | Path) -> list[Path]:
	city_root = Path(city_root)
	if not city_root.exists():
		raise FileNotFoundError(f"city_root does not exist: {city_root}")
	paths = sorted(city_root.glob("*__*/full_year/gen.nc"))
	if not paths:
		raise FileNotFoundError(f"No city full_year gen.nc files found under: {city_root}")
	return paths


def _split_neimenggu(city_key: str) -> tuple[str, str]:
	if city_key in NEIMENGGU_EAST_CITY_KEYS:
		return "mengdong", "蒙东"
	return "mengxi", "蒙西"


def _resolve_province_group(province_key: str, province_label: str, city_key: str) -> tuple[str, str]:
	if province_key == "neimenggu" or province_label == "内蒙古":
		return _split_neimenggu(city_key)
	return province_key, province_label


def _load_city_daily_long(city_root: str | Path) -> pd.DataFrame:
	records: list[pd.DataFrame] = []
	for nc_path in _discover_city_nc_files(city_root):
		city_dir = nc_path.parent.parent.name
		province_key, city_key = city_dir.split("__", maxsplit=1)
		ds = _open_dataset_with_fallback(nc_path)
		if "total_gen" not in ds.data_vars or "time" not in ds["total_gen"].dims:
			raise KeyError(f"dataset missing hourly total_gen on time dimension: {nc_path}")

		province_label_raw = ds.attrs.get("province_short_name", province_key)
		city_label_raw = ds.attrs.get("city_short_name", city_key)
		province_label = _normalize_label(province_label_raw)
		city_label = _normalize_label(city_label_raw)
		# 研究范围过滤（唯一入口）: 香港 整体不入研究，不计入南方电网，
		# 也不出现在任何省/城市序列、排名、图例或面板中。
		if province_label in EXCLUDED_REGIONS:
			continue
		province_group_key, province_group_label = _resolve_province_group(
			province_key=province_key,
			province_label=province_label,
			city_key=city_key,
		)
		grid_label = PROVINCE_GRID_MAP.get(province_group_label)

		daily_series = ds["total_gen"].to_series().resample(DAILY_FREQ).sum(min_count=1).fillna(0.0)
		frame = daily_series.rename("generation_mwh").reset_index()
		frame["province_key"] = province_key
		frame["province_label"] = province_label
		frame["province_group_key"] = province_group_key
		frame["province_group_label"] = province_group_label
		frame["province_group_en"] = _to_english_name(province_group_label)
		frame["city_key"] = city_key
		frame["city_label"] = city_label
		frame["city_en"] = _to_english_name(city_label, fallback=city_key)
		frame["grid_label"] = grid_label
		frame["grid_en"] = GRID_EN_MAP.get(grid_label, "") if grid_label is not None else ""
		frame["is_grid_covered"] = bool(pd.notna(grid_label))
		frame = _enforce_province_earliest_nonzero_start(frame, province_col="province_label")
		records.append(frame)

	if not records:
		raise ValueError("No city generation records were loaded")

	daily_long = pd.concat(records, ignore_index=True)
	daily_long["time"] = pd.to_datetime(daily_long["time"])
	daily_long["generation_mwh"] = pd.to_numeric(daily_long["generation_mwh"], errors="coerce").fillna(0.0)
	return daily_long.sort_values(["province_group_label", "city_label", "time"]).reset_index(drop=True)


def _build_share_frame(
	group_df: pd.DataFrame,
	entity_col: str,
	parent_col: str,
	value_col: str = "generation_mwh",
) -> pd.DataFrame:
	entity_daily = (
		group_df.groupby([parent_col, "time", entity_col], as_index=False)[value_col]
		.sum()
	)
	parent_daily = (
		group_df.groupby([parent_col, "time"], as_index=False)[value_col]
		.sum()
		.rename(columns={value_col: "parent_generation_mwh"})
	)
	merged = entity_daily.merge(parent_daily, on=[parent_col, "time"], how="left")
	merged["share"] = (merged[value_col] / merged["parent_generation_mwh"]).where(merged["parent_generation_mwh"] > 0, 0.0)
	return merged.sort_values([parent_col, "time", entity_col]).reset_index(drop=True)


def _build_grid_share_frame(province_daily: pd.DataFrame) -> pd.DataFrame:
	grid_daily = (
		province_daily.groupby(["grid_label", "grid_en", "time"], as_index=False)["generation_mwh"]
		.sum()
	)
	national_daily = (
		grid_daily.groupby("time", as_index=False)["generation_mwh"]
		.sum()
		.rename(columns={"generation_mwh": "national_generation_mwh"})
	)
	merged = grid_daily.merge(national_daily, on="time", how="left")
	merged["share"] = (merged["generation_mwh"] / merged["national_generation_mwh"]).where(merged["national_generation_mwh"] > 0, 0.0)
	merged["grid_label"] = pd.Categorical(merged["grid_label"], categories=GRID_ORDER, ordered=True)
	return merged.sort_values(["time", "grid_label"]).reset_index(drop=True)


def _wide_value_table(df: pd.DataFrame, index_col: str, column_col: str, value_col: str) -> pd.DataFrame:
	wide = df.pivot(index=index_col, columns=column_col, values=value_col).sort_index()
	return wide.fillna(0.0)


def _stacked_peak(value_wide: pd.DataFrame) -> float:
	if value_wide.empty:
		return 0.0
	return float(value_wide.sum(axis=1).max())


def _ordered_plot_df(value_wide: pd.DataFrame, *, is_share: bool) -> pd.DataFrame:
	plot_df = value_wide.copy()
	last_values = plot_df.ffill().iloc[-1].sort_values(ascending=False)
	ordered_cols = last_values.index.tolist()
	plot_df = plot_df.loc[:, ordered_cols]
	if not is_share:
		plot_df = plot_df / 1000.0
	return plot_df


def _build_year_ticks(index: pd.Index, *, label_every: int, label_all: bool = False, force_endpoints: bool = False) -> tuple[list[pd.Timestamp], list[str]]:
	if len(index) == 0:
		return [], []

	start_year = int(pd.Timestamp(index.min()).year)
	end_year = int(pd.Timestamp(index.max()).year)
	tickvals = [pd.Timestamp(year=year, month=1, day=1) for year in range(start_year, end_year + 1)]
	ticktext: list[str] = []
	for offset, year in enumerate(range(start_year, end_year + 1)):
		show_label = label_all or offset % label_every == 0 or (force_endpoints and year in {start_year, end_year})
		ticktext.append(str(year) if show_label else "")
	return tickvals, ticktext


def _build_reverse_year_ticks(index: pd.Index, *, label_every: int) -> tuple[list[pd.Timestamp], list[str]]:
	if len(index) == 0:
		return [], []

	start_year = int(pd.Timestamp(index.min()).year)
	end_year = int(pd.Timestamp(index.max()).year)
	years = list(range(start_year, end_year + 1))
	tickvals = [pd.Timestamp(year=year, month=1, day=1) for year in years]
	visible_years = set(range(end_year, start_year - 1, -label_every))
	ticktext = [str(year) if year in visible_years else "" for year in years]
	return tickvals, ticktext


def _build_share_ticks() -> tuple[list[float], list[str]]:
	tickvals = [round(step * 0.05, 10) for step in range(21)]
	ticktext = [f"{int(value * 100)}%" if index % 2 == 0 else "" for index, value in enumerate(tickvals)]
	return tickvals, ticktext


def _nice_number(value: float) -> float:
	if value <= 0:
		return 1.0
	exponent = math.floor(math.log10(value))
	fraction = value / (10 ** exponent)
	if fraction <= 1:
		nice_fraction = 1
	elif fraction <= 2:
		nice_fraction = 2
	elif fraction <= 5:
		nice_fraction = 5
	else:
		nice_fraction = 10
	return float(nice_fraction * (10 ** exponent))


def _format_generation_tick(value: float) -> str:
	if value >= 100:
		return f"{value:,.0f}"
	if value >= 10:
		return f"{value:,.1f}".rstrip("0").rstrip(".")
	if value >= 1:
		return f"{value:,.2f}".rstrip("0").rstrip(".")
	return f"{value:,.3f}".rstrip("0").rstrip(".")


def _build_generation_ticks(max_value: float) -> tuple[list[float], list[str], list[float], list[str], float]:
	if max_value <= 0:
		return [0.0, 1.0], ["0", "1"], [1.0], ["1"], 1.0

	base_step = _nice_number(max_value / 25.0)
	axis_max = math.ceil(max_value / base_step) * base_step
	tickvals = [float(step * base_step) for step in range(int(round(axis_max / base_step)) + 1)]
	ticktext = [_format_generation_tick(value) if index % 5 == 0 or index == len(tickvals) - 1 else "" for index, value in enumerate(tickvals)]
	positive_tickvals = [value for value in tickvals if value > 0]
	positive_ticktext = [ticktext[index] for index, value in enumerate(tickvals) if value > 0]
	return tickvals, ticktext, positive_tickvals, positive_ticktext, base_step


def _legend_config() -> dict[str, Any]:
	return {
		"orientation": "v",
		"yanchor": "top",
		"y": 1.0,
		"xanchor": "left",
		"x": LEGEND_X,
		"font": {"size": PLOT_FONT_SIZE},
		"traceorder": "reversed",
	}


def _legend_config_inside_top_left() -> dict[str, Any]:
	return {
		"orientation": "v",
		"yanchor": "top",
		"y": 0.99,
		"xanchor": "left",
		"x": 0.01,
		"font": {"size": PLOT_FONT_SIZE},
		"traceorder": "reversed",
		"bgcolor": "rgba(255,255,255,0.78)",
		"bordercolor": "rgba(120,120,120,0.45)",
		"borderwidth": 1,
	}


def _base_axis_style() -> dict[str, Any]:
	return {
		"showline": True,
		"linewidth": 1.0,
		"linecolor": "#444444",
		"ticks": "outside",
		"ticklen": 5,
		"tickwidth": 1,
		"tickcolor": "#444444",
		"showticklabels": True,
		"automargin": True,
		"zeroline": False,
	}


def _build_share_axis_config(index: pd.Index, *, ylabel: str) -> tuple[dict[str, Any], dict[str, Any]]:
	tickvals, ticktext = _build_year_ticks(index, label_every=2)
	xaxis = {
		"title": "Year",
		"tickmode": "array",
		"tickvals": tickvals,
		"ticktext": ticktext,
		"range": [pd.Timestamp(index.min()), pd.Timestamp(index.max())],
		"showgrid": True,
		"gridcolor": "#ececec",
		"gridwidth": 0.3,
		"griddash": "dash",
		"minor": {
			"showgrid": True,
			"dtick": "M1",
			"gridcolor": "#f6f6f6",
			"gridwidth": 0.12,
			"griddash": "dash",
		},
		**_base_axis_style(),
	}
	yaxis = {
		"title": ylabel,
		"tickmode": "linear",
		"tick0": 0,
		"dtick": 0.1,
		"tickformat": ".0%",
		"range": [0, 1],
		"showgrid": True,
		"gridcolor": "#e9e9e9",
		"gridwidth": 0.35,
		"griddash": "dash",
		"minor": {
			"showgrid": True,
			"tick0": 0,
			"dtick": 0.05,
			"gridcolor": "#f7f7f7",
			"gridwidth": 0.12,
			"griddash": "dash",
		},
		**_base_axis_style(),
	}
	return xaxis, yaxis


def _build_generation_axis_config(index: pd.Index, max_value: float) -> tuple[dict[str, Any], dict[str, Any]]:
	tickvals, ticktext = _build_year_ticks(index, label_every=2)
	linear_tickvals, linear_ticktext, _, _, base_step = _build_generation_ticks(max_value)
	xaxis = {
		"title": "Year",
		"tickmode": "array",
		"tickvals": tickvals,
		"ticktext": ticktext,
		"range": [pd.Timestamp(index.min()), pd.Timestamp(index.max())],
		"showgrid": True,
		"gridcolor": "#ececec",
		"gridwidth": 0.3,
		"griddash": "dash",
		"minor": {
			"showgrid": True,
			"dtick": "M1",
			"gridcolor": "#f6f6f6",
			"gridwidth": 0.12,
			"griddash": "dash",
		},
		**_base_axis_style(),
	}
	yaxis = {
		"title": "Generation (GWh)",
		"tickmode": "array",
		"tickvals": linear_tickvals,
		"ticktext": linear_ticktext,
		"tickformat": ",.0f",
		"range": [0, (linear_tickvals[-1] if linear_tickvals else max_value) * 1.08],
		"showgrid": True,
		"gridcolor": "#e9e9e9",
		"gridwidth": 0.35,
		"griddash": "dash",
		"minor": {
			"showgrid": True,
			"tick0": 0,
			"dtick": base_step,
			"gridcolor": "#f7f7f7",
			"gridwidth": 0.12,
			"griddash": "dash",
		},
		**_base_axis_style(),
	}
	return xaxis, yaxis




def _add_stacked_traces(
	figure: go.Figure,
	plot_df: pd.DataFrame,
	*,
	is_share: bool,
	row: int | None = None,
	col: int | None = None,
	showlegend: bool = True,
	legendgroup_suffix: str = "main",
) -> None:
	palette = _build_palette(len(plot_df.columns))
	for index, column_name in enumerate(plot_df.columns):
		trace = go.Scatter(
			x=plot_df.index,
			y=plot_df[column_name],
			name=str(column_name),
			mode="lines",
			line={"width": 0.6, "color": palette[index]},
			stackgroup=f"stack_{row}_{col}_{legendgroup_suffix}" if row is not None and col is not None else f"stack_{legendgroup_suffix}",
			groupnorm="fraction" if is_share else None,
			legendgroup=str(column_name),
			showlegend=showlegend,
			hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2f}" + ("<br>Share: %{y:.2%}<extra></extra>" if is_share else "<extra></extra>"),
		)
		if row is not None and col is not None:
			figure.add_trace(trace, row=row, col=col)
		else:
			figure.add_trace(trace)


def _write_figure(figure: go.Figure, output_path: str | Path, figsize: tuple[float, float]) -> None:
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	html_path = output_path.with_suffix(".html")
	figure.write_html(str(html_path), include_plotlyjs="cdn")


def _write_matplotlib_figure(figure: plt.Figure, output_path: str | Path) -> None:
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	png_path = output_path.with_suffix(".png")
	figure.savefig(str(png_path), dpi=PRINT_DPI, facecolor="white")
	plt.close(figure)


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
	color = color.lstrip("#")
	return tuple(int(color[index:index + 2], 16) / 255.0 for index in (0, 2, 4))


def _build_matplotlib_palette(n_colors: int) -> list[tuple[float, float, float]]:
	return [_hex_to_rgb(color) for color in _build_palette(n_colors)]


def _plot_matplotlib_stacked_area(
	value_wide: pd.DataFrame,
	output_path: str | Path,
	*,
	is_share: bool,
	ylabel: str,
	figsize: tuple[float, float],
) -> None:
	if value_wide.empty:
		return

	plot_df = _ordered_plot_df(value_wide, is_share=is_share)
	if plot_df.empty:
		return

	figure, axis = plt.subplots(figsize=_print_figsize(figsize))
	_apply_print_margins(figure)
	font_family = "Arial"
	tick_font_size = PLOT_FONT_SIZE
	legend_font_size = PLOT_FONT_SIZE
	label_font_size = PLOT_FONT_SIZE
	palette = _build_matplotlib_palette(len(plot_df.columns))
	x_values = plot_df.index.to_pydatetime()
	year_label_every = _year_label_every(plot_df.index, axis.get_position().width * PRINT_WIDTH_MM)
	year_ticks, year_labels = _build_year_ticks(plot_df.index, label_every=year_label_every)
	handles = axis.stackplot(
		x_values,
		*[plot_df[column].to_numpy() for column in plot_df.columns],
		colors=palette,
		linewidth=0.0,
		alpha=0.95,
	)
	axis.set_xlim(plot_df.index.min().to_pydatetime(), plot_df.index.max().to_pydatetime())
	axis.xaxis.set_major_locator(FixedLocator(mdates.date2num([tick.to_pydatetime() for tick in year_ticks])))
	axis.set_xticklabels(year_labels, fontsize=tick_font_size, fontfamily=font_family)
	axis.tick_params(axis="both", labelsize=tick_font_size, colors="#444444")
	axis.spines["top"].set_visible(False)
	axis.spines["right"].set_visible(False)
	axis.spines["left"].set_color("#444444")
	axis.spines["bottom"].set_color("#444444")
	axis.grid(False)
	axis.set_xlabel("Year", fontsize=label_font_size, fontfamily=font_family)
	axis.set_ylabel(ylabel, fontsize=label_font_size, fontfamily=font_family)

	if is_share:
		share_tickvals, _ = _build_share_ticks()
		axis.set_ylim(0, 1)
		axis.set_yticks(share_tickvals)
		axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value * 100))}%"))
	else:
		max_value = _stacked_peak(plot_df)
		tickvals, ticktext, _, _, _ = _build_generation_ticks(max_value)
		axis.set_ylim(0, (tickvals[-1] if tickvals else max_value) * 1.08)
		axis.set_yticks(tickvals)
		axis.set_yticklabels(ticktext, fontsize=tick_font_size, fontfamily=font_family)

	axis.legend(
		list(reversed(handles)),
		list(reversed(plot_df.columns.tolist())),
		loc="center left",
		bbox_to_anchor=(1.01, 0.5),
		prop={"family": font_family, "size": legend_font_size},
		frameon=True,
		facecolor="white",
		framealpha=0.85,
		edgecolor="#c0c0c0",
		handlelength=0.8,
		handletextpad=0.4,
	)

	_write_matplotlib_figure(figure, output_path)


def _plot_generation_share_biplot_matplotlib(
	generation_wide: pd.DataFrame,
	share_wide: pd.DataFrame,
	share_ylabel: str,
	output_path: str | Path,
	figsize: tuple[float, float] = (20, 8),
	use_fixed_rgb: bool = False,
	width_mm: float = PRINT_WIDTH_MM,
	right_mm: float = AXES_RIGHT_NOLEGEND_MM,
	major_minor_year_ticks: bool = False,
	panel_titles: tuple[str, str] | None = None,
) -> None:
	if generation_wide.empty or share_wide.empty:
		return

	generation_plot_df = _ordered_plot_df(generation_wide, is_share=False)
	share_plot_df = _ordered_plot_df(share_wide, is_share=True)
	if generation_plot_df.empty or share_plot_df.empty:
		return

	# 图例一律放图框内左上角：条目多到面板装不下时，只把画布加高，宽度不动。
	legend_fit_height_mm = _legend_height_mm(len(generation_plot_df.columns)) + PANEL_TOP_MM + PANEL_BOTTOM_MM + 1.0
	if legend_fit_height_mm > _print_figsize(figsize, width_mm)[1] * MM_PER_INCH:
		figsize = (figsize[0], figsize[0] * legend_fit_height_mm / width_mm)

	figure, axes = plt.subplots(1, 2, figsize=_print_figsize(figsize, width_mm), squeeze=False)
	generation_ax = axes[0, 0]
	share_ax = axes[0, 1]
	_apply_print_margins(figure, columns=2, right_mm=right_mm, width_mm=width_mm)
	font_family = "Arial"
	tick_font_size = PLOT_FONT_SIZE
	legend_font_size = PLOT_FONT_SIZE
	label_font_size = PLOT_FONT_SIZE
	palette = _build_province_grid_matplotlib_palette(len(generation_plot_df.columns)) if use_fixed_rgb else _build_matplotlib_palette(len(generation_plot_df.columns))
	x_generation = generation_plot_df.index.to_pydatetime()
	x_share = share_plot_df.index.to_pydatetime()
	year_label_every = _year_label_every(generation_plot_df.index, generation_ax.get_position().width * width_mm)
	share_tickvals = [step / 5 for step in range(6)]

	generation_handles = generation_ax.stackplot(
		x_generation,
		*[generation_plot_df[column].to_numpy() for column in generation_plot_df.columns],
		colors=palette,
		linewidth=0.0,
		alpha=0.95,
	)
	share_ax.stackplot(
		x_share,
		*[share_plot_df[column].to_numpy() for column in share_plot_df.columns],
		colors=palette,
		linewidth=0.0,
		alpha=0.95,
	)

	generation_max_value = _stacked_peak(generation_plot_df)
	generation_tickvals, generation_ticktext, _, _, _ = _build_generation_ticks(generation_max_value)
	generation_year_ticks, generation_year_labels = _build_reverse_year_ticks(generation_plot_df.index, label_every=year_label_every)
	share_year_ticks, share_year_labels = _build_reverse_year_ticks(share_plot_df.index, label_every=year_label_every)
	if generation_ticktext:
		generation_ticktext = list(generation_ticktext)
		for tick_index in range(len(generation_ticktext) - 1, -1, -1):
			if generation_ticktext[tick_index]:
				generation_ticktext[tick_index] = ""
				break

	generation_ax.set_xlim(generation_plot_df.index.min().to_pydatetime(), generation_plot_df.index.max().to_pydatetime())
	share_ax.set_xlim(share_plot_df.index.min().to_pydatetime(), share_plot_df.index.max().to_pydatetime())
	generation_ax.set_ylim(0, (generation_tickvals[-1] if generation_tickvals else generation_max_value) * 1.08)
	share_ax.set_ylim(0, 1)

	generation_ax.xaxis.set_major_locator(FixedLocator(mdates.date2num([tick.to_pydatetime() for tick in generation_year_ticks])))
	share_ax.xaxis.set_major_locator(FixedLocator(mdates.date2num([tick.to_pydatetime() for tick in share_year_ticks])))
	if major_minor_year_ticks:
		# 用户 23:00 指定：有年份文字的做长刻度、没有的做短刻度，且首尾年份必须标出来。
		years = [int(tick.year) for tick in generation_year_ticks]
		# 只强制标出首年份，其余按等间隔往后排（不强制尾年份，否则末尾间距会不均匀）。
		start_year, end_year = years[0], years[-1]
		panel_w_mm = generation_ax.get_position().width * width_mm
		max_labels = max(2, int(panel_w_mm / (YEAR_LABEL_WIDTH_MM + YEAR_LABEL_GAP_MM)) + 1)
		step_years = max(1, math.ceil((end_year - start_year) / max(1, max_labels - 1)))
		labeled_years = list(range(start_year, end_year + 1, step_years))
		labeled_set = set(labeled_years)
		minor_years = [year for year in years if year not in labeled_set]
		major_ticks = mdates.date2num([pd.Timestamp(year=year, month=1, day=1) for year in labeled_years])
		minor_ticks = mdates.date2num([pd.Timestamp(year=year, month=1, day=1) for year in minor_years])
		for axis in (generation_ax, share_ax):
			axis.xaxis.set_major_locator(FixedLocator(major_ticks))
			axis.xaxis.set_minor_locator(FixedLocator(minor_ticks))
			axis.set_xticklabels([str(year) for year in labeled_years], fontsize=tick_font_size, fontfamily=font_family)
			axis.tick_params(axis="x", which="major", length=3.5, color="#444444")
			axis.tick_params(axis="x", which="minor", length=1.6, color="#444444", bottom=True)
	else:
		generation_ax.set_xticklabels(generation_year_labels, fontsize=tick_font_size, fontfamily=font_family)
		share_ax.set_xticklabels(share_year_labels, fontsize=tick_font_size, fontfamily=font_family)
	generation_ax.set_yticks(generation_tickvals)
	generation_ax.set_yticklabels(generation_ticktext, fontsize=tick_font_size, fontfamily=font_family)
	share_ax.set_yticks(share_tickvals)
	share_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value * 100))}%"))

	for axis in (generation_ax, share_ax):
		axis.tick_params(axis="both", labelsize=tick_font_size, colors="#444444")
		axis.spines["top"].set_visible(False)
		axis.spines["right"].set_visible(False)
		axis.spines["left"].set_color("#444444")
		axis.spines["bottom"].set_color("#444444")
		axis.grid(False)
		axis.set_xlabel("", fontsize=label_font_size, fontfamily=font_family)

	generation_ax.set_ylabel("Generation (GWh)", fontsize=label_font_size, fontfamily=font_family)
	share_ax.set_ylabel(share_ylabel, fontsize=label_font_size, fontfamily=font_family)
	generation_ax.legend(
		list(reversed(generation_handles)),
		list(reversed(generation_plot_df.columns.tolist())),
		loc="upper left",
		prop={"family": font_family, "size": legend_font_size},
		frameon=True,
		facecolor="white",
		framealpha=0.85,
		edgecolor="#c0c0c0",
		handlelength=0.8,
		handletextpad=0.4,
	)

	if panel_titles:
		figure_height_mm = figure.get_size_inches()[1] * MM_PER_INCH
		for title_text, title_axis in zip(panel_titles, (generation_ax, share_ax)):
			title_axis.text(
				0.5,
				1.0 + PANEL_TITLE_OFFSET_MM / (title_axis.get_position().height * figure_height_mm),
				title_text,
				transform=title_axis.transAxes,
				ha="center",
				va="bottom",
				fontsize=label_font_size,
				fontfamily=font_family,
				clip_on=False,
			)

	_write_matplotlib_figure(figure, output_path)


def _plot_share_stacked_area(
	value_wide: pd.DataFrame,
	output_path: str | Path,
	ylabel: str,
	figsize: tuple[float, float] = (16, 8),
) -> None:
	if value_wide.empty:
		return

	plot_df = _ordered_plot_df(value_wide, is_share=True)
	xaxis, yaxis = _build_share_axis_config(plot_df.index, ylabel=ylabel)
	figure = go.Figure()
	_add_stacked_traces(figure, plot_df, is_share=True)
	figure.update_layout(
		template="plotly_white",
		width=int(figsize[0] * 100),
		height=int(figsize[1] * 100),
		hovermode="x unified",
		font={"family": "Arial", "size": 11},
		margin={"l": 100, "r": 220, "t": 30, "b": 90},
		legend=_legend_config(),
		xaxis=xaxis,
		yaxis=yaxis,
	)
	_write_figure(figure, output_path, figsize)
	_plot_matplotlib_stacked_area(value_wide=value_wide, output_path=output_path, is_share=True, ylabel=ylabel, figsize=figsize)


def _plot_generation_stacked_area(
	value_wide: pd.DataFrame,
	output_path: str | Path,
	figsize: tuple[float, float] = (16, 8),
) -> None:
	if value_wide.empty:
		return

	plot_df = _ordered_plot_df(value_wide, is_share=False)
	max_value = _stacked_peak(plot_df)
	xaxis, yaxis = _build_generation_axis_config(plot_df.index, max_value)
	figure = go.Figure()
	_add_stacked_traces(figure, plot_df, is_share=False, showlegend=True, legendgroup_suffix="generation")
	figure.update_layout(
		template="plotly_white",
		width=int(figsize[0] * 100),
		height=int(figsize[1] * 100),
		hovermode="x unified",
		font={"family": "Arial", "size": 11},
		margin={"l": 100, "r": 40, "t": 30, "b": 90},
		legend=_legend_config_inside_top_left(),
		xaxis=xaxis,
		yaxis=yaxis,
	)
	_write_figure(figure, output_path, figsize)
	_plot_matplotlib_stacked_area(value_wide=value_wide, output_path=output_path, is_share=False, ylabel="Generation (GWh)", figsize=figsize)


def _plot_generation_share_biplot(
	generation_wide: pd.DataFrame,
	share_wide: pd.DataFrame,
	share_ylabel: str,
	output_path: str | Path,
	figsize: tuple[float, float] = (20, 8),
	use_fixed_rgb: bool = False,
	width_mm: float = PRINT_WIDTH_MM,
	right_mm: float = AXES_RIGHT_NOLEGEND_MM,
	major_minor_year_ticks: bool = False,
	panel_titles: tuple[str, str] | None = None,
) -> None:
	if generation_wide.empty or share_wide.empty:
		return

	generation_plot_df = _ordered_plot_df(generation_wide, is_share=False)
	share_plot_df = _ordered_plot_df(share_wide, is_share=True)
	if generation_plot_df.empty or share_plot_df.empty:
		return

	generation_max_value = _stacked_peak(generation_plot_df)
	generation_xaxis, generation_yaxis = _build_generation_axis_config(generation_plot_df.index, generation_max_value)
	share_xaxis, share_yaxis = _build_share_axis_config(share_plot_df.index, ylabel=share_ylabel)
	figure = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.08)
	_add_stacked_traces(figure, generation_plot_df, is_share=False, row=1, col=1, showlegend=True, legendgroup_suffix="biplot_generation")
	_add_stacked_traces(figure, share_plot_df, is_share=True, row=1, col=2, showlegend=False, legendgroup_suffix="biplot_share")
	figure.update_layout(
		template="plotly_white",
		width=int(figsize[0] * 100),
		height=int(figsize[1] * 100),
		hovermode="x unified",
		font={"family": "Arial", "size": 11},
		margin={"l": 100, "r": 40, "t": 30, "b": 90},
		legend=_legend_config_inside_top_left(),
	)
	figure.update_xaxes(row=1, col=1, **generation_xaxis)
	figure.update_yaxes(row=1, col=1, **generation_yaxis)
	figure.update_xaxes(row=1, col=2, **share_xaxis)
	figure.update_yaxes(row=1, col=2, **share_yaxis)
	_write_figure(figure, output_path, figsize)
	_plot_generation_share_biplot_matplotlib(
		generation_wide=generation_wide,
		share_wide=share_wide,
		share_ylabel=share_ylabel,
		output_path=output_path,
		figsize=figsize,
		use_fixed_rgb=use_fixed_rgb,
		width_mm=width_mm,
		right_mm=right_mm,
		major_minor_year_ticks=major_minor_year_ticks,
		panel_titles=panel_titles,
	)


def _plot_grid_national_generation_share_biplot_matplotlib(
	generation_wide: pd.DataFrame,
	share_wide: pd.DataFrame,
	share_ylabel: str,
	output_path: str | Path,
	figsize: tuple[float, float] = (20, 8),
	width_mm: float = PRINT_WIDTH_MM,
	right_mm: float = AXES_RIGHT_NOLEGEND_MM,
	major_minor_year_ticks: bool = False,
	panel_titles: tuple[str, str] | None = None,
) -> None:
	if generation_wide.empty or share_wide.empty:
		return

	generation_plot_df = _ordered_plot_df(generation_wide, is_share=False)
	share_plot_df = _ordered_plot_df(share_wide, is_share=True)
	if generation_plot_df.empty or share_plot_df.empty:
		return

	# 图例一律放图框内左上角：条目多到面板装不下时，只把画布加高，宽度不动。
	legend_fit_height_mm = _legend_height_mm(len(generation_plot_df.columns)) + PANEL_TOP_MM + PANEL_BOTTOM_MM + 1.0
	if legend_fit_height_mm > _print_figsize(figsize, width_mm)[1] * MM_PER_INCH:
		figsize = (figsize[0], figsize[0] * legend_fit_height_mm / width_mm)

	figure, axes = plt.subplots(1, 2, figsize=_print_figsize(figsize, width_mm), squeeze=False)
	generation_ax = axes[0, 0]
	share_ax = axes[0, 1]
	_apply_print_margins(figure, columns=2, right_mm=right_mm, width_mm=width_mm)
	font_family = "Arial"
	tick_font_size = PLOT_FONT_SIZE
	legend_font_size = PLOT_FONT_SIZE
	label_font_size = PLOT_FONT_SIZE
	palette = _build_province_grid_matplotlib_palette(len(generation_plot_df.columns))
	x_generation = generation_plot_df.index.to_pydatetime()
	x_share = share_plot_df.index.to_pydatetime()
	year_label_every = _year_label_every(generation_plot_df.index, generation_ax.get_position().width * width_mm)
	share_tickvals = [step / 5 for step in range(6)]

	generation_handles = generation_ax.stackplot(
		x_generation,
		*[generation_plot_df[column].to_numpy() for column in generation_plot_df.columns],
		colors=palette,
		linewidth=0.0,
		alpha=0.95,
	)
	share_ax.stackplot(
		x_share,
		*[share_plot_df[column].to_numpy() for column in share_plot_df.columns],
		colors=palette,
		linewidth=0.0,
		alpha=0.95,
	)

	generation_max_value = _stacked_peak(generation_plot_df)
	generation_tickvals, generation_ticktext, _, _, _ = _build_generation_ticks(generation_max_value)
	generation_year_ticks, generation_year_labels = _build_reverse_year_ticks(generation_plot_df.index, label_every=year_label_every)
	share_year_ticks, share_year_labels = _build_reverse_year_ticks(share_plot_df.index, label_every=year_label_every)
	if generation_ticktext:
		generation_ticktext = list(generation_ticktext)
		for tick_index in range(len(generation_ticktext) - 1, -1, -1):
			if generation_ticktext[tick_index]:
				generation_ticktext[tick_index] = ""
				break

	generation_ax.set_xlim(generation_plot_df.index.min().to_pydatetime(), generation_plot_df.index.max().to_pydatetime())
	share_ax.set_xlim(share_plot_df.index.min().to_pydatetime(), share_plot_df.index.max().to_pydatetime())
	generation_ax.set_ylim(0, (generation_tickvals[-1] if generation_tickvals else generation_max_value) * 1.08)
	share_ax.set_ylim(0, 1)

	generation_ax.xaxis.set_major_locator(FixedLocator(mdates.date2num([tick.to_pydatetime() for tick in generation_year_ticks])))
	share_ax.xaxis.set_major_locator(FixedLocator(mdates.date2num([tick.to_pydatetime() for tick in share_year_ticks])))
	if major_minor_year_ticks:
		# 用户 23:00 指定：有年份文字的做长刻度、没有的做短刻度，且首尾年份必须标出来。
		years = [int(tick.year) for tick in generation_year_ticks]
		# 只强制标出首年份，其余按等间隔往后排（不强制尾年份，否则末尾间距会不均匀）。
		start_year, end_year = years[0], years[-1]
		panel_w_mm = generation_ax.get_position().width * width_mm
		max_labels = max(2, int(panel_w_mm / (YEAR_LABEL_WIDTH_MM + YEAR_LABEL_GAP_MM)) + 1)
		step_years = max(1, math.ceil((end_year - start_year) / max(1, max_labels - 1)))
		labeled_years = list(range(start_year, end_year + 1, step_years))
		labeled_set = set(labeled_years)
		minor_years = [year for year in years if year not in labeled_set]
		major_ticks = mdates.date2num([pd.Timestamp(year=year, month=1, day=1) for year in labeled_years])
		minor_ticks = mdates.date2num([pd.Timestamp(year=year, month=1, day=1) for year in minor_years])
		for axis in (generation_ax, share_ax):
			axis.xaxis.set_major_locator(FixedLocator(major_ticks))
			axis.xaxis.set_minor_locator(FixedLocator(minor_ticks))
			axis.set_xticklabels([str(year) for year in labeled_years], fontsize=tick_font_size, fontfamily=font_family)
			axis.tick_params(axis="x", which="major", length=3.5, color="#444444")
			axis.tick_params(axis="x", which="minor", length=1.6, color="#444444", bottom=True)
	else:
		generation_ax.set_xticklabels(generation_year_labels, fontsize=tick_font_size, fontfamily=font_family)
		share_ax.set_xticklabels(share_year_labels, fontsize=tick_font_size, fontfamily=font_family)
	generation_ax.set_yticks(generation_tickvals)
	generation_ax.set_yticklabels(generation_ticktext, fontsize=tick_font_size, fontfamily=font_family)
	share_ax.set_yticks(share_tickvals)
	share_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value * 100))}%"))

	for axis in (generation_ax, share_ax):
		axis.tick_params(axis="both", labelsize=tick_font_size, colors="#444444")
		axis.spines["top"].set_visible(False)
		axis.spines["right"].set_visible(False)
		axis.spines["left"].set_color("#444444")
		axis.spines["bottom"].set_color("#444444")
		axis.grid(False)
		axis.set_xlabel("", fontsize=label_font_size, fontfamily=font_family)

	generation_ax.set_ylabel("Generation (GWh)", fontsize=label_font_size, fontfamily=font_family)
	share_ax.set_ylabel(share_ylabel, fontsize=label_font_size, fontfamily=font_family)
	generation_ax.legend(
		list(reversed(generation_handles)),
		list(reversed(generation_plot_df.columns.tolist())),
		loc="upper left",
		prop={"family": font_family, "size": legend_font_size},
		frameon=True,
		facecolor="white",
		framealpha=0.85,
		edgecolor="#c0c0c0",
		handlelength=0.8,
		handletextpad=0.4,
	)

	_write_matplotlib_figure(figure, output_path)


def _subplot_panel_label(index: int) -> str:
	return f"({chr(ord('a') + index)})"


def _subplot_axis_layout_keys(row: int, col: int, n_cols: int = 2) -> tuple[str, str]:
	axis_number = (row - 1) * n_cols + col
	suffix = "" if axis_number == 1 else str(axis_number)
	return f"xaxis{suffix}", f"yaxis{suffix}"


def _build_subplot_legend_annotation(labels: list[str]) -> str:
	palette = _build_palette(len(labels))
	return "<br>".join(
		f"<span style='color:{color}; font-weight:700'>■</span> {label}"
		for label, color in zip(labels, palette)
	)


def _plot_province_grid_generation_share_biplot_composite(
	province_generation_df: pd.DataFrame,
	province_share_df: pd.DataFrame,
	output_path: str | Path,
	figsize: tuple[float, float] = BIPLOT_PROVINCE_GRID_COMPOSITE_SIZE,
) -> None:
	available_grid_labels = [
		grid_label
		for grid_label in GRID_ORDER
		if grid_label in set(province_share_df["grid_label"].dropna().astype(str))
	]
	if not available_grid_labels:
		return

	plotly_titles: list[str] = []
	panel_index = 0
	for grid_label in available_grid_labels:
		grid_en = GRID_EN_MAP.get(grid_label, str(grid_label))
		plotly_titles.append(f"{_subplot_panel_label(panel_index)} Generation Dynamics of {grid_en}")
		panel_index += 1
		plotly_titles.append(f"{_subplot_panel_label(panel_index)} Share Dynamics of {grid_en}")
		panel_index += 1

	plotly_figure = make_subplots(
		rows=len(available_grid_labels),
		cols=2,
		shared_xaxes=False,
		horizontal_spacing=0.08,
		vertical_spacing=0.06,
		subplot_titles=plotly_titles,
	)
	for row_index, grid_label in enumerate(available_grid_labels, start=1):
		share_df = province_share_df.loc[province_share_df["grid_label"] == grid_label].copy()
		generation_df = province_generation_df.loc[province_generation_df["grid_label"] == grid_label].copy()
		share_wide = _wide_value_table(share_df, index_col="time", column_col="province_group_en", value_col="share")
		generation_wide = _wide_value_table(generation_df, index_col="time", column_col="province_group_en", value_col="generation_mwh")
		if share_wide.empty or generation_wide.empty:
			continue
		generation_plot_df = _ordered_plot_df(generation_wide, is_share=False)
		share_plot_df = _ordered_plot_df(share_wide, is_share=True)
		generation_max_value = _stacked_peak(generation_plot_df)
		generation_xaxis, generation_yaxis = _build_generation_axis_config(generation_plot_df.index, generation_max_value)
		share_xaxis, share_yaxis = _build_share_axis_config(share_plot_df.index, ylabel="Share of Regional Grid Generation")
		_add_stacked_traces(plotly_figure, generation_plot_df, is_share=False, row=row_index, col=1, showlegend=False, legendgroup_suffix=f"province_grid_generation_{row_index}")
		_add_stacked_traces(plotly_figure, share_plot_df, is_share=True, row=row_index, col=2, showlegend=False, legendgroup_suffix=f"province_grid_share_{row_index}")
		plotly_figure.update_xaxes(row=row_index, col=1, **generation_xaxis)
		plotly_figure.update_yaxes(row=row_index, col=1, **generation_yaxis)
		plotly_figure.update_xaxes(row=row_index, col=2, **share_xaxis)
		plotly_figure.update_yaxes(row=row_index, col=2, **share_yaxis)
	plotly_figure.update_layout(
		template="plotly_white",
		width=int(figsize[0] * 100),
		height=int(figsize[1] * 100),
		hovermode="x unified",
		font={"family": "Arial", "size": 11},
		margin={"l": 100, "r": 40, "t": 60, "b": 80},
		showlegend=False,
	)
	_write_figure(plotly_figure, output_path, figsize)

	row_count = len(available_grid_labels)
	# 用户指定：每个子图固定 30 mm 高（5 省图例 23.4 mm 能完整进框；总高超出 A4 没关系）。
	province_panel_mm = 30.0
	needed_height_mm = province_panel_mm * row_count + (row_count - 1) * ROW_GAP_MM + PANEL_TOP_MM + PANEL_BOTTOM_MM
	figsize = (figsize[0], figsize[0] * needed_height_mm / PRINT_WIDTH_MM)
	figure, axes = plt.subplots(row_count, 2, figsize=_print_figsize(figsize), squeeze=False)
	_apply_print_margins(figure, rows=row_count, columns=2, right_mm=AXES_RIGHT_NOLEGEND_MM, row_gap_mm=ROW_GAP_MM)
	figure_height_mm = figure.get_size_inches()[1] * MM_PER_INCH
	panel_h_mm = axes[0, 0].get_position().height * figure_height_mm
	panel_title_y = 1.0 + PANEL_TITLE_OFFSET_MM / panel_h_mm
	share_tickvals = [step / 5 for step in range(6)]
	font_family = "Arial"
	tick_font_size = PLOT_FONT_SIZE
	legend_font_size = PLOT_FONT_SIZE
	label_font_size = PLOT_FONT_SIZE
	title_font_size = PLOT_FONT_SIZE

	for row_index, grid_label in enumerate(available_grid_labels, start=1):
		share_df = province_share_df.loc[province_share_df["grid_label"] == grid_label].copy()
		generation_df = province_generation_df.loc[province_generation_df["grid_label"] == grid_label].copy()
		share_wide = _wide_value_table(share_df, index_col="time", column_col="province_group_en", value_col="share")
		generation_wide = _wide_value_table(generation_df, index_col="time", column_col="province_group_en", value_col="generation_mwh")
		if share_wide.empty or generation_wide.empty:
			continue

		generation_plot_df = _ordered_plot_df(generation_wide, is_share=False)
		share_plot_df = _ordered_plot_df(share_wide, is_share=True)
		generation_ax = axes[row_index - 1, 0]
		share_ax = axes[row_index - 1, 1]
		palette = _build_province_grid_matplotlib_palette(len(generation_plot_df.columns))
		x_generation = generation_plot_df.index.to_pydatetime()
		x_share = share_plot_df.index.to_pydatetime()
		year_label_every = _year_label_every(generation_plot_df.index, generation_ax.get_position().width * PRINT_WIDTH_MM)

		generation_handles = generation_ax.stackplot(
			x_generation,
			*[generation_plot_df[column].to_numpy() for column in generation_plot_df.columns],
			colors=palette,
			linewidth=0.0,
			alpha=0.95,
		)
		share_ax.stackplot(
			x_share,
			*[share_plot_df[column].to_numpy() for column in share_plot_df.columns],
			colors=palette,
			linewidth=0.0,
			alpha=0.95,
		)

		generation_max_value = _stacked_peak(generation_plot_df)
		generation_tickvals, generation_ticktext, _, _, _ = _build_generation_ticks(generation_max_value)
		generation_year_ticks, generation_year_labels = _build_reverse_year_ticks(generation_plot_df.index, label_every=year_label_every)
		share_year_ticks, share_year_labels = _build_reverse_year_ticks(share_plot_df.index, label_every=year_label_every)
		if generation_ticktext:
			generation_ticktext = list(generation_ticktext)
			for tick_index in range(len(generation_ticktext) - 1, -1, -1):
				if generation_ticktext[tick_index]:
					generation_ticktext[tick_index] = ""
					break

		generation_ax.set_xlim(generation_plot_df.index.min().to_pydatetime(), generation_plot_df.index.max().to_pydatetime())
		share_ax.set_xlim(share_plot_df.index.min().to_pydatetime(), share_plot_df.index.max().to_pydatetime())
		generation_ax.set_ylim(0, (generation_tickvals[-1] if generation_tickvals else generation_max_value) * 1.08)
		share_ax.set_ylim(0, 1)

		generation_ax.xaxis.set_major_locator(FixedLocator(mdates.date2num([tick.to_pydatetime() for tick in generation_year_ticks])))
		share_ax.xaxis.set_major_locator(FixedLocator(mdates.date2num([tick.to_pydatetime() for tick in share_year_ticks])))
		generation_ax.set_xticklabels(generation_year_labels, fontsize=tick_font_size, fontfamily=font_family)
		share_ax.set_xticklabels(share_year_labels, fontsize=tick_font_size, fontfamily=font_family)
		generation_ax.set_yticks(generation_tickvals)
		generation_ax.set_yticklabels(generation_ticktext, fontsize=tick_font_size, fontfamily=font_family)
		share_ax.set_yticks(share_tickvals)
		share_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value * 100))}%"))

		for axis in (generation_ax, share_ax):
			axis.tick_params(axis="both", labelsize=tick_font_size, colors="#444444")
			axis.spines["top"].set_visible(False)
			axis.spines["right"].set_visible(False)
			axis.spines["left"].set_color("#444444")
			axis.spines["bottom"].set_color("#444444")
			axis.grid(False)
			axis.set_xlabel("", fontsize=label_font_size, fontfamily=font_family)

		generation_ax.set_ylabel("")
		share_ax.set_ylabel("")

		generation_ax.legend(
			list(reversed(generation_handles)),
			list(reversed(generation_plot_df.columns.tolist())),
			loc="upper left",
			prop={"family": font_family, "size": legend_font_size},
			frameon=True,
			facecolor="white",
			framealpha=0.85,
			edgecolor="#c0c0c0",
			handlelength=0.8,
			handletextpad=0.4,
		)

		grid_en = GRID_EN_MAP.get(grid_label, str(grid_label))
		generation_ax.text(
			0.5,
			panel_title_y,
			f"{_subplot_panel_label((row_index - 1) * 2)} Generation Dynamics of {grid_en}",
			transform=generation_ax.transAxes,
			ha="center",
			va="top",
			fontsize=title_font_size,
			fontfamily=font_family,
			clip_on=False,
		)
		share_ax.text(
			0.5,
			panel_title_y,
			f"{_subplot_panel_label((row_index - 1) * 2 + 1)} Share Dynamics of {grid_en}",
			transform=share_ax.transAxes,
			ha="center",
			va="top",
			fontsize=title_font_size,
			fontfamily=font_family,
			clip_on=False,
		)

	generation_position = axes[0, 0].get_position()
	share_position = axes[0, 1].get_position()
	figure.text(generation_position.x0 - YLABEL_OFFSET_MM / PRINT_WIDTH_MM, 0.5, "Generation (GWh)", va="center", rotation="vertical", fontsize=label_font_size, fontfamily=font_family)
	figure.text(share_position.x0 - YLABEL_OFFSET_MM / PRINT_WIDTH_MM, 0.5, "Share of Regional Grid Generation", va="center", rotation="vertical", fontsize=label_font_size, fontfamily=font_family)

	_write_matplotlib_figure(figure, output_path)


def _plot_city_province_representative_generation_share_biplot_composite(
	city_generation_df: pd.DataFrame,
	city_share_df: pd.DataFrame,
	output_path: str | Path,
	figsize: tuple[float, float] = BIPLOT_CITY_PROVINCE_REPRESENTATIVE_COMPOSITE_SIZE,
) -> None:
	available_representatives = [
		(province_en, category_label)
		for province_en, category_label in CITY_PROVINCE_REPRESENTATIVES
		if province_en in set(city_generation_df["province_group_en"].dropna().astype(str))
	]
	if not available_representatives:
		return

	plotly_titles: list[str] = []
	panel_index = 0
	for province_en, category_label in available_representatives:
		plotly_titles.append(f"{_subplot_panel_label(panel_index)} Generation Dynamics of {province_en} ({category_label})")
		panel_index += 1
		plotly_titles.append(f"{_subplot_panel_label(panel_index)} Share Dynamics of {province_en} ({category_label})")
		panel_index += 1

	plotly_figure = make_subplots(
		rows=len(available_representatives),
		cols=2,
		shared_xaxes=False,
		horizontal_spacing=0.08,
		vertical_spacing=0.06,
		subplot_titles=plotly_titles,
	)
	for row_index, (province_en, _category_label) in enumerate(available_representatives, start=1):
		share_df = city_share_df.loc[city_share_df["province_group_en"] == province_en].copy()
		generation_df = city_generation_df.loc[city_generation_df["province_group_en"] == province_en].copy()
		share_wide = _wide_value_table(share_df, index_col="time", column_col="city_en", value_col="share")
		generation_wide = _wide_value_table(generation_df, index_col="time", column_col="city_en", value_col="generation_mwh")
		if share_wide.empty or generation_wide.empty:
			continue
		generation_plot_df = _ordered_plot_df(generation_wide, is_share=False)
		share_plot_df = _ordered_plot_df(share_wide, is_share=True)
		generation_max_value = _stacked_peak(generation_plot_df)
		generation_xaxis, generation_yaxis = _build_generation_axis_config(generation_plot_df.index, generation_max_value)
		share_xaxis, share_yaxis = _build_share_axis_config(share_plot_df.index, ylabel="Share of Provincial Generation")
		_add_stacked_traces(plotly_figure, generation_plot_df, is_share=False, row=row_index, col=1, showlegend=False, legendgroup_suffix=f"city_generation_{row_index}")
		_add_stacked_traces(plotly_figure, share_plot_df, is_share=True, row=row_index, col=2, showlegend=False, legendgroup_suffix=f"city_share_{row_index}")
		plotly_figure.update_xaxes(row=row_index, col=1, **generation_xaxis)
		plotly_figure.update_yaxes(row=row_index, col=1, **generation_yaxis)
		plotly_figure.update_xaxes(row=row_index, col=2, **share_xaxis)
		plotly_figure.update_yaxes(row=row_index, col=2, **share_yaxis)
	plotly_figure.update_layout(
		template="plotly_white",
		width=int(figsize[0] * 100),
		height=int(figsize[1] * 100),
		hovermode="x unified",
		font={"family": "Arial", "size": 11},
		margin={"l": 100, "r": 40, "t": 60, "b": 80},
		showlegend=False,
	)
	_write_figure(plotly_figure, output_path, figsize)

	row_count = len(available_representatives)
	# 图例一律放图框内：某一行条目多到面板装不下，就把画布加高（宽度锁 183 不变）。
	# 每行子图高度跟着该行图例的高度走：既不切图例，也不出现某一行过高过矮。
	row_entry_counts = [
		int(city_generation_df.loc[city_generation_df["province_group_en"] == province_en, "city_en"].nunique())
		for province_en, _ in available_representatives
	]
	row_heights_mm = [_legend_height_mm(count) + 1.5 for count in row_entry_counts]
	needed_height_mm = sum(row_heights_mm) + (row_count - 1) * ROW_GAP_MM + PANEL_TOP_MM + PANEL_BOTTOM_MM
	figsize = (figsize[0], figsize[0] * needed_height_mm / PRINT_WIDTH_MM)

	figure, axes = plt.subplots(
		row_count, 2, figsize=_print_figsize(figsize), squeeze=False, gridspec_kw={"height_ratios": row_heights_mm}
	)
	_apply_print_margins(figure, rows=row_count, columns=2, right_mm=AXES_RIGHT_NOLEGEND_MM, row_gap_mm=ROW_GAP_MM)
	figure_height_mm = figure.get_size_inches()[1] * MM_PER_INCH
	panel_h_mm = axes[0, 0].get_position().height * figure_height_mm
	panel_title_y = 1.0 + PANEL_TITLE_OFFSET_MM / panel_h_mm
	share_tickvals = [step / 5 for step in range(6)]
	font_family = "Arial"
	tick_font_size = PLOT_FONT_SIZE
	legend_font_size = PLOT_FONT_SIZE
	label_font_size = PLOT_FONT_SIZE
	title_font_size = PLOT_FONT_SIZE

	for row_index, (province_en, category_label) in enumerate(available_representatives, start=1):
		share_df = city_share_df.loc[city_share_df["province_group_en"] == province_en].copy()
		generation_df = city_generation_df.loc[city_generation_df["province_group_en"] == province_en].copy()
		share_wide = _wide_value_table(share_df, index_col="time", column_col="city_en", value_col="share")
		generation_wide = _wide_value_table(generation_df, index_col="time", column_col="city_en", value_col="generation_mwh")
		if share_wide.empty or generation_wide.empty:
			continue

		generation_plot_df = _ordered_plot_df(generation_wide, is_share=False)
		share_plot_df = _ordered_plot_df(share_wide, is_share=True)
		generation_ax = axes[row_index - 1, 0]
		share_ax = axes[row_index - 1, 1]
		palette = _build_matplotlib_palette(len(generation_plot_df.columns))
		x_generation = generation_plot_df.index.to_pydatetime()
		x_share = share_plot_df.index.to_pydatetime()
		year_label_every = _year_label_every(generation_plot_df.index, generation_ax.get_position().width * PRINT_WIDTH_MM)

		generation_handles = generation_ax.stackplot(
			x_generation,
			*[generation_plot_df[column].to_numpy() for column in generation_plot_df.columns],
			colors=palette,
			linewidth=0.0,
			alpha=0.95,
		)
		share_ax.stackplot(
			x_share,
			*[share_plot_df[column].to_numpy() for column in share_plot_df.columns],
			colors=palette,
			linewidth=0.0,
			alpha=0.95,
		)

		generation_max_value = _stacked_peak(generation_plot_df)
		generation_tickvals, generation_ticktext, _, _, _ = _build_generation_ticks(generation_max_value)
		generation_year_ticks, generation_year_labels = _build_reverse_year_ticks(generation_plot_df.index, label_every=year_label_every)
		share_year_ticks, share_year_labels = _build_reverse_year_ticks(share_plot_df.index, label_every=year_label_every)
		if generation_ticktext:
			generation_ticktext = list(generation_ticktext)
			for tick_index in range(len(generation_ticktext) - 1, -1, -1):
				if generation_ticktext[tick_index]:
					generation_ticktext[tick_index] = ""
					break

		generation_ax.set_xlim(generation_plot_df.index.min().to_pydatetime(), generation_plot_df.index.max().to_pydatetime())
		share_ax.set_xlim(share_plot_df.index.min().to_pydatetime(), share_plot_df.index.max().to_pydatetime())
		generation_ax.set_ylim(0, (generation_tickvals[-1] if generation_tickvals else generation_max_value) * 1.08)
		share_ax.set_ylim(0, 1)

		generation_ax.xaxis.set_major_locator(FixedLocator(mdates.date2num([tick.to_pydatetime() for tick in generation_year_ticks])))
		share_ax.xaxis.set_major_locator(FixedLocator(mdates.date2num([tick.to_pydatetime() for tick in share_year_ticks])))
		generation_ax.set_xticklabels(generation_year_labels, fontsize=tick_font_size, fontfamily=font_family)
		share_ax.set_xticklabels(share_year_labels, fontsize=tick_font_size, fontfamily=font_family)
		generation_ax.set_yticks(generation_tickvals)
		generation_ax.set_yticklabels(generation_ticktext, fontsize=tick_font_size, fontfamily=font_family)
		share_ax.set_yticks(share_tickvals)
		share_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value * 100))}%"))

		for axis in (generation_ax, share_ax):
			axis.tick_params(axis="both", labelsize=tick_font_size, colors="#444444")
			axis.spines["top"].set_visible(False)
			axis.spines["right"].set_visible(False)
			axis.spines["left"].set_color("#444444")
			axis.spines["bottom"].set_color("#444444")
			axis.grid(False)
			axis.set_xlabel("", fontsize=label_font_size, fontfamily=font_family)

		generation_ax.set_ylabel("")
		share_ax.set_ylabel("")

		generation_ax.legend(
			list(reversed(generation_handles)),
			list(reversed(generation_plot_df.columns.tolist())),
			loc="upper left",
			prop={"family": font_family, "size": legend_font_size},
			frameon=True,
			facecolor="white",
			framealpha=0.85,
			edgecolor="#c0c0c0",
			handlelength=0.8,
			handletextpad=0.4,
		)

		generation_ax.text(
			0.5,
			panel_title_y,
			f"{_subplot_panel_label((row_index - 1) * 2)} Generation Dynamics of {province_en} ({category_label})",
			transform=generation_ax.transAxes,
			ha="center",
			va="top",
			fontsize=title_font_size,
			fontfamily=font_family,
			clip_on=False,
		)
		share_ax.text(
			0.5,
			panel_title_y,
			f"{_subplot_panel_label((row_index - 1) * 2 + 1)} Share Dynamics of {province_en} ({category_label})",
			transform=share_ax.transAxes,
			ha="center",
			va="top",
			fontsize=title_font_size,
			fontfamily=font_family,
			clip_on=False,
		)

	generation_position = axes[0, 0].get_position()
	share_position = axes[0, 1].get_position()
	figure.text(generation_position.x0 - YLABEL_OFFSET_MM / PRINT_WIDTH_MM, 0.5, "Generation (GWh)", va="center", rotation="vertical", fontsize=label_font_size, fontfamily=font_family)
	figure.text(share_position.x0 - YLABEL_OFFSET_MM / PRINT_WIDTH_MM, 0.5, "Share of Provincial Generation", va="center", rotation="vertical", fontsize=label_font_size, fontfamily=font_family)

	_write_matplotlib_figure(figure, output_path)


def _write_share_tables(
	city_share_df: pd.DataFrame,
	province_share_df: pd.DataFrame,
	grid_share_df: pd.DataFrame,
	output_root: str | Path,
) -> None:
	table_dir = Path(output_root) / "tables"
	table_dir.mkdir(parents=True, exist_ok=True)
	for old_csv in table_dir.glob("*.csv"):
		old_csv.unlink()
	city_share_df.to_csv(table_dir / "city_share_in_province_daily.csv", index=False, encoding="utf-8-sig")
	province_share_df.to_csv(table_dir / "province_share_in_grid_daily.csv", index=False, encoding="utf-8-sig")
	grid_share_df.to_csv(table_dir / "grid_share_in_national_daily.csv", index=False, encoding="utf-8-sig")


def _write_generation_tables(
	city_generation_df: pd.DataFrame,
	province_generation_df: pd.DataFrame,
	grid_generation_df: pd.DataFrame,
	output_root: str | Path,
) -> None:
	table_dir = Path(output_root) / "tables"
	table_dir.mkdir(parents=True, exist_ok=True)
	for old_csv in table_dir.glob("*.csv"):
		old_csv.unlink()
	city_generation_df.to_csv(table_dir / "city_generation_in_province.csv", index=False, encoding="utf-8-sig")
	province_generation_df.to_csv(table_dir / "province_generation_in_grid.csv", index=False, encoding="utf-8-sig")
	grid_generation_df.to_csv(table_dir / "grid_generation_in_national.csv", index=False, encoding="utf-8-sig")


def _clear_plot_outputs(*output_dirs: Path) -> None:
	for output_dir in output_dirs:
		output_dir.mkdir(parents=True, exist_ok=True)
		for pattern in ("*.png", "*.html"):
			for old_file in output_dir.glob(pattern):
				old_file.unlink()


def _remove_legacy_outputs(output_root: Path) -> None:
	legacy_dirs = [
		output_root / "city_share_in_province",
		output_root / "province_share_in_grid",
		output_root / "tables",
	]
	legacy_files = [output_root / "grid_share_in_national.png"]
	for legacy_dir in legacy_dirs:
		if legacy_dir.exists():
			shutil.rmtree(legacy_dir)
	for legacy_file in legacy_files:
		if legacy_file.exists():
			legacy_file.unlink()


def generate_generation_dynamics_plots(
	city_root: str | Path = DEFAULT_CITY_ROOT,
	output_root: str | Path = DEFAULT_OUTPUT_ROOT,
	granularity: Literal["day", "week", "month"] = DEFAULT_PLOT_GRANULARITY,
	start: str | pd.Timestamp = DEFAULT_START,
	end: str | pd.Timestamp = DEFAULT_END,
	render_target: str = RENDER_TARGET_ALL,
) -> dict[str, Any]:
	output_root = Path(output_root)
	if render_target not in RENDER_TARGET_CHOICES:
		raise ValueError(f"Unsupported render target: {render_target}")
	_remove_legacy_outputs(output_root)
	daily_long = _load_city_daily_long(city_root)
	# 口径安全网（不改变任何计算，仅断言排除生效）：香港 一旦泄漏进来就立刻失败。
	_leaked = sorted(set(daily_long["province_label"].astype(str)) & EXCLUDED_REGIONS)
	if _leaked:
		raise AssertionError(f"EXCLUDED_REGIONS leaked into the dataset: {_leaked}")
	daily_long = _filter_time_range(daily_long, start=start, end=end)
	if daily_long.empty:
		raise ValueError("No generation records remain after applying the selected time range")
	daily_long = _resample_generation_long(
		daily_long,
		group_cols=[
			"province_key",
			"province_label",
			"province_group_key",
			"province_group_label",
			"province_group_en",
			"city_key",
			"city_label",
			"city_en",
			"grid_label",
			"grid_en",
		],
		granularity=granularity,
	)

	province_daily = (
		daily_long.groupby(["province_group_label", "province_group_en", "grid_label", "grid_en", "time"], as_index=False)["generation_mwh"]
		.sum()
	)
	covered_province_daily = province_daily.dropna(subset=["grid_label"]).copy()
	excluded_provinces = sorted(
		set(province_daily.loc[province_daily["grid_label"].isna(), "province_group_label"].astype(str).tolist())
	)

	if render_target == RENDER_TARGET_BIPLOT_PROVINCE_IN_GRID:
		province_share_df = _build_share_frame(covered_province_daily, entity_col="province_group_label", parent_col="grid_label")
		province_share_df = province_share_df.merge(
			covered_province_daily[["grid_label", "province_group_label", "province_group_en"]].drop_duplicates(),
			on=["grid_label", "province_group_label"],
			how="left",
		)
		province_generation_df = covered_province_daily[["grid_label", "grid_en", "province_group_label", "province_group_en", "time", "generation_mwh"]].copy()
		biplot_province_plot_dir = output_root / "biplot" / "province_in_grid"
		_clear_plot_outputs(biplot_province_plot_dir)
		_plot_province_grid_generation_share_biplot_composite(
			province_generation_df=province_generation_df,
			province_share_df=province_share_df,
			output_path=biplot_province_plot_dir / "province_in_grid_biplot_composite",
		)
		return {
			"render_target": render_target,
			"granularity": granularity,
			"start": str(_parse_timestamp(start).date()),
			"end": str(_parse_timestamp(end).date()),
			"province_group_count": int(daily_long["province_group_label"].nunique()),
			"grid_count": int(province_share_df["grid_label"].nunique()),
			"output_root": str(output_root),
			"excluded_from_grid_aggregation": excluded_provinces,
		}

	if render_target == RENDER_TARGET_BIPLOT_GRID_IN_NATIONAL:
		grid_share_df = _build_grid_share_frame(covered_province_daily)
		grid_generation_df = grid_share_df[["grid_label", "grid_en", "time", "generation_mwh", "national_generation_mwh"]].copy()
		grid_share_wide = _wide_value_table(grid_share_df, index_col="time", column_col="grid_en", value_col="share")
		grid_share_wide = grid_share_wide.loc[:, [GRID_EN_MAP[label] for label in GRID_ORDER if GRID_EN_MAP[label] in grid_share_wide.columns]]
		grid_generation_wide = _wide_value_table(grid_generation_df, index_col="time", column_col="grid_en", value_col="generation_mwh")
		grid_generation_wide = grid_generation_wide.loc[:, [GRID_EN_MAP[label] for label in GRID_ORDER if GRID_EN_MAP[label] in grid_generation_wide.columns]]
		biplot_grid_plot_path = output_root / "biplot" / "grid_in_national"
		for suffix in (".png", ".html"):
			plot_file = biplot_grid_plot_path.with_suffix(suffix)
			if plot_file.exists():
				plot_file.unlink()
		# 与全量分支保持同一规格（210 mm 宽 / 子图 80 mm / 长短刻度）
		grid_national_panel_mm = 80.0
		grid_national_width_mm = 210.0
		_plot_generation_share_biplot(
			generation_wide=grid_generation_wide,
			share_wide=grid_share_wide,
			share_ylabel="Share",
			output_path=biplot_grid_plot_path,
			figsize=(20, 20 * (grid_national_panel_mm + PANEL_TOP_MM + PANEL_BOTTOM_MM) / grid_national_width_mm),
			use_fixed_rgb=True,
			width_mm=grid_national_width_mm,
			right_mm=grid_national_width_mm - 5.0,
			major_minor_year_ticks=True,
			panel_titles=GRID_NATIONAL_PANEL_TITLES,
		)
		return {
			"render_target": render_target,
			"granularity": granularity,
			"start": str(_parse_timestamp(start).date()),
			"end": str(_parse_timestamp(end).date()),
			"grid_count": int(grid_share_df["grid_label"].nunique()),
			"output_root": str(output_root),
			"excluded_from_grid_aggregation": excluded_provinces,
		}

	if render_target == RENDER_TARGET_BIPLOT_CITY_IN_PROVINCE_REPRESENTATIVES:
		city_share_df = _build_share_frame(daily_long, entity_col="city_label", parent_col="province_group_label")
		city_share_df = city_share_df.merge(
			daily_long[["province_group_label", "province_group_en", "city_label", "city_en"]].drop_duplicates(),
			on=["province_group_label", "city_label"],
			how="left",
		)
		city_generation_df = daily_long[["province_group_label", "province_group_en", "city_label", "city_en", "time", "generation_mwh"]].copy()
		biplot_city_plot_dir = output_root / "biplot" / "city_in_province"
		_clear_plot_outputs(biplot_city_plot_dir)
		_plot_city_province_representative_generation_share_biplot_composite(
			city_generation_df=city_generation_df,
			city_share_df=city_share_df,
			output_path=biplot_city_plot_dir / "city_in_province_representative_biplot_composite",
		)
		return {
			"render_target": render_target,
			"granularity": granularity,
			"start": str(_parse_timestamp(start).date()),
			"end": str(_parse_timestamp(end).date()),
			"province_group_count": int(daily_long["province_group_label"].nunique()),
			"representative_provinces": [province_en for province_en, _category_label in CITY_PROVINCE_REPRESENTATIVES],
			"output_root": str(output_root),
		}

	city_share_df = _build_share_frame(daily_long, entity_col="city_label", parent_col="province_group_label")
	province_share_df = _build_share_frame(covered_province_daily, entity_col="province_group_label", parent_col="grid_label")
	grid_share_df = _build_grid_share_frame(covered_province_daily)
	city_share_df = city_share_df.merge(
		daily_long[["province_group_label", "city_label", "city_en"]].drop_duplicates(),
		on=["province_group_label", "city_label"],
		how="left",
	)
	province_share_df = province_share_df.merge(
		covered_province_daily[["grid_label", "province_group_label", "province_group_en"]].drop_duplicates(),
		on=["grid_label", "province_group_label"],
		how="left",
	)
	city_generation_df = daily_long[["province_group_label", "province_group_en", "city_label", "city_en", "time", "generation_mwh"]].copy()
	province_generation_df = covered_province_daily[["grid_label", "grid_en", "province_group_label", "province_group_en", "time", "generation_mwh"]].copy()
	grid_generation_df = grid_share_df[["grid_label", "grid_en", "time", "generation_mwh", "national_generation_mwh"]].copy()

	_write_share_tables(
		city_share_df=city_share_df,
		province_share_df=province_share_df,
		grid_share_df=grid_share_df,
		output_root=output_root / "share",
	)
	_write_generation_tables(
		city_generation_df=city_generation_df,
		province_generation_df=province_generation_df,
		grid_generation_df=grid_generation_df,
		output_root=output_root / "generation",
	)

	share_city_plot_dir = output_root / "share" / "city_in_province"
	share_province_plot_dir = output_root / "share" / "province_in_grid"
	generation_city_plot_dir = output_root / "generation" / "city_in_province"
	generation_province_plot_dir = output_root / "generation" / "province_in_grid"
	biplot_city_plot_dir = output_root / "biplot" / "city_in_province"
	biplot_province_plot_dir = output_root / "biplot" / "province_in_grid"
	_clear_plot_outputs(
		share_city_plot_dir,
		share_province_plot_dir,
		generation_city_plot_dir,
		generation_province_plot_dir,
		biplot_city_plot_dir,
		biplot_province_plot_dir,
	)
	share_grid_plot_path = output_root / "share" / "grid_in_national"
	generation_grid_plot_path = output_root / "generation" / "grid_in_national"
	biplot_grid_plot_path = output_root / "biplot" / "grid_in_national"
	legacy_share_composite_plot_path = output_root / "share" / "grid_and_provinces_composite"
	for plot_path in (share_grid_plot_path, generation_grid_plot_path, biplot_grid_plot_path, legacy_share_composite_plot_path):
		for suffix in (".png", ".html"):
			plot_file = plot_path.with_suffix(suffix)
			if plot_file.exists():
				plot_file.unlink()

	for province_label, province_df in city_share_df.groupby("province_group_label", sort=True):
		share_wide = _wide_value_table(province_df, index_col="time", column_col="city_en", value_col="share")
		_plot_share_stacked_area(
			value_wide=share_wide,
			output_path=share_city_plot_dir / f"{_safe_slug(str(province_label))}_city_share",
			ylabel="Share of Provincial Generation",
			figsize=(16, 8.5),
		)
		province_generation_wide = _wide_value_table(
			city_generation_df[city_generation_df["province_group_label"] == province_label],
			index_col="time",
			column_col="city_en",
			value_col="generation_mwh",
		)
		_plot_generation_stacked_area(
			value_wide=province_generation_wide,
			output_path=generation_city_plot_dir / f"{_safe_slug(str(province_label))}_city_generation",
			figsize=(16, 8.5),
		)
		_plot_generation_share_biplot(
			generation_wide=province_generation_wide,
			share_wide=share_wide,
			share_ylabel="Share of Provincial Generation",
			output_path=biplot_city_plot_dir / f"{_safe_slug(str(province_label))}_city_biplot",
			figsize=(20, 8.5),
		)

	for grid_label, grid_df in province_share_df.groupby("grid_label", sort=False):
		share_wide = _wide_value_table(grid_df, index_col="time", column_col="province_group_en", value_col="share")
		_plot_share_stacked_area(
			value_wide=share_wide,
			output_path=share_province_plot_dir / f"{_safe_slug(str(grid_label))}_province_share",
			ylabel="Share of Regional Grid Generation",
			figsize=(15, 7.5),
		)
		grid_generation_wide = _wide_value_table(
			province_generation_df[province_generation_df["grid_label"] == grid_label],
			index_col="time",
			column_col="province_group_en",
			value_col="generation_mwh",
		)
		_plot_generation_stacked_area(
			value_wide=grid_generation_wide,
			output_path=generation_province_plot_dir / f"{_safe_slug(str(grid_label))}_province_generation",
			figsize=(15, 7.5),
		)

	_plot_province_grid_generation_share_biplot_composite(
		province_generation_df=province_generation_df,
		province_share_df=province_share_df,
		output_path=biplot_province_plot_dir / "province_in_grid_biplot_composite",
	)
	_plot_city_province_representative_generation_share_biplot_composite(
		city_generation_df=city_generation_df,
		city_share_df=city_share_df.merge(
			city_generation_df[["province_group_label", "province_group_en"]].drop_duplicates(),
			on="province_group_label",
			how="left",
		),
		output_path=biplot_city_plot_dir / "city_in_province_representative_biplot_composite",
	)

	grid_share_wide = _wide_value_table(grid_share_df, index_col="time", column_col="grid_en", value_col="share")
	grid_share_wide = grid_share_wide.loc[:, [GRID_EN_MAP[label] for label in GRID_ORDER if GRID_EN_MAP[label] in grid_share_wide.columns]]
	_plot_share_stacked_area(
		value_wide=grid_share_wide,
		output_path=share_grid_plot_path,
		ylabel="Share of National Generation",
		figsize=(16, 8),
	)
	grid_generation_wide = _wide_value_table(grid_generation_df, index_col="time", column_col="grid_en", value_col="generation_mwh")
	grid_generation_wide = grid_generation_wide.loc[:, [GRID_EN_MAP[label] for label in GRID_ORDER if GRID_EN_MAP[label] in grid_generation_wide.columns]]
	_plot_generation_stacked_area(
		value_wide=grid_generation_wide,
		output_path=generation_grid_plot_path,
		figsize=(16, 8),
	)
	# 用户 22:29 + 23:00 指定（只改这一张）：
	#   宽度 183 → 210 mm（A4 纸宽）；子图高 80 mm → 画布 210×105 mm；
	#   x 轴改成「有年份文字的长刻度 + 没文字的小刻度」，且首尾年份必须标出。
	grid_national_panel_mm = 80.0
	grid_national_width_mm = 210.0
	_plot_generation_share_biplot(
		generation_wide=grid_generation_wide,
		share_wide=grid_share_wide,
		share_ylabel="Share",
		output_path=biplot_grid_plot_path,
		figsize=(20, 20 * (grid_national_panel_mm + PANEL_TOP_MM + PANEL_BOTTOM_MM) / grid_national_width_mm),
		use_fixed_rgb=True,
		width_mm=grid_national_width_mm,
		right_mm=grid_national_width_mm - 5.0,
		major_minor_year_ticks=True,
		panel_titles=GRID_NATIONAL_PANEL_TITLES,
	)

	return {
		"render_target": render_target,
		"granularity": granularity,
		"start": str(_parse_timestamp(start).date()),
		"end": str(_parse_timestamp(end).date()),
		"city_file_count": int(daily_long[["province_key", "city_key"]].drop_duplicates().shape[0]),
		"province_group_count": int(daily_long["province_group_label"].nunique()),
		"grid_count": int(grid_share_df["grid_label"].nunique()),
		"output_root": str(output_root),
		"excluded_from_grid_aggregation": excluded_provinces,
	}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot daily generation share dynamics from final_gen city outputs")
	parser.add_argument("--city-root", type=Path, default=DEFAULT_CITY_ROOT, help="Path to final_gen/city root")
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory to store outputs")
	parser.add_argument("--granularity", choices=["day", "week", "month"], default=DEFAULT_PLOT_GRANULARITY, help="Plot granularity")
	parser.add_argument("--start", type=str, default=DEFAULT_START, help="Inclusive start date, e.g. 1989-01-01")
	parser.add_argument("--end", type=str, default=DEFAULT_END, help="Inclusive end date, e.g. 2023-12-31")
	parser.add_argument("--target", choices=RENDER_TARGET_CHOICES, default=RENDER_TARGET_ALL, help="Limit rendering to a specific output target")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	result = generate_generation_dynamics_plots(
		city_root=args.city_root,
		output_root=args.output_root,
		granularity=args.granularity,
		start=args.start,
		end=args.end,
		render_target=args.target,
	)
	print(result)


if __name__ == "__main__":
	main()
