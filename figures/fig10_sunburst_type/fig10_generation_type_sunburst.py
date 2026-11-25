"""Fig. 10 - 2023 年中国风电发电量旭日图（ESSD 出图规范重制版）。

旧版：plot/fig10_generation_sunburst.py。旧版的问题：

1. 字号太小 —— 画布是 1800x1400 的"逻辑像素"，旭日字号 18 px、右侧清单 9 px；
   按 ESSD 满栏宽 177 mm 印刷折算只有 5.0 pt 和 2.5 pt；
2. 版式重叠 —— 旭日圆盘居中、直径占满画高（0.754 倍画宽，右缘到 x=0.877），
   而清单从 x=0.68 起排两列，实测右半张纸 99% 的文字压在圆盘上；
3. 在本机跑不起来 —— netCDF4 引擎打不开中文路径 D:\\Data\\市级风电发电数据\\...。

本版的改动：

1. 画布按最终印刷尺寸作画（ESSD 满栏宽 177 mm，1 px = 1/96 in），文字 Arial 9 pt，
   字号即印刷实际值，不靠后期缩放；**画布是正方形，圆盘填满**（177x177、盘径 171 mm）；
2. **图上只有这一个圆盘**：不加右侧说明文字、不叠任何文字栏（用户明确禁止）；
3. 盘内标签：「9 pt 优先、放不下按扇区自动缩、缩到 5.5 pt 以下才不写」（uniformtext）。
   内环电网名 23~41 mm，要在 28.5 mm 环厚里写全只能自动缩字号——这是刻意的取舍：
   **内环必须写全**，外圈写不下就不写；
4. 城名不带省前缀（父级环已经是省），少数民族地名走国际通用转写：
   锡林郭勒 Xilingol、鄂尔多斯 Ordos、呼伦贝尔 Hulunbuir、乌鲁木齐 Ürümqi、
   吐鲁番 Turpan、和田 Hotan、那曲 Nagqu 等 18 个（内蒙古 8 个与 fig9 逐字一致）；
5. 配色用本图原来那套（深色相）；2026-09-24 试过 fig8 的马卡龙/莫兰迪，用户反馈"太浅"
   已回退，那套色板仍在 plot/figure8/fig8_generation_dynamics.py 的 MUTED_COLORS 里备查；
6. 导出 600 dpi PNG（写入 dpi 元数据，Word/LaTeX 里按 177 mm 落版）+ 同尺寸矢量
   PDF + 交互 HTML；
7. 读 nc 走 h5netcdf 引擎（中文路径下 netCDF4 引擎会报 FileNotFoundError）；
8. 脚本末尾用渲染结果实测像素自检：盘径、越界、被裁、以及「盘内每层实际写出几个标签」。

输出到脚本所在目录。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Literal, Sequence

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from PIL import Image, ImageFont
from scipy import ndimage


_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
if str(_HERE.parent / "_lib") not in sys.path:
    sys.path.insert(0, str(_HERE.parent / "_lib"))
import config_paths as _cfg  # noqa: E402
import grid_map as _grid_map  # noqa: E402  (single source of truth for 香港 / 蒙东)


# Original default: ROOT_DIR = D:\HKU PhD\wind data paper\final_gen
# (this script expects the *results root*, i.e. the parent of final_gen\city).
ROOT_DIR = _cfg.DATA_ROOT / "final_gen"
OUTPUT_DIR = _HERE / "output"
# Output stem prefix (plot_paper_v2: one file per paper figure directory).
# Original value: "fig10_generation_sunburst" / "fig10_generation_type_sunburst".
FIG_STEM = "fig10_generation_sunburst"
FIG_TYPE_STEM = "fig10_generation_type_sunburst"

# ------------------------------------------------------------------ 出图规范
FONT_PT = 9.0                    # 盘内标签字号（上限，放不下的按扇区自动缩到 LABEL_MIN_PT）
PRINT_WIDTH_MM = 177.0           # ESSD 满栏宽
EXPORT_DPI = 300                 # PNG 分辨率（用户定：300 dpi，不加码）
PX_PER_MM = 96.0 / 25.4          # 画布按印刷尺寸作画：1 mm = 96/25.4 px
FONT_PX = FONT_PT / 72.0 * 96.0
SECTOR_LINE_PT = 0.75            # 扇区白色分隔线宽
MARGIN_MM = 3.0                  # 圆盘外留的边
LABEL_MIN_PT = 8.0               # 字号下限（用户硬指标：最小 8 pt Arial，不能妥协）
                                 # → 放不下 8 pt 的外圈标签直接不写；内环靠下面两条保证写全
LABEL_AUDIT_DOWNSAMPLE = 4       # 数标签时降采样倍数（maximum_filter 在 4181² 上跑不动）
FONT_FILE_EN = str(_cfg.FONT_CANDIDATES["arial"][0])

# 研究范围排除地区（2026-10 用户裁定）: 香港 整体排除出本研究 —— 不计入南方电网，
# 不出现在任何图例/扇区/份额分母里。与 fig05/fig06/fig07 同口径。
# 单一事实来源: plot_paper_v2\_lib\grid_map.py（EXCLUDED_REGIONS）。
EXCLUDED_REGIONS = set(_grid_map.EXCLUDED_REGIONS)

GRID_TO_PROVINCES: dict[str, tuple[str, ...]] = {
	"东北电网": ("黑龙江", "吉林", "辽宁", "蒙东"),
	"华北电网": ("山西", "河北", "山东", "北京", "天津", "蒙西"),
	"西北电网": ("陕西", "甘肃", "青海", "宁夏", "新疆"),
	"西南电网": ("四川", "重庆", "西藏"),
	"华东电网": ("江苏", "安徽", "上海", "浙江", "福建"),
	"华中电网": ("河南", "湖北", "湖南", "江西"),
	"南方电网": ("云南", "贵州", "广西", "广东", "海南"),  # "香港" 已按 EXCLUDED_REGIONS 排除
}

# 内环标签的显示名：8.37 pt 下 plotly 只接受 ≤ 约 24 mm 的标签（环厚 28.5 mm），
# 超过就整条不写、断成两行也仍被丢。西南电网原名 28.8 mm，必须缩到 24 mm 以内才写得出来。
# 南方电网按用户要求固定写全称 "China Southern Power Grid"（旧的短形写法已废弃，不得改回）；
# 它本身 36.2 mm 放不进内环，由 _wrap_long_label 在空格处断成两行画进去。
GRID_LABEL_SHORT = {
	"Southwest China Grid": "Southwest China",
}

GRID_NAME_EN = {
	"东北电网": "Northeast China Grid",
	"华北电网": "North China Grid",
	"西北电网": "Northwest China Grid",
	"西南电网": "Southwest China Grid",
	"华东电网": "East China Grid",
	"华中电网": "Central China Grid",
	"南方电网": "China Southern Power Grid",
}
PROVINCE_EARLIEST_NONZERO_START = {
	"新疆": pd.Timestamp("1989-10-01"),
	"内蒙古": pd.Timestamp("1990-01-01"),
}

# 配色：沿用本图原来这套（深色相，扇区面积大也压得住）。
# 2026-09-24 试过 plot/figure8 那套马卡龙/莫兰迪（MUTED_COLORS，前 7 色给 7 个电网），
# 用户反馈"太浅了"已回退；那套色板仍在 fig8_generation_dynamics.py 里备查。
GRID_COLOR_MAP_ZH = {
	"东北电网": "#5B8FF9",
	"华北电网": "#5D7092",
	"西北电网": "#F6BD16",
	"西南电网": "#E8684A",
	"华东电网": "#6DC8EC",
	"华中电网": "#9270CA",
	"南方电网": "#269A99",
}

GRID_COLOR_MAP = {GRID_NAME_EN[name]: color for name, color in GRID_COLOR_MAP_ZH.items()}

INNER_MONGOLIA_EAST_CITIES = {
	"呼伦贝尔",
	"兴安",
	"兴安盟",
	"通辽",
	"赤峰",
}

INNER_MONGOLIA_WEST_CITIES = {
	"呼和浩特",
	"包头",
	"乌海",
	"鄂尔多斯",
	"巴彦淖尔",
	"乌兰察布",
	"锡林郭勒",
	"锡林郭勒盟",
	"阿拉善",
	"阿拉善盟",
}

# 少数民族地区地名的国际通用转写，不用汉语拼音硬拼。
# 内蒙古这 8 个与 plot/city generation ranking/fig9_city_generation_ranking.py 的 CITY_LABEL_OVERRIDES 逐字一致，全篇统一。
# 新疆/西藏以英文维基的行政区划页为准（Ürümqi、Turpan、Hotan、Kizilsu、Bayingolin、Bortala、Ili、Altay、Nagqu）。
CITY_NAME_EN = {
	"阿拉善": "Alxa",
	"巴彦淖尔": "Bayannur",
	"鄂尔多斯": "Ordos",
	"呼和浩特": "Hohhot",
	"呼伦贝尔": "Hulunbuir",
	"乌兰察布": "Ulaanqab",
	"锡林郭勒": "Xilingol",
	"兴安": "Hinggan",
	"乌鲁木齐": "Ürümqi",
	"克拉玛依": "Karamay",
	"吐鲁番": "Turpan",
	"博尔塔拉": "Bortala",
	"巴音郭楞": "Bayingolin",
	"克孜勒苏": "Kizilsu",
	"和田": "Hotan",
	"伊犁": "Ili",
	"阿勒泰": "Altay",
	"那曲": "Nagqu",
}

PROVINCE_ALIASES = {
	"北京市": "北京",
	"天津市": "天津",
	"上海市": "上海",
	"重庆市": "重庆",
	"河北省": "河北",
	"山西省": "山西",
	"辽宁省": "辽宁",
	"吉林省": "吉林",
	"黑龙江省": "黑龙江",
	"江苏省": "江苏",
	"浙江省": "浙江",
	"安徽省": "安徽",
	"福建省": "福建",
	"江西省": "江西",
	"山东省": "山东",
	"河南省": "河南",
	"湖北省": "湖北",
	"湖南省": "湖南",
	"广东省": "广东",
	"海南省": "海南",
	"四川省": "四川",
	"贵州省": "贵州",
	"云南省": "云南",
	"陕西省": "陕西",
	"甘肃省": "甘肃",
	"青海省": "青海",
	"台湾省": "台湾",
	"内蒙古自治区": "内蒙古",
	"广西壮族自治区": "广西",
	"西藏自治区": "西藏",
	"宁夏回族自治区": "宁夏",
	"新疆维吾尔自治区": "新疆",
	"香港特别行政区": "香港",
	"澳门特别行政区": "澳门",
}

PROVINCE_NAME_EN = {
	"北京": "Beijing",
	"天津": "Tianjin",
	"上海": "Shanghai",
	"重庆": "Chongqing",
	"河北": "Hebei",
	"山西": "Shanxi",
	"辽宁": "Liaoning",
	"吉林": "Jilin",
	"黑龙江": "Heilongjiang",
	"江苏": "Jiangsu",
	"浙江": "Zhejiang",
	"安徽": "Anhui",
	"福建": "Fujian",
	"江西": "Jiangxi",
	"山东": "Shandong",
	"河南": "Henan",
	"湖北": "Hubei",
	"湖南": "Hunan",
	"广东": "Guangdong",
	"海南": "Hainan",
	"四川": "Sichuan",
	"贵州": "Guizhou",
	"云南": "Yunnan",
	"陕西": "Shaanxi",
	"甘肃": "Gansu",
	"青海": "Qinghai",
	"台湾": "Taiwan",
	"内蒙古": "Inner Mongolia",
	"广西": "Guangxi",
	"西藏": "Xizang",
	"宁夏": "Ningxia",
	"新疆": "Xinjiang",
	"香港": "Hong Kong",
	"澳门": "Macao",
	"蒙东": "Inner Mongolia East",
	"蒙西": "Inner Mongolia West",
}

PROVINCE_TO_GRID = {
	province: grid
	for grid, provinces in GRID_TO_PROVINCES.items()
	for province in provinces
}

TYPE_NAME_EN = {
	"onshore": "Onshore",
	"offshore": "Offshore",
}

TYPE_NAME_ZH = {
	"onshore": "陆上",
	"offshore": "海上",
}

# 陆上 / 海上：沿用原来的深青 / 深绿一对
TYPE_COLOR_MAP = {
	"en": {"Onshore": "#2E86AB", "Offshore": "#4C9A5A"},
	"zh": {"陆上": "#2E86AB", "海上": "#4C9A5A"},
}


def normalize_text(value: object) -> str | None:
	def _cjk_ratio(text: str) -> float:
		if not text:
			return 0.0
		return sum(1 for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF) / len(text)

	def _repair_mojibake(text: str) -> str:
		candidates = [text]
		for src_enc, dst_enc in (("gbk", "utf-8"), ("gb18030", "utf-8"), ("latin-1", "utf-8"), ("cp1252", "utf-8")):
			try:
				candidates.append(text.encode(src_enc).decode(dst_enc))
			except Exception:
				continue

		best = text
		best_ratio = _cjk_ratio(text)
		for candidate in candidates[1:]:
			ratio = _cjk_ratio(candidate)
			if ratio > best_ratio + 0.2:
				best = candidate
				best_ratio = ratio
		return best

	if value is None:
		return None
	if isinstance(value, str):
		text = value.strip()
	else:
		if hasattr(value, "item"):
			value = value.item()
		text = str(value).strip()
	if not text:
		return None
	return _repair_mojibake(text)


def _format_city_slug(city_slug: str) -> str:
	parts = [part for part in city_slug.replace("-", "_").split("_") if part]
	if not parts:
		return city_slug
	return " ".join(part.capitalize() for part in parts)


def _language_suffix(language: Literal["en", "zh"]) -> str:
	return "en" if language == "en" else "zh"


def _open_dataset_lazy(path: Path) -> xr.Dataset:
	"""按顺序试各引擎；中文路径下 netCDF4 引擎会误报文件不存在，故优先 h5netcdf。"""
	resolved_path = Path(path).resolve()
	open_attempts = ({"engine": "h5netcdf"}, {"engine": "netcdf4"}, {"engine": "scipy"})
	errors: list[str] = []

	for kwargs in open_attempts:
		try:
			return xr.open_dataset(str(resolved_path), **kwargs)
		except Exception as exc:  # noqa: BLE001
			errors.append(f"engine={kwargs['engine']}: {exc}")

	raise OSError(f"failed to open NetCDF file: {resolved_path}; errors={errors}")


def _discover_city_nc_files(root_dir: Path) -> list[Path]:
	city_root = Path(root_dir) / "city"
	if not city_root.exists():
		raise FileNotFoundError(f"city directory does not exist: {city_root}")

	nc_paths = sorted(city_root.glob("*/full_year/gen.nc"))
	if not nc_paths:
		raise FileNotFoundError(f"no city generation files found under: {city_root}")
	return nc_paths


def _normalize_time_selector(time_selector: int | Sequence[int] | None) -> tuple[str, tuple[int, ...]]:
	if time_selector is None:
		return "all", ()

	if isinstance(time_selector, int):
		parts = (int(time_selector),)
	elif isinstance(time_selector, Sequence) and not isinstance(time_selector, (str, bytes)):
		parts = tuple(int(part) for part in time_selector)
	else:
		raise TypeError("time_selector must be None, an int, or a sequence of ints")

	if not 1 <= len(parts) <= 4:
		raise ValueError("time_selector must contain 1 to 4 integers")

	year = parts[0]
	if not 1900 <= year <= 2100:
		raise ValueError(f"invalid year in time_selector: {year}")
	if len(parts) >= 2 and not 1 <= parts[1] <= 12:
		raise ValueError(f"invalid month in time_selector: {parts[1]}")
	if len(parts) >= 3 and not 1 <= parts[2] <= 31:
		raise ValueError(f"invalid day in time_selector: {parts[2]}")
	if len(parts) >= 4 and not 0 <= parts[3] <= 23:
		raise ValueError(f"invalid hour in time_selector: {parts[3]}")

	selector_kind = {1: "year", 2: "month", 3: "day", 4: "hour"}[len(parts)]
	return selector_kind, parts


def _selector_file_tag(selector_kind: str, parts: tuple[int, ...]) -> str:
	if selector_kind == "all":
		return "all_1999_2023"
	if selector_kind == "year":
		return f"year_{parts[0]}"
	if selector_kind == "month":
		return f"month_{parts[0]}_{parts[1]:02d}"
	if selector_kind == "day":
		return f"day_{parts[0]}_{parts[1]:02d}_{parts[2]:02d}"
	return f"hour_{parts[0]}_{parts[1]:02d}_{parts[2]:02d}_{parts[3]:02d}"


def _canonical_province_name(raw_name: object) -> str:
	province_name = normalize_text(raw_name)
	if province_name is None:
		raise KeyError("dataset is missing province_short_name")

	province_name = PROVINCE_ALIASES.get(province_name, province_name)
	suffixes = ("维吾尔自治区", "壮族自治区", "回族自治区", "特别行政区", "自治区", "省", "市")
	for suffix in suffixes:
		if province_name.endswith(suffix):
			trimmed_name = province_name[: -len(suffix)]
			if trimmed_name:
				province_name = PROVINCE_ALIASES.get(trimmed_name, trimmed_name)
				break
	return province_name


def _canonical_city_name(raw_name: object) -> str:
	city_name = normalize_text(raw_name)
	if city_name is None:
		raise KeyError("dataset is missing city_short_name")
	return city_name


def _province_label_and_grid(province_name: str, city_name: str) -> tuple[str, str]:
	if province_name == "内蒙古":
		if city_name in INNER_MONGOLIA_EAST_CITIES:
			province_label = "蒙东"
		elif city_name in INNER_MONGOLIA_WEST_CITIES:
			province_label = "蒙西"
		else:
			raise KeyError(f"unable to classify Inner Mongolia city into 蒙东/蒙西: {city_name}")
	else:
		province_label = province_name

	grid_name = PROVINCE_TO_GRID.get(province_label)
	if grid_name is None:
		raise KeyError(f"province is not mapped to a grid: {province_label}")
	return province_label, grid_name


def _province_earliest_nonzero_start(province_name: str) -> pd.Timestamp | None:
	return PROVINCE_EARLIEST_NONZERO_START.get(province_name)


def _display_name(value: str, mapping: dict[str, str], language: Literal["en", "zh"]) -> str:
	if language == "zh":
		return value
	return mapping.get(value, value)


def _city_display_name(city_slug: str, city_cn: str, language: Literal["en", "zh"]) -> str:
	"""城名不带省前缀（父级环已经是省），少数民族地名走 CITY_NAME_EN 的国际通用转写。"""
	if language == "zh":
		return city_cn
	if city_cn in CITY_NAME_EN:
		return CITY_NAME_EN[city_cn]
	return _format_city_slug(city_slug.split("__")[-1])


def _type_display_name(type_name: str, language: Literal["en", "zh"]) -> str:
	mapping = TYPE_NAME_EN if language == "en" else TYPE_NAME_ZH
	return mapping.get(type_name, type_name)


def _font_family(language: Literal["en", "zh"]) -> str:
	if language == "zh":
		return "Microsoft YaHei, SimHei, sans-serif"
	return "Arial, Helvetica, sans-serif"


def _format_gwh(value_mwh: float) -> str:
	return f"{value_mwh / 1000.0:,.0f} GWh"


def _figure_layout() -> dict[str, object]:
	"""版式：画布按 ESSD 满栏宽做成正方形，圆盘填满，四边留 MARGIN_MM。"""
	side_mm = PRINT_WIDTH_MM
	box_mm = side_mm - 2 * MARGIN_MM
	return {
		"width_mm": side_mm,
		"height_mm": side_mm,
		"disk_box_mm": box_mm,
		"disk_diameter_mm": box_mm,
		"domain": {"x": (MARGIN_MM / side_mm, 1 - MARGIN_MM / side_mm),
		           "y": (MARGIN_MM / side_mm, 1 - MARGIN_MM / side_mm)},
	}


def _layout_report(layout: dict[str, object], df: pd.DataFrame) -> None:
	cities = len(df.groupby([column for column in ("grid", "type") if column in df.columns] + ["province", "city"]))
	print(
		f"  [版式] 画布 {layout['width_mm']:.1f} x {layout['height_mm']:.1f} mm（正方形）| "
		f"旭日盘直径 {layout['disk_diameter_mm']:.1f} mm | 标签 {FONT_PT:.1f} pt（下限 {LABEL_MIN_PT:.1f} pt）| "
		f"城市扇区 {cities} 个"
	)


def _build_sunburst_node_table(df: pd.DataFrame, language: Literal["en", "zh"]) -> pd.DataFrame:
	total_mwh = float(df["generation_mwh"].sum())
	grid_df = df.groupby("grid", as_index=False)["generation_mwh"].sum()
	province_df = df.groupby(["grid", "province"], as_index=False)["generation_mwh"].sum()

	grid_total_map = {row["grid"]: float(row["generation_mwh"]) for _, row in grid_df.iterrows()}
	province_total_map = {
		(row["grid"], row["province"]): float(row["generation_mwh"]) for _, row in province_df.iterrows()
	}

	nodes: list[dict[str, object]] = []
	for _, row in grid_df.iterrows():
		grid_name = str(row["grid"])
		value_mwh = float(row["generation_mwh"])
		nodes.append(
			{
				"id": f"grid::{grid_name}",
				"label": grid_name,
				"parent": "",
				"value": value_mwh,
				"color": grid_name,
				"hover": _build_hover_text(grid_name, "grid", value_mwh, total_mwh, None, None, language),
			}
		)

	for _, row in province_df.iterrows():
		grid_name = str(row["grid"])
		province_name = str(row["province"])
		value_mwh = float(row["generation_mwh"])
		nodes.append(
			{
				"id": f"province::{grid_name}::{province_name}",
				"label": province_name,
				"parent": f"grid::{grid_name}",
				"value": value_mwh,
				"color": grid_name,
				"hover": _build_hover_text(province_name, "province", value_mwh, total_mwh, grid_total_map.get(grid_name), None, language),
			}
		)

	for _, row in df.iterrows():
		grid_name = str(row["grid"])
		province_name = str(row["province"])
		city_name = str(row["city"])
		value_mwh = float(row["generation_mwh"])
		nodes.append(
			{
				"id": f"city::{grid_name}::{province_name}::{city_name}",
				"label": city_name,
				"parent": f"province::{grid_name}::{province_name}",
				"value": value_mwh,
				"color": grid_name,
				"hover": _build_hover_text(
					city_name,
					"city",
					value_mwh,
					total_mwh,
					grid_total_map.get(grid_name),
					province_total_map.get((grid_name, province_name)),
					language,
				),
			}
		)

	return pd.DataFrame(nodes)


def _share_lines(
	level: Literal["grid", "province", "city", "type"],
	value_mwh: float,
	total_mwh: float,
	grid_mwh: float | None,
	province_mwh: float | None,
	type_mwh: float | None,
	language: Literal["en", "zh"],
) -> list[str]:
	def pct(numerator: float, denominator: float | None) -> str:
		return f"{0.0 if not denominator else numerator / denominator:.2%}"

	if language == "zh":
		names = {"grid": "占全国比例", "province": "占电网比例", "city": "占本省比例", "type": "占全国比例"}
		scope = {"grid": total_mwh, "province": grid_mwh, "city": province_mwh, "type": total_mwh}
		lines = [f"{names[level]}: {pct(value_mwh, scope[level])}"]
		if level in {"province", "city"}:
			lines.append(f"占全国比例: {pct(value_mwh, total_mwh)}")
		if level == "city":
			lines.insert(1, f"占电网比例: {pct(value_mwh, grid_mwh)}")
		return lines

	names = {
		"grid": "Share of National Total",
		"province": "Share of Grid",
		"city": "Share of Province",
		"type": "Share of National Total",
	}
	scope = {"grid": total_mwh, "province": grid_mwh, "city": province_mwh, "type": total_mwh}
	lines = [f"{names[level]}: {pct(value_mwh, scope[level])}"]
	if level in {"province", "city"}:
		lines.append(f"Share of National Total: {pct(value_mwh, total_mwh)}")
	if level == "city":
		lines.insert(1, f"Share of Grid: {pct(value_mwh, grid_mwh)}")
	return lines


def _build_hover_text(
	label: str,
	level: Literal["grid", "province", "city", "type"],
	value_mwh: float,
	total_mwh: float,
	grid_mwh: float | None,
	province_mwh: float | None,
	language: Literal["en", "zh"],
) -> str:
	head = "发电量" if language == "zh" else "Generation"
	lines = [f"<b>{label}</b>", f"{head}: {_format_gwh(value_mwh)}"]
	if level == "type":
		return "<br>".join(lines + _share_lines("type", value_mwh, total_mwh, grid_mwh, province_mwh, None, language))
	return "<br>".join(lines + _share_lines(level, value_mwh, total_mwh, grid_mwh, province_mwh, None, language))


def _build_type_sunburst_node_table(df: pd.DataFrame, language: Literal["en", "zh"]) -> pd.DataFrame:
	total_mwh = float(df["generation_mwh"].sum())
	type_df = df.groupby("type", as_index=False)["generation_mwh"].sum()
	province_df = df.groupby(["type", "province"], as_index=False)["generation_mwh"].sum()

	type_total_map = {row["type"]: float(row["generation_mwh"]) for _, row in type_df.iterrows()}
	province_total_map = {
		(row["type"], row["province"]): float(row["generation_mwh"]) for _, row in province_df.iterrows()
	}

	nodes: list[dict[str, object]] = []
	for _, row in type_df.iterrows():
		type_name = str(row["type"])
		value_mwh = float(row["generation_mwh"])
		nodes.append(
			{
				"id": f"type::{type_name}",
				"label": type_name,
				"parent": "",
				"value": value_mwh,
				"color": type_name,
				"hover": _build_hover_text(type_name, "type", value_mwh, total_mwh, type_total_map.get(type_name), None, language),
			}
		)

	for _, row in province_df.iterrows():
		type_name = str(row["type"])
		province_name = str(row["province"])
		value_mwh = float(row["generation_mwh"])
		nodes.append(
			{
				"id": f"province::{type_name}::{province_name}",
				"label": province_name,
				"parent": f"type::{type_name}",
				"value": value_mwh,
				"color": type_name,
				"hover": _build_hover_text(
					province_name,
					"province",
					value_mwh,
					total_mwh,
					type_total_map.get(type_name),
					None,
					language,
				),
			}
		)

	for _, row in df.iterrows():
		type_name = str(row["type"])
		province_name = str(row["province"])
		city_name = str(row["city"])
		value_mwh = float(row["generation_mwh"])
		nodes.append(
			{
				"id": f"city::{type_name}::{province_name}::{city_name}",
				"label": city_name,
				"parent": f"province::{type_name}::{province_name}",
				"value": value_mwh,
				"color": type_name,
				"hover": _build_hover_text(
					city_name,
					"city",
					value_mwh,
					total_mwh,
					type_total_map.get(type_name),
					province_total_map.get((type_name, province_name)),
					language,
				),
			}
		)

	return pd.DataFrame(nodes)


def _sum_monthly_generation(
	ds: xr.Dataset,
	variable_prefix: str,
	selector_kind: str,
	parts: tuple[int, ...],
	province_name: str,
) -> float | None:
	monthly_var_name = f"{variable_prefix}_monthly_gen"
	if monthly_var_name not in ds.data_vars or "date" not in ds.coords:
		return None

	date_index = pd.DatetimeIndex(pd.to_datetime(ds["date"].values))
	values = np.asarray(ds[monthly_var_name].values, dtype=float)

	if selector_kind == "all":
		mask = np.ones(len(date_index), dtype=bool)
	elif selector_kind == "year":
		mask = date_index.year == parts[0]
	elif selector_kind == "month":
		mask = (date_index.year == parts[0]) & (date_index.month == parts[1])
	else:
		return None

	earliest_start = _province_earliest_nonzero_start(province_name)
	if earliest_start is not None:
		mask = mask & (date_index >= earliest_start)

	return float(np.nansum(values[mask]))


def _sum_hourly_generation(
	ds: xr.Dataset,
	variable_prefix: str,
	selector_kind: str,
	parts: tuple[int, ...],
	province_name: str,
) -> float:
	hourly_var_name = f"{variable_prefix}_gen"
	if hourly_var_name not in ds.data_vars or "time" not in ds.coords:
		raise KeyError(f"dataset does not contain {hourly_var_name} with time coordinate")

	time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
	values = np.asarray(ds[hourly_var_name].values, dtype=float)

	if selector_kind == "all":
		mask = np.ones(len(time_index), dtype=bool)
	elif selector_kind == "year":
		mask = time_index.year == parts[0]
	elif selector_kind == "month":
		mask = (time_index.year == parts[0]) & (time_index.month == parts[1])
	elif selector_kind == "day":
		mask = (
			(time_index.year == parts[0])
			& (time_index.month == parts[1])
			& (time_index.day == parts[2])
		)
	else:
		mask = (
			(time_index.year == parts[0])
			& (time_index.month == parts[1])
			& (time_index.day == parts[2])
			& (time_index.hour == parts[3])
		)

	earliest_start = _province_earliest_nonzero_start(province_name)
	if earliest_start is not None:
		mask = mask & (time_index >= earliest_start)

	return float(np.nansum(values[mask]))


def _extract_generation_value(
	ds: xr.Dataset,
	selector_kind: str,
	parts: tuple[int, ...],
	variable_prefix: str,
	province_name: str,
) -> float:
	monthly_value = _sum_monthly_generation(ds, variable_prefix, selector_kind, parts, province_name)
	if monthly_value is not None:
		return monthly_value
	return _sum_hourly_generation(ds, variable_prefix, selector_kind, parts, province_name)


def _extract_generation_components(
	ds: xr.Dataset,
	selector_kind: str,
	parts: tuple[int, ...],
	variable_prefix: str,
	province_name: str,
	language: Literal["en", "zh"],
) -> list[tuple[str, float]]:
	components: list[tuple[str, float]] = []
	if variable_prefix in {"onshore", "offshore"}:
		component_prefixes = (variable_prefix,)
	else:
		component_prefixes = ("onshore", "offshore")

	for component_prefix in component_prefixes:
		monthly_var_name = f"{component_prefix}_monthly_gen"
		hourly_var_name = f"{component_prefix}_gen"
		if monthly_var_name not in ds.data_vars and hourly_var_name not in ds.data_vars:
			continue
		value = _extract_generation_value(ds, selector_kind, parts, component_prefix, province_name)
		if value <= 0:
			continue
		components.append((_type_display_name(component_prefix, language), value))
	return components


def build_generation_sunburst_dataframe(
	root_dir: str | Path = ROOT_DIR,
	time_selector: int | Sequence[int] | None = None,
	variable_prefix: str = "total",
	language: Literal["en", "zh"] = "en",
	min_value: float = 0.0,
) -> pd.DataFrame:
	selector_kind, selector_parts = _normalize_time_selector(time_selector)
	records: list[dict[str, object]] = []

	for nc_path in _discover_city_nc_files(Path(root_dir)):
		with _open_dataset_lazy(nc_path) as ds:
			province_name = _canonical_province_name(ds.attrs.get("province_short_name"))
			if province_name in EXCLUDED_REGIONS:
				# 研究范围过滤: 香港 不入本研究（与 fig05/fig07 同口径）。
				continue
			city_name = _canonical_city_name(ds.attrs.get("city_short_name"))
			province_label, grid_name = _province_label_and_grid(province_name, city_name)
			generation_value = _extract_generation_value(ds, selector_kind, selector_parts, variable_prefix, province_name)
			city_label = _format_city_slug(nc_path.parent.parent.name)

		if generation_value <= min_value:
			continue

		records.append(
			{
				"grid": _display_name(grid_name, GRID_NAME_EN, language),
				"province": _display_name(province_label, PROVINCE_NAME_EN, language),
				"city": _city_display_name(nc_path.parent.parent.name, city_name, language),
				"city_cn": city_name,
				"city_en": city_label,
				"generation_mwh": generation_value,
			}
		)

	if not records:
		raise ValueError("no city generation data matched the requested selector")

	df = pd.DataFrame(records)
	df = df.sort_values(["grid", "province", "generation_mwh", "city"], ascending=[True, True, False, True])
	df.reset_index(drop=True, inplace=True)
	return df


def build_generation_type_sunburst_dataframe(
	root_dir: str | Path = ROOT_DIR,
	time_selector: int | Sequence[int] | None = None,
	variable_prefix: str = "total",
	language: Literal["en", "zh"] = "en",
	min_value: float = 0.0,
) -> pd.DataFrame:
	selector_kind, selector_parts = _normalize_time_selector(time_selector)
	records: list[dict[str, object]] = []

	for nc_path in _discover_city_nc_files(Path(root_dir)):
		with _open_dataset_lazy(nc_path) as ds:
			province_name = _canonical_province_name(ds.attrs.get("province_short_name"))
			if province_name in EXCLUDED_REGIONS:
				# 研究范围过滤: 香港 不入本研究（与 fig05/fig07 同口径）。
				continue
			city_name = _canonical_city_name(ds.attrs.get("city_short_name"))
			province_label, grid_name = _province_label_and_grid(province_name, city_name)
			generation_components = _extract_generation_components(
				ds,
				selector_kind,
				selector_parts,
				variable_prefix,
				province_name,
				language,
			)

		for type_label, generation_value in generation_components:
			if generation_value <= min_value:
				continue
			records.append(
				{
					"province": _display_name(province_label, PROVINCE_NAME_EN, language),
					"city": _city_display_name(nc_path.parent.parent.name, city_name, language),
					"type": type_label,
					"generation_mwh": generation_value,
				}
			)

	if not records:
		raise ValueError("no city generation data matched the requested selector")

	df = pd.DataFrame(records)
	df = df.sort_values(["type", "province", "generation_mwh", "city"], ascending=[True, True, False, True])
	df.reset_index(drop=True, inplace=True)
	return df


def _label_width_mm(text: str, pt: float, language: Literal["en", "zh"] = "en") -> float:
	"""量一行文字在给定字号下的印刷宽度（mm），用来判断内环放不放得下。"""
	path = str(_cfg.FONT_CANDIDATES["arial"][0]) if language == "en" else str(_cfg.FONT_CANDIDATES["msyh"][0])
	font = ImageFont.truetype(path, round(pt / 72.0 * 96.0 * 8))
	return font.getlength(text) / 8.0 / 96.0 * 25.4


def _wrap_long_label(label: str, ring_mm: float, language: Literal["en", "zh"]) -> str:
	"""内环某些电网名比环厚还长，plotly 会整条不写；在空格处断成两行（字面不变）就能画进去。

	断点选「使最长那行最短」的位置；两行都还放不下就原样返回。
	"""
	if _label_width_mm(label, FONT_PT, language) <= ring_mm * 0.85:
		return label
	parts = label.split(" ")
	best: tuple[float, str] | None = None
	for index in range(1, len(parts)):
		first, second = " ".join(parts[:index]), " ".join(parts[index:])
		longest = max(_label_width_mm(first, FONT_PT, language), _label_width_mm(second, FONT_PT, language))
		if best is None or longest < best[0]:
			best = (longest, f"{first}<br>{second}")
	if best and best[0] <= ring_mm * 0.85:
		return best[1]
	return label


def _figure_labels(node_df: pd.DataFrame, layout: dict[str, object], language: Literal["en", "zh"]) -> list[str]:
	"""出图和自检共用的「图上实际显示的标签」。

	第 1 层：先套短名（plotly 只接受 ≤ 约 24 mm 的标签），再给放不下的断两行。
	全部文字最终都由 plotly 自己渲染 —— 图上不加任何 annotation。
	"""
	ring_mm = layout["disk_diameter_mm"] / 6.0
	labels = node_df["label"].tolist()
	levels = node_df["id"].str.count("::")
	for index, level in enumerate(levels):
		if level == 1:
			label = GRID_LABEL_SHORT.get(labels[index], labels[index]) if language == "en" else labels[index]
			labels[index] = _wrap_long_label(label, ring_mm, language)
	return labels


def _sunburst_figure(node_df: pd.DataFrame, colors: list[str], layout: dict[str, object], language: Literal["en", "zh"]) -> go.Figure:
	labels = _figure_labels(node_df, layout, language)

	fig = go.Figure(
		go.Sunburst(
			ids=node_df["id"],
			labels=labels,
			parents=node_df["parent"],
			values=node_df["value"],
			branchvalues="total",
			insidetextorientation="radial",
			textinfo="label",
			# ★ plotly 默认给叶子节点（外圈城市扇区）opacity=0.7：叠在白底上等于掺 30% 白，
			# 外环发灰、像蒙了一层半透明板浮在图上。必须关掉。
			leaf={"opacity": 1.0},
			customdata=[[text] for text in node_df["hover"].tolist()],
			hovertemplate="%{customdata[0]}<extra></extra>",
			marker={
				"colors": colors,
				"line": {"color": "white", "width": SECTOR_LINE_PT / 72.0 * 96.0},
			},
			insidetextfont={"size": FONT_PX},
			outsidetextfont={"size": FONT_PX},
			domain={"x": list(layout["domain"]["x"]), "y": list(layout["domain"]["y"])},
		)
	)
	fig.update_layout(
		margin={"t": 0, "l": 0, "r": 0, "b": 0},
		font={"family": _font_family(language), "size": FONT_PX},
		paper_bgcolor="white",
		uniformtext={"minsize": LABEL_MIN_PT / 72.0 * 96.0, "mode": "hide"},
	)
	return fig


def _set_png_dpi(path: Path, dpi: int) -> None:
	"""写入 dpi 元数据，Word/LaTeX 里才会按 177 mm 落版，而不是按 96 dpi 撑满整页。"""
	image = Image.open(path)
	image.save(path, dpi=(dpi, dpi))


def _canvas_px(layout: dict[str, object]) -> tuple[int, int]:
	return round(layout["width_mm"] * PX_PER_MM), round(layout["height_mm"] * PX_PER_MM)


def _report_label_fonts(fig, node_df: pd.DataFrame, layout: dict[str, object],
                        language: Literal["en", "zh"]) -> None:
	"""把图导成 SVG，逐个 text 元素量「实际字号 × 内容」——这是「全图 ≥ 8 pt」的唯一硬证据。

	plotly 缩字是靠 transform 的 scale()，所以实际字号 = font-size × scale；
	像素法（数墨团/数黑字）会漏掉深底白字、又会把扇区白缝算成字，不足为凭。
	"""
	width_px, height_px = _canvas_px(layout)
	with tempfile.TemporaryDirectory() as tmp:
		svg_path = Path(tmp) / "labels.svg"
		fig.write_image(svg_path, format="svg", width=width_px, height=height_px)
		svg = svg_path.read_text(encoding="utf-8", errors="ignore")

	sizes: list[float] = []
	families: set[str] = set()
	drawn: set[str] = set()
	for element in re.finditer(r"<text([^>]*)>(.*?)</text>", svg, re.S):
		attributes, body = element.group(1), element.group(2)
		scale = re.search(r"scale\(([0-9.]+)\)", attributes)
		size = re.search(r"font-size: ([0-9.]+)px", attributes)
		family = re.search(r"font-family: ([^;\"]+)", attributes)
		if size:
			sizes.append(float(size.group(1)) * (float(scale.group(1)) if scale else 1.0) * 72.0 / 96.0)
		if family:
			families.add(family.group(1).split(",")[0].strip())
		lines = [line.strip() for line in re.sub(r"<[^>]*>", "\n", body).split("\n") if line.strip()]
		if lines:
			drawn.add(" ".join(" ".join(lines).replace("<br>", " ").split()))   # 断两行的合起来才算一句

	levels = node_df["id"].str.count("::")
	shown = _figure_labels(node_df, layout, language)
	counts: list[str] = []
	for level in sorted(levels.unique()):
		indexes = [i for i, value in enumerate(levels) if value == level]
		written = sum(1 for i in indexes if " ".join(shown[i].replace("<br>", " ").split()) in drawn)
		counts.append(f"第{level}层 {written}/{len(indexes)}")
	sizes.sort()
	if sizes:
		print(f"    · 标签自查：{' | '.join(counts)}")
		print(f"    · 实际字号 {sizes[0]:.2f}~{sizes[-1]:.2f} pt（中位 {sizes[len(sizes) // 2]:.2f}）"
		      f" | 字体 {'/'.join(sorted(families))} | 低于 {LABEL_MIN_PT:.1f} pt 的 {sum(1 for s in sizes if s < LABEL_MIN_PT - 0.05)} 个"
		      f" | 文本元素 {len(sizes)} 个")


def _audit_render(png_path: Path, layout: dict[str, object]) -> None:
	"""量渲染结果：盘的位置/大小对不对、有没有出画布、每层写出了几个标签。"""
	array = np.asarray(Image.open(png_path).convert("RGB")).astype(np.int16)
	height, width = array.shape[:2]
	scale = width / layout["width_mm"]           # 图内像素 / 印刷 mm
	mx = array.max(axis=2)
	mn = array.min(axis=2)
	white = mn >= 238
	colored = (mx - mn > 40) & ~white
	dark = (mx < 130) & (mx - mn <= 40) & ~white

	disk_colored = colored
	colored_x = np.nonzero(disk_colored)[1]
	disk_right_mm = (colored_x.max() + 1) / scale
	disk_diameter_mm = (colored_x.max() - colored_x.min() + 1) / scale
	ink = np.nonzero(dark | colored)
	touch = ink[0].min() <= 1 or ink[0].max() >= height - 2 or ink[1].min() <= 1 or ink[1].max() >= width - 2

	checks = [
		("圆盘直径与反解值一致", abs(disk_diameter_mm - layout["disk_diameter_mm"]) <= 1.0,
		 f"实测 {disk_diameter_mm:.1f} mm vs 反解 {layout['disk_diameter_mm']:.1f} mm"),
		("圆盘落在自己的区间内", disk_right_mm <= layout["domain"]["x"][1] * layout["width_mm"] + 0.3,
		 f"盘右缘 {disk_right_mm:.1f} mm ≤ 区间右界 {layout['domain']['x'][1] * layout['width_mm']:.1f} mm"),
		("没有内容被画布裁掉", not touch,
		 f"纵向墨迹 {(ink[0].max() - ink[0].min() + 1) / scale:.1f} mm / {(height - 1) / scale:.1f} mm"),
	]
	print(f"    · 图内墨迹 {len(ink[0]):,} px（占画布 {len(ink[0]) / (width * height):.2%}）| 文件 {png_path.name}")

	for name, ok, detail in checks:
		print(f"    {'✓' if ok else '✗'} {name}：{detail}")


def _export(fig: go.Figure, output_html, output_png, output_pdf, default_stem: str, layout: dict[str, object],
            node_df: pd.DataFrame | None = None, language: Literal["en", "zh"] = "en") -> None:
	width_px, height_px = _canvas_px(layout)
	targets = ((".html", output_html), (".png", output_png), (".pdf", output_pdf))
	paths: dict[str, Path] = {}
	for suffix, target in targets:
		if target is None:
			continue
		path = Path(target)
		if path.suffix.lower() != suffix:
			path = path / f"{default_stem}{suffix}"
		path.parent.mkdir(parents=True, exist_ok=True)
		paths[suffix] = path

	if ".html" in paths:
		fig.write_html(paths[".html"])
	if ".png" in paths:
		fig.write_image(paths[".png"], format="png", width=width_px, height=height_px, scale=EXPORT_DPI / 96.0)
		_set_png_dpi(paths[".png"], EXPORT_DPI)
	if ".pdf" in paths:
		fig.write_image(paths[".pdf"], format="pdf", width=width_px, height=height_px)

	for suffix in (".png", ".pdf"):
		if suffix in paths:
			print(f"    写出 {paths[suffix].name}（{paths[suffix].stat().st_size / 1024:,.0f} KB）")
	if ".png" in paths:
		_audit_render(paths[".png"], layout)
	_report_label_fonts(fig, node_df, layout, language)



def plot_generation_sunburst(
	root_dir: str | Path = ROOT_DIR,
	time_selector: int | Sequence[int] | None = None,
	variable_prefix: str = "total",
	language: Literal["en", "zh"] = "en",
	min_value: float = 0.0,
	output_html: str | Path | None = None,
	output_png: str | Path | None = None,
	output_pdf: str | Path | None = None,
	show: bool = True,
):
	selector_kind, selector_parts = _normalize_time_selector(time_selector)
	file_tag = _selector_file_tag(selector_kind, selector_parts)

	df = build_generation_sunburst_dataframe(
		root_dir=root_dir,
		time_selector=time_selector,
		variable_prefix=variable_prefix,
		language=language,
		min_value=min_value,
	)
	layout = _figure_layout()
	_layout_report(layout, df)

	node_df = _build_sunburst_node_table(df, language=language)
	color_map = GRID_COLOR_MAP if language == "en" else GRID_COLOR_MAP_ZH
	fig = _sunburst_figure(node_df, [color_map[str(color)] for color in node_df["color"].tolist()], layout, language)

	stem = f"{FIG_STEM}_{_language_suffix(language)}_{file_tag}"
	_export(fig, output_html, output_png, output_pdf, stem, layout, node_df, language)
	if show:
		fig.show()
	return fig


def plot_generation_type_sunburst(
	root_dir: str | Path = ROOT_DIR,
	time_selector: int | Sequence[int] | None = None,
	variable_prefix: str = "total",
	language: Literal["en", "zh"] = "en",
	min_value: float = 0.0,
	output_html: str | Path | None = None,
	output_png: str | Path | None = None,
	output_pdf: str | Path | None = None,
	show: bool = True,
):
	selector_kind, selector_parts = _normalize_time_selector(time_selector)
	file_tag = _selector_file_tag(selector_kind, selector_parts)

	df = build_generation_type_sunburst_dataframe(
		root_dir=root_dir,
		time_selector=time_selector,
		variable_prefix=variable_prefix,
		language=language,
		min_value=min_value,
	)
	layout = _figure_layout()
	_layout_report(layout, df)

	node_df = _build_type_sunburst_node_table(df, language=language)
	type_colors = TYPE_COLOR_MAP[language]
	fig = _sunburst_figure(
		node_df,
		[type_colors.get(str(color), "#808080") for color in node_df["color"].tolist()],
		layout,
		language,
	)

	stem = f"{FIG_TYPE_STEM}_{_language_suffix(language)}_{file_tag}"
	_export(fig, output_html, output_png, output_pdf, stem, layout, node_df, language)
	if show:
		fig.show()
	return fig


if __name__ == "__main__":
	# plot_paper_v2: default outputs go to <script dir>/output/; pass --html to also
	# write the (large, ~4.7 MB) interactive HTML next to the PNG/PDF.
	import argparse

	_parser = argparse.ArgumentParser(description="Fig. 10 (technology-type sunburst)")
	_parser.add_argument("--year", type=int, default=2023, help="Year to visualise (single year selector).")
	_parser.add_argument("--language", choices=["en", "zh"], default="en")
	_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
	_parser.add_argument("--html", action="store_true", help="Also export the interactive HTML.")
	_args = _parser.parse_args()

	_args.output_dir.mkdir(parents=True, exist_ok=True)
	_html = _args.output_dir if _args.html else None
	plot_generation_type_sunburst(
		time_selector=_args.year,
		language=_args.language,
		output_html=_html,
		output_png=_args.output_dir,
		output_pdf=_args.output_dir,
		show=False,
	)
