"""Fig. 12 - 各省分季节 full-data optimal (k, b) 搜索图（4x2 面板）。

出图口径对齐 fig.13 / fig.14（ESSD）：画布按印刷尺寸作画（满栏宽 177 mm）、
全图文字统一 Arial 8 pt。
配色 = 单一色系顺序色带（色相锁死，只走明度）+ 同色相深一档的 optimal path 线条，
三条候选见 PALETTES，用 --palette 切换。
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import matplotlib

if not os.environ.get("MPLBACKEND"):
	matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "_lib") not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT / "_lib"))

import config_paths as _cfg  # noqa: E402

# Vendored copy of the pipeline helper module (byte-identical to
# D:\HKU PhD\wind data paper\CalculationCode\find_optimal_kb.py, which lived in
# the `code` package that does not ship with the plot directory).
from find_optimal_kb import (  # noqa: E402
	_draw_path,
	_style_axis,
	find_min_wind_on_gen_op,
	fit_surface_from_dataframe,
	get_optimal_path_from_dataframe,
	interpolate_smape_value,
	load_smape_csv,
	should_force_spawn,
)


FULL_GEN_PATTERN = re.compile(r"^(?P<prov>.+?)_(?P<season>[^_]+)_full_gen_smape\.csv$")
# plot_paper_v2 note: the GBK-mojibake names on disk (内蒙古 / 宁夏 / 黑龙江) have been
# repaired by rename -- 24 files, sha256-verified against the remote originals, some
# of which had also lost the '_' separator.  Every name now reads
# <Chinese province>_<season>_<suffix>, so the regex above matches all of them:
# nothing is reported as skipped / unrecognized_filename, and fig04 consumes
# 120/120 gen CSVs + 132 wind CSVs across 30 provinces with skipped_count = 0.
# PROVINCE_TOKEN_FIXES / canonical_province_name() below stay as a permanent guard.
# See README "Known data problems".
SEASON_ORDER = ("spring", "summer", "autumn", "winter")
SEASON_LABELS = {
	"spring": "Spring",
	"summer": "Summer",
	"autumn": "Autumn",
	"winter": "Winter",
}
PANEL_TITLES = (
	"Full Generation SMAPE Surface",
	"Wind Speed SMAPE Surface & Optimal (k, b)",
)

DEFAULT_GEN_DIR = _cfg.PROV_GEN_FULL_SMAPE_DIR
DEFAULT_WIND_DIR = _cfg.PROV_W_SMAPE_ENUMS_DIR
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ---------------------------------------------------------------- 输出文件名规范化（只影响文件名的拼接，不碰任何数值逻辑）
# 上游 CSV 文件名曾被 GBK 误码污染过（青海 -> 闈掓捣），目前绝大部分已还原为正确中文。
# 极少数仍未还原（宁夏 -> 瀹佸\ue634，内蒙古 -> 鍐呰挋鍙\ue605）且丢失了 "_" 分隔符，
# 对这些名字正则本就匹配不上（会被记为 skipped）。为了不让误码名再落到成图上，
# 输出名统一走 "读取到的 token -> 正确中文" 的对照表；表中没有的 token 原样使用。
PROVINCE_TOKEN_FIXES = {
	# 青海 的 GBK 误码写法
	"闈掓捣": "青海",
	# 宁夏 的 GBK 误码写法（部分字节无法还原，含私用区字符 U+E634）
	"瀹佸\ue634": "宁夏",
	"瀹佸": "宁夏",
	# 内蒙古 的 GBK 误码写法（含私用区字符 U+E605），且丢失 "_" 分隔符
	"鍐呰挋鍙\ue605": "内蒙古",
	"鍐呰挋鍙": "内蒙古",
	# 黑龙江 曾出现的误码写法
	"榛戦緳姹焈": "黑龙江",
}


def canonical_province_name(token: str) -> str:
	"""把从文件名读到的省份 token 规范成正确中文（仅用于输出文件名）。"""
	text = str(token).strip()
	if text in PROVINCE_TOKEN_FIXES:
		return PROVINCE_TOKEN_FIXES[text]
	for bad, good in PROVINCE_TOKEN_FIXES.items():
		if text.startswith(bad) and len(text) > len(bad):
			# 误码把 "_season" 的分隔符吞进了省份 token（如 鍐呰挋鍙\ue605autumn）：
			# 只保留省份，后面的季节由 filename -> season 字段单独提供。
			return good
	return text


# ---------------------------------------------------------------- 出图规范（对齐 fig.13 / fig.14 的 ESSD 口径）
FONT_FAMILY = "Arial"
FONT_SIZE = 8.0
TICK_FONT_SIZE = FONT_SIZE
LEGEND_FONT_SIZE = FONT_SIZE
TITLE_FONT_SIZE = FONT_SIZE

MM_PER_INCH = 25.4
OUTPUT_WIDTH_MM = 177.0           # 交付图最终宽度 = ESSD 满栏版心；画布即印刷尺寸，8 pt 就是印刷 8 pt
CANVAS_WIDTH_MM = 168.32          # 画布宽；加 bbox_inches="tight" 后成品宽恰为 OUTPUT_WIDTH_MM（渲染实测标定）
BASE_FIGSIZE = (13.5, 22.0)       # 只用来锁住原来的面板长宽比
FIGSIZE = (
	CANVAS_WIDTH_MM / MM_PER_INCH,
	CANVAS_WIDTH_MM / MM_PER_INCH * BASE_FIGSIZE[1] / BASE_FIGSIZE[0],
)

# SMAPE 顺序色带，三种写法：
#  a) matplotlib 常规顺序色系名（"YlGnBu" / "Blues" / "viridis" …）；
#  b) hex 列表 = 自配色带；
#  c) ("desat", 色系名, 比例) = 把该色系与**同明度的灰**混合，明度保持单调、只压彩度。
# 线条：常规色系占满色域，只能取无彩色近黑 #1A1A1A + 白描边；自配单色相时取同色相深一档。
class Palette(NamedTuple):
	label: str
	ramp: tuple | str
	line: str


def _desaturate(colors, amount: float):
	arr = np.asarray(colors, dtype=float)[:, :3]
	luma = (arr * np.array([0.2126, 0.7152, 0.0722])).sum(axis=1, keepdims=True)
	return arr * (1.0 - amount) + np.repeat(luma, 3, axis=1) * amount


PALETTES = {
	"ylgnbu_m40": Palette("YlGnBu 去饱和40%", ("desat", "YlGnBu", 0.40), "#1A1A1A"),
	"gnbu_m40": Palette("GnBu 去饱和40%", ("desat", "GnBu", 0.40), "#1A1A1A"),
	"blues_m30": Palette("Blues 去饱和30%", ("desat", "Blues", 0.30), "#1A1A1A"),
	"viridis": Palette("viridis · 对照", "viridis", "#1A1A1A"),
	"blues": Palette("Blues · 冷·常规", "Blues", "#1A1A1A"),
	"ylgnbu": Palette("YlGnBu · 冷·常规", "YlGnBu", "#1A1A1A"),
	"cividis": Palette("cividis · 冷·色盲友好", "cividis", "#1A1A1A"),
	"ylorrd": Palette("YlOrRd · 暖·常规", "YlOrRd", "#1A1A1A"),
	"blue": Palette(
		"雾蓝灰 · 单色相",
		("#EEF4FB", "#D0DEED", "#B4C9DE", "#9AB6D1", "#82A4C3", "#6A92B5"),
		"#31658A",
	),
	"green": Palette(
		"青灰绿 · 单色相",
		("#F1F5ED", "#D7E0CF", "#BECAB4", "#A8B89B", "#94A685", "#7F956F"),
		"#50673F",
	),
	"taupe": Palette(
		"暖灰茶 · 单色相",
		("#FBF2EB", "#EBDACB", "#D9C3AE", "#CAAE95", "#BA9B7D", "#AA8867"),
		"#7B5A37",
	),
}
DEFAULT_PALETTE = "ylgnbu_m40"


def build_smape_colormap(palette: Palette) -> LinearSegmentedColormap:
	ramp = palette.ramp
	if isinstance(ramp, str):
		return plt.get_cmap(ramp)
	if len(ramp) == 3 and ramp[0] == "desat":
		_, name, amount = ramp
		base = plt.get_cmap(name)(np.linspace(0.0, 1.0, 256))
		return LinearSegmentedColormap.from_list(f"{name}_muted", _desaturate(base, float(amount)))
	return LinearSegmentedColormap.from_list("smape_custom", ramp)


OPTIMAL_PATH_CASING_COLOR = "#FFFFFF"
OPTIMAL_PATH_LINEWIDTH = 1.1
OPTIMAL_MARKER_FACE = "#FFFFFF"
OPTIMAL_MARKER_SIZE = 26.0
BOX_EDGE_COLOR = "#C9C9C9"
SMAPE_VMIN = 0.0
SMAPE_VMAX = 100.0

plt.rcParams["font.family"] = ["Arial", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def _resolve_cjk_font() -> FontProperties | None:
	font_candidates = [
		Path("C:/Windows/Fonts/msyh.ttc"),
		Path("C:/Windows/Fonts/msyhbd.ttc"),
		Path("C:/Windows/Fonts/simhei.ttf"),
		Path("C:/Windows/Fonts/simsun.ttc"),
	]
	for font_path in font_candidates:
		if font_path.exists():
			return FontProperties(fname=str(font_path))
	return None


CJK_FONT_PROP = _resolve_cjk_font()


def parse_full_gen_filename(gen_csv: Path) -> dict[str, str] | None:
	match = FULL_GEN_PATTERN.match(gen_csv.name)
	if match is None:
		return None
	return {
		"prov": match.group("prov"),
		"season": match.group("season"),
	}


def _add_r2_label(ax, r2: float) -> None:
	ax.text(
		0.03,
		0.97,
		f"R\u00b2={r2:.4f}",
		transform=ax.transAxes,
		fontsize=LEGEND_FONT_SIZE,
		fontfamily=FONT_FAMILY,
		va="top",
		ha="left",
		bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.92, edgecolor=BOX_EDGE_COLOR, linewidth=0.5),
		zorder=30,
	)


def _add_panel_title_below(ax, title: str) -> None:
	ax.text(
		0.5,
		-0.14,
		title,
		transform=ax.transAxes,
		fontsize=TITLE_FONT_SIZE,
		fontfamily="Arial",
		va="top",
		ha="center",
	)


def _add_column_titles_bottom(fig) -> None:
	fig.text(
		0.32,
		0.04,
		PANEL_TITLES[0],
		fontsize=TITLE_FONT_SIZE,
		fontfamily="Arial",
		ha="center",
		va="bottom",
	)
	fig.text(
		0.72,
		0.04,
		PANEL_TITLES[1],
		fontsize=TITLE_FONT_SIZE,
		fontfamily="Arial",
		ha="center",
		va="bottom",
	)


def _apply_axis_text_style(ax) -> None:
	ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, length=2.2, width=0.5, pad=1.6)
	for label in ax.get_xticklabels():
		label.set_fontfamily(FONT_FAMILY)
		label.set_fontsize(TICK_FONT_SIZE)
	for label in ax.get_yticklabels():
		label.set_fontfamily(FONT_FAMILY)
		label.set_fontsize(TICK_FONT_SIZE)
	ax.xaxis.label.set_fontfamily(FONT_FAMILY)
	ax.yaxis.label.set_fontfamily(FONT_FAMILY)
	ax.xaxis.label.set_fontsize(FONT_SIZE)
	ax.yaxis.label.set_fontsize(FONT_SIZE)
	for spine in ax.spines.values():
		spine.set_linewidth(0.5)


def _compute_selection(*, gen_csv: Path, wind_csv: Path, poly_degree: int) -> dict[str, object]:
	gen_df = load_smape_csv(gen_csv)
	wind_df = load_smape_csv(wind_csv)

	gen_k, gen_b, gen_model, gen_poly, gen_k_fine, gen_b_fine, gen_S_pred, gen_r2 = get_optimal_path_from_dataframe(
		gen_df,
		poly_degree=poly_degree,
	)
	if gen_k is None or gen_model is None or gen_poly is None:
		raise RuntimeError(f"无法从 full generation csv 中提取有效 optimal path: {gen_csv}")

	wind_model, wind_poly, wind_k_fine, wind_b_fine, wind_S_pred, wind_r2 = fit_surface_from_dataframe(
		wind_df,
		poly_degree=poly_degree,
	)
	if wind_model is None or wind_poly is None:
		raise RuntimeError(f"无法从 wind csv 中拟合有效模型: {wind_csv}")

	min_result = find_min_wind_on_gen_op(gen_k, gen_b, wind_model, wind_poly)
	final_k = float(min_result["k"])
	final_b = float(min_result["b"])
	final_gen_smape = float(interpolate_smape_value(gen_model, gen_poly, final_k, final_b))

	return {
		"gen_k": np.asarray(gen_k, dtype=float),
		"gen_b": np.asarray(gen_b, dtype=float),
		"gen_k_fine": np.asarray(gen_k_fine, dtype=float),
		"gen_b_fine": np.asarray(gen_b_fine, dtype=float),
		"gen_S_pred": np.asarray(gen_S_pred, dtype=float),
		"gen_r2": float(gen_r2),
		"wind_k_fine": np.asarray(wind_k_fine, dtype=float),
		"wind_b_fine": np.asarray(wind_b_fine, dtype=float),
		"wind_S_pred": np.asarray(wind_S_pred, dtype=float),
		"wind_r2": float(wind_r2),
		"min_result": min_result,
		"final_row": {
			"w_smape": float(min_result["smape"]),
			"k": final_k,
			"b": final_b,
			"gen_smape": final_gen_smape,
			"candidate_point_count": int(len(gen_k)),
		},
	}


def _draw_unavailable_row(axes_row, *, show_titles: bool, season_label: str, message: str) -> None:
	for idx, ax in enumerate(axes_row):
		if idx == 0:
			ax.text(
				-0.3,
				0.5,
				season_label,
				transform=ax.transAxes,
				fontsize=FONT_SIZE,
				fontfamily="Arial",
				va="center",
				ha="center",
				zorder=30,
			)
		ax.set_axis_off()
		ax.text(
			0.5,
			0.5,
			message,
			transform=ax.transAxes,
			fontsize=FONT_SIZE,
			fontfamily=FONT_FAMILY,
			va="center",
			ha="center",
		)


def _draw_cased_path(ax, path_k, path_b, *, label: str, color: str) -> None:
	"""同色相深一档的路径 + 白色描边：在色带浅端和深端都清楚。"""
	_draw_path(
		ax,
		path_k,
		path_b,
		color=OPTIMAL_PATH_CASING_COLOR,
		label="_nolegend_",
		linewidth=OPTIMAL_PATH_LINEWIDTH * 2.2,
		zorder=5,
	)
	_draw_path(
		ax,
		path_k,
		path_b,
		color=color,
		label=label,
		linewidth=OPTIMAL_PATH_LINEWIDTH,
		zorder=6,
	)


def _draw_selection_row(
	axes_row,
	selection: dict[str, object],
	*,
	show_titles: bool,
	season_label: str,
	palette: Palette,
):
	ax1, ax2 = axes_row
	colormap = build_smape_colormap(palette)
	line_color = palette.line

	gen_k = np.asarray(selection["gen_k"], dtype=float)
	gen_b = np.asarray(selection["gen_b"], dtype=float)
	gen_k_fine = np.asarray(selection["gen_k_fine"], dtype=float)
	gen_b_fine = np.asarray(selection["gen_b_fine"], dtype=float)
	gen_S_pred = np.asarray(selection["gen_S_pred"], dtype=float)
	wind_k_fine = np.asarray(selection["wind_k_fine"], dtype=float)
	wind_b_fine = np.asarray(selection["wind_b_fine"], dtype=float)
	wind_S_pred = np.asarray(selection["wind_S_pred"], dtype=float)
	gen_r2 = float(selection["gen_r2"])
	wind_r2 = float(selection["wind_r2"])
	min_result = selection["min_result"]

	im1 = ax1.imshow(
		np.clip(gen_S_pred, SMAPE_VMIN, SMAPE_VMAX),
		cmap=colormap,
		origin="lower",
		extent=[gen_k_fine.min(), gen_k_fine.max(), gen_b_fine.min(), gen_b_fine.max()],
		aspect="auto",
		vmin=SMAPE_VMIN,
		vmax=SMAPE_VMAX,
	)
	_style_axis(ax1, title=None, season_label=season_label)
	ax1.set_xlim(0.5, 3.0)
	ax1.set_ylim(-2.0, 2.0)
	_apply_axis_text_style(ax1)
	_add_r2_label(ax1, gen_r2)
	_draw_cased_path(ax1, gen_k, gen_b, label="Optimal Path", color=line_color)
	ax1.legend(
		handles=[Line2D([0], [0], color=line_color, linewidth=OPTIMAL_PATH_LINEWIDTH)],
		labels=["Optimal Path"],
		loc="upper right",
		frameon=True,
		facecolor="white",
		framealpha=1.0,
		edgecolor=BOX_EDGE_COLOR,
		borderpad=0.3,
		labelspacing=0.3,
		handlelength=1.8,
		handletextpad=0.5,
		borderaxespad=0.4,
		prop={"family": FONT_FAMILY, "size": LEGEND_FONT_SIZE},
	).set_zorder(30)

	ax2.imshow(
		np.clip(wind_S_pred, SMAPE_VMIN, SMAPE_VMAX),
		cmap=colormap,
		origin="lower",
		extent=[wind_k_fine.min(), wind_k_fine.max(), wind_b_fine.min(), wind_b_fine.max()],
		aspect="auto",
		vmin=SMAPE_VMIN,
		vmax=SMAPE_VMAX,
	)
	_style_axis(ax2, title=None)
	ax2.set_xlim(0.5, 3.0)
	ax2.set_ylim(-2.0, 2.0)
	_apply_axis_text_style(ax2)
	_add_r2_label(ax2, wind_r2)
	_draw_cased_path(ax2, gen_k, gen_b, label="Optimal Path", color=line_color)

	legend_handles = [Line2D([0], [0], color=line_color, linewidth=OPTIMAL_PATH_LINEWIDTH)]
	legend_labels = ["Optimal Path"]
	if min_result is not None:
		ax2.scatter(
			min_result["k"],
			min_result["b"],
			color=OPTIMAL_MARKER_FACE,
			edgecolors=line_color,
			linewidth=0.8,
			s=OPTIMAL_MARKER_SIZE,
			zorder=10,
		)
		legend_handles.append(
			Line2D(
				[0],
				[0],
				marker="o",
				markersize=4.5,
				linestyle="None",
				markerfacecolor=OPTIMAL_MARKER_FACE,
				markeredgecolor=line_color,
				markeredgewidth=0.8,
			)
		)
		legend_labels.append(f"Optimal (k, b):\n({min_result['k']:.2f}, {min_result['b']:.2f})")
	ax2.legend(
		handles=legend_handles,
		labels=legend_labels,
		loc="upper right",
		frameon=True,
		facecolor="white",
		framealpha=1.0,
		edgecolor=BOX_EDGE_COLOR,
		borderpad=0.3,
		labelspacing=0.3,
		handlelength=1.8,
		handletextpad=0.5,
		borderaxespad=0.4,
		prop={"family": FONT_FAMILY, "size": LEGEND_FONT_SIZE},
	).set_zorder(30)

	season_texts = [text for text in ax1.texts if text.get_text() == season_label]
	for text in season_texts:
		text.set_x(-0.25)
		text.set_fontfamily(FONT_FAMILY)
		text.set_fontsize(FONT_SIZE)

	return im1


def build_province_figure(
	*,
	prov: str,
	season_entries: dict[str, dict[str, object]],
	output_plot: Path,
	palette: Palette,
) -> None:
	fig, axes = plt.subplots(4, 2, figsize=FIGSIZE)
	im_for_cbar = None

	for row_idx, season in enumerate(SEASON_ORDER):
		row_axes = axes[row_idx]
		season_label = SEASON_LABELS.get(season, season)
		show_titles = row_idx == 0
		entry = season_entries.get(season)
		if entry is None or entry.get("selection") is None:
			message = "Unavailable"
			if entry is not None and entry.get("failure_message"):
				message = str(entry["failure_message"])
			_draw_unavailable_row(row_axes, show_titles=show_titles, season_label=season_label, message=message)
			continue

		im_current = _draw_selection_row(
			row_axes,
			entry["selection"],
			show_titles=show_titles,
			season_label=season_label,
			palette=palette,
		)
		if im_for_cbar is None:
			im_for_cbar = im_current

	_add_column_titles_bottom(fig)
	fig.subplots_adjust(left=0.12, right=0.93, top=0.97, bottom=0.08, wspace=0.2, hspace=0.2)
	if im_for_cbar is not None:
		cbar_ax = fig.add_axes([0.945, 0.10, 0.014, 0.82])
		cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
		cbar.set_ticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
		cbar.set_label("SMAPE (%)", fontsize=FONT_SIZE, fontfamily=FONT_FAMILY, labelpad=3.0)
		cbar.ax.tick_params(labelsize=FONT_SIZE, length=2.2, width=0.5, pad=2.0)
		for label in cbar.ax.get_yticklabels():
			label.set_fontfamily(FONT_FAMILY)
			label.set_fontsize(FONT_SIZE)
		cbar.outline.set_linewidth(0.5)
	output_plot.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_plot, dpi=1000, bbox_inches="tight", format="jpg")
	plt.close()


def _process_selection_task(task: tuple[str, str, int]) -> dict[str, object]:
	gen_csv_str, wind_csv_str, poly_degree = task
	gen_csv = Path(gen_csv_str)
	wind_csv = Path(wind_csv_str)
	meta = parse_full_gen_filename(gen_csv)
	try:
		selection = _compute_selection(gen_csv=gen_csv, wind_csv=wind_csv, poly_degree=poly_degree)
		return {
			"ok": True,
			"prov": meta["prov"] if meta is not None else None,
			"season": meta["season"] if meta is not None else None,
			"selection": selection,
			"gen_csv": str(gen_csv),
			"wind_csv": str(wind_csv),
		}
	except Exception as exc:
		return {
			"ok": False,
			"prov": meta["prov"] if meta is not None else None,
			"season": meta["season"] if meta is not None else None,
			"selection": None,
			"gen_csv": str(gen_csv),
			"wind_csv": str(wind_csv),
			"error": str(exc),
		}


def generate_province_optimal_kb_plots(
	gen_dir: Path,
	wind_dir: Path,
	output_dir: Path,
	poly_degree: int = 9,
	max_workers: int | None = None,
	force_spawn: bool | None = None,
	provinces: list[str] | None = None,
	palette: str = DEFAULT_PALETTE,
) -> dict[str, object]:
	if poly_degree < 1:
		raise ValueError(f"poly_degree 必须 >= 1, 当前为: {poly_degree}")
	if palette not in PALETTES:
		raise ValueError(f"未知配色 {palette!r}，可选: {sorted(PALETTES)}")
	active_palette = PALETTES[palette]

	effective_force_spawn = should_force_spawn(force_spawn)
	if effective_force_spawn:
		try:
			multiprocessing.set_start_method("spawn")
		except RuntimeError:
			pass

	gen_dir = Path(gen_dir).resolve()
	wind_dir = Path(wind_dir).resolve()
	output_dir = Path(output_dir).resolve()

	if not gen_dir.exists():
		raise FileNotFoundError(f"未找到 gen_dir: {gen_dir}")
	if not wind_dir.exists():
		raise FileNotFoundError(f"未找到 wind_dir: {wind_dir}")
	output_dir.mkdir(parents=True, exist_ok=True)

	province_filter = {item.strip() for item in provinces if item.strip()} if provinces else None
	# plot_paper_v2 patch: the province tokens recovered from the downloaded CSV file
	# names are mojibake (the remote UTF-8 names were re-decoded as GBK during
	# transfer, e.g. 青海 -> 闈掓捣).  Accept either spelling so `--province 青海`
	# keeps working.  No numerical logic is touched.
	if province_filter is not None:
		province_filter |= {
			item.encode("utf-8").decode("gbk", errors="ignore") for item in province_filter
		}
	gen_csv_list = sorted(gen_dir.glob("*_full_gen_smape.csv"), key=lambda path: path.name)
	if not gen_csv_list:
		raise FileNotFoundError(f"未找到任何 *_full_gen_smape.csv: {gen_dir}")

	tasks: list[tuple[str, str, int]] = []
	season_entries: dict[str, dict[str, dict[str, object]]] = {}
	skipped_items: list[dict[str, object]] = []

	for gen_csv in gen_csv_list:
		meta = parse_full_gen_filename(gen_csv)
		if meta is None:
			skipped_items.append({"gen_csv": str(gen_csv), "reason": "unrecognized_filename"})
			continue
		prov = meta["prov"]
		season = meta["season"]
		if province_filter is not None and prov not in province_filter:
			continue
		wind_csv = wind_dir / f"{prov}_{season}_w_smape_enums.csv"
		if not wind_csv.exists():
			season_entries.setdefault(prov, {})[season] = {
				"selection": None,
				"failure_message": f"Missing wind CSV: {wind_csv.name}",
			}
			skipped_items.append({
				"gen_csv": str(gen_csv),
				"wind_csv": str(wind_csv),
				"reason": "missing_wind_csv",
			})
			continue
		tasks.append((str(gen_csv), str(wind_csv), poly_degree))

	if not tasks and not season_entries:
		raise ValueError("没有可用于绘图的省季组合")

	worker_count = len(tasks) if max_workers is None else max(1, min(int(max_workers), len(tasks)))
	if tasks:
		with ProcessPoolExecutor(max_workers=worker_count) as executor:
			futures = [executor.submit(_process_selection_task, task) for task in tasks]
			for future in as_completed(futures):
				result = future.result()
				prov = str(result.get("prov"))
				season = str(result.get("season"))
				if result["ok"]:
					season_entries.setdefault(prov, {})[season] = {
						"selection": result["selection"],
						"failure_message": None,
					}
				else:
					season_entries.setdefault(prov, {})[season] = {
						"selection": None,
						"failure_message": str(result.get("error", "Unavailable")),
					}

	plots_written: list[str] = []
	for prov in sorted(season_entries):
		# 输出文件名用规范化后的正确中文省份名（数值与绘图口径完全不变）
		output_stem = canonical_province_name(prov)
		output_plot = output_dir / f"{output_stem}_optimal_kb.jpg"
		build_province_figure(
			prov=prov,
			season_entries=season_entries[prov],
			output_plot=output_plot,
			palette=active_palette,
		)
		plots_written.append(str(output_plot))

	return {
		"gen_dir": str(gen_dir),
		"wind_dir": str(wind_dir),
		"output_dir": str(output_dir),
		"palette": f"{palette} {active_palette.label}",
		"province_count": len(plots_written),
		"plot_count": len(plots_written),
		"plots_written": plots_written,
		"skipped_count": len(skipped_items),
		"skipped_items": skipped_items,
	}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="按省汇总 full-data optimal k,b 搜索图，输出四季 4x2 面板图。",
	)
	parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR, help="prov_gen_full_smape 目录")
	parser.add_argument("--wind-dir", type=Path, default=DEFAULT_WIND_DIR, help="prov_w_smape_enums 目录")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="图片输出目录")
	parser.add_argument("--poly-degree", type=int, default=9, help="曲面拟合多项式阶数")
	parser.add_argument("--max-workers", type=int, default=max((os.cpu_count() or 1) // 2, 1), help="并行 worker 数")
	parser.add_argument("--province", nargs="*", default=None, help="仅绘制这些省简称，例如 河北 江苏")
	parser.add_argument(
		"--palette",
		choices=sorted(PALETTES),
		default=DEFAULT_PALETTE,
		help="单一色系色带: " + " | ".join(f"{k}={v.label}" for k, v in sorted(PALETTES.items())),
	)
	return parser


def main() -> None:
	args = build_parser().parse_args()
	result = generate_province_optimal_kb_plots(
		gen_dir=args.gen_dir,
		wind_dir=args.wind_dir,
		output_dir=args.output_dir,
		poly_degree=args.poly_degree,
		max_workers=args.max_workers,
		provinces=args.province,
		palette=args.palette,
	)
	print(result)


if __name__ == "__main__":
	main()