"""fig8 文字合规自检：Arial / 8 pt / 文字不压图。

两路独立证据：
  A 设计闸门：渲染器实测每个 Text / Legend 的包围盒，判是否与坐标区相交、文字互压、越出画布。
  B 像素证伪：把全部文字隐藏后重渲染，与原图逐像素求差得到「纯文字墨迹」，
    断言没有墨迹落在任何坐标区内，并报告文字到最近图框的距离（mm）。

用法：D:/python/python.exe _check_fig8_text_overlap.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import findfont
from matplotlib.legend import Legend
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "fig05_generation_dynamics.py"
CHECK_DPI = 300

_spec = importlib.util.spec_from_file_location("fig05_generation_dynamics", MODULE_PATH)
fig8 = importlib.util.module_from_spec(_spec)
sys.modules["fig05_generation_dynamics"] = fig8
_spec.loader.exec_module(fig8)

PROBLEMS: list[str] = []
NOTES: list[str] = []


def _mm(value_px: float, dpi: int) -> float:
	return value_px / dpi * 25.4


def _overlap(a, b) -> tuple[float, float]:
	dx = min(a.x1, b.x1) - max(a.x0, b.x0)
	dy = min(a.y1, b.y1) - max(a.y0, b.y0)
	return (dx, dy) if dx > 0 and dy > 0 else (0.0, 0.0)


def _outside_distance(px: float, py: float, box) -> float:
	dx = max(box.x0 - px, px - box.x1, 0.0)
	dy = max(box.y0 - py, py - box.y1, 0.0)
	return float(np.hypot(dx, dy))


def _visible_texts(fig) -> list[Text]:
	return [t for t in fig.findobj(Text) if t.get_visible() and str(t.get_text()).strip()]


def _audit(fig: plt.Figure, output_path: Path) -> None:
	fig.set_dpi(CHECK_DPI)
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	dpi = fig.dpi
	tag = output_path.name
	font_sizes = {round(t.get_fontsize(), 3) for t in _visible_texts(fig)}
	font_files = {Path(findfont(t.get_fontproperties())).name for t in _visible_texts(fig)}

	axes_boxes = [(ax, ax.get_window_extent(renderer)) for ax in fig.axes]
	texts = _visible_texts(fig)
	legends = [lg for lg in fig.findobj(Legend) if lg.get_visible()]

	# --- 判据 1：文字 / 图例是否压在坐标区内 ---
	for t in texts:
		bb = t.get_window_extent(renderer)
		for _ax, ab in axes_boxes:
			dx, dy = _overlap(bb, ab)
			if dx > 0 and dy > 0:
				PROBLEMS.append(
					f"[{tag}] 文字压在坐标区内 {str(t.get_text())[:40]!r} 重叠 {_mm(dx, dpi):.1f}×{_mm(dy, dpi):.1f} mm"
				)
	for lg in legends:
		bb = lg.get_window_extent(renderer)
		for _ax, ab in axes_boxes:
			dx, dy = _overlap(bb, ab)
			if dx > 0 and dy > 0:
				PROBLEMS.append(f"[{tag}] 图例压在坐标区内 重叠 {_mm(dx, dpi):.1f}×{_mm(dy, dpi):.1f} mm")

	# --- 判据 2：文字互压 ---
	for i in range(len(texts)):
		for j in range(i + 1, len(texts)):
			dx, dy = _overlap(texts[i].get_window_extent(renderer), texts[j].get_window_extent(renderer))
			if dx > 0 and dy > 0:
				PROBLEMS.append(
					f"[{tag}] 文字互压 {str(texts[i].get_text())[:20]!r} × {str(texts[j].get_text())[:20]!r} "
					f"重叠 {_mm(dx, dpi):.1f}×{_mm(dy, dpi):.1f} mm"
				)

	# --- 判据 3：文字越出画布 ---
	width, height = fig.canvas.get_width_height()
	for t in texts:
		bb = t.get_window_extent(renderer)
		if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > width + 1 or bb.y1 > height + 1:
			PROBLEMS.append(
				f"[{tag}] 文字越出画布 {str(t.get_text())[:30]!r} x=[{_mm(bb.x0, dpi):.1f},{_mm(bb.x1, dpi):.1f}] "
				f"y=[{_mm(bb.y0, dpi):.1f},{_mm(bb.y1, dpi):.1f}] 画布 {_mm(width, dpi):.1f}×{_mm(height, dpi):.1f} mm"
			)

	# --- 判据 4（像素证伪）：纯文字墨迹不得进入坐标区内 ---
	figure_pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
	hidden = []
	for obj in texts + legends:
		obj.set_visible(False)
		hidden.append(obj)
	fig.canvas.draw()
	plain_pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
	for obj in hidden:
		obj.set_visible(True)
	ink = (figure_pixels != plain_pixels).any(axis=2)
	ink_pixels = int(ink.sum())
	y_rows, x_cols = np.nonzero(ink)
	scaled_y = figure_pixels.shape[0] - 1 - y_rows

	for _ax, ab in axes_boxes:
		x0, x1 = int(max(ab.x0, 0)), int(min(ab.x1, figure_pixels.shape[1]))
		y0, y1 = int(max(ab.y0, 0)), int(min(ab.y1, figure_pixels.shape[0]))
		inside = int(ink[figure_pixels.shape[0] - y1 : figure_pixels.shape[0] - y0, x0:x1].sum())
		if inside > 0:
			PROBLEMS.append(f"[{tag}] 像素证伪：{inside} 个文字墨迹像素落在坐标区内")

	min_distance = float("inf")
	if len(x_cols) > 20000:
		step = len(x_cols) // 20000
		x_sample, y_sample = x_cols[::step].astype(float), scaled_y[::step].astype(float)
	else:
		x_sample, y_sample = x_cols.astype(float), scaled_y.astype(float)
	for _ax, ab in axes_boxes:
		dx = np.maximum(np.maximum(ab.x0 - x_sample, x_sample - ab.x1), 0.0)
		dy = np.maximum(np.maximum(ab.y0 - y_sample, y_sample - ab.y1), 0.0)
		min_distance = min(min_distance, float(np.hypot(dx, dy).min()))

	fig_width_mm = _mm(figure_pixels.shape[1], dpi)
	fig_height_mm = _mm(figure_pixels.shape[0], dpi)
	fonts_ok = font_sizes == {8.0} and all(name.lower().startswith("arial") for name in font_files)
	# 分区电网总图（biplot）按用户指定用 210 mm（A4 纸宽）；同名的 share/generation 单幅仍是 183 mm。
	width_ok = abs(fig_width_mm - 183.0) <= 0.5 or ("grid_in_national" in tag and abs(fig_width_mm - 210.0) <= 0.5)
	height_ok = fig_height_mm <= 250.0
	NOTES.append(
		f"[{tag}] 画布 {fig_width_mm:.1f}×{fig_height_mm:.1f} mm | 坐标区 {len(axes_boxes)} 个 | "
		f"文字对象 {len(texts)} 个（其中图例 {len(legends)} 个）| 字号 {sorted(font_sizes)} | "
		f"字体文件 {sorted(font_files)} | 纯文字墨迹 {ink_pixels} px | "
		f"文字到最近图框 {_mm(min_distance, dpi):.2f} mm | 字体合规 {'是' if fonts_ok else '否'} | "
		f"宽 183mm {'是' if width_ok else '否'} | 高≤250mm {'是' if height_ok else '否'}"
	)
	if not fonts_ok:
		PROBLEMS.append(f"[{tag}] 字号/字体不合规：fontsize={sorted(font_sizes)} fontfile={sorted(font_files)}")
	if not width_ok:
		PROBLEMS.append(f"[{tag}] 画布宽度不是 183 mm：实测 {fig_width_mm:.1f} mm")
	if not height_ok:
		PROBLEMS.append(f"[{tag}] 画布高度超过 250 mm：实测 {fig_height_mm:.1f} mm")


_original_write = fig8._write_matplotlib_figure


def _patched_write(figure, output_path):
	_audit(figure, Path(output_path).with_suffix(".png"))
	_original_write(figure, output_path)


fig8._write_matplotlib_figure = _patched_write

# 本机对「一次批量删除超过 50 个文件」有拦截。输出文件名是确定的，新图会原地覆盖同名旧图，
# 所以本轮把删除动作整体改成空操作：不删任何文件，只覆盖。
fig8._clear_plot_outputs = lambda *dirs: None
fig8._remove_legacy_outputs = lambda root: None
_original_unlink = Path.unlink
Path.unlink = lambda self, missing_ok=False: None

_load_cache: dict[str, object] = {}
_original_load = fig8._load_city_daily_long


def _patched_load(city_root):
	key = str(city_root)
	if key not in _load_cache:
		start = time.time()
		_load_cache[key] = _original_load(city_root)
		print(f"  （加载城市数据 {time.time() - start:.0f}s）", flush=True)
	return _load_cache[key]


fig8._load_city_daily_long = _patched_load


def _synthetic_stacked_area() -> None:
	import pandas as pd

	index = pd.date_range("1991-01-01", "2023-01-01", freq="YS")
	columns = [f"City{i:02d}" for i in range(13)]
	rng = np.random.default_rng(0)
	wide = pd.DataFrame(rng.random((len(index), len(columns))) * 500 + 200, index=index, columns=columns)
	for is_share, name in ((False, "generation"), (True, "share")):
		fig8._plot_matplotlib_stacked_area(
			value_wide=wide if not is_share else wide.div(wide.sum(axis=1), axis=0),
			output_path=Path(tempfile.gettempdir()) / "fig8_synthetic" / f"synthetic_stack_{name}",
			is_share=is_share,
			ylabel="Generation (GWh)" if not is_share else "Share of Provincial Generation",
			figsize=(16, 8.5),
		)


def main() -> None:
	targets = sys.argv[1:] or ["all"]
	output_root = HERE / "output"
	output_root.mkdir(parents=True, exist_ok=True)
	_synthetic_stacked_area()
	for target in targets:
		print(f"== {target} ==", flush=True)
		fig8.generate_generation_dynamics_plots(
			render_target=target,
			output_root=output_root,
		)

	lines = ["===== 逐图明细 =====", *NOTES, ""]
	if PROBLEMS:
		lines += [f"===== 结论：{len(PROBLEMS)} 处问题 =====", *PROBLEMS]
	else:
		lines.append("===== 结论：全部通过（0 处问题）=====")
	report = "\n".join(lines)
	print("\n" + report)
	(output_root / "_check_fig05_text_overlap.log").write_text(report, encoding="utf-8")


if __name__ == "__main__":
	main()
