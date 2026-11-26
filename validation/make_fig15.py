"""Fig. 14 - 各省分季节 holdout SMAPE 箱线图（ESSD 出图规范重制版）。

旧版：plot/kb_validation_holdout_smape_boxplots.py（画布 391 mm 宽、字号 10.8~16.2 pt）。
本版改动：

1. 画布按最终印刷尺寸作画（ESSD 满栏宽 177 mm），全图文字统一 Arial 8 pt；
2. 每个子图左上角加面板标签 (a)(b)...（ESSD 规定：括号内小写字母），与省名同行，
   省名由旧版的"图框下方"移到"图框上方"，腾出下方空间给季节刻度；
3. 版式重排：每个格子给 y 轴刻度留出内侧字槽、季节刻度按实测字宽反解 x 轴范围，
   脚本末尾用渲染器实测所有文字包围盒自检（文字不压图框、不互相重叠、不越界）。

输出：fig14_holdout_smape_boxplots.png（300 dpi）+ .pdf（矢量）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import pandas as pd
from matplotlib.patches import Patch

from config_paths import OUTDIR, KB_VALIDATION_DETAIL_CSV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = OUTDIR
DETAIL_CANDIDATES = (KB_VALIDATION_DETAIL_CSV,)

SEASON_ORDER = ("spring", "summer", "autumn", "winter")
SEASON_LABELS = {"spring": "MAM", "summer": "JJA", "autumn": "SON", "winter": "DJF"}
SEASON_COLORS = {
    "spring": "#7AA874",
    "summer": "#F4A261",
    "autumn": "#C97C5D",
    "winter": "#5C88A5",
}
PROVINCE_NAME_EN = {
    "北京": "Beijing", "天津": "Tianjin", "上海": "Shanghai", "重庆": "Chongqing",
    "河北": "Hebei", "山西": "Shanxi", "辽宁": "Liaoning", "吉林": "Jilin",
    "黑龙江": "Heilongjiang", "江苏": "Jiangsu", "浙江": "Zhejiang", "安徽": "Anhui",
    "福建": "Fujian", "江西": "Jiangxi", "山东": "Shandong", "河南": "Henan",
    "湖北": "Hubei", "湖南": "Hunan", "广东": "Guangdong", "海南": "Hainan",
    "四川": "Sichuan", "贵州": "Guizhou", "云南": "Yunnan", "陕西": "Shaanxi",
    "甘肃": "Gansu", "青海": "Qinghai", "内蒙古": "Inner Mongolia", "广西": "Guangxi",
    "宁夏": "Ningxia", "新疆": "Xinjiang",
}

# ---------------------------------------------------------------- 出图规范常量
FS_PT = 8.0
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,            # 矢量图字体嵌入
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",  # 保持"全图只用一种字体族"
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

MM = 1.0 / 25.4
FIG_W_MM = 177.0               # ESSD 满栏文本宽
GRID_COLS, GRID_ROWS = 5, 6
M_LEFT_MM, M_RIGHT_MM, M_TOP_MM, M_BOTTOM_MM = 5.0, 2.5, 2.0, 12.0
GUTTER_X_MM, GUTTER_Y_MM = 4.0, 5.5
TITLE_H_MM, TICK_H_MM = 3.4, 4.7
BOX_H_MM = 25.0
TICK_LEN_PT, TICK_PAD_PT = 2.0, 3.4
Y_LIMITS = (0.0, 40.0)
Y_TICK_STEP = 10.0
BOX_WIDTH = 0.58
JITTER = 0.11
LABEL_GAP_MM = 0.5             # 相邻季节刻度文字之间至少留的间隙
LINE_GRID, LINE_SPINE = 0.35, 0.4
S_POINT = 4.5


# ---------------------------------------------------------------- 数据
def _resolve_existing_path(candidates: tuple[Path, ...], explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {explicit_path}")
        return explicit_path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate any candidate file: {[str(p) for p in candidates]}")


def _load_detail(detail_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(detail_csv)
    required = {"prov", "season", "holdout_smape", "status"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Detail CSV is missing columns: {sorted(missing)}")
    df = df[df["status"].astype(str).str.lower() == "success"].copy()
    df = df[df["season"].isin(SEASON_ORDER)].copy()
    df["holdout_smape"] = pd.to_numeric(df["holdout_smape"], errors="coerce")
    df = df.dropna(subset=["holdout_smape"]).copy()
    if df.empty:
        raise ValueError("No successful holdout_smape rows found in detail CSV.")
    return df


def _province_order(detail_df: pd.DataFrame) -> list[str]:
    provinces = sorted(detail_df["prov"].dropna().astype(str).unique().tolist())
    return sorted(provinces, key=lambda item: PROVINCE_NAME_EN.get(item, item))


def _panel_labels(count: int) -> list[str]:
    """(a)..(z), (aa)..(ad) —— ESSD：面板标签用括号内小写字母。"""
    return [f"({chr(97 + i)})" if i < 26 else f"(a{chr(97 + i - 26)})" for i in range(count)]


_TEXT_CACHE: dict[str, tuple[float, float]] = {}


def _text_metrics_mm(text: str) -> tuple[float, float]:
    """用渲染器实测 8 pt 文本的宽/高（mm），版式据此反解，不靠估。"""
    if text not in _TEXT_CACHE:
        probe = plt.figure(figsize=(2, 1))
        artist = probe.text(0.0, 0.5, text, fontsize=FS_PT, fontfamily="Arial")
        bbox = artist.get_window_extent(renderer=probe.canvas.get_renderer())
        plt.close(probe)
        _TEXT_CACHE[text] = (bbox.width / probe.dpi * 25.4, bbox.height / probe.dpi * 25.4)
    return _TEXT_CACHE[text]


def _make_layout(y_label_slot_mm: float) -> dict:
    cell_w = (FIG_W_MM - M_LEFT_MM - M_RIGHT_MM - (GRID_COLS - 1) * GUTTER_X_MM) / GRID_COLS
    block_h = TITLE_H_MM + BOX_H_MM + TICK_H_MM
    fig_h = M_TOP_MM + GRID_ROWS * block_h + (GRID_ROWS - 1) * GUTTER_Y_MM + M_BOTTOM_MM
    return {"cell_w": cell_w, "slot": y_label_slot_mm,
            "box_w": cell_w - y_label_slot_mm, "block_h": block_h, "fig_h": fig_h}


def _panel_geometry(layout: dict, index: int) -> tuple[float, float, float, float]:
    """返回格子 (格左边 mm, 图框左 mm, 图框底 mm, 图框顶 mm)。"""
    row, col = divmod(index, GRID_COLS)
    cell_left = M_LEFT_MM + col * (layout["cell_w"] + GUTTER_X_MM)
    block_top = layout["fig_h"] - M_TOP_MM - row * (layout["block_h"] + GUTTER_Y_MM)
    box_top = block_top - TITLE_H_MM
    return cell_left, cell_left + layout["slot"], box_top - BOX_H_MM, box_top


# ---------------------------------------------------------------- 布局自检
def _verify_layout(fig: plt.Figure, panel_boxes: list[plt.Axes], fig_h_mm: float) -> None:
    """实测所有文字包围盒：不得压图框、不得互相重叠、不得越出画布。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()

    def mm(bb: object) -> str:
        return (f"x {bb.x0 * FIG_W_MM:6.2f}~{bb.x1 * FIG_W_MM:6.2f} "
                f"y {bb.y0 * fig_h_mm:6.2f}~{bb.y1 * fig_h_mm:6.2f}")

    texts = []
    for artist in fig.findobj(mtext.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        owner = getattr(artist, "axes", None)
        texts.append((f"{owner.get_label() if owner is not None else 'fig'}:{artist.get_text()[:16]}", artist))

    extents = [(name, t.get_window_extent(renderer).transformed(to_fig)) for name, t in texts]
    boxes = [(ax.get_label(), ax.get_window_extent(renderer).transformed(to_fig)) for ax in panel_boxes]

    on_axes = [(name, axname) for name, bb in extents for axname, abb in boxes if bb.overlaps(abb)]
    on_text = [(extents[i], extents[j]) for i in range(len(extents))
               for j in range(i + 1, len(extents)) if extents[i][1].overlaps(extents[j][1])]
    outside = [(name, mm(bb)) for name, bb in extents
               if bb.x0 < -1e-6 or bb.y0 < -1e-6 or bb.x1 > 1 + 1e-6 or bb.y1 > 1 + 1e-6]
    legend_out = []
    for i, legend in enumerate(fig.legends):
        bb = legend.get_window_extent(renderer).transformed(to_fig)
        if bb.x0 < -1e-6 or bb.y0 < -1e-6 or bb.x1 > 1 + 1e-6 or bb.y1 > 1 + 1e-6:
            legend_out.append(i)

    print(f"  [布局自检] 文字压图框 {len(on_axes)} | 文字相互重叠 {len(on_text)} | "
          f"越出画布 {len(outside)} | 图例越界 {len(legend_out)} | "
          f"文字对象 {len(extents)} | 画布 {FIG_W_MM:.1f} x {fig_h_mm:.1f} mm")
    for name, axname in on_axes[:6]:
        print(f"      ! {name} 压在 {axname}")
    for (name_a, bb_a), (name_b, bb_b) in on_text[:6]:
        print(f"      ! {name_a} [{mm(bb_a)}] 与 {name_b} [{mm(bb_b)}] 重叠")
    for name, where in outside[:6]:
        print(f"      ! {name} 越界 [{where}]")
    if on_axes or on_text or outside or legend_out:
        raise AssertionError("布局自检未通过，见上方明细")


# ---------------------------------------------------------------- 出图
def build_figure(detail_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    provinces = _province_order(detail_df)
    if len(provinces) > GRID_COLS * GRID_ROWS:
        raise ValueError(f"Too many provinces for the {GRID_COLS}x{GRID_ROWS} grid: {len(provinces)}")

    y_labels = [f"{int(v)}" for v in np.arange(Y_LIMITS[0], Y_LIMITS[1] + 1e-9, Y_TICK_STEP)]
    slot = max(_text_metrics_mm(t)[0] for t in y_labels) + 0.6
    layout = _make_layout(slot)
    fig_h = layout["fig_h"]
    print(f"  [版式] 画布 {FIG_W_MM:.1f} x {fig_h:.1f} mm | 格子宽 {layout['cell_w']:.2f} mm | "
          f"图框 {layout['box_w']:.2f} x {BOX_H_MM:.1f} mm | y 刻度字槽 {slot:.2f} mm")

    # 季节刻度文字宽度反解 x 轴范围：先保证相邻刻度文字留出 LABEL_GAP_MM 间隙，
    # 再确认最外侧箱体（含抖动散点）仍在图框内。
    widest = max(_text_metrics_mm(SEASON_LABELS[s])[0] for s in SEASON_ORDER)
    unit_min = BOX_WIDTH / 2.0 + JITTER
    unit_for_gap = (layout["box_w"] / (widest + LABEL_GAP_MM) - (len(SEASON_ORDER) - 1)) / 2.0
    pos_pad = max(unit_min + 0.02, min(unit_for_gap, 0.60))
    half_pad = pos_pad
    print(f"  [版式] 季节刻度文字宽 {widest:.2f} mm | 箱位留白 {pos_pad:.3f} 单位 | "
          f"刻度文字间隙 {layout['box_w'] / (len(SEASON_ORDER) - 1 + 2 * pos_pad) - widest:.2f} mm")

    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    axes: list[plt.Axes] = []
    positions = np.arange(len(SEASON_ORDER), dtype=float)
    rng = np.random.default_rng(42)

    for index, (province, label) in enumerate(zip(provinces, _panel_labels(len(provinces)))):
        province_df = detail_df[detail_df["prov"] == province]
        cell_left, box_left, box_bottom, box_top = _panel_geometry(layout, index)
        ax = fig.add_axes((box_left / FIG_W_MM, box_bottom / fig_h,
                           layout["box_w"] / FIG_W_MM, BOX_H_MM / fig_h))
        ax.set_label(f"{label} {province}")
        axes.append(ax)

        season_series = [province_df.loc[province_df["season"] == season, "holdout_smape"].to_numpy(float)
                         for season in SEASON_ORDER]
        box = ax.boxplot(
            season_series, positions=positions, widths=BOX_WIDTH, patch_artist=True, showfliers=False,
            medianprops={"color": "#2A2A2A", "linewidth": 0.6},
            whiskerprops={"color": "#777777", "linewidth": 0.45},
            capprops={"color": "#777777", "linewidth": 0.45},
            boxprops={"edgecolor": "#666666", "linewidth": 0.4},
        )
        for patch, season in zip(box["boxes"], SEASON_ORDER):
            patch.set_facecolor(SEASON_COLORS[season])
            patch.set_alpha(0.58)
        for position, season, values in zip(positions, SEASON_ORDER, season_series):
            if len(values) == 0:
                continue
            ax.scatter(np.full(len(values), position) + rng.uniform(-JITTER, JITTER, len(values)), values,
                       s=S_POINT, color=SEASON_COLORS[season], alpha=0.32, edgecolors="none", zorder=3)

        ax.set_xlim(-half_pad, len(SEASON_ORDER) - 1 + half_pad)
        ax.set_ylim(*Y_LIMITS)
        ax.set_box_aspect(BOX_H_MM / layout["box_w"])
        ax.set_xticks(positions, [SEASON_LABELS[s] for s in SEASON_ORDER])
        ax.set_yticks(np.arange(Y_LIMITS[0], Y_LIMITS[1] + 1e-9, Y_TICK_STEP), y_labels)
        ax.tick_params(labelsize=FS_PT, length=TICK_LEN_PT, width=0.4, pad=TICK_PAD_PT)
        ax.grid(True, axis="y", color="#E3E3E3", linewidth=LINE_GRID)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#BBBBBB")
            spine.set_linewidth(LINE_SPINE)

        # 面板标签 + 省名：图框上方，横向居中于图框
        fig.text((box_left + layout["box_w"] / 2.0) / FIG_W_MM, (box_top + 0.65) / fig_h,
                 f"{label} {PROVINCE_NAME_EN.get(province, province)}",
                 ha="center", va="bottom", fontsize=FS_PT, color="black")

    grid_left = M_LEFT_MM + layout["slot"]
    grid_cx = (grid_left + layout["box_w"] * GRID_COLS + GUTTER_X_MM * (GRID_COLS - 1) / 2.0) / FIG_W_MM
    grid_cy = (M_TOP_MM + (fig_h - M_TOP_MM - M_BOTTOM_MM) / 2.0) / fig_h
    fig.text(grid_cx, 9.4 / fig_h, "Season", ha="center", va="center", fontsize=FS_PT)
    fig.text(1.7 / FIG_W_MM, grid_cy, "Holdout SMAPE", ha="center", va="center",
             rotation="vertical", fontsize=FS_PT)

    handles = [Patch(facecolor=SEASON_COLORS[s], edgecolor="#666666", linewidth=0.4, alpha=0.58,
                     label=SEASON_LABELS[s]) for s in SEASON_ORDER]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 3.2 / fig_h), ncol=4,
               frameon=False, fontsize=FS_PT, columnspacing=2.2, handletextpad=0.9,
               handlelength=1.4, borderpad=0.2)

    _verify_layout(fig, axes, fig_h)

    png_path = output_dir / "fig14_holdout_smape_boxplots.png"
    pdf_path = output_dir / "fig14_holdout_smape_boxplots.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fig. 14: province-level seasonal holdout SMAPE boxplots (ESSD style).")
    parser.add_argument("--detail-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail_csv = _resolve_existing_path(DETAIL_CANDIDATES, args.detail_csv)
    detail_df = _load_detail(detail_csv)
    png_path, pdf_path = build_figure(detail_df, args.output_dir)
    print(f"Detail CSV: {detail_csv}")
    print(f"Output PNG: {png_path}")
    print(f"Output PDF: {pdf_path}")


if __name__ == "__main__":
    main()
