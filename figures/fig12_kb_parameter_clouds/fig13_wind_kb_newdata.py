"""Fig. 13 - 各省分季节 (k, b) 参数云（ESSD 出图规范重制版）。

旧版：plot/kb_validation_parameter_clouds.py（画布 391 mm 宽、字号 12~16 pt，
RMS 统计块压在散点与椭圆上）。本版改动：

1. 画布按最终印刷尺寸作画（ESSD 满栏宽 177 mm），全图文字统一 Arial 8 pt，
   不靠后期缩放 —— 字号与线宽即印刷实际值；
2. 每个子图左上角加面板标签 (a)(b)...（ESSD 规定：括号内小写字母），与省名同行；
3. 版式重排：每个格子里给 y 轴刻度留出内侧字槽，RMS 统计块移到图框**外部**
   （图框正下方 2x2）；脚本末尾用渲染器实测所有文字包围盒做自检，
   确保文字既不压图框、也不互相重叠。

输出：fig13_parameter_clouds.png（300 dpi）+ .pdf（矢量）+ 统计明细 csv。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import config_paths as _cfg  # noqa: E402

PROJECT_ROOT = _HERE.parent
DEFAULT_OUTPUT_DIR = Path(r"D:\HKU PhD\wind data paper\plot\sec44_diag\newfigs")
# Original candidates:
#   <paper_root>/kb_validation_detail.csv
#   <paper_root>/results/kb_validation/kb_validation_detail.csv
DETAIL_CANDIDATES = (
    Path(r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\validation44\kb_validation_detail_new.csv"),
)

SEASON_ORDER = ("spring", "summer", "autumn", "winter")
SEASON_LABELS = {"spring": "MAM", "summer": "JJA", "autumn": "SON", "winter": "DJF"}
SEASON_COLORS = {
    "spring": "#7AA874",
    "summer": "#F4A261",
    "autumn": "#C97C5D",
    "winter": "#5C88A5",
}
SEASON_MARKERS = {"spring": "o", "summer": "^", "autumn": "s", "winter": "X"}
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
M_LEFT_MM, M_RIGHT_MM, M_TOP_MM, M_BOTTOM_MM = 5.0, 3.8, 2.0, 12.0
GUTTER_X_MM, GUTTER_Y_MM = 5.0, 4.0
TITLE_H_MM, TICK_H_MM = 3.4, 4.2
BOX_H_MM = 18.8
STATS_PAD_MM, STATS_PITCH_MM = 1.2, 4.2   # RMS 块：上留白 / 两行行距
STATS_GAP_MM = 1.2                        # RMS 块两列之间的间隙
TICK_LEN_PT, TICK_PAD_PT = 2.0, 2.0

LINE_GRID, LINE_SPINE, LINE_ELLIPSE = 0.9, 1.0, 2.4
S_POINT, S_OPT_HALO, S_OPT_FACE = 8.0, 32.0, 19.0
MIN_WINDOW, WINDOW_PAD, ENVELOPE_SCALE = 0.12, 1.18, 2.15
TICK_STEP_CHOICES = ((0.20, 0.05), (0.30, 0.10), (0.40, 0.10), (0.50, 0.10), (1.00, 0.25), (1.25, 0.25))
FOOTNOTE = "RMS: distance from each seasonal cloud to the full-data optimum"

# ---------------------------------------------------- 等比放大（版式完全不变）
_S = 2.40
FS_PT = 8.0 * _S
FIG_W_MM = 177.0 * _S
M_LEFT_MM, M_RIGHT_MM, M_TOP_MM, M_BOTTOM_MM = 5.0*_S, 3.8*_S, 2.0*_S, 12.0*_S
GUTTER_X_MM, GUTTER_Y_MM = 5.0*_S, 4.0*_S
TITLE_H_MM, TICK_H_MM = 3.4*_S, 4.2*_S
BOX_H_MM = 22.5 * _S
STATS_PAD_MM, STATS_PITCH_MM = 1.2*_S, 4.2*_S
STATS_GAP_MM = 1.2 * _S
TICK_LEN_PT, TICK_PAD_PT = 2.0*_S, 2.0*_S
S_POINT, S_OPT_HALO, S_OPT_FACE = 13.0*_S*_S, 48.0*_S*_S, 30.0*_S*_S



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
    required = {"prov", "season", "selected_k", "selected_b", "full_data_k", "full_data_b", "status"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Detail CSV is missing columns: {sorted(missing)}")
    df = df[df["status"].astype(str).str.lower() == "success"].copy()
    for col in ["selected_k", "selected_b", "full_data_k", "full_data_b"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["season"].isin(SEASON_ORDER)].copy()
    if df.empty:
        raise ValueError("No successful seasonal validation rows found in detail CSV.")
    return df


def _province_order(detail_df: pd.DataFrame) -> list[str]:
    provinces = sorted(detail_df["prov"].dropna().astype(str).unique().tolist())
    return sorted(provinces, key=lambda item: PROVINCE_NAME_EN.get(item, item))


def _panel_labels(count: int) -> list[str]:
    """(a)..(z), (aa)..(ad) —— ESSD：面板标签用括号内小写字母。"""
    return [f"({chr(97 + i)})" if i < 26 else f"(a{chr(97 + i - 26)})" for i in range(count)]


def _window(province_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """(k_center, b_center, half_span, tick_step)。

    k 与 b 用相同窗宽（便于跨省比较）；窗宽再向外各放半个刻度步长，
    使首末刻度文字落在图框内 —— 否则它会被相邻子图抢位压住。
    """
    k_values = pd.concat([province_df["selected_k"], province_df["full_data_k"]]).dropna()
    b_values = pd.concat([province_df["selected_b"], province_df["full_data_b"]]).dropna()
    if k_values.empty or b_values.empty:
        raise ValueError("Province panel has no valid k/b values to plot.")
    span = max(float(k_values.max() - k_values.min()),
               float(b_values.max() - b_values.min()),
               MIN_WINDOW) * WINDOW_PAD
    step = TICK_STEP_CHOICES[-1][1]
    for bound, candidate in TICK_STEP_CHOICES:
        if span <= bound + 1e-9:
            step = float(candidate)
            break
    return (float(k_values.min() + k_values.max()) / 2.0,
            float(b_values.min() + b_values.max()) / 2.0,
            span / 2.0 + step / 2.0,
            step)


def _ticks(lower: float, upper: float, step: float) -> list[float]:
    lo = int(np.ceil(lower / step - 1e-9))
    hi = int(np.floor(upper / step + 1e-9))
    return [i * step for i in range(lo, hi + 1)]


def _tick_label(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


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


def _stride_for_spacing(step_mm: float, needed_mm: float, count: int) -> int:
    """刻度文字互不重叠所需的抽样步长：每隔几个刻度标一次。

    判据是"相邻刻度的物理间距 ≥ 文字宽度 + 间隙"，不是"n 个字加起来不超过图框"——
    后者会在刻度只占图框一部分时误判通过。
    """
    if step_mm <= 0 or count <= 1:
        return 1
    return max(1, min(count, int(np.ceil(needed_mm / step_mm - 1e-9))))


def _panel_geometry(layout: dict, index: int) -> tuple[float, float, float, float]:
    """返回格子 (格左边 mm, 图框左 mm, 图框底 mm, 图框顶 mm)。"""
    row, col = divmod(index, GRID_COLS)
    cell_left = M_LEFT_MM + col * (layout["cell_w"] + GUTTER_X_MM)
    block_top = layout["fig_h"] - M_TOP_MM - row * (layout["block_h"] + GUTTER_Y_MM)
    box_top = block_top - TITLE_H_MM
    return cell_left, cell_left + layout["slot"], box_top - BOX_H_MM, box_top


def _make_layout(y_label_slot_mm: float) -> dict:
    cell_w = (FIG_W_MM - M_LEFT_MM - M_RIGHT_MM - (GRID_COLS - 1) * GUTTER_X_MM) / GRID_COLS
    stats_h = STATS_PAD_MM + _text_metrics_mm("0")[1] + STATS_PITCH_MM
    block_h = TITLE_H_MM + BOX_H_MM + TICK_H_MM + stats_h
    fig_h = M_TOP_MM + GRID_ROWS * block_h + (GRID_ROWS - 1) * GUTTER_Y_MM + M_BOTTOM_MM
    return {"cell_w": cell_w, "slot": y_label_slot_mm, "stats_h": stats_h,
            "box_w": cell_w - y_label_slot_mm, "block_h": block_h, "fig_h": fig_h}


def _y_label_slot(windows: dict[str, tuple[float, float, float, float]]) -> float:
    """y 刻度字槽宽 = 实际用到的刻度文字最大宽度 + 间隙（按实测字宽定，不靠估）。"""
    widths = []
    for k_center, b_center, half_span, step in windows.values():
        for center in (k_center, b_center):
            widths += [_text_metrics_mm(_tick_label(v))[0] for v in _ticks(center - half_span, center + half_span, step)]
    return max(widths) + 0.5


def _add_envelope(ax: plt.Axes, season_df: pd.DataFrame, season: str) -> None:
    points = season_df[["selected_k", "selected_b"]].dropna().to_numpy(dtype=float)
    if len(points) < 3:
        return
    center = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    ax.add_patch(Ellipse(
        xy=(center[0], center[1]),
        width=2.0 * ENVELOPE_SCALE * float(np.sqrt(eigvals[0])),
        height=2.0 * ENVELOPE_SCALE * float(np.sqrt(eigvals[1])),
        angle=float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))),
        facecolor="none", edgecolor=SEASON_COLORS[season],
        linestyle=(0, (3, 2)), linewidth=LINE_ELLIPSE, zorder=3,
    ))


def _dispersion_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (province, season), season_df in detail_df.groupby(["prov", "season"], sort=False):
        points = season_df[["selected_k", "selected_b"]].dropna()
        center = season_df[["full_data_k", "full_data_b"]].dropna().head(1)
        if points.empty or center.empty:
            continue
        ck, cb = float(center.iloc[0]["full_data_k"]), float(center.iloc[0]["full_data_b"])
        arr = points.to_numpy(dtype=float)
        dist = np.sqrt((arr[:, 0] - ck) ** 2 + (arr[:, 1] - cb) ** 2)
        rows.append({
            "prov": str(province), "season": str(season), "n_points": int(len(arr)),
            "full_data_k": ck, "full_data_b": cb,
            "cloud_centroid_k": float(arr[:, 0].mean()), "cloud_centroid_b": float(arr[:, 1].mean()),
            "centroid_shift": float(np.hypot(arr[:, 0].mean() - ck, arr[:, 1].mean() - cb)),
            "mean_distance_to_center": float(dist.mean()),
            "median_distance_to_center": float(np.median(dist)),
            "p90_distance_to_center": float(np.percentile(dist, 90)),
            "rms_distance_to_center": float(np.sqrt(np.mean(dist ** 2))),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["season"] = pd.Categorical(out["season"], categories=SEASON_ORDER, ordered=True)
        out = out.sort_values(["prov", "season"]).reset_index(drop=True)
    return out


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
def build_figure(detail_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    provinces = _province_order(detail_df)
    if len(provinces) > GRID_COLS * GRID_ROWS:
        raise ValueError(f"Too many provinces for the {GRID_COLS}x{GRID_ROWS} grid: {len(provinces)}")
    summary = _dispersion_summary(detail_df)
    windows = {p: _window(detail_df[detail_df["prov"] == p]) for p in provinces}
    layout = _make_layout(_y_label_slot(windows))
    fig_h = layout["fig_h"]
    print(f"  [版式] 画布 {FIG_W_MM:.1f} x {fig_h:.1f} mm | 格子宽 {layout['cell_w']:.2f} mm | "
          f"图框 {layout['box_w']:.2f} x {BOX_H_MM:.1f} mm | y 刻度字槽 {layout['slot']:.2f} mm")
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    axes: list[plt.Axes] = []

    for index, (province, label) in enumerate(zip(provinces, _panel_labels(len(provinces)))):
        province_df = detail_df[detail_df["prov"] == province].copy()
        k_center, b_center, half_span, step = windows[province]
        cell_left, box_left, box_bottom, box_top = _panel_geometry(layout, index)

        ax = fig.add_axes((box_left / FIG_W_MM, box_bottom / fig_h,
                           layout["box_w"] / FIG_W_MM, BOX_H_MM / fig_h))
        ax.set_label(f"{label} {province}")
        axes.append(ax)

        jitter = min(max(2 * half_span * 0.022, 0.006), 0.015)
        for season in SEASON_ORDER:
            season_df = province_df[province_df["season"] == season]
            if season_df.empty:
                continue
            _add_envelope(ax, season_df, season)
            ax.scatter(
                season_df["selected_k"].to_numpy(float) + rng.uniform(-jitter, jitter, len(season_df)),
                season_df["selected_b"].to_numpy(float) + rng.uniform(-jitter, jitter, len(season_df)),
                s=S_POINT, color=SEASON_COLORS[season], alpha=0.34,
                marker=SEASON_MARKERS[season], edgecolors="white", linewidths=0.2, zorder=2,
            )
            optimum = season_df[["full_data_k", "full_data_b"]].dropna().head(1)
            if not optimum.empty:
                ax.scatter(optimum["full_data_k"], optimum["full_data_b"], s=S_OPT_HALO,
                           facecolors="white", edgecolors="white", marker="D", zorder=4)
                ax.scatter(optimum["full_data_k"], optimum["full_data_b"], s=S_OPT_FACE,
                           facecolors=SEASON_COLORS[season], edgecolors="#222222",
                           linewidths=0.35, marker="D", alpha=0.96, zorder=5)

        ax.set_xlim(k_center - half_span, k_center + half_span)
        ax.set_ylim(b_center - half_span, b_center + half_span)
        ax.set_box_aspect(BOX_H_MM / layout["box_w"])

        xticks, yticks = _ticks(*ax.get_xlim(), step), _ticks(*ax.get_ylim(), step)
        xlabels = [_tick_label(v) for v in xticks]
        ylabels = [_tick_label(v) for v in yticks]
        x_step_mm = layout["box_w"] * step / (2 * half_span)
        y_step_mm = BOX_H_MM * step / (2 * half_span)
        xstride = _stride_for_spacing(
            x_step_mm, max((_text_metrics_mm(t)[0] for t in xlabels), default=0.0) + 0.5, len(xlabels))
        ystride = _stride_for_spacing(
            y_step_mm, _text_metrics_mm("0")[1] + 0.6, len(ylabels))
        ax.set_xticks(xticks, [t if i % xstride == 0 else "" for i, t in enumerate(xlabels)])
        ax.set_yticks(yticks, [t if i % ystride == 0 else "" for i, t in enumerate(ylabels)])
        ax.tick_params(labelsize=FS_PT, length=TICK_LEN_PT, width=0.4, pad=TICK_PAD_PT)
        ax.grid(True, color="#E6E6E6", linewidth=LINE_GRID, alpha=0.9)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#BEBEBE")
            spine.set_linewidth(LINE_SPINE)

        # 面板标签 + 省名：图框上方，横向居中于图框
        fig.text((box_left + layout["box_w"] / 2.0) / FIG_W_MM, (box_top + 0.65 * _S) / fig_h,
                 f"{label} {PROVINCE_NAME_EN.get(province, province)}",
                 ha="center", va="bottom", fontsize=FS_PT, color="black")

    grid_left = M_LEFT_MM + layout["slot"]
    grid_cx = (grid_left + layout["box_w"] * GRID_COLS + GUTTER_X_MM * (GRID_COLS - 1) / 2.0) / FIG_W_MM
    grid_cy = (M_TOP_MM + (fig_h - M_TOP_MM - M_BOTTOM_MM) / 2.0) / fig_h
    fig.text(grid_cx, 10.4 * _S / fig_h, "k", ha="center", va="center", fontsize=FS_PT)
    fig.text(1.7 * _S / FIG_W_MM, grid_cy, "b", ha="center", va="center", rotation="vertical", fontsize=FS_PT)

    handles = [Line2D([0], [0], marker=SEASON_MARKERS[s], linestyle="", color=SEASON_COLORS[s],
                      markeredgecolor="white", markeredgewidth=0.25, markersize=3.6 * _S, alpha=0.75)
               for s in SEASON_ORDER]
    handles.append(Line2D([0], [0], marker="D", linestyle="", markerfacecolor="#8A8A8A",
                          markeredgecolor="#222222", markeredgewidth=0.35, markersize=3.2 * _S))
    fig.legend(handles, [*(SEASON_LABELS[s] for s in SEASON_ORDER), "Full Data Result"],
               loc="lower center", bbox_to_anchor=(0.5, 4.4 * _S / fig_h), ncol=5, frameon=False,
               fontsize=FS_PT, columnspacing=2.2, handletextpad=0.9, handlelength=1.4, borderpad=0.2)

    _verify_layout(fig, axes, fig_h)

    # plot_paper_v2: output stem follows the paper's figure number (was "fig13_...").
    png_path = output_dir / "Fig13_wind_kb_stability.png"
    pdf_path = output_dir / "Fig13_wind_kb_stability.pdf"
    metrics_path = output_dir / "Fig13_wind_kb_stability_metrics.csv"
    fig.savefig(png_path, dpi=200, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    summary.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    return png_path, pdf_path, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fig. 13: province-level seasonal k/b parameter clouds (ESSD style).")
    parser.add_argument("--detail-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail_csv = _resolve_existing_path(DETAIL_CANDIDATES, args.detail_csv)
    detail_df = _load_detail(detail_csv)
    png_path, pdf_path, metrics_path = build_figure(detail_df, args.output_dir)
    print(f"Detail CSV: {detail_csv}")
    print(f"Output PNG: {png_path}")
    print(f"Output PDF: {pdf_path}")
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
