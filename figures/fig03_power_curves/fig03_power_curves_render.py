# -*- coding: utf-8 -*-
"""
fig4 功率曲线图 —— 生成脚本（可复算）

输入（同目录 / 数据盘）：
    fig4_turbine_power_curves.json            机型匹配表
    D:/Data/turbines/wind_turbines.csv        机型参数（Rated power / Cut-in / Rated / Cut-out wind speed）

输出（同目录）：
    fig4_v2_darker.png / .pdf

相对原脚本的改动：
    1) 5 台机型 = 每个额定功率等级取装机容量最大的那一台（同档不再重叠）
    2) 实线 = 风场聚合曲线（对场内风速空间分布做高斯平滑）
       虚线 = 同一机型的单机曲线（未平滑），与所属实线同色
    3) 高斯平滑改为「以 i 为中心重新生成核」的正确实现
       —— 原脚本 kernel[: end - start] 在左边界会把核整体右移 (4*sigma - w)，
          使低风速段出力被高估（w=5 m/s 时约高估 2 倍）
    4) 横轴 0-30 m/s
    5) 配色 = 用户指定莫兰迪色相，亮度重排到 L* 58-76（比原色深、仍可印）
    6) **保留原图的两级虚线网格**（major 深 / minor 浅），网格置于数据之下
"""
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
FS_PT = 8.0          # ESSD：全图统一 8 pt（用户 2026-09-23 定）
XMAX_MS = 35.0       # 横轴上限 (m/s)
VERBOSE_SIZES = False

# plot_paper_v2: keep the original file names, route them into output/.
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "fig4_essd.png"
OUT_PDF = OUT_DIR / "fig4_essd.pdf"

# ------------------------------------------------------------------ 配色
HUE_SRC = ["#B6E2DC", "#B8FABF", "#C6C3E1", "#FFCFD1", "#E9E4AF"]  # 薄荷青 嫩绿 淡紫 浅粉 芥末黄
L_TARGETS = np.linspace(58, 76, 5)

M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
M_INV = np.linalg.inv(M_XYZ)
M_PRO = np.array([[0.152286, 1.052583, -0.204868],
                  [0.114503, 0.786281, 0.099216],
                  [-0.003882, -0.048116, 1.051998]])
M_DEU = np.array([[0.367322, 0.860646, -0.227968],
                  [0.280085, 0.672501, 0.047413],
                  [-0.011820, 0.042940, 0.968881]])


def hx2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def rgb2hx(c):
    return "#" + "".join(f"{int(round(min(max(v, 0), 1) * 255)):02X}" for v in c)


def s2l(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def l2s(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def rgb2lab(rgb):
    xyz = M_XYZ @ s2l(np.asarray(rgb)) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab2rgb(lab):
    L, a, b = lab
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787)
    return np.clip(l2s(M_INV @ (xyz * np.array([0.95047, 1.0, 1.08883]))), 0, 1)


def sim(rgb, M):
    return np.clip(l2s(M @ s2l(np.asarray(rgb))), 0, 1)


def minpair(v):
    return min(float(np.linalg.norm(np.asarray(v[i]) - np.asarray(v[j])))
               for i, j in itertools.combinations(range(len(v)), 2))


def pick_colours():
    """保持色相、重排亮度，使色盲下最小色差最大（120 种分配穷举）"""
    best = None
    for sc in (0.9, 1.0, 1.1):
        for perm in itertools.permutations(range(5)):
            labs = [np.array([L_TARGETS[perm[k]]] + list(rgb2lab(hx2rgb(HUE_SRC[k]))[1:] * sc))
                    for k in range(5)]
            rl = [lab2rgb(x) for x in labs]
            w = min(minpair([rgb2lab(sim(c, M_DEU)) for c in rl]),
                    minpair([rgb2lab(sim(c, M_PRO)) for c in rl]))
            if best is None or w > best[0]:
                best = (w, sc, [rgb2hx(c) for c in rl])
    return best


# ------------------------------------------------------------------ 曲线
DW = 0.01
WG = np.arange(0.0, 40.0 + DW, DW)


def raw_cf(cin, vr, co, w=WG):
    return np.where(w < cin, 0.0,
                    np.where(w < vr, (w ** 3 - cin ** 3) / (vr ** 3 - cin ** 3),
                             np.where(w < co, 1.0, 0.0)))


def farm_smooth(cfr, si=0.6, ss=0.2, trunc=4.0):
    """边界正确的变 sigma 高斯卷积：核以 i 为中心重新生成"""
    sig = (si + ss * WG) / DW
    out = np.zeros_like(cfr)
    n = len(WG)
    for i in range(n):
        r = int(np.ceil(trunc * sig[i]))
        j = np.arange(max(0, i - r), min(n, i + r + 1))
        k = np.exp(-0.5 * ((j - i) / sig[i]) ** 2)
        out[i] = np.sum(cfr[j] * k) / k.sum()
    return out


def load_models():
    spec = importlib.util.spec_from_file_location(
        "f4", str(HERE / "fig03_turbine_power_curves.py"))
    f4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f4)
    cfg = json.load(open(HERE / "fig03_turbine_power_curves.json", encoding="utf-8"))
    ref = f4._prepare_reference_turbines(f4.CSV_PATH)
    rows = []
    for s in cfg["selections"]:
        r = ref[(ref["Company"].astype(str).str.strip() == s["reference_company"]) &
                (ref["Model"].astype(str).str.strip() == s["reference_model"])].iloc[0]
        m = dict(inv=s["inventory_model"], company=s["reference_company"], model=str(r["Model"]),
                 P=float(r["Rated power"]), cin=float(r["Cut-in wind speed"]),
                 vr=float(r["Rated wind speed"]), cout=float(r["Cut-out wind speed"]),
                 cap=float(s["inventory_capacity_mw"]))
        cfr = raw_cf(m["cin"], m["vr"], m["cout"])
        m["single"] = cfr * m["P"]
        m["farm"] = farm_smooth(cfr) * m["P"]
        rows.append(m)
    byc = {}
    for m in sorted(rows, key=lambda d: -d["cap"]):
        byc.setdefault(m["P"], m)
    return [byc[p] for p in sorted(byc)]


# ------------------------------------------------------------------ 渲染
def render(models, colours, legend_anchor, axes_rect, save=False,
           fs=7.0, hl=2.0, label_mode="full", loc="upper right",
           note="Single turbine power curve\n(without Gaussian smoothing)"):
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         # Arial 缺 U+207B（上标减号），负指数用 mathtext 写；
                         # fontset 必须是 custom，否则会引入 DejaVu → 违反 ESSD「只用一种字体族」
                         "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold"})
    MM = 1 / 25.4
    fig = plt.figure(figsize=(177 * MM, 106 * MM))
    ax = fig.add_axes(axes_rect)

    ax.set_xlim(0, XMAX_MS)
    ax.set_ylim(0, 3.2)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # ---- 原图的网格：major 深 / minor 浅，两级虚线；置于数据之下
    ax.set_axisbelow(True)
    ax.grid(which="major", linestyle="--", linewidth=0.55, alpha=0.35, color="#8f8f8f")
    ax.grid(which="minor", linestyle="--", linewidth=0.55, alpha=0.22, color="#b8b8b8")

    ax.tick_params(which="major", direction="out", length=2.4, width=0.5,
                   labelsize=FS_PT, pad=1.8)
    ax.tick_params(axis="x", which="minor", direction="out", length=1.3, width=0.4)
    ax.tick_params(axis="y", which="minor", length=0)   # y 次刻度太密，只留网格线不留刻度

    # 单机参照层（先画，在下）
    for m, c in zip(models, colours):
        ax.plot(WG, m["single"], color=c, lw=0.5, ls=(0, (4, 2.5)), zorder=2)

    # 主曲线（按平台高度从高到低进图例，便于和图里的位置对上）
    order = sorted(range(len(models)), key=lambda i: -models[i]["P"])
    handles = []
    for i in order:
        m, c = models[i], colours[i]
        if label_mode == "full":
            lab = f'{m["company"]} | {m["model"]}'
        elif label_mode == "code":
            lab = m["inv"]
        else:
            lab = m["model"]
        h, = ax.plot(WG, m["farm"], color=c, lw=1.3, zorder=3,
                     label=lab, solid_capstyle="round")
        handles.append(h)
    hd, = ax.plot([], [], color="#8A8A8A", lw=0.5, ls=(0, (4, 2.5)),
                  label=note)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.5)
    ax.set_xlabel("Wind speed (m s$^{-1}$)", fontsize=FS_PT, labelpad=2.5)
    ax.set_ylabel("Power (MW)", fontsize=FS_PT, labelpad=3.0)

    leg = ax.legend(handles=handles + [hd], loc=loc,
                    bbox_to_anchor=legend_anchor, fontsize=fs, frameon=True,
                    handlelength=hl, handletextpad=0.35, labelspacing=0.40,
                    borderpad=0.28, borderaxespad=0.0)
    leg.get_frame().set_edgecolor("#C8C8C8")
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)

    # 图例是否压到曲线：在图例 bbox 内数曲线采样点
    fig.canvas.draw()
    bb_fig = leg.get_window_extent().transformed(fig.transFigure.inverted())
    bb = leg.get_window_extent().transformed(ax.transData.inverted())
    hit = 0
    for m in models:
        for y in (m["farm"], m["single"]):
            hit += int(((WG >= bb.x0) & (WG <= bb.x1) & (y >= bb.y0) & (y <= bb.y1)).sum())

    if VERBOSE_SIZES:
        got = []
        got += [t.get_fontsize() for t in ax.texts]
        got += [t.get_fontsize() for t in ax.get_xticklabels() + ax.get_yticklabels()]
        got += [ax.xaxis.label.get_fontsize(), ax.yaxis.label.get_fontsize()]
        got += [t.get_fontsize() for t in leg.get_texts()]
        print(f"   [字号自检] 图中所有文字对象 {len(got)} 个，取值集合 = {sorted(set(got))}")
    if save:
        fig.savefig(OUT_PNG, dpi=300, facecolor="white")
        fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)
    return hit, bb, bb_fig


if __name__ == "__main__":
    SEL = load_models()
    score, chroma, COLS = pick_colours()

    print("= 曲线")
    for k, (m, c) in enumerate(zip(SEL, COLS)):
        print(f'   {m["P"]:.1f} MW  {m["inv"]:<9} {m["company"]:<9} {c}   '
              f'ΔL*={100 - rgb2lab(hx2rgb(c))[0]:5.1f}  (原色 {HUE_SRC[k]} '
              f'ΔL*={100 - rgb2lab(hx2rgb(HUE_SRC[k]))[0]:4.1f})')
    labD = [rgb2lab(sim(hx2rgb(c), M_DEU)) for c in COLS]
    labP = [rgb2lab(sim(hx2rgb(c), M_PRO)) for c in COLS]
    print(f'\n   色盲最差 ΔE = {min(minpair(labD), minpair(labP)):.1f}   (彩度系数 {chroma})')
    print('   网格: major #8f8f8f alpha0.35 / minor #b8b8b8 alpha0.22，均 0.55pt 虚线，置于数据之下')

    # ---- 定稿几何：横轴拉到 40 m/s，图例放回绘图区右上角（25 m/s 右侧是空白区）----
    CANVAS_MM = 177.0
    LEFT, BOT, HGT = 0.070, 0.115, 0.855
    RIGHT_MARGIN_MM = 3.0                 # 绘图区右边界到画布右边的余量（图例在区内，不需要大 gap）
    LEG_INSET = 1.0                       # 图例相对绘图区右边界内缩（mm）
    x0 = LEFT * CANVAS_MM
    axw = CANVAS_MM - x0 - RIGHT_MARGIN_MM
    anc_x = 1.0 - LEG_INSET / axw
    rect = [LEFT, BOT, axw / CANVAS_MM, HGT]

    h, bb, bf = render(SEL, COLS, (anc_x, 1.0), rect, save=False, fs=FS_PT, hl=1.0)
    print(f"\n= 定稿几何｜画布 {CANVAS_MM:.0f} mm，横轴 0–{XMAX_MS:.0f} m/s，全图 {FS_PT} pt")
    print(f"   绘图区 {axw:.1f} mm（{x0:.1f} → {x0+axw:.1f} mm），右侧仅留 {RIGHT_MARGIN_MM:.0f} mm")
    print(f"   图例在图内：x {bb.x0:.1f} → {bb.x1:.1f} m/s")
    print(f"   ★ 图例左边界 = {bb.x0:.1f} m/s，25 m/s 那根线在 {25.0:.0f} m/s"
          f" → 相距 {bb.x0-25.0:+.1f} m/s  " + ("✅ 没碰到" if bb.x0 > 25.0 else "❌ 仍然碰到"))
    print(f"   图例框内曲线点 {h} 个  " + ("✅ 不压任何线" if h == 0 else "❌ 仍压线"))
    print(f"   画面边距：上 {(1-BOT-HGT)*106:.1f} / 下 {BOT*106:.1f} / 左 {x0:.1f} mm")
    render(SEL, COLS, (anc_x, 1.0), rect, save=True, fs=FS_PT, hl=1.0)
    print(f"\n已输出: {OUT_PNG.name} / {OUT_PDF.name}")
