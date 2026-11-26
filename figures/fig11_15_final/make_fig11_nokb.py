# -*- coding: utf-8 -*-
"""Two standalone per-province scatter figures (February-alignment-fix variant) (English, large fonts, few ticks, big points).

Fig 1: X = calibrated theoretical (K*x+B), Y = official; 2013-2023, coloured by year.
Fig 2: 2024 (blue) / 2025 (red); 28 provinces (Beijing/Hainan not shown), two empty slots bottom-right.
In-panel text: sMAPE, R^2, ratio (no box, no outline).
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from figure_paths import (
    M4, PROD23, PROD24, KB, OFF, CAP, UTIL, OUTDIR,
    CV_WIND, FULL_WIND, CV_GEN, GEN_PROD23, GEN_PROD24,
    GEN_OFF, GEN_CAP, GEN_UTIL,
)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors as mcolors, lines as mlines
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

OOS_EXCLUDE = ["北京", "海南"]

PY2CN = {"anhui":"安徽","beijing":"北京","chongqing":"重庆","fujian":"福建","gansu":"甘肃","guangdong":"广东",
 "guangxi":"广西","guizhou":"贵州","hainan":"海南","hebei":"河北","heilongjiang":"黑龙江","henan":"河南",
 "hubei":"湖北","hunan":"湖南","jiangsu":"江苏","jiangxi":"江西","jilin":"吉林","liaoning":"辽宁","neimenggu":"内蒙古",
 "ningxia":"宁夏","qinghai":"青海","shaanxi":"陕西","shandong":"山东","shanghai":"上海","shanxi":"山西",
 "sichuan":"四川","tianjin":"天津","xinjiang":"新疆","yunnan":"云南","zhejiang":"浙江"}
EN = {"安徽":"Anhui","北京":"Beijing","重庆":"Chongqing","福建":"Fujian","甘肃":"Gansu","广东":"Guangdong",
      "广西":"Guangxi","贵州":"Guizhou","海南":"Hainan","河北":"Hebei","黑龙江":"Heilongjiang","河南":"Henan",
      "湖北":"Hubei","湖南":"Hunan","江苏":"Jiangsu","江西":"Jiangxi","吉林":"Jilin","辽宁":"Liaoning",
      "内蒙古":"Inner Mongolia","宁夏":"Ningxia","青海":"Qinghai","陕西":"Shaanxi","山东":"Shandong",
      "上海":"Shanghai","山西":"Shanxi","四川":"Sichuan","天津":"Tianjin","新疆":"Xinjiang",
      "云南":"Yunnan","浙江":"Zhejiang"}

rcParams.update({"font.family": "Arial", "axes.unicode_minus": False, "font.size": 6})

TITLE_FS = 6.0
ANNOT_FS = 6.0
TICK_FS = 6.0
LABEL_FS = 7.0
SUPTITLE_FS = 6.0


def load_prod(path, factor=False):
    d = pd.read_csv(path, encoding="utf-8-sig")
    d["cn"] = d["prov"].map(PY2CN)
    d = d.dropna(subset=["cn"]).copy()
    d["year"] = d["ym"].astype(str).str[:4].astype(int)
    d["month"] = d["ym"].astype(str).str[4:6].astype(int)
    if factor:
        cap = pd.read_csv(CAP, encoding="utf-8-sig")
        capf = cap[["省简称","年","月","cap_ratio"]].rename(columns={"省简称":"cn","年":"year","月":"month","cap_ratio":"cap_f"})
        util = pd.read_csv(UTIL, encoding="utf-8-sig")
        ucol = "factor" if "factor" in util.columns else ("util_applied" if "util_applied" in util.columns else util.columns[-1])
        uf = util[["省简称","年","月",ucol]].rename(columns={"省简称":"cn","年":"year","月":"month",ucol:"util_f"})
        f = capf.merge(uf, on=["cn","year","month"], how="outer")
        f["factor"] = f["cap_f"].fillna(1.0) * f["util_f"].fillna(1.0)
        d = d.merge(f[["cn","year","month","factor"]], on=["cn","year","month"], how="left")
        d["factor"] = d["factor"].fillna(1.0)
        d["gen_MWh"] = d["gen_MWh"] * d["factor"]
    d["mk"] = d["month"].apply(lambda m: 1 if m <= 2 else int(m))
    return d.groupby(["cn","year","mk"], as_index=False)["gen_MWh"].sum()


def load_obs():
    o = pd.read_csv(OFF, encoding="utf-8-sig")
    o = o[o["value_MWh"] > 0].copy()
    o["mk"] = o["month"].apply(lambda m: 1 if m <= 2 else int(m))
    return o.groupby(["cn","year","mk"], as_index=False)["value_MWh"].sum()


def smape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat); m = den > 0
    return 100.0 * np.mean(np.abs(yhat[m] - y[m]) / den[m]) if m.any() else np.nan


def r2score(y, pred):
    """Squared Pearson correlation between simulated and observed series (always in [0,1])."""
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return np.nan
    return float(np.corrcoef(y, pred)[0, 1] ** 2)


def fmt_compact(v, _pos=None):
    av = abs(v)
    if av >= 1e6:
        return "%.1fM" % (v / 1e6)
    if av >= 1e4:
        return "%.0fk" % (v / 1e3)
    if av >= 1e3:
        return "%.1fk" % (v / 1e3)
    if av >= 10:
        return "%.0f" % v
    return ("%.1f" % v).rstrip("0").rstrip(".")


def setup_axis(ax, g):
    lo = min(g.calib_GWh.min(), g.y_GWh.min())
    hi = max(g.calib_GWh.max(), g.y_GWh.max())
    pad = (hi - lo) * 0.08 + 1e-6
    lo = max(0, lo - pad); hi = hi + pad
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.1, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(lw=0.4, color="#E9E9E9")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TICK_FS, length=2.5, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_compact))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_compact))


def annotate(ax, lines, fs=ANNOT_FS):
    ax.text(0.035, 0.968, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=fs, linespacing=1.12, zorder=6, color="black")


_FP16 = FontProperties(family="Arial", size=18.0)


def put_metrics(ax, fig, rows):
    """Fixed top-left block: label column, value column starts one space after the widest label."""
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer)
    y0 = 0.985
    dy = 0.085
    # measure the widest label once, in axes fraction
    probe = ax.text(0.035, y0, "SMAPE", transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.0, zorder=6, color="black")
    bb = probe.get_window_extent(renderer=renderer)
    vx = (bb.x1 - ax_bb.x0) / ax_bb.width + 0.075
    probe.remove()
    for i, (lab, val) in enumerate(rows):
        yy = y0 - i * dy
        ax.text(0.035, yy, lab, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.0, zorder=6, color="black")
        ax.text(vx, yy, val, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.0, zorder=6, color="black")


def panel_letter(i):
    """(a)...(z), (aa), (ab), ..."""
    s = ""
    n = i
    while True:
        s = chr(ord("a") + n % 26) + s
        n = n // 26 - 1
        if n < 0:
            break
    return "(%s)" % s


def annotate_auto(ax, g, lines, fs=ANNOT_FS, exclude=()):
    """Put the text in the emptiest corner of the panel (no box, no outline)."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = (g.calib_GWh.values - x0) / (x1 - x0)
    y = (g.y_GWh.values - y0) / (y1 - y0)
    regions = {"TL": int(((x < 0.42) & (y > 0.58)).sum()),
               "TR": int(((x > 0.58) & (y > 0.58)).sum()),
               "BL": int(((x < 0.42) & (y < 0.42)).sum()),
               "BR": int(((x > 0.58) & (y < 0.42)).sum())}
    for k in exclude:
        regions.pop(k, None)
    corner = min(regions, key=regions.get)
    fx, fy, ha, va = {"TL": (0.035, 0.965, "left", "top"),
                      "TR": (0.965, 0.965, "right", "top"),
                      "BL": (0.035, 0.035, "left", "bottom"),
                      "BR": (0.965, 0.035, "right", "bottom")}[corner]
    ax.text(fx, fy, "\n".join(lines), transform=ax.transAxes, va=va, ha=ha,
            fontsize=fs, linespacing=1.12, zorder=6, color="black")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    m23 = load_prod(PROD23).rename(columns={"gen_MWh": "x"})
    m24 = load_prod(PROD24, factor=True).rename(columns={"gen_MWh": "x"})
    obs = load_obs().rename(columns={"value_MWh": "y"})
    kb = pd.read_csv(KB, encoding="utf-8-sig").set_index("prov")
    provs = sorted(set(m23.cn) & set(kb.index), key=lambda c: EN.get(c, c))

    d23 = m23.merge(obs, on=["cn","year","mk"], how="inner")
    d23 = d23[(d23.cn.isin(provs)) & (d23.year >= 2013) & (d23.year <= 2023)].copy()
    # --- Feb alignment fix (option C): 2014/2015 Jan-Feb period uses model February only ---
    _feb = pd.read_csv(PROD23, encoding="utf-8-sig")
    _feb["cn"] = _feb["prov"].map(PY2CN)
    _feb = _feb.dropna(subset=["cn"]).copy()
    _feb["year"] = _feb["ym"].astype(str).str[:4].astype(int)
    _feb["month"] = _feb["ym"].astype(str).str[4:6].astype(int)
    _feb = _feb[_feb["month"] == 2].groupby(["cn", "year"], as_index=False)["gen_MWh"].sum().rename(columns={"gen_MWh": "feb_x"})
    d23 = d23.merge(_feb, on=["cn", "year"], how="left")
    _fm = (d23["mk"] == 1) & d23["year"].isin([2014, 2015]) & d23["feb_x"].notna()
    d23.loc[_fm, "x"] = d23.loc[_fm, "feb_x"]
    d23 = d23.drop(columns=["feb_x"])
    print("Feb-fix rows corrected in Fig 11 data:", int(_fm.sum()))
    # -----------------------------------------------------------------------------------------
    d23["K"] = d23.cn.map(kb["K"]); d23["B"] = d23.cn.map(kb["B"])
    d23["calib_GWh"] = (d23.K * d23.x + d23.B) / 1e3; d23["y_GWh"] = d23.y / 1e3
    d23["x_GWh"] = d23.x / 1e3

    d24 = m24.merge(obs, on=["cn","year","mk"], how="inner")
    d24 = d24[d24.year >= 2024].copy()
    d24["K"] = d24.cn.map(kb["K"]); d24["B"] = d24.cn.map(kb["B"])
    d24 = d24.dropna(subset=["K"])
    d24["calib_GWh"] = (d24.K * d24.x + d24.B) / 1e3; d24["y_GWh"] = d24.y / 1e3

    # ================= Figure 1 =================
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=2013, vmax=2023)
    fig, axes = plt.subplots(6, 5, figsize=(6.768, 9.57))
    fig.subplots_adjust(left=0.068, right=0.985, top=0.976, bottom=0.088, wspace=0.32, hspace=0.42)
    fig.canvas.draw()
    for i, (ax, cn) in enumerate(zip(axes.ravel(), provs)):
        g = d23[d23.cn == cn]
        sc = ax.scatter(g.calib_GWh, g.y_GWh, c=g.year, cmap=cmap, norm=norm, s=20, alpha=0.85,
                        edgecolors="white", linewidths=0.35, zorder=3)
        setup_axis(ax, g)
        ax.set_title("%s %s" % (panel_letter(i), EN.get(cn, cn)), fontsize=TITLE_FS, pad=6)
        rows = [("SMAPE", "%.1f%%" % smape(g.y_GWh.values, g.calib_GWh.values)),
                ("R$^2$", "%.3f" % r2score(g.y_GWh.values, g.calib_GWh.values))]
        put_metrics(ax, fig, rows)
    for ax in axes.ravel()[len(provs):]:
        ax.axis("off")
    fig.supxlabel("Monthly generation (GWh)", fontsize=LABEL_FS, y=0.006)
    fig.supylabel("Official monthly generation (GWh)", fontsize=LABEL_FS, x=0.013)
    p_left = axes[5, 0].get_position().x0
    p_right = axes[5, -1].get_position().x1
    cax = fig.add_axes([p_left, 0.050, (p_right - p_left) * 0.985, 0.007])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([2013, 2015, 2017, 2019, 2021, 2023])
    cb.ax.tick_params(labelsize=6.0, length=2.0, width=0.4, pad=1.5)
    cb.set_label("", fontsize=6, labelpad=2)
    cb.outline.set_linewidth(0.7)
    fig.savefig(os.path.join(OUTDIR, "Fig11_calibration_sample_scatter_febfix.png"), dpi=600)
    plt.close(fig)

    # ================= Figure 2 =================
    c24, c25 = "#1F77B4", "#D62728"
    panel_handles = [mlines.Line2D([], [], marker="o", linestyle="", color=c24, label="2024", markersize=6),
                     mlines.Line2D([], [], marker="o", linestyle="", color=c25, label="2025", markersize=6),
                     mlines.Line2D([], [], color="0.45", ls="--", lw=1.2, label="1:1")]
    fig, axes = plt.subplots(6, 5, figsize=(6.768, 9.57))
    fig.subplots_adjust(left=0.068, right=0.985, top=0.976, bottom=0.050, wspace=0.32, hspace=0.42)
    fig.canvas.draw()
    provs_oos = [cn for cn in provs if cn not in OOS_EXCLUDE]
    for i, (ax, cn) in enumerate(zip(axes.ravel(), provs_oos)):
        g1 = d24[(d24.cn == cn) & (d24.year == 2024)]
        g2 = d24[(d24.cn == cn) & (d24.year == 2025)]
        ax.scatter(g1.calib_GWh, g1.y_GWh, color=c24, s=34, alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
        ax.scatter(g2.calib_GWh, g2.y_GWh, color=c25, s=34, alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
        setup_axis(ax, pd.concat([g1, g2]))
        ax.set_title("%s %s" % (panel_letter(i), EN.get(cn, cn)), fontsize=TITLE_FS, pad=6)
        both = pd.concat([g1, g2])
        rows = [("SMAPE", "%.1f%%" % smape(both.y_GWh.values, both.calib_GWh.values)),
                ("R$^2$", "%.3f" % r2score(both.y_GWh.values, both.calib_GWh.values))]
        put_metrics(ax, fig, rows)
    empty_axes = list(axes.ravel()[len(provs_oos):])
    for ax in empty_axes:
        ax.axis("off")
    if empty_axes:
        legend_ax = axes[5, -1]
        legend_ax.legend(handles=panel_handles, labels=["2024", "2025", "1:1"],
                         loc="center right", bbox_to_anchor=(0.99, 0.5),
                         frameon=False, fontsize=8.0,
                         ncol=1, handlelength=1.0, handletextpad=0.3,
                         labelspacing=0.40, borderaxespad=0.0)
    fig.supxlabel("Calibrated monthly generation (GWh)", fontsize=LABEL_FS, y=0.006)
    fig.supylabel("Official monthly generation (GWh)", fontsize=LABEL_FS, x=0.013)
    fig.savefig(os.path.join(OUTDIR, "Fig12_outofsample_scatter_febfix.png"), dpi=600)
    plt.close(fig)

    # chat previews
    from PIL import Image
    os.makedirs(r"D:\_dsh_recover\drafts", exist_ok=True)
    for name in ("Fig11_calibration_sample_scatter_febfix.png", "Fig12_outofsample_scatter_febfix.png"):
        p = os.path.join(OUTDIR, name)
        im = Image.open(p); im.thumbnail((2000, 2000))
        im.save(os.path.join(r"D:\_dsh_recover\drafts", "prev_febfix_" + name))
    print("done", OUTDIR)


if __name__ == "__main__":
    main()
