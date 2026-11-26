# -*- coding: utf-8 -*-
"""Unified province-panel template for Fig 13 and Fig 14 (Scheme A: deviation coordinates).

Template = same canvas/typography as Fig 11/12:
  canvas 21.0 x 29.7 in (A4 ratio) @200 dpi, 5 cols x 6 rows,
  margins left 0.068 / right 0.985 / top 0.976 / bottom 0.050,
  wspace 0.13, hspace 0.26, title pad 14,
  title 23 pt, ticks 17 pt, axis labels 27 pt, legend 19 pt.

All panels share identical axes and tick steps, so panels are directly comparable:
  Fig 13 : x = k - k*   (wind-speed multiplier deviation)
           y = b - b*   (m s-1 deviation)
  Fig 14 : x = (K - K*)/K* * 100                     (%)
           y = (B - B*)/mean_monthly_generation * 100 (% of the province's typical month)
The full-sample optimum is a single red diamond at the origin (0, 0).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
from config_paths import (
    M4, OUTD, OUTDIR, OUT, PROD23, PROD24, KB, KB_FULL, OFF, CAP, UTIL, CV_WIND, FULL_WIND, CV_GEN,
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter, MultipleLocator

GEN_PROD23 = r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\prov_month_from_city.csv"
GEN_PROD24 = r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\prov_month_2425.csv"
GEN_OFF = r"D:\HongyuEnergy Copilot\_run\prod_2b\work\official_prov_month.csv"
GEN_CAP = r"D:\能源系统软件\_finals\deliver\cap_adj_applied_2425.csv"
GEN_UTIL = r"D:\能源系统软件\_finals\deliver\util_applied_2425.csv"

EN = {"安徽":"Anhui","北京":"Beijing","重庆":"Chongqing","福建":"Fujian","甘肃":"Gansu","广东":"Guangdong",
      "广西":"Guangxi","贵州":"Guizhou","海南":"Hainan","河北":"Hebei","黑龙江":"Heilongjiang","河南":"Henan",
      "湖北":"Hubei","湖南":"Hunan","江苏":"Jiangsu","江西":"Jiangxi","吉林":"Jilin","辽宁":"Liaoning",
      "内蒙古":"Inner Mongolia","宁夏":"Ningxia","青海":"Qinghai","陕西":"Shaanxi","山东":"Shandong",
      "上海":"Shanghai","山西":"Shanxi","四川":"Sichuan","天津":"Tianjin","新疆":"Xinjiang",
      "云南":"Yunnan","浙江":"Zhejiang"}
PY2CN = {"anhui":"安徽","beijing":"北京","chongqing":"重庆","fujian":"福建","gansu":"甘肃","guangdong":"广东",
 "guangxi":"广西","guizhou":"贵州","hainan":"海南","hebei":"河北","heilongjiang":"黑龙江","henan":"河南",
 "hubei":"湖北","hunan":"湖南","jiangsu":"江苏","jiangxi":"江西","jilin":"吉林","liaoning":"辽宁","neimenggu":"内蒙古",
 "ningxia":"宁夏","qinghai":"青海","shaanxi":"陕西","shandong":"山东","shanghai":"上海","shanxi":"山西",
 "sichuan":"四川","tianjin":"天津","xinjiang":"新疆","yunnan":"云南","zhejiang":"浙江"}
SEASONS = ["spring", "summer", "autumn", "winter"]
SEASON_LABEL = {"spring": "MAM", "summer": "JJA", "autumn": "SON", "winter": "DJF"}
SEASON_COLOR = {"spring": "#7AA874", "summer": "#F4A261", "autumn": "#C97C5D", "winter": "#5C88A5"}
SEASON_MARKER = {"spring": "o", "summer": "^", "autumn": "s", "winter": "X"}
OPT_COLOR = "#B03A2E"

FS_TITLE, FS_TICK, FS_LABEL, FS_LEG = 6.0, 6.0, 7.0, 6.0
GRID_COLS, GRID_ROWS = 5, 6
FIG13_XLIM, FIG13_XSTEP = (-0.035, 0.035), 0.015
FIG13_YLIM, FIG13_YSTEP = (-0.08, 0.08), 0.04
FIG14_XLIM, FIG14_XSTEP = (-24.0, 24.0), 10.0
FIG14_YLIM, FIG14_YSTEP = (-13.0, 13.0), 5.0

plt.rcParams.update({"font.family": "Arial", "axes.unicode_minus": False})


def fmt_compact(v, _pos=None):
    av = abs(v)
    if v == 0:
        return "0"
    if av >= 1e4:
        return "%.0fk" % (v / 1e3)
    if av >= 1e3:
        return "%.1fk" % (v / 1e3)
    if av >= 10:
        return "%.0f" % v
    if av >= 1:
        return "%.1f" % v
    if av >= 0.01:
        return "%.2f" % v
    return "%.3f" % v


def panel_letter(i):
    s, n = "", i
    while True:
        s = chr(ord("a") + n % 26) + s
        n = n // 26 - 1
        if n < 0:
            break
    return "(%s)" % s


def new_figure(bottom=0.050):
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(6.768, 9.57))
    fig.subplots_adjust(left=0.068, right=0.985, top=0.976, bottom=bottom,
                        wspace=0.32, hspace=0.42)
    return fig, axes


def tight_window(lo, hi, step):
    """Tick-aligned window around the data, minimal span; at least 3 ticks."""
    start = np.floor(lo / step) * step
    end = np.ceil(hi / step) * step
    while (end - start) / step < 2 - 1e-9:
        mid = 0.5 * (lo + hi)
        start = np.floor((mid - step) / step) * step
        end = start + 2 * step
    return float(start), float(end)


def shifted_window(lo, hi, span, step):
    """A window of fixed span, shifted to contain [lo, hi] and aligned to step."""
    start = np.floor((0.5 * (lo + hi) - 0.5 * span) / step) * step
    while start > lo:
        start -= step
    while start + span < hi:
        start += step
    return float(start), float(start + span)


def fixed_ticks_alt(ax, xstep, ystep, xdec=1, ydec=0):
    """Fixed tick positions; label every second tick only."""
    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    def fmt(step, dec):
        def f(v, _p=None):
            idx = int(round(v / step))
            if idx % 2:
                return ""
            return f"{v:.{dec}f}"
        return FuncFormatter(f)
    ax.xaxis.set_major_formatter(fmt(xstep, xdec))
    ax.yaxis.set_major_formatter(fmt(ystep, ydec))


def fixed_ticks_only(ax, xstep, ystep, xdec=2, ydec=2):
    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"{v:.{xdec}f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"{v:.{ydec}f}"))


def fixed_step_axis(ax, xstep, ystep, xdec=2, ydec=2):
    def expand(lo, hi, step):
        lo = np.floor(lo / step) * step
        hi = np.ceil(hi / step) * step
        # at least 2 intervals => 3 tick positions
        while (hi - lo) / step < 2 - 1e-9:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - step, mid + step
        return float(lo), float(hi)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(*expand(x0, x1, xstep))
    ax.set_ylim(*expand(y0, y1, ystep))
    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"{v:.{xdec}f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"{v:.{ydec}f}"))


def fixed_axis(ax, xlim, xstep, ylim, ystep):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_compact))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_compact))
    ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    labels = ax.get_xticklabels()
    if labels:
        labels[0].set_horizontalalignment("left")
        labels[-1].set_horizontalalignment("right")
    ax.grid(lw=0.5, color="#E9E9E9")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel_title(ax, i, name_cn):
    ax.set_title("%s %s" % (panel_letter(i), EN.get(name_cn, name_cn)), fontsize=FS_TITLE, pad=6)


def add_ellipse(ax, xs, ys, color, lw=2.0):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    nx = (np.asarray(xs) - x0) / (x1 - x0)
    ny = (np.asarray(ys) - y0) / (y1 - y0)
    pts = np.column_stack([nx, ny])
    cov = np.cov(pts, rowvar=False)
    if not np.isfinite(cov).all():
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    ax.add_patch(Ellipse(xy=pts.mean(axis=0),
                         width=2.0 * 2.15 * float(np.sqrt(vals[0])),
                         height=2.0 * 2.15 * float(np.sqrt(vals[1])), angle=angle,
                         transform=ax.transAxes, facecolor="none", edgecolor=color,
                         linestyle=(0, (4, 3)), linewidth=lw, alpha=0.95, zorder=4))


# =============================================================== Fig 13 (absolute k, b)
def window_limits(vals, pad_frac=0.10, min_span=0.02):
    lo = float(np.min(vals)); hi = float(np.max(vals))
    span = max(hi - lo, min_span)
    return lo - span * pad_frac, hi + span * pad_frac


def _fmt_for_span(span):
    if span >= 10:
        dec = 0
    elif span >= 1:
        dec = 1
    elif span >= 0.1:
        dec = 2
    elif span >= 0.01:
        dec = 3
    else:
        dec = 4

    def _f(v, _pos=None):
        if v == 0:
            return "0"
        return ("%." + str(dec) + "f") % v

    return FuncFormatter(_f)


def style_axis_auto(ax, nbins=3):
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.xaxis.set_major_formatter(_fmt_for_span(xspan))
    ax.yaxis.set_major_formatter(_fmt_for_span(yspan))
    ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.grid(lw=0.5, color="#E9E9E9")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def make_fig13():
    cv = pd.read_csv(CV_WIND, encoding="utf-8-sig")
    full = pd.read_csv(FULL_WIND, encoding="utf-8-sig").set_index(["prov", "season"])
    provs = sorted(cv.prov.unique(), key=lambda p: EN.get(p, p))
    _k = np.concatenate([cv["k"].values.astype(float), full["k"].values.astype(float)])
    _b = np.concatenate([cv["b"].values.astype(float), full["b"].values.astype(float)])
    _kx0, _kx1 = float(np.nanmin(_k)), float(np.nanmax(_k))
    _by0, _by1 = float(np.nanmin(_b)), float(np.nanmax(_b))
    _kxpad = 0.04 * (_kx1 - _kx0); _bypad = 0.04 * (_by1 - _by0)
    GX13 = (_kx0 - _kxpad, _kx1 + _kxpad)
    GY13 = (_by0 - _bypad, _by1 + _bypad)
    fig, axes = new_figure()
    rng = np.random.default_rng(42)

    for i, (ax, prov) in enumerate(zip(axes.ravel(), provs)):
        g = cv[cv.prov == prov]
        ks, bs = [], []
        for season in SEASONS:
            gs = g[g.season == season]
            ks.extend(gs.k.values.tolist()); bs.extend(gs.b.values.tolist())
            ks.append(float(full.loc[(prov, season), "k"])); bs.append(float(full.loc[(prov, season), "b"]))
        klo, khi = window_limits(ks); blo, bhi = window_limits(bs)
        # Same X/Y span in every panel => identical physical scale.
        # Tick spacing chosen from the most dispersed panel; labels every 2nd tick.
        x0, x1 = shifted_window(klo, khi, 0.25, 0.05)
        y0, y1 = shifted_window(blo, bhi, 0.40, 0.10)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        style_axis_auto(ax)
        fixed_ticks_alt(ax, 0.05, 0.10, xdec=2, ydec=2)
        jx = 0.008 * (x1 - x0); jy = 0.008 * (y1 - y0)
        for season in SEASONS:
            gs = g[g.season == season]
            if gs.empty:
                continue
            ax.scatter(gs.k.values + rng.uniform(-jx, jx, len(gs)),
                       gs.b.values + rng.uniform(-jy, jy, len(gs)),
                       s=20, color=SEASON_COLOR[season], alpha=0.55,
                       marker=SEASON_MARKER[season], edgecolors="white", linewidths=0.3, zorder=3)
            add_ellipse(ax, gs.k.values, gs.b.values, SEASON_COLOR[season])
            fk = float(full.loc[(prov, season), "k"]); fb = float(full.loc[(prov, season), "b"])
            ax.scatter([fk], [fb], s=32, marker="D", facecolors=SEASON_COLOR[season],
                       edgecolors="#222222", linewidths=0.3, zorder=6)
        panel_title(ax, i, prov)

    for ax in axes.ravel()[len(provs):]:
        ax.axis("off")
    fig.supxlabel("k (wind-speed multiplier)", fontsize=FS_LABEL, y=0.004)
    fig.supylabel("b (m s$^{-1}$)", fontsize=FS_LABEL, x=0.012)
    handles = [Line2D([], [], marker=SEASON_MARKER[s], linestyle="", color=SEASON_COLOR[s],
                      markersize=5, label=SEASON_LABEL[s]) for s in SEASONS]
    handles.append(Line2D([], [], marker="D", linestyle="", markerfacecolor="#8A8A8A",
                          markeredgecolor="#222222", markeredgewidth=1.0, markersize=5,
                          label="Full Data Result"))
    handles.append(Line2D([], [], color="#666666", linestyle=(0, (4, 3)), linewidth=2.0,
                          label="2σ envelope"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.016), ncol=6,
               frameon=False, fontsize=FS_LEG, handletextpad=0.6, columnspacing=2.2)
    out = os.path.join(OUTDIR, "Fig13_wind_kb_stability_febfix.png")
    fig.savefig(out, dpi=600, facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
    plt.close(fig)
    print("saved", out)


# =============================================================== Fig 14
def _load_gen_pool():
    def load_prod(path, factor=False):
        d = pd.read_csv(path, encoding="utf-8-sig")
        d["cn"] = d["prov"].map(PY2CN)
        d = d.dropna(subset=["cn"]).copy()
        d["year"] = d["ym"].astype(str).str[:4].astype(int)
        d["month"] = d["ym"].astype(str).str[4:6].astype(int)
        if factor:
            cap = pd.read_csv(GEN_CAP, encoding="utf-8-sig")
            capf = cap[["省简称","年","月","cap_ratio"]].rename(columns={"省简称":"cn","年":"year","月":"month","cap_ratio":"cap_f"})
            util = pd.read_csv(GEN_UTIL, encoding="utf-8-sig")
            ucol = "factor" if "factor" in util.columns else ("util_applied" if "util_applied" in util.columns else util.columns[-1])
            uf = util[["省简称","年","月",ucol]].rename(columns={"省简称":"cn","年":"year","月":"month",ucol:"util_f"})
            f = capf.merge(uf, on=["cn","year","month"], how="outer")
            f["factor"] = f["cap_f"].fillna(1.0) * f["util_f"].fillna(1.0)
            d = d.merge(f[["cn","year","month","factor"]], on=["cn","year","month"], how="left")
            d["factor"] = d["factor"].fillna(1.0)
            d["gen_MWh"] = d["gen_MWh"] * d["factor"]
        d["mk"] = d["month"].apply(lambda m: 1 if m <= 2 else int(m))
        return d.groupby(["cn","year","mk"], as_index=False)["gen_MWh"].sum()

    m23 = load_prod(GEN_PROD23).rename(columns={"gen_MWh": "x"})
    m23 = m23[(m23.year >= 2013) & (m23.year <= 2023)]
    _feb = pd.read_csv(GEN_PROD23, encoding="utf-8-sig")
    _feb["cn"] = _feb["prov"].map(PY2CN)
    _feb = _feb.dropna(subset=["cn"]).copy()
    _feb["year"] = _feb["ym"].astype(str).str[:4].astype(int)
    _feb["month"] = _feb["ym"].astype(str).str[4:6].astype(int)
    _feb = _feb[_feb["month"] == 2].groupby(["cn", "year"], as_index=False)["gen_MWh"].sum().rename(columns={"gen_MWh": "feb_x"})
    m23 = m23.merge(_feb, on=["cn", "year"], how="left")
    _fm = (m23.mk == 1) & m23.year.isin([2014, 2015]) & m23.feb_x.notna()
    m23.loc[_fm, "x"] = m23.loc[_fm, "feb_x"]
    m23 = m23.drop(columns=["feb_x"])
    m24 = load_prod(GEN_PROD24, factor=True).rename(columns={"gen_MWh": "x"})
    m24 = m24[m24.year >= 2024]
    model = pd.concat([m23, m24], ignore_index=True)
    o = pd.read_csv(GEN_OFF, encoding="utf-8-sig")
    o = o[o["value_MWh"] > 0].copy()
    o["mk"] = o["month"].apply(lambda m: 1 if m <= 2 else int(m))
    obs = o.groupby(["cn","year","mk"], as_index=False)["value_MWh"].sum().rename(columns={"value_MWh":"y"})
    obs = obs[(obs.year >= 2013) & (obs.year <= 2025)]
    df = model.merge(obs, on=["cn","year","mk"], how="inner")
    rows = []
    for cn, g in df.groupby("cn"):
        a = g.x.values.astype(float); b = g.y.values.astype(float)
        K = np.cov(a, b, ddof=1)[0, 1] / np.var(a, ddof=1)
        B = b.mean() - K * a.mean()
        if B < 0:
            K = float((a * b).sum() / (a * a).sum()); B = 0.0
        rows.append(dict(prov=cn, K=float(K), B=float(B), meangen=float(b.mean() / 1e3)))
    return pd.DataFrame(rows).set_index("prov")


def make_fig14():
    cv = pd.read_csv(CV_GEN, encoding="utf-8-sig")
    full = _load_gen_pool()
    cv["K"] = cv["K"].astype(float)
    cv["B_GWh"] = cv["B"].astype(float) / 1e3
    provs = sorted(cv.prov.unique(), key=lambda p: EN.get(p, p))
    _K = np.concatenate([cv["K"].values.astype(float), full["K"].values.astype(float)])
    _B = np.concatenate([cv["B_GWh"].values.astype(float), full["B"].values.astype(float) / 1e3])
    _Kx0, _Kx1 = float(np.nanmin(_K)), float(np.nanmax(_K))
    _By0, _By1 = float(np.nanmin(_B)), float(np.nanmax(_B))
    _Kxpad = 0.04 * (_Kx1 - _Kx0); _Bypad = 0.04 * (_By1 - _By0)
    GX14 = (_Kx0 - _Kxpad, _Kx1 + _Kxpad)
    GY14 = (_By0 - _Bypad, _By1 + _Bypad)
    fig, axes = new_figure()
    rng = np.random.default_rng(42)

    for i, (ax, prov) in enumerate(zip(axes.ravel(), provs)):
        g = cv[cv.prov == prov]
        fk = float(full.loc[prov, "K"]); fb = float(full.loc[prov, "B"]) / 1e3
        klo, khi = window_limits(list(g.K.values) + [fk], min_span=0.08)
        blo, bhi = window_limits(list(g.B_GWh.values) + [fb], min_span=0.04)
        # Same axes logic as Fig13: identical span in every panel, shifted windows,
        # fixed tick spacing, labels every other tick.
        x0, x1 = shifted_window(klo, khi, 1.20, 0.30)
        y0, y1 = shifted_window(blo, bhi, 800.0, 200.0)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        style_axis_auto(ax)
        fixed_ticks_alt(ax, 0.30, 200.0, xdec=1, ydec=0)
        jx = 0.008 * (x1 - x0); jy = 0.008 * (y1 - y0)
        ax.scatter(g.K.values + rng.uniform(-jx, jx, len(g)),
                   g.B_GWh.values + rng.uniform(-jy, jy, len(g)),
                   s=20, color="#4C72B0", alpha=0.50, edgecolors="white", linewidths=0.3, zorder=3)
        add_ellipse(ax, g.K.values, g.B_GWh.values, "#3A6EA5")
        ax.scatter([fk], [fb], s=32, marker="D", facecolors=OPT_COLOR,
                   edgecolors="#222222", linewidths=0.3, zorder=6)
        panel_title(ax, i, prov)

    for ax in axes.ravel()[len(provs):]:
        ax.axis("off")
    fig.supxlabel("K (generation multiplier)", fontsize=FS_LABEL, y=0.004)
    fig.supylabel("B (GWh)", fontsize=FS_LABEL, x=0.012)
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#4C72B0", markersize=5, alpha=0.6,
               label="50 x 80/20 resamples"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor=OPT_COLOR,
               markeredgecolor="#222222", markeredgewidth=1.0, markersize=5,
               label="Full Data Result"),
        Line2D([], [], color="#3A6EA5", linestyle=(0, (4, 3)), linewidth=2.0, label="2σ envelope"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.016), ncol=3,
               frameon=False, fontsize=FS_LEG, handletextpad=0.6, columnspacing=2.6)
    out = os.path.join(OUTDIR, "Fig14_generation_kb_stability_febfix.png")
    fig.savefig(out, dpi=600, facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
    plt.close(fig)
    print("saved", out)


# =============================================================== Fig 15
GEN_BOX_COLOR = "#8E44AD"


def make_fig15():
    w = pd.read_csv(CV_WIND, encoding="utf-8-sig")
    g = pd.read_csv(CV_GEN, encoding="utf-8-sig")
    w["smape_pct"] = 100.0 * w.ho_smape
    provs = sorted(w.prov.unique(), key=lambda p: EN.get(p, p))
    fig, axes = new_figure()
    positions = np.arange(5)
    labels = [SEASON_LABEL[s] for s in SEASONS] + ["Gen"]
    colors = [SEASON_COLOR[s] for s in SEASONS] + [GEN_BOX_COLOR]

    for i, (ax, prov) in enumerate(zip(axes.ravel(), provs)):
        data = [w[(w.prov == prov) & (w.season == s)].smape_pct.values for s in SEASONS]
        data.append(g[g.prov == prov].smape.values.astype(float))
        bp = ax.boxplot(data, positions=positions, widths=0.58, patch_artist=True,
                        showfliers=True,
                        medianprops={"color": "#2A2A2A", "linewidth": 1.3},
                        whiskerprops={"color": "#666666", "linewidth": 0.9},
                        capprops={"color": "#666666", "linewidth": 0.9},
                        boxprops={"edgecolor": "#666666", "linewidth": 0.8},
                        flierprops={"marker": "o", "markersize": 3.0,
                                    "markerfacecolor": "none", "markeredgecolor": "#888888",
                                    "markeredgewidth": 0.6})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.62)
        rng = np.random.default_rng(42 + i)
        for pos, vals, c in zip(positions, data, colors):
            ax.scatter(pos + rng.uniform(-0.16, 0.16, len(vals)), vals,
                       s=5, color=c, alpha=0.25, linewidths=0, zorder=2)
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(5, 40)
        ax.set_xticks(positions, labels, fontsize=FS_TICK)
        ax.set_yticks([5, 15, 25, 35])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: "%.0f" % v))
        ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.5, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.grid(axis="y", lw=0.35, color="#E9E9E9")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        panel_title(ax, i, prov)

    for ax in axes.ravel()[len(provs):]:
        ax.axis("off")
    fig.supylabel("Holdout SMAPE (%)", fontsize=FS_LABEL, x=0.012)
    handles = [Line2D([], [], marker="s", linestyle="", color=SEASON_COLOR[s],
                      markersize=5, alpha=0.62, label=SEASON_LABEL[s]) for s in SEASONS]
    handles.append(Line2D([], [], marker="s", linestyle="", color=GEN_BOX_COLOR,
                          markersize=5, alpha=0.62, label="Gen"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.016), ncol=5,
               frameon=False, fontsize=FS_LEG, handletextpad=0.6, columnspacing=2.2)
    out = os.path.join(OUTDIR, "Fig15_holdout_smape_boxplots_febfix.png")
    fig.savefig(out, dpi=600, facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    make_fig13()
    make_fig14()
    make_fig15()
