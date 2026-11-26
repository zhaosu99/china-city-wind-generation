# -*- coding: utf-8 -*-
"""Fig 4 定稿样式（用户 2026-11-23 指定）：
  · 子图标题居中、写全：Spring (MAM) wind speed SMAPE surface
  · 去掉大标题
  · 配色 viridis_r
  · 横轴只写 k（无单位括号）
  · 双色等高线（黑+白半透明），解决深端看不清
  · 最优点：白底黑边实心圆
"""
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF = Path(os.environ.get(
    "WINDPAPER_SURF_DIR", Path(__file__).resolve().parents[2] / "data" / "wind_surfaces"
))
OUT = Path(os.environ.get(
    "WINDPAPER_FIG_OUT", Path(__file__).resolve().parent / "output"
))
os.makedirs(OUT, exist_ok=True)

SEASONS = ["spring", "summer", "autumn", "winter"]
TITLE = {"spring": "(a) Spring (MAM) wind speed SMAPE surface",
         "summer": "(b) Summer (JJA) wind speed SMAPE surface",
         "autumn": "(c) Autumn (SON) wind speed SMAPE surface",
         "winter": "(d) Winter (DJF) wind speed SMAPE surface"}
CN2EN = {"河北": "Hebei", "吉林": "Jilin", "安徽": "Anhui", "河南": "Henan",
         "内蒙古": "Inner Mongolia", "上海": "Shanghai", "云南": "Yunnan",
         "北京": "Beijing", "四川": "Sichuan", "天津": "Tianjin", "宁夏": "Ningxia",
         "山东": "Shandong", "山西": "Shanxi", "广东": "Guangdong", "广西": "Guangxi",
         "新疆": "Xinjiang", "江苏": "Jiangsu", "江西": "Jiangxi", "浙江": "Zhejiang",
         "海南": "Hainan", "湖北": "Hubei", "湖南": "Hunan", "甘肃": "Gansu",
         "福建": "Fujian", "贵州": "Guizhou", "辽宁": "Liaoning", "重庆": "Chongqing",
         "陕西": "Shaanxi", "青海": "Qinghai", "黑龙江": "Heilongjiang"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.8,
})

PROV = "吉林"
CMAP = "viridis_r"


def load(prov, season):
    z = np.load(os.path.join(SURF, "%s_%s.npz" % (prov, season)))
    return dict(k=z["k_grid"], b=z["b_grid"], W=z["W"].astype(float),
                pk=z["path_k"], pb=z["path_b"],
                km4=float(z["k_m4"]), bm4=float(z["b_m4"]),
                wm4=float(z["w_at_m4"]))


def draw(prov, tag=""):
    data = {s: load(prov, s) for s in SEASONS}
    vals = np.concatenate([data[s]["W"][np.isfinite(data[s]["W"])].ravel()
                           for s in SEASONS])
    # 色标取整到 5 的倍数，让 colorbar 刻度是干净数字
    vmin = float(np.floor(np.percentile(vals, 2) / 5.0) * 5)
    vmax = float(np.ceil(np.percentile(vals, 98) / 5.0) * 5)

    # 无大标题 → 顶部只需给子图标题留空间
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.855, bottom=0.085,
                          top=0.945, wspace=0.22, hspace=0.30)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    cmap = plt.get_cmap(CMAP)
    step = 5.0
    levels = np.arange(vmin, vmax + 1e-9, step)

    cf = None
    for ax, season in zip(axes, SEASONS):
        d = data[season]
        W = np.ma.masked_invalid(d["W"])
        cf = ax.pcolormesh(d["k"], d["b"], W, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        # 双色等高线：浅底见黑、深底见白
        cl = np.arange(vmin + 2.5, vmax, 2.5)
        ax.contour(d["k"], d["b"], W, levels=cl, colors="black",
                   linewidths=0.35, alpha=0.30)
        ax.contour(d["k"], d["b"], W, levels=cl, colors="white",
                   linewidths=0.35, alpha=0.40)
        ax.plot(d["pk"], d["pb"], color="#ff2d2d", lw=1.5, zorder=5,
                solid_capstyle="round", label=r"path $b^{*}(k)$")
        ax.plot([d["km4"]], [d["bm4"]], marker="o", ms=5.5,
                mfc="white", mec="black", mew=1.1, zorder=6,
                label=r"optimum $(k^{*},b^{*})$")
        ax.set_xlim(d["k"].min(), d["k"].max())
        ax.set_ylim(d["b"].min(), d["b"].max())
        ax.set_title(TITLE[season], fontsize=8.6, loc="center", pad=4)
        if season in ("spring", "autumn"):
            ax.set_ylabel(r"$b$ (m s$^{-1}$)")
        if season in ("autumn", "winter"):
            ax.set_xlabel(r"$k$")
        ax.tick_params(direction="out", length=3, width=0.7)
        ax.text(0.03, 0.06,
                r"$k^{*}$=%.2f  $b^{*}$=%+.2f" % (d["km4"], d["bm4"]) + "\n"
                + r"min SMAPE = %.1f %%" % d["wm4"],
                transform=ax.transAxes, fontsize=7.2, va="bottom", zorder=7,
                bbox=dict(fc="white", ec="0.6", lw=0.5, alpha=0.72, pad=2))
    axes[0].legend(loc="upper right", fontsize=7.0, framealpha=0.92,
                   borderpad=0.3, handlelength=1.5)

    cax = fig.add_axes([0.878, 0.085, 0.020, 0.86])
    cb = fig.colorbar(cf, cax=cax, extend="both")
    cb.set_label("SMAPE of wind speed (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    base = os.path.join(OUT, "final_%s%s" % (CN2EN.get(prov, prov), tag))
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print("  ✓", os.path.basename(base))


if __name__ == "__main__":
    draw(PROV, "")
