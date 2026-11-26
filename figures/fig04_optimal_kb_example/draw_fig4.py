# -*- coding: utf-8 -*-
"""Fig 4 版式：每省 2×2（春夏秋冬），单一风速面 W(k,b)。

版式要求（用户 2026-11-23）：
  · 只画风速面，不再四行两列
  · 2×2，子图标题放季节，标号 (a)(b)(c)(d)
  · 画填色等高线形状
  · 四个子图用【统一色标范围】
  · 图例（colorbar）在最右侧、顶天立地
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

SURF = r"D:\HongyuEnergy Copilot\_run\essd_survey\surfaces"
OUT = r"D:\HongyuEnergy Copilot\_run\essd_survey\paper_figs"
os.makedirs(OUT, exist_ok=True)

SEASONS = ["spring", "summer", "autumn", "winter"]
CN2EN = {
 "上海":"Shanghai","云南":"Yunnan","内蒙古":"Inner Mongolia","北京":"Beijing",
 "吉林":"Jilin","四川":"Sichuan","天津":"Tianjin","宁夏":"Ningxia","安徽":"Anhui",
 "山东":"Shandong","山西":"Shanxi","广东":"Guangdong","广西":"Guangxi",
 "新疆":"Xinjiang","江苏":"Jiangsu","江西":"Jiangxi","河北":"Hebei",
 "河南":"Henan","浙江":"Zhejiang","海南":"Hainan","湖北":"Hubei",
 "湖南":"Hunan","甘肃":"Gansu","福建":"Fujian","贵州":"Guizhou",
 "辽宁":"Liaoning","重庆":"Chongqing","陕西":"Shaanxi","青海":"Qinghai",
 "黑龙江":"Heilongjiang"}
def en(prov):
    return CN2EN.get(prov, prov)
LABEL = {"spring": "Spring (MAM)", "summer": "Summer (JJA)",
         "autumn": "Autumn (SON)", "winter": "Winter (DJF)"}
TAGS = ["(a)", "(b)", "(c)", "(d)"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
})


def load(prov, season):
    p = os.path.join(SURF, "%s_%s.npz" % (prov, season))
    if not os.path.isfile(p):
        return None
    try:
        z = np.load(p)
        d = dict(k=z["k_grid"], b=z["b_grid"], W=z["W"].astype(float),
                 pk=z["path_k"], pb=z["path_b"], pw=z["path_W"],
                 km4=float(z["k_m4"]), bm4=float(z["b_m4"]),
                 wm4=float(z["w_at_m4"]), nper=int(z["n_wind_periods"]))
        z.close()
        return d
    except Exception as e:
        print("  ! 跳过损坏/不可读文件 %s_%s.npz: %s"
              % (prov, season, type(e).__name__), flush=True)
        return None


def draw(prov, data, vmin, vmax):
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.86, bottom=0.09,
                          top=0.93, wspace=0.20, hspace=0.22)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    levels = np.linspace(vmin, vmax, 26)
    cmap = plt.get_cmap("viridis")

    cf = None
    for ax, season, tag in zip(axes, SEASONS, TAGS):
        d = data[season]
        k, b, W = d["k"], d["b"], np.ma.masked_invalid(d["W"])
        cf = ax.contourf(k, b, W, levels=levels, cmap=cmap, extend="both")
        ax.contour(k, b, W, levels=levels[::5], colors="k",
                   linewidths=0.35, alpha=0.45)
        # 路径 b*(k)
        ax.plot(d["pk"], d["pb"], color="#ff3b30", lw=1.4, zorder=5,
                label=r"path $b^{*}(k)$")
        # 最优点
        ax.plot([d["km4"]], [d["bm4"]], marker="*", ms=15, mfc="#ffffff",
                mec="#ff3b30", mew=1.3, zorder=6,
                label=r"optimum $(k^{*},b^{*})$")
        # 搜索框
        ax.set_xlim(k.min(), k.max())
        ax.set_ylim(b.min(), b.max())
        ax.set_title("%s %s" % (tag, LABEL[season]), fontsize=9.5, loc="left")
        if season in ("spring", "autumn"):
            ax.set_ylabel(r"$b$ (m s$^{-1}$)")
        if season in ("autumn", "winter"):
            ax.set_xlabel(r"$k$ (–)")
        ax.tick_params(direction="out", length=3, width=0.7)
        ax.text(0.03, 0.06,
                r"$k^{*}$=%.2f  $b^{*}$=%+.2f" % (d["km4"], d["bm4"]) + "\n"
                + r"$%s$=%.1f %%" % ("\\mathcal{E}^{W}", d["wm4"]),
                transform=ax.transAxes, fontsize=7.4, va="bottom",
                bbox=dict(fc="white", ec="0.6", lw=0.5, alpha=0.85, pad=2))
    axes[0].legend(loc="upper right", fontsize=7.2, framealpha=0.9,
                   borderpad=0.3, handlelength=1.4)

    # 顶天立地的 colorbar
    cax = fig.add_axes([0.885, 0.09, 0.022, 0.84])
    cb = fig.colorbar(cf, cax=cax, extend="both")
    cb.set_label(r"Wind-speed SMAPE $\mathcal{E}^{W}$ (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("%s — province–season wind-speed calibration surface"
                 % en(prov), fontsize=10.5, y=0.975)
    base = os.path.join(OUT, "fig4_%s" % en(prov).replace(" ",""))
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print("  ✓ %s" % base)


def main():
    provs = sorted({os.path.basename(f).rsplit("_", 1)[0]
                    for f in glob.glob(os.path.join(SURF, "*.npz"))})
    print("本地有面的省:", provs)
    cache = {}
    for prov in provs:
        cache[prov] = {}
        for s in SEASONS:
            d = load(prov, s)
            if d is None:
                continue
            cache[prov][s] = d
    skipped = []
    for prov in provs:
        if len(cache[prov]) != 4:
            miss = [s for s in SEASONS if s not in cache[prov]]
            skipped.append((prov, miss))
            print("  - 跳过 %s：仅 %d/4 季，缺 %s"
                  % (prov, len(cache[prov]), ",".join(miss)), flush=True)
            continue
        # 每张图内 4 个子图统一色标（2–98 分位，避免极端角点压扁对比）
        vals = np.concatenate(
            [cache[prov][s]["W"][np.isfinite(cache[prov][s]["W"])].ravel()
             for s in SEASONS])
        vmin = float(np.floor(np.percentile(vals, 2)))
        vmax = float(np.ceil(np.percentile(vals, 98)))
        print("  %s colorbar %.0f - %.0f %%" % (en(prov), vmin, vmax))
        draw(prov, cache[prov], vmin, vmax)

    print("\n生成 %d 张；跳过 %d 个省（数据不全）" % (len(provs) - len(skipped), len(skipped)))
    for prov, miss in skipped:
        print("  SKIP %s  缺 %s" % (prov, ",".join(miss)))
    with open(os.path.join(OUT, "fig4_skipped.txt"), "w",
              encoding="utf-8") as f:
        f.write("生成 %d 张图（省）\n" % (len(provs) - len(skipped)))
        for prov, miss in skipped:
            f.write("跳过 %s：缺 %s\n" % (prov, ",".join(miss)))


if __name__ == "__main__":
    main()
