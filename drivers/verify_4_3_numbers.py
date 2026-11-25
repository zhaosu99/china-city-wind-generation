# -*- coding: utf-8 -*-
"""核实 §4.3（Figure 9 / Figure 10）要引用的全部数字。"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, r"D:\HKU PhD\wind data paper\plot\piling")
import fig8_generation_dynamics as F  # noqa: E402

CITY = r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\data\01_city_final_1989_2023"


def op(p):
    try:
        return xr.open_dataset(p)
    except Exception:
        return xr.open_dataset(p, engine="scipy")


# ---------- 逐城 2023 的 onshore / offshore ----------
rows = []
for f in sorted(glob.glob(os.path.join(CITY, "*__*", "full_year", "gen.nc"))):
    d = op(f)
    t = d["time"].values
    yr = np.array([str(x)[:4] for x in t], dtype="U4")
    m = yr == "2023"
    tot = float(np.nansum(d["total_gen"].values[m])) if m.any() else 0.0
    on = float(np.nansum(d["onshore_gen"].values[m])) if ("onshore_gen" in d and m.any()) else 0.0
    off = float(np.nansum(d["offshore_gen"].values[m])) if ("offshore_gen" in d and m.any()) else 0.0
    attrs = d.attrs
    d.close()
    rows.append(dict(prov=attrs.get("province_short_name"), city=attrs.get("city_short_name"),
                     total=tot, onshore=on, offshore=off))
c = pd.DataFrame(rows)
print("城市数 %d ｜ 2023 合计 %.2f TWh ｜ 陆上 %.2f ｜ 海上 %.2f"
      % (len(c), c.total.sum() / 1e6, c.onshore.sum() / 1e6, c.offshore.sum() / 1e6))
print("  陆上占比 %.2f%% ｜ 海上占比 %.2f%%"
      % (100 * c.onshore.sum() / c.total.sum(), 100 * c.offshore.sum() / c.total.sum()))

ok = []


def chk(label, stated, actual, tol=0.08):
    good = abs(stated - actual) <= tol
    ok.append(good)
    print("   %-40s 文中 %7s ｜ 实测 %7.2f ｜ %s" % (label, stated, actual, "OK" if good else "**不符**"))


print()
print("=== Figure 10：海上发电量的省级份额 ===")
off_p = c.groupby("prov").offshore.sum().sort_values(ascending=False)
off_tot = off_p.sum()
for p, v in off_p.head(6).items():
    print("   %-6s %7.2f TWh  %5.2f%%" % (p, v / 1e6, 100 * v / off_tot))
S = lambda p: 100 * off_p.get(p, 0) / off_tot
print()
chk("江苏占海上", 31.1, S("江苏"))
chk("广东占海上", 25.9, S("广东"))
chk("福建占海上", 16.7, S("福建"))
chk("山东占海上", 12.7, S("山东"))
chk("浙江占海上", 8.5, S("浙江"))
chk("海上五省合计", 94.8, sum(S(p) for p in ["江苏", "广东", "福建", "山东", "浙江"]))

print()
print("=== Figure 10：海上发电量的城市份额 ===")
for prov, cities, stated in [("江苏", ["南通", "盐城"], [49.4, 48.1]),
                             ("广东", ["阳江"], [39.7]),
                             ("福建", ["福州", "莆田"], [58.4, 40.8]),
                             ("浙江", ["宁波"], [41.4])]:
    base = c[(c.prov == prov)].offshore.sum()
    for ct, st in zip(cities, stated):
        v = c[(c.prov == prov) & (c.city == ct)].offshore.sum()
        chk("%s %s 占本省海上" % (prov, ct), st, 100 * v / base if base else np.nan)

print()
print("=== Figure 10：海上占本省总发电量 ===")
byp = c.groupby("prov")[["total", "offshore"]].sum()
for p, st in [("江苏", 68.3), ("广东", 83.5), ("福建", 75.0), ("浙江", 78.7),
              ("河北", 1.8), ("辽宁", 9.7), ("山东", 27.8)]:
    if p in byp.index:
        chk("%s 海上占本省" % p, st, 100 * byp.loc[p, "offshore"] / byp.loc[p, "total"])

# ---------- Fig 9：电网 / 省级 / 城市份额 ----------
print()
print("=== Figure 9：电网与省级份额 ===")
p_ = F._load_city_daily_long(CITY)
p_["time"] = pd.to_datetime(p_["time"])
p_["year"] = p_["time"].dt.year
d23 = p_[p_["year"] == 2023]
g = d23.groupby("grid_label")["generation_mwh"].sum().sort_values(ascending=False)
gt = g.sum()
for k, v in g.items():
    print("   电网 %-4s %8.2f TWh  %5.2f%%" % (k, v / 1e6, 100 * v / gt))
pv = d23.groupby("province_group_label")["generation_mwh"].sum().sort_values(ascending=False)
pt = pv.sum()
print()
print("   省级前六：")
for k, v in pv.head(6).items():
    print("     %-6s %8.2f TWh  %5.2f%%" % (k, v / 1e6, 100 * v / pt))
chk("华北占全国", 29.5, 100 * g.get("华北", 0) / gt)
chk("西北占全国", 20.9, 100 * g.get("西北", 0) / gt)
chk("华北+西北", 50.3, 100 * (g.get("华北", 0) + g.get("西北", 0)) / gt)
chk("其余五网上限", 14.0, max(100 * g[k] / gt for k in g.index if k not in ("华北", "西北")))
chk("其余五网下限", 2.5, min(100 * g[k] / gt for k in g.index if k not in ("华北", "西北")))
chk("蒙西占全国", 10.8, 100 * pv.get("蒙西", 0) / pt)
chk("新疆占全国", 8.3, 100 * pv.get("新疆", 0) / pt)
chk("河北占全国", 7.3, 100 * pv.get("河北", 0) / pt)
chk("山东占全国", 5.7, 100 * pv.get("山东", 0) / pt)
chk("江苏占全国", 5.7, 100 * pv.get("江苏", 0) / pt)
chk("省级前五合计", 37.8, sum(100 * pv.get(k, 0) / pt for k in ["蒙西", "新疆", "河北", "山东", "江苏"]))

print()
print("=== Figure 9：城市占本省份额 ===")
for prov, ct, st in [("河北", "张家口", 64.3), ("新疆", "哈密", 55.4),
                     ("甘肃", "酒泉", 74.0), ("江苏", "盐城", 46.2)]:
    sub = d23[d23["province_label"] == prov]
    tot = sub["generation_mwh"].sum()
    v = sub[sub["city_label"] == ct]["generation_mwh"].sum()
    chk("%s %s 占本省" % (prov, ct), st, 100 * v / tot if tot else np.nan)

print()
print("全部核对:", all(ok), "（%d/%d）" % (sum(ok), len(ok)))
