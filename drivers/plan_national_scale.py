# -*- coding: utf-8 -*-
"""若 2005–2012 按全国官方年度值直接缩放，缩放因子与结果如何。"""
import numpy as np
import pandas as pd

D = r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod"
OFF_NAT = {2005: 1.927, 2006: 3.675, 2007: 5.710, 2008: 12.800,
           2009: 27.609, 2010: 44.620, 2011: 70.330, 2012: 95.980}   # TWh

PY2CN = {"anhui": "安徽", "beijing": "北京", "chongqing": "重庆", "fujian": "福建",
         "gansu": "甘肃", "guangdong": "广东", "guangxi": "广西", "guizhou": "贵州",
         "hainan": "海南", "hebei": "河北", "heilongjiang": "黑龙江", "henan": "河南",
         "hubei": "湖北", "hunan": "湖南", "jiangsu": "江苏", "jiangxi": "江西",
         "jilin": "吉林", "liaoning": "辽宁", "neimenggu": "内蒙古", "ningxia": "宁夏",
         "qinghai": "青海", "shaanxi": "陕西", "shandong": "山东", "shanghai": "上海",
         "shanxi": "山西", "sichuan": "四川", "tianjin": "天津", "xinjiang": "新疆",
         "yunnan": "云南", "zhejiang": "浙江"}
pm = pd.read_csv(D + r"\prov_month_from_city.csv", encoding="utf-8-sig")
pm["cn"] = pm["prov"].map(PY2CN)
pm = pm.dropna(subset=["cn"])
pm["year"] = pm["ym"].astype(str).str[:4].astype(int)
raw = pm.groupby("year")["gen_MWh"].sum() / 1e6

print("=== 2005–2012：全国年度缩放因子 ===")
rows = []
for y, off in OFF_NAT.items():
    r = raw.get(y, np.nan)
    f = off / r if r and r > 0 else np.nan
    rows.append(dict(year=y, model=r, official=off, factor=f))
t = pd.DataFrame(rows)
print(t.to_string(index=False, float_format=lambda v: "%.3f" % v))
print()
print("因子范围 %.2f–%.2f ｜ 中位 %.2f"
      % (t.factor.min(), t.factor.max(), t.factor.median()))

print()
print("=== 缩放后与 2013 起的衔接（全国 TWh）===")
for y in range(2005, 2016):
    if y in OFF_NAT:
        v = OFF_NAT[y]
        tag = "scaled"
    elif y in raw.index:
        v = raw.loc[y]
        tag = "raw"
    else:
        continue
    print("  %d  %8.2f  (%s)" % (y, v, tag))

print()
print("=== 2004 及以前的原始量级（无官方值，不缩放）===")
for y in [1989, 1995, 2000, 2003, 2004]:
    if y in raw.index:
        print("  %d  %.4f TWh" % (y, raw.loc[y]))
