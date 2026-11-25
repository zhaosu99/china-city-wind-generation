# -*- coding: utf-8 -*-
"""重拟合 (K,B)：约束截距非负（B>=0），保证 K*M+B 恒非负。

对江西/河南这两省，无约束 LS 给出 B<0，会在 2013–2015 小水量月份产生负发电。
约束解：若无约束 B<0，则取 B=0、K=Σxy/Σx²（过原点最小二乘）。
其余 28 省保持无约束解不变。
"""
import os

import numpy as np
import pandas as pd

D = r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod"
PROD = os.path.join(D, "prov_month_from_city.csv")
OFF = r"D:\HongyuEnergy Copilot\_run\prod_2b\work\official_prov_month.csv"
OUT = os.path.join(D, "refit", "gen_correction_constrained.csv")

PY2CN = {"anhui": "安徽", "beijing": "北京", "chongqing": "重庆", "fujian": "福建",
         "gansu": "甘肃", "guangdong": "广东", "guangxi": "广西", "guizhou": "贵州",
         "hainan": "海南", "hebei": "河北", "heilongjiang": "黑龙江", "henan": "河南",
         "hubei": "湖北", "hunan": "湖南", "jiangsu": "江苏", "jiangxi": "江西",
         "jilin": "吉林", "liaoning": "辽宁", "neimenggu": "内蒙古", "ningxia": "宁夏",
         "qinghai": "青海", "shaanxi": "陕西", "shandong": "山东", "shanghai": "上海",
         "shanxi": "山西", "sichuan": "四川", "tianjin": "天津", "xinjiang": "新疆",
         "yunnan": "云南", "zhejiang": "浙江"}

pm = pd.read_csv(PROD, encoding="utf-8-sig")
pm["cn"] = pm["prov"].map(PY2CN)
pm = pm.dropna(subset=["cn"])
pm["year"] = pm["ym"].astype(str).str[:4].astype(int)
pm["mk"] = pm["ym"].astype(str).str[4:6].astype(int).apply(lambda m: 1 if m <= 2 else m)
x = pm.groupby(["cn", "year", "mk"], as_index=False)["gen_MWh"].sum()

o = pd.read_csv(OFF, encoding="utf-8-sig")
o = o[o["value_MWh"] > 0].copy()
o["mk"] = o["month"].apply(lambda m: 1 if m <= 2 else m)
y = o.groupby(["cn", "year", "mk"], as_index=False)["value_MWh"].sum()
y = y[(y["year"] >= 2013) & (y["year"] <= 2023)]

m = x.merge(y, on=["cn", "year", "mk"], how="inner")
print("拟合样本: %d 行, %d 省" % (len(m), m["cn"].nunique()))

rows = []
for prov, g in m.groupby("cn"):
    xa = g["gen_MWh"].values.astype(float)
    ya = g["value_MWh"].values.astype(float)
    K = np.cov(xa, ya, ddof=1)[0, 1] / np.var(xa, ddof=1)
    B = ya.mean() - K * xa.mean()
    mode = "unconstrained"
    if B < 0:
        # 过原点最小二乘
        K = float((xa * ya).sum() / (xa * xa).sum())
        B = 0.0
        mode = "B_constrained_to_0"
    yhat = K * xa + B
    r2 = 1 - ((ya - yhat) ** 2).sum() / ((ya - ya.mean()) ** 2).sum()
    rows.append(dict(prov=prov, K=K, B=B, n=len(xa), r2=r2, mode=mode,
                     sum_x=xa.sum(), sum_y=ya.sum()))
fit = pd.DataFrame(rows).sort_values("prov")
fit.to_csv(OUT, index=False, encoding="utf-8-sig")

print()
print("=== 被约束的省 ===")
print(fit[fit["mode"] != "unconstrained"].to_string(index=False))
print()
print("其余 28 省：K 中位 %.4f ｜ B 中位 %.4g（与无约束解一致）"
      % (fit[fit["mode"] == "unconstrained"]["K"].median(),
         fit[fit["mode"] == "unconstrained"]["B"].median()))
print()
print("输出:", OUT)
