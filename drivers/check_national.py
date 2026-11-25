# -*- coding: utf-8 -*-
"""检查新口径（2013 起仿射、之前纯 K）下的全国年度总量是否合理。"""
import numpy as np
import pandas as pd

D = r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod"
pm = pd.read_csv(D + r"\prov_month_from_city.csv", encoding="utf-8-sig")
kb = pd.read_csv(D + r"\refit\gen_correction_M4prod.csv", encoding="utf-8-sig").set_index("prov")
PY2CN = {"anhui": "安徽", "beijing": "北京", "chongqing": "重庆", "fujian": "福建",
         "gansu": "甘肃", "guangdong": "广东", "guangxi": "广西", "guizhou": "贵州",
         "hainan": "海南", "hebei": "河北", "heilongjiang": "黑龙江", "henan": "河南",
         "hubei": "湖北", "hunan": "湖南", "jiangsu": "江苏", "jiangxi": "江西",
         "jilin": "吉林", "liaoning": "辽宁", "neimenggu": "内蒙古", "ningxia": "宁夏",
         "qinghai": "青海", "shaanxi": "陕西", "shandong": "山东", "shanghai": "上海",
         "shanxi": "山西", "sichuan": "四川", "tianjin": "天津", "xinjiang": "新疆",
         "yunnan": "云南", "zhejiang": "浙江"}
pm["cn"] = pm["prov"].map(PY2CN)
pm = pm.dropna(subset=["cn"])
pm["year"] = pm["ym"].astype(str).str[:4].astype(int)
pm["K"] = pm["cn"].map(kb["K"])
pm["B"] = pm["cn"].map(kb["B"])
is_aff = pm["ym"].astype(str) >= "201301"
pm["after"] = np.where(is_aff, pm["K"] * pm["gen_MWh"] + pm["B"],
                       pm["K"] * pm["gen_MWh"])

g = pm.groupby("year").agg(before_TWh=("gen_MWh", lambda x: x.sum() / 1e6),
                           after_TWh=("after", lambda x: x.sum() / 1e6))
print("=== 全国年度总量（TWh）===")
print(g.tail(40).to_string(float_format=lambda v: "%.2f" % v))
print()
print("== 与已发表数字对照 ==")
cmp = {"2006": 3.675, "2010": 44.620, "2023": 905.230}
for y, v in cmp.items():
    a = g.loc[int(y), "after_TWh"] if int(y) in g.index else float("nan")
    print("  %s  本版 %8.2f ｜ 已发表/官方 %8.2f ｜ 比 %+.0f%%"
          % (y, a, v, 100 * (a / v - 1)))
