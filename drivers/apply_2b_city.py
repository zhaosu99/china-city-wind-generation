#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 (K,B) 施加到市级逐时序列（论文用的未对齐版）。

口径（用户 2026-11-23 指定）：
  · 2013–2023：s = (K·M + B) / M      （仿射，K 与 B 都用）
  · 2013 以前：s = K                  （只用 K，丢掉截距 B）
  逐省逐月一个标量，乘到该省所有城市的逐时与月度序列上。
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

W = "/geogfs1/groups/geog_liulb/hongyu/m4prod"
SRC = os.path.join(W, "results/calibrated_gen/gen/city")
DST = os.path.join(W, "results/city_2b")
PROV_MONTH = os.path.join(W, "prov_month_from_city.csv")
KB = os.path.join(W, "gen_correction_M4prod.csv")
CUT = "201301"          # 从这个月起用仿射

PY2CN = {
    "anhui": "安徽", "beijing": "北京", "chongqing": "重庆", "fujian": "福建",
    "gansu": "甘肃", "guangdong": "广东", "guangxi": "广西", "guizhou": "贵州",
    "hainan": "海南", "hebei": "河北", "heilongjiang": "黑龙江", "henan": "河南",
    "hubei": "湖北", "hunan": "湖南", "jiangsu": "江苏", "jiangxi": "江西",
    "jilin": "吉林", "liaoning": "辽宁", "neimenggu": "内蒙古", "ningxia": "宁夏",
    "qinghai": "青海", "shaanxi": "陕西", "shandong": "山东", "shanghai": "上海",
    "shanxi": "山西", "sichuan": "四川", "tianjin": "天津", "xinjiang": "新疆",
    "yunnan": "云南", "zhejiang": "浙江",
}


def open_nc(p):
    try:
        return xr.open_dataset(p)
    except Exception:
        return xr.open_dataset(p, engine="scipy")


pm = pd.read_csv(PROV_MONTH, encoding="utf-8-sig")
pm["cn"] = pm["prov"].map(PY2CN)
kb = pd.read_csv(KB, encoding="utf-8-sig").set_index("prov")

scale = {}
n_aff = n_mul = 0
for r in pm.itertuples(index=False):
    if r.cn not in kb.index:
        continue
    K = float(kb.loc[r.cn, "K"])
    B = float(kb.loc[r.cn, "B"])
    M = float(r.gen_MWh)
    ym = str(r.ym)
    if ym >= CUT:
        if M > 0:
            scale[(r.prov, ym)] = (K * M + B) / M
            n_aff += 1
    else:
        scale[(r.prov, ym)] = K
        n_mul += 1

print("标量条目 %d（仿射 %d ｜ 纯乘性 %d）" % (len(scale), n_aff, n_mul))
vals = np.array(list(scale.values()))
print("s 分位: p05 %.4f ｜ 中位 %.4f ｜ p95 %.4f ｜ max %.4f"
      % (np.percentile(vals, 5), np.median(vals), np.percentile(vals, 95), vals.max()))

files = sorted(glob.glob(os.path.join(SRC, "*__*", "full_year", "gen.nc")))
print("city 文件: %d" % len(files))
for i, f in enumerate(files):
    base = os.path.basename(os.path.dirname(os.path.dirname(f)))
    prov, city = base.split("__")
    K = float(kb.loc[PY2CN.get(prov, ""), "K"]) if PY2CN.get(prov) in kb.index else 1.0

    ds = open_nc(f)
    ym = np.array([str(x)[:7].replace("-", "") for x in ds["time"].values])
    s = np.array([scale.get((prov, y), K) for y in ym])
    out = {}
    for v in ("total_gen", "onshore_gen", "offshore_gen"):
        if v in ds:
            out[v] = (("time",), np.asarray(ds[v].values, dtype="float64") * s)
    dym = np.array([str(x)[:7].replace("-", "") for x in ds["date"].values])
    sm = np.array([scale.get((prov, y), K) for y in dym])
    for v in ("total_monthly_gen", "onshore_monthly_gen", "offshore_monthly_gen"):
        if v in ds:
            out[v] = (("date",), np.asarray(ds[v].values, dtype="float64") * sm)
    ds2 = xr.Dataset(out, coords={"time": ds["time"].values,
                                  "date": ds["date"].values},
                     attrs=dict(ds.attrs))
    ds2.attrs["gen_correction_applied"] = "K_affine_2013plus_K_only_before"
    ds2.attrs["aligned_to_official"] = "0"
    d = os.path.join(DST, base, "full_year")
    os.makedirs(d, exist_ok=True)
    ds2.to_netcdf(os.path.join(d, "gen.nc"))
    ds.close()
    if i % 60 == 0:
        print("  ...%d/%d" % (i, len(files)), flush=True)

print("完成 -> %s" % DST)
