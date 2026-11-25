#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成论文最终 city 级数据集（三段口径，用户 2026-11-24 指定）：

  1989–2004：只做风速四季校正（calibrated_gen 原样），不做 generation 校正
  2005–2012：全国年度统一缩放 f_y = 官方全国总量 / 模型全国总量
  2013–2023：逐省仿射 G_corr = K·G + B（逐省逐月标量应用到市级逐时序列）

输出：results/city_final/<prov>__<city>/full_year/gen.nc
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

W = "/geogfs1/groups/geog_liulb/hongyu/m4prod"
SRC = os.path.join(W, "results/calibrated_gen/gen/city")
DST = os.path.join(W, "results/city_final")
PROV_MONTH = os.path.join(W, "prov_month_from_city.csv")
KB = os.path.join(W, "gen_correction_M4prod.csv")

# 2005–2012 全国官方风电发电量（TWh），来源：data_v2/2012年以前风电发电量(亿千瓦时).csv
OFF_NAT = {2005: 1.927, 2006: 3.675, 2007: 5.710, 2008: 12.800,
           2009: 27.609, 2010: 44.620, 2011: 70.330, 2012: 95.980}

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


# ---------- 1. 逐省逐月 M 与 (K,B) ----------
pm = pd.read_csv(PROV_MONTH, encoding="utf-8-sig")
pm["cn"] = pm["prov"].map(PY2CN)
pm = pm.dropna(subset=["cn"])
kb = pd.read_csv(KB, encoding="utf-8-sig").set_index("prov")

# 模型全国年度总量（2005–2012 用）
pm["year"] = pm["ym"].astype(str).str[:4].astype(int)
nat_model = pm.groupby("year")["gen_MWh"].sum() / 1e6
f_year = {}
for y, off in OFF_NAT.items():
    m = nat_model.get(y, np.nan)
    if m and m > 0:
        f_year[y] = off / m
print("全国年度缩放因子:", {y: round(v, 3) for y, v in f_year.items()})

# 逐省逐月仿射标量（仅 2013 起）
scale = {}
for r in pm.itertuples(index=False):
    if r.cn not in kb.index:
        continue
    ym = str(r.ym)
    if ym < "201301":
        continue
    K = float(kb.loc[r.cn, "K"]); B = float(kb.loc[r.cn, "B"])
    M = float(r.gen_MWh)
    if M > 0:
        scale[(r.prov, ym)] = (K * M + B) / M
print("仿射标量条目:", len(scale))
vals = np.array(list(scale.values()))
print("  分位 p05 %.3f ｜ 中位 %.3f ｜ p95 %.3f ｜ max %.1f"
      % (np.percentile(vals, 5), np.median(vals), np.percentile(vals, 95), vals.max()))


def factor_for(prov, ym):
    y = int(str(ym)[:4])
    if ym >= "201301":
        return scale.get((prov, ym), np.nan)
    if 2005 <= y <= 2012:
        return f_year.get(y, 1.0)
    return 1.0          # 1989–2004：不校正


# ---------- 2. 逐城市应用 ----------
files = sorted(glob.glob(os.path.join(SRC, "*__*", "full_year", "gen.nc")))
print("city 文件: %d" % len(files))
n_fb = 0
for i, f in enumerate(files):
    base = os.path.basename(os.path.dirname(os.path.dirname(f)))
    prov, city = base.split("__")
    ds = open_nc(f)

    ym = np.array([str(x)[:7].replace("-", "") for x in ds["time"].values])
    s = np.array([factor_for(prov, y) for y in ym], dtype="float64")
    if np.isnan(s).any():
        n_fb += 1
        s = np.where(np.isnan(s), 1.0, s)

    out = {}
    for v in ("total_gen", "onshore_gen", "offshore_gen"):
        if v in ds:
            out[v] = (("time",), np.asarray(ds[v].values, dtype="float64") * s)

    dym = np.array([str(x)[:7].replace("-", "") for x in ds["date"].values])
    sm = np.array([factor_for(prov, y) for y in dym], dtype="float64")
    sm = np.where(np.isnan(sm), 1.0, sm)
    for v in ("total_monthly_gen", "onshore_monthly_gen", "offshore_monthly_gen"):
        if v in ds:
            out[v] = (("date",), np.asarray(ds[v].values, dtype="float64") * sm)

    ds2 = xr.Dataset(out, coords={"time": ds["time"].values,
                                  "date": ds["date"].values},
                     attrs=dict(ds.attrs))
    ds2.attrs["generation_calibration"] = (
        "1989-2004 none; 2005-2012 national_annual_scale; 2013-2023 per_province_affine_K_B")
    ds2.attrs["aligned_to_official_province_month"] = "0"
    d = os.path.join(DST, base, "full_year")
    os.makedirs(d, exist_ok=True)
    ds2.to_netcdf(os.path.join(d, "gen.nc"))
    ds.close()
    if i % 60 == 0:
        print("  ...%d/%d" % (i, len(files)), flush=True)

print("完成 -> %s" % DST)
print("仿射月份缺失回退的城市文件数: %d" % n_fb)
