#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 HPC 上：把 city 级 gen.nc 汇总成 省×月，并拟合逐省 (K,B) 与 M4_gen_correction.csv 对比。"""
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

BASE = "/geogfs1/groups/geog_liulb/hongyu/m4prod/results_2425/calibrated_gen/gen/city"
OUT = "/geogfs1/groups/geog_liulb/hongyu/m4prod/prov_month_2425.csv"
REF = "/geogfs1/groups/geog_liulb/hongyu/m4prod/code/M4_gen_correction.csv"


def open_nc(p):
    try:
        return xr.open_dataset(p)
    except Exception:
        return xr.open_dataset(p, engine="scipy")


files = sorted(glob.glob(os.path.join(BASE, "*__*", "full_year", "gen.nc")))
print("city 文件数:", len(files), flush=True)

acc = defaultdict(float)          # (prov, ym) -> MWh
for i, f in enumerate(files):
    prov = os.path.basename(os.path.dirname(os.path.dirname(f))).split("__")[0]
    ds = open_nc(f)
    g = ds["total_gen"].values.astype("float64")
    t = ds["time"].values
    ds.close()
    # 月标签 YYYYMM
    ym = np.array([str(x)[:7].replace("-", "") for x in t])
    uniq, inv = np.unique(ym, return_inverse=True)
    sums = np.bincount(inv, weights=g, minlength=len(uniq))
    for u, s in zip(uniq, sums):
        acc[(prov, u)] += float(s)
    if i % 60 == 0:
        print("  ...%d/%d" % (i, len(files)), flush=True)

rows = [{"prov": k[0], "ym": k[1], "gen_MWh": v} for k, v in acc.items()]
df = pd.DataFrame(rows).sort_values(["prov", "ym"])
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print("写出:", OUT, len(df), "行", flush=True)

# 拟合 (K,B)：2013–2023 全部月份
d = df[(df["ym"] >= "201301") & (df["ym"] <= "202312")].copy()
print("拟合期行数:", len(d), flush=True)
res = []
for prov, g in d.groupby("prov"):
    x = g["gen_MWh"].values
    # 官方值从 M4_gen_correction.csv 的来源推不出来，这里只输出 x 的统计供本地核对
    res.append(dict(prov=prov, n=len(x), mean=x.mean(), total=x.sum(),
                    min=x.min(), max=x.max()))
pd.DataFrame(res).to_csv(
    "/geogfs1/groups/geog_liulb/hongyu/m4prod/prov_month_stats.csv",
    index=False, encoding="utf-8-sig")
print("省数:", len(res), flush=True)
print("DONE")
