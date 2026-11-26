#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the released city-level dataset (1989-2023).

This is the final generation-correction stage after
``pipeline/panel.py --steps 9`` has produced ``calibrated_gen``:

* 1989-2004: no generation-level correction;
* 2005-2012: national annual scaling from ``parameters/pre2012_official.csv``;
* 2013-2023: per-province affine ``G_corr = K G + B`` from
  ``parameters/gen_correction_M4prod_final.csv`` applied as a monthly
  scalar to each city's hourly series.

Paths are configurable through CLI arguments or WINDPAPER_* variables.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

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


def open_nc(path):
    try:
        return xr.open_dataset(path)
    except Exception:
        return xr.open_dataset(path, engine="scipy")


def read_pre2012(path: Path) -> dict[int, float]:
    """Read the national annual table (values in 亿 kWh) as TWh."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")
    row = df[df.iloc[:, 0].astype(str).str.strip() == "全国"]
    if row.empty:
        raise ValueError(f"no 全国 row in {path}")
    out = {}
    for col in df.columns:
        try:
            year = int(str(col).strip())
        except ValueError:
            continue
        if 2005 <= year <= 2012:
            out[year] = float(row.iloc[0][col]) / 10.0
    missing = sorted(set(range(2005, 2013)) - set(out))
    if missing:
        raise ValueError(f"missing years in {path}: {missing}")
    return out


def build_final_city(
    src_dir: Path,
    dst_dir: Path,
    model_monthly: Path,
    kb_csv: Path,
    pre2012_csv: Path,
) -> int:
    kb = pd.read_csv(kb_csv, encoding="utf-8-sig").set_index("prov")
    pm = pd.read_csv(model_monthly, encoding="utf-8-sig")
    pm["cn"] = pm["prov"].map(PY2CN)
    pm = pm.dropna(subset=["cn"]).copy()
    pm["year"] = pm["ym"].astype(str).str[:4].astype(int)

    off_nat = read_pre2012(pre2012_csv)
    nat_model = pm.groupby("year")["gen_MWh"].sum() / 1e6  # TWh
    f_year = {
        year: off_nat[year] / float(nat_model[year])
        for year in off_nat
        if year in nat_model.index and nat_model[year] > 0
    }

    scale: dict[tuple[str, str], float] = {}
    for row in pm.itertuples(index=False):
        if row.cn not in kb.index:
            continue
        ym = str(row.ym)
        if ym < "201301":
            continue
        M = float(row.gen_MWh)
        if M > 0:
            K = float(kb.loc[row.cn, "K"])
            B = float(kb.loc[row.cn, "B"])
            scale[(row.prov, ym)] = (K * M + B) / M

    def factor(prov: str, ym: str) -> float:
        year = int(str(ym)[:4])
        if ym >= "201301":
            return scale.get((prov, ym), 1.0)
        if 2005 <= year <= 2012:
            return f_year.get(year, 1.0)
        return 1.0

    files = sorted(glob.glob(str(src_dir / "*__*" / "full_year" / "gen.nc")))
    if not files:
        raise FileNotFoundError(f"no city gen.nc files under {src_dir}")
    print(f"city files: {len(files)}")
    for i, path in enumerate(files, 1):
        case = Path(path).parent.parent.name
        prov, _city = case.split("__")
        ds = open_nc(path)
        ym = np.array([str(x)[:7].replace("-", "") for x in ds["time"].values])
        s = np.array([factor(prov, y) for y in ym], dtype="float64")
        dym = np.array([str(x)[:7].replace("-", "") for x in ds["date"].values])
        sm = np.array([factor(prov, y) for y in dym], dtype="float64")

        out = {}
        for var in ("total_gen", "onshore_gen", "offshore_gen"):
            if var in ds:
                out[var] = (("time",), np.asarray(ds[var].values, dtype="float64") * s)
        for var in ("total_monthly_gen", "onshore_monthly_gen", "offshore_monthly_gen"):
            if var in ds:
                out[var] = (("date",), np.asarray(ds[var].values, dtype="float64") * sm)

        ds2 = xr.Dataset(out, coords={"time": ds["time"].values, "date": ds["date"].values},
                         attrs=dict(ds.attrs))
        ds2.attrs["generation_calibration"] = (
            "1989-2004 none; 2005-2012 national_annual_scale; "
            "2013-2023 per_province_affine_K_B"
        )
        ds2.attrs["aligned_to_official_province_month"] = "0"
        out_dir = dst_dir / case / "full_year"
        out_dir.mkdir(parents=True, exist_ok=True)
        ds2.to_netcdf(out_dir / "gen.nc")
        ds.close()
        if i % 60 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)
    print(f"wrote {len(files)} files to {dst_dir}")
    return len(files)


def main() -> None:
    root = Path(os.environ.get("WINDPAPER_ROOT", ".")).resolve()
    parser = argparse.ArgumentParser(description="Build the released city dataset")
    parser.add_argument("--src", type=Path,
                        default=Path(os.environ.get("WINDPAPER_CALIBRATED_CITY_DIR",
                                                    root / "results/calibrated_gen/gen/city")))
    parser.add_argument("--dst", type=Path,
                        default=Path(os.environ.get("WINDPAPER_CITY_FINAL_DIR",
                                                    root / "results/city_final")))
    parser.add_argument("--model-monthly", type=Path,
                        default=Path(os.environ.get("WINDPAPER_MODEL_MONTHLY",
                                                    root / "prov_month_from_city.csv")))
    parser.add_argument("--kb", type=Path,
                        default=Path(os.environ.get("WINDPAPER_KB",
                                                    root / "parameters/gen_correction_M4prod_final.csv")))
    parser.add_argument("--pre2012", type=Path,
                        default=Path(os.environ.get("WINDPAPER_PRE2012_OFFICIAL",
                                                    root / "parameters/pre2012_official.csv")))
    args = parser.parse_args()
    build_final_city(args.src, args.dst, args.model_monthly, args.kb, args.pre2012)


if __name__ == "__main__":
    main()
