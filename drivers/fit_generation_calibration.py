#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit the production generation calibration (K, B), 2013-2023.

This is the production-level fit behind
``parameters/gen_correction_M4prod_final.csv``.  It is separate from the
2013-2025 cross-validation in ``drivers/cv_m4.py``:

* production fit: all available official province-months 2013-2023;
* CV: 2013-2025 resampling pool, with (K, B) re-fitted on each training fold.

The intercept is constrained to be non-negative.  The province-month
mapping and the 1-2 February merge follow the manuscript.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def _fit_constrained(x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    K = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    B = float(y.mean() - K * x.mean())
    mode = "unconstrained"
    if B < 0:
        K = float(np.sum(x * y) / np.sum(x * x))
        B = 0.0
        mode = "B_constrained_to_0"
    return K, B, mode


def fit_generation_calibration(
    model_monthly: Path,
    official_monthly: Path,
    out_csv: Path,
    *,
    start_year: int = 2013,
    end_year: int = 2023,
) -> pd.DataFrame:
    model = pd.read_csv(model_monthly, encoding="utf-8-sig")
    official = pd.read_csv(official_monthly, encoding="utf-8-sig")

    model["cn"] = model["prov"].map(PY2CN)
    model = model.dropna(subset=["cn"]).copy()
    model["year"] = model["ym"].astype(str).str[:4].astype(int)
    model["month"] = model["ym"].astype(str).str[4:6].astype(int)
    model["mk"] = np.where(model["month"] <= 2, 1, model["month"])
    model = model[(model["year"] >= start_year) & (model["year"] <= end_year)]
    x = (
        model.groupby(["cn", "year", "mk"], as_index=False)["gen_MWh"]
        .sum()
        .rename(columns={"gen_MWh": "gen_MWh"})
    )

    if not {"cn", "year", "month", "value_MWh"}.issubset(official.columns):
        raise ValueError(
            "official_monthly must contain cn, year, month, value_MWh; "
            f"got {list(official.columns)}"
        )
    official = official[official["value_MWh"] > 0].copy()
    official["mk"] = np.where(official["month"] <= 2, 1, official["month"])
    official = official[(official["year"] >= start_year) & (official["year"] <= end_year)]
    y = (
        official.groupby(["cn", "year", "mk"], as_index=False)["value_MWh"]
        .sum()
        .rename(columns={"value_MWh": "value_MWh"})
    )

    matched = x.merge(y, on=["cn", "year", "mk"], how="inner")
    rows = []
    for prov, group in matched.groupby("cn", sort=True):
        xa = group["gen_MWh"].to_numpy(dtype=float)
        ya = group["value_MWh"].to_numpy(dtype=float)
        if len(xa) < 2 or np.var(xa) <= 0:
            continue
        K, B, mode = _fit_constrained(xa, ya)
        yhat = K * xa + B
        sst = float(np.sum((ya - ya.mean()) ** 2))
        r2 = 1.0 - float(np.sum((ya - yhat) ** 2)) / sst if sst > 0 else float("nan")
        rows.append(
            {
                "prov": prov,
                "K": K,
                "B": B,
                "n": int(len(xa)),
                "r2": r2,
                "mode": mode,
                "sum_x": float(np.sum(xa)),
                "sum_y": float(np.sum(ya)),
            }
        )
    result = pd.DataFrame(rows).sort_values("prov").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit production generation calibration")
    parser.add_argument("--model-monthly", type=Path, required=True,
                        help="CSV with prov, ym, gen_MWh")
    parser.add_argument("--official-monthly", type=Path, required=True,
                        help="CSV with cn, year, month, value_MWh")
    parser.add_argument("--out", type=Path, default=Path("parameters/gen_correction_M4prod_final.csv"))
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2023)
    args = parser.parse_args()
    result = fit_generation_calibration(
        args.model_monthly, args.official_monthly, args.out,
        start_year=args.start_year, end_year=args.end_year,
    )
    print(f"wrote {args.out} ({len(result)} provinces)")
    print(result[["prov", "K", "B", "n", "r2", "mode"]].to_string(index=False))


if __name__ == "__main__":
    main()
