#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4 wind-only calibration and 2013-2025 cross-validation.

The wind-speed selection is fixed before any generation data are seen:
for each province and season, the M4 point is the minimum of the
wind-speed error surface along the wind-valley path.  The generation
calibration is then fitted separately, province by province.

Cross-validation uses the final 143-period grid (2013-2025; 1-2 January
February merged), with a seasonal 80/20 split repeated 50 times.  The
production (K, B) parameters are fitted on all available 2013-2023
periods, as in the manuscript.

Expected layout
---------------
``--paths-dir``:
    ``<prov>_<season>.npz`` with arrays ``k``, ``b`` and
    ``wind_at_path`` (created by the wind-surface preparation step).

``--enum-dir``:
    ``<prov>_<season>_monthly_gen_enums.csv`` with ``official`` and
    ``k{k}b{b}`` generation columns.

The output files are ``selection.csv``, ``per_run.csv``,
``per_prov.csv``, ``summary.csv`` and ``gen_correction.csv``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv_base as base  # noqa: E402

K_LO, K_HI = 0.5, 3.0
B_LO, B_HI = -2.0, 2.0


def in_box(k: float, b: float) -> bool:
    return (
        K_LO - 1e-9 <= k <= K_HI + 1e-9
        and B_LO - 1e-9 <= b <= B_HI + 1e-9
    )


def pick_m4_from_npz(path_file: Path) -> dict:
    """Load one wind-valley path and select the M4 point."""
    data = np.load(path_file)
    k_path = np.asarray(data["k"], dtype=float)
    b_path = np.asarray(data["b"], dtype=float)
    wind = np.asarray(data["wind_at_path"], dtype=float)
    if not (k_path.size == b_path.size == wind.size) or k_path.size == 0:
        raise ValueError(f"invalid path arrays in {path_file}")
    box = np.array([in_box(k, b) for k, b in zip(k_path, b_path)], dtype=bool)
    if not box.any():
        raise ValueError(f"no path point inside the box in {path_file}")
    values = np.where(box, wind, np.inf)
    if not np.isfinite(values).any():
        raise ValueError(f"all wind values are non-finite in {path_file}")
    idx = int(np.argmin(values))
    k = float(k_path[idx])
    b = float(b_path[idx])
    on_boundary = bool(
        np.isclose(k, K_LO, atol=1e-9)
        or np.isclose(k, K_HI, atol=1e-9)
        or np.isclose(b, B_LO, atol=1e-9)
        or np.isclose(b, B_HI, atol=1e-9)
    )
    return {
        "k": k,
        "b": b,
        "w_smape": float(wind[idx]),
        "candidate_point_count": int(box.sum()),
        "selection_mode": "M4_wind_only",
        "on_boundary": on_boundary,
    }


def build_selection(paths_dir: Path) -> pd.DataFrame:
    rows = []
    for path_file in sorted(paths_dir.glob("*.npz")):
        prov, season = path_file.stem.rsplit("_", 1)
        if season not in base.SEASONS:
            continue
        item = pick_m4_from_npz(path_file)
        rows.append({"prov": prov, "season": season, **item})
    if not rows:
        raise FileNotFoundError(f"no <prov>_<season>.npz files found in {paths_dir}")
    return (
        pd.DataFrame(rows)
        .sort_values(["prov", "season"], kind="stable")
        .reset_index(drop=True)
    )


def fit_kb_constrained(x, y) -> tuple[float, float, int, str]:
    """Province-level affine fit with a non-negative intercept.

    The paper constrains B >= 0 so that small generation months cannot
    receive negative generation.  If unconstrained OLS gives B < 0, the
    fit is re-done through the origin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < 2 or np.var(x) <= 0:
        return float("nan"), float("nan"), n, "invalid"
    K = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    B = float(y.mean() - K * x.mean())
    mode = "unconstrained"
    if B < 0:
        K = float(np.sum(x * y) / np.sum(x * x))
        B = 0.0
        mode = "B_constrained_to_0"
    return K, B, n, mode


def _r2(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 3:
        return float("nan")
    sst = float(np.sum((y[mask] - y[mask].mean()) ** 2))
    if sst <= 0:
        return float("nan")
    sse = float(np.sum((y[mask] - yhat[mask]) ** 2))
    return 1.0 - sse / sst


def _period_codes(periods):
    return np.array([base.CODE_OF.get(p, -1) for p in periods], dtype=np.int64)


def run_cv(
    selection: pd.DataFrame,
    enum_dir: Path,
    out_dir: Path,
    *,
    n_splits: int = 50,
) -> dict[str, pd.DataFrame]:
    base.ENUM_DIR = Path(enum_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out_dir / "selection.csv", index=False, encoding="utf-8-sig")

    # Load each province/season once.  The M4 selection is fixed for all splits.
    cache = {}
    for row in selection.itertuples(index=False):
        prov, season = row.prov, row.season
        enum = base.load_enum(prov, season)
        theory, _ = base.theory_at(enum, float(row.k), float(row.b))
        cache[(prov, season)] = (enum, theory)

    splits = base.make_splits(n_splits=n_splits)
    recs = []
    for split in splits:
        split_id = split["split"]
        train_codes = np.asarray(split["train_codes"], dtype=np.int64)
        test_codes = np.asarray(split["test_codes"], dtype=np.int64)
        # Fit one (K, B) per province using all seasons and all training periods.
        for prov in sorted(selection["prov"].unique()):
            train_theory, train_official = [], []
            season_data = {}
            for season in base.SEASONS:
                enum, theory = cache[(prov, season)]
                official = enum["official"]
                pcode = _period_codes(enum["periods"])
                tr = np.isin(pcode, train_codes)
                te = np.isin(pcode, test_codes)
                season_data[season] = (theory, official, tr, te)
                train_theory.append(theory[tr])
                train_official.append(official[tr])
            K, B, n_fit, _mode = fit_kb_constrained(
                np.concatenate(train_theory), np.concatenate(train_official)
            )
            for season, (theory, official, tr, te) in season_data.items():
                for sample, mask in (("train", tr), ("test", te)):
                    valid = mask & np.isfinite(theory) & np.isfinite(official)
                    n = int(valid.sum())
                    if n < 2 or not np.isfinite(K):
                        recs.append(
                            dict(split=split_id, prov=prov, season=season, sample=sample,
                                 smape=np.nan, ratio=np.nan, n=n, n_valid=0, K=K, B=B,
                                 n_fit=n_fit, sum_official=0.0, sum_adj=0.0)
                        )
                        continue
                    adjusted = K * theory[valid] + B
                    observed = official[valid]
                    smape, n_valid = base.smape_count(observed, adjusted)
                    total_o = float(np.sum(observed))
                    total_a = float(np.sum(adjusted))
                    recs.append(
                        dict(split=split_id, prov=prov, season=season, sample=sample,
                             smape=smape,
                             ratio=(total_a / total_o if total_o > 0 else np.nan),
                             n=n, n_valid=n_valid, K=K, B=B, n_fit=n_fit,
                             sum_official=total_o, sum_adj=total_a)
                    )
    per_run = pd.DataFrame(recs)
    per_run.to_csv(out_dir / "per_run.csv", index=False, encoding="utf-8-sig")

    # Aggregate seasons to province level, weighting by valid periods.
    prov_rows = []
    for (split_id, prov, sample), group in per_run.groupby(
        ["split", "prov", "sample"], sort=True
    ):
        valid = group["n_valid"].to_numpy(float)
        weight = np.where(np.isfinite(valid), valid, 0.0)
        smape = group["smape"].to_numpy(float)
        ok = np.isfinite(smape) & (weight > 0)
        smape_prov = float(np.sum(smape[ok] * weight[ok]) / np.sum(weight[ok])) if ok.any() else np.nan
        sum_o = float(np.nansum(group["sum_official"]))
        sum_a = float(np.nansum(group["sum_adj"]))
        prov_rows.append(dict(split=split_id, prov=prov, sample=sample,
                              smape=smape_prov,
                              ratio=(sum_a / sum_o if sum_o > 0 else np.nan),
                              n_valid=int(group["n_valid"].sum()),
                              sum_official=sum_o, sum_adj=sum_a))
    per_prov = pd.DataFrame(prov_rows)
    per_prov.to_csv(out_dir / "per_prov.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for sample in ("train", "test"):
        group = per_prov[per_prov["sample"] == sample]
        for metric in ("smape", "ratio"):
            values = group[metric].dropna()
            if len(values):
                summary_rows.append(dict(sample=sample, metric=metric, n_prov=len(values),
                                         mean=float(values.mean()), median=float(values.median()),
                                         sd=float(values.std()), p05=float(values.quantile(0.05)),
                                         p95=float(values.quantile(0.95))))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    # Production generation parameters: fit on all 2013-2023 periods.
    prod_codes = np.array(
        [base.CODE_OF[(y, mk)] for (y, mk, _s) in base.GRID if y <= 2023],
        dtype=np.int64,
    )
    prod_rows = []
    for prov in sorted(selection["prov"].unique()):
        theory_all, official_all = [], []
        for season in base.SEASONS:
            enum, theory = cache[(prov, season)]
            official = enum["official"]
            pcode = _period_codes(enum["periods"])
            mask = np.isin(pcode, prod_codes)
            theory_all.append(theory[mask])
            official_all.append(official[mask])
        theory = np.concatenate(theory_all)
        official = np.concatenate(official_all)
        K, B, n, mode = fit_kb_constrained(theory, official)
        prod_rows.append(dict(prov=prov, K=K, B=B, n=n, r2=_r2(official, K * theory + B),
                              mode=mode, sum_x=float(np.nansum(theory)),
                              sum_y=float(np.nansum(official))))
    gen_correction = pd.DataFrame(prod_rows).sort_values("prov").reset_index(drop=True)
    gen_correction.to_csv(out_dir / "gen_correction.csv", index=False, encoding="utf-8-sig")
    return {"per_run": per_run, "per_prov": per_prov, "summary": summary,
            "gen_correction": gen_correction, "selection": selection}


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 wind-only calibration and 2013-2025 CV")
    parser.add_argument("--paths-dir", type=Path, required=True)
    parser.add_argument("--enum-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("cv_m4_out"))
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=1,
                        help="reserved for compatibility; the public version runs serially")
    args = parser.parse_args()

    selection = build_selection(args.paths_dir)
    results = run_cv(selection, args.enum_dir, args.out, n_splits=args.splits)
    print("selection rows:", len(results["selection"]))
    print(results["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
