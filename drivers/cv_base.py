# -*- coding: utf-8 -*-
"""Shared calibration/CV utilities for the M4 workflow.

This module contains the generic functions used by ``drivers/cv_m4.py``:
period grid (2013-2025), SMAPE, polynomial wind surfaces, wind-valley
paths, enumeration loading and generation-calibration fitting.

The old M2 selection is not present here; the M2 scripts are archived
under ``legacy/`` for transparency.

All paths are configurable with the WINDPAPER_* environment variables.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WINDPAPER_ROOT", ".")).resolve()
RES = Path(os.environ.get("WINDPAPER_RESULTS", ROOT / "results"))
DATA = Path(os.environ.get("WINDPAPER_DATA", ROOT / "data"))
WORK = Path(os.environ.get("WINDPAPER_CV_WORK", ROOT / "cv_m4_work"))
OUT = Path(os.environ.get("WINDPAPER_CV_OUT", WORK / "out"))
CACHE = Path(os.environ.get("WINDPAPER_CACHE", WORK / "cache"))
SURF = Path(os.environ.get("WINDPAPER_SURF_DIR", ROOT / "data" / "wind_surfaces"))

ENUM_DIR = Path(os.environ.get("WINDPAPER_ENUM_DIR", ROOT / "data" / "gen_enums_2013_2025"))
WC_DIR = Path(os.environ.get("WINDPAPER_WIND_DIR", RES / "prov_w_compare"))
INVENTORY = Path(os.environ.get("WINDPAPER_INVENTORY", DATA / "wind_farms_inventory_till_2023_with_elevation.csv"))
TIME_AXIS = Path(os.environ.get("WINDPAPER_TIME_AXIS", RES / "_farm_time_axis.csv"))

SEASONS = ("spring", "summer", "autumn", "winter")
KV = np.round(np.arange(0.1, 3.0 + 1e-9, 0.1), 10)
BV = np.round(np.arange(-2.0, 2.0 + 1e-9, 0.1), 10)
POLY_DEGREE = 9
MIN_OPTIMAL_K = 0.5
K_MAX = 3.0
B_MIN, B_MAX = -2.0, 2.0
FINE_STEP = 0.01

YEAR_MIN, YEAR_MAX = 2013, 2025
WIND_YEARS = tuple(range(2013, 2021))          # CDMet 真实可用年份
EXCLUDE_PROV = {"西藏", "香港", "澳门"}
N_SPLITS = 50
N_TEST_PER_SEASON = {"spring": 8, "summer": 7, "autumn": 7, "winter": 7}   # 29 期
SEED0 = 20261116


# ================================================================== 期点网格（143）
def build_period_grid():
    per = []
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        for mk in (3, 4, 5):
            per.append((y, mk, "spring"))
        for mk in (6, 7, 8):
            per.append((y, mk, "summer"))
        for mk in (9, 10, 11):
            per.append((y, mk, "autumn"))
        per.append((y, 0, "winter"))
        per.append((y, 12, "winter"))
    return per


GRID = build_period_grid()
PERIOD_SEASON = {(y, mk): s for (y, mk, s) in GRID}
CODE_OF = {(y, mk): y * 100 + mk for (y, mk, _) in GRID}
PERIOD_OF_CODE = {v: k for k, v in CODE_OF.items()}
ALL_CODES = np.array(sorted(CODE_OF.values()), dtype=np.int32)


def season_codes(season):
    return [CODE_OF[(y, mk)] for (y, mk, s) in GRID if s == season]


# ================================================================== 指标与拟合
def smape_count(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.abs(y_true) + np.abs(y_pred)
    m = den > 0
    if not np.any(m):
        return float("nan"), 0
    return float(np.mean(100.0 * np.abs(y_pred - y_true)[m] / den[m])), int(m.sum())


def smape(y_true, y_pred):
    return smape_count(y_true, y_pred)[0]


def _poly_terms(degree):
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]


def _design(k, b, terms):
    k = np.asarray(k, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return np.column_stack([(k ** pk) * (b ** pb) for pk, pb in terms])


class PolyModel:
    __slots__ = ("terms", "coef", "r2", "n_fit")

    def __init__(self, terms, coef, r2, n_fit):
        self.terms, self.coef, self.r2, self.n_fit = terms, coef, r2, n_fit

    def eval(self, k, b):
        return _design(k, b, self.terms) @ self.coef


def fit_poly(surf, degree=POLY_DEGREE):
    ks = surf.columns.values.astype(float)
    bs = surf.index.values.astype(float)
    K, B = np.meshgrid(ks, bs)
    y = surf.values.ravel()
    m = np.isfinite(y)
    terms = _poly_terms(degree)
    if m.sum() < len(terms):
        return None
    X = np.column_stack([K.ravel()[m], B.ravel()[m]])
    yv = y[m]
    Xp = _design(X[:, 0], X[:, 1], terms)
    coef, *_ = np.linalg.lstsq(Xp, yv, rcond=None)
    pred = Xp @ coef
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return PolyModel(terms, coef, r2, int(m.sum()))


def optimal_path(model):
    k_fine = np.arange(MIN_OPTIMAL_K, K_MAX + FINE_STEP / 2, FINE_STEP)
    b_fine = np.arange(B_MIN, B_MAX + FINE_STEP / 2, FINE_STEP)
    if len(k_fine) < 2 or len(b_fine) < 2:
        return None
    KB, BB = np.meshgrid(k_fine, b_fine)
    S = model.eval(KB.ravel(), BB.ravel()).reshape(BB.shape)
    b_opt = b_fine[np.argmin(S, axis=0)]
    tol = 1e-9
    in_op = (b_opt >= B_MIN - tol) & (b_opt <= B_MAX + tol)
    if not np.any(in_op):
        return None
    k_path, b_path = k_fine[in_op], b_opt[in_op]
    boundary = (np.isclose(b_path, B_MIN, atol=tol) | np.isclose(b_path, B_MAX, atol=tol))
    start = end = None
    started = False
    n = len(k_path)
    for idx, is_b in enumerate(boundary):
        nb = boundary[idx + 1] if idx + 1 < n else None
        if not started:
            if is_b:
                if nb is False:
                    start, started = idx, True
            else:
                start, started = idx, True
            continue
        if idx != start and is_b:
            end = idx
            break
    if start is None:
        return None
    if end is None:
        k_out, b_out = k_path[start:], b_path[start:]
    else:
        k_out, b_out = k_path[start:end + 1], b_path[start:end + 1]
    if len(k_out) == 0:
        return None
    return {"k": k_out, "b": b_out, "S": S, "k_fine": k_fine, "b_fine": b_fine}


# ================================================================== wind 面（固定 2013–2020）
def surface_from_acc(acc, cnt):
    with np.errstate(divide="ignore", invalid="ignore"):
        surf = np.where(cnt > 0, 100.0 * acc / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(surf, index=BV, columns=KV)


def wind_surface_fixed(prov, season):
    """只用 2013–2020 的 CDMet 期点累加（不随 split 变）。"""
    zf = np.load(SURF / f"{prov}_{season}.npz")
    periods_arr = zf["periods"]
    keep = np.array([(int(p) // 100) in WIND_YEARS for p in periods_arr], dtype=bool)
    if not keep.any():
        return None, 0
    acc = zf["acc"][keep].sum(axis=0)
    cnt = zf["cnt"][keep].sum(axis=0)
    return surface_from_acc(acc, cnt), int(keep.sum())


# ================================================================== gen 枚举
_KB_COL_RE = re.compile(r"^k(-?\d+(?:\.\d+)?)b(-?\d+(?:\.\d+)?)$")


def load_enum(prov, season, merge_jf=True):
    fp = ENUM_DIR / f"{prov}_{season}_monthly_gen_enums.csv"
    df = pd.read_csv(fp, encoding="utf-8-sig")
    time_col = df.columns[0]
    kb_cols, kb_idx = [], []
    for c in df.columns:
        m = _KB_COL_RE.match(str(c).strip())
        if m:
            kb_cols.append(str(c).strip())
            kb_idx.append((round(float(m.group(1)), 4), round(float(m.group(2)), 4)))
    order = sorted(range(len(kb_idx)), key=lambda i: (kb_idx[i][1], kb_idx[i][0]))
    kb_cols = [kb_cols[i] for i in order]
    kb_idx = [kb_idx[i] for i in order]

    t = df[time_col].astype(str).str.strip()
    years = np.full(len(t), -1, dtype=np.int64)
    pkeys = np.full(len(t), -1, dtype=np.int64)
    for i, s in enumerate(t):
        m = re.match(r"^(\d{4})年1-2月$", s)
        if m:
            years[i] = int(m.group(1)); pkeys[i] = 0; continue
        m = re.match(r"^(\d{4})年(\d{1,2})月$", s)
        if m:
            y, mk = int(m.group(1)), int(m.group(2))
            years[i] = y
            pkeys[i] = (0 if mk in (1, 2) else mk) if merge_jf else mk
            continue
    ok = (years > 0) & (pkeys >= 0)
    df2 = df.loc[ok]
    years, pkeys = years[ok], pkeys[ok]
    num = df2[["official"] + kb_cols].apply(pd.to_numeric, errors="coerce")
    key = years * 100 + pkeys
    num = num.copy()
    num["__key"] = key
    g = num.groupby("__key", sort=True)
    sums = g.sum()
    cnt = g.count()
    size = g.size()
    valid = cnt.eq(size, axis=0)
    agg = sums.where(valid)
    periods = [(int(k // 100), int(k % 100)) for k in agg.index.to_numpy()]
    official = agg["official"].to_numpy(dtype=float)
    vals = agg[kb_cols].to_numpy(dtype=float)
    return {"periods": periods, "official": official, "vals": vals,
            "kb_idx": kb_idx, "kb_cols": kb_cols}


def gen_f_matrix(en):
    """F[p, j] = |off - val| / (|off| + |val|)；valid[p, j] = 有限。用于任意训练子集的精确 SMAPE。"""
    off = en["official"][:, None]
    vals = en["vals"]
    den = np.abs(off) + np.abs(vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.divide(np.abs(vals - off), den, out=np.full_like(den, np.nan), where=den > 0)
    return F


def gen_surface_from_F(F, sel_rows):
    """在 sel_rows 选中的期点上，按列算 SMAPE → 30×41 面。"""
    sub = F[sel_rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 100.0 * np.nanmean(sub, axis=0)
    en_kb = None
    return s


def _cn(k, b):
    k = round(float(k), 4); b = round(float(b), 4)
    k = 0.0 if k == 0 else k
    b = 0.0 if b == 0 else b
    return "k%gb%g" % (k, b)


def theory_at(en, k, b):
    k0 = float(np.clip(np.floor(k * 10) / 10, 0.1, 2.9)); k1 = k0 + 0.1
    b0 = float(np.clip(np.floor(b * 10) / 10, -2.0, 1.9)); b1 = b0 + 0.1
    wk = (k - k0) / (k1 - k0) if k1 > k0 else 0.0
    wb = (b - b0) / (b1 - b0) if b1 > b0 else 0.0
    need = [_cn(k0, b0), _cn(k0, b1), _cn(k1, b0), _cn(k1, b1)]
    ci = {c: i for i, c in enumerate(en["kb_cols"])}
    missing = [c for c in need if c not in ci]
    if missing:
        raise KeyError(f"枚举表缺列 {missing}")
    v = [en["vals"][:, ci[c]] for c in need]
    out = ((1 - wk) * (1 - wb) * v[0] + (1 - wk) * wb * v[1]
           + wk * (1 - wb) * v[2] + wk * wb * v[3])
    return out, dict(k0=k0, k1=k1, b0=b0, b1=b1, wk=wk, wb=wb)


def fit_kb(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 2 or np.var(x) <= 0:
        return float("nan"), float("nan"), n
    A = np.column_stack([x, np.ones(n)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1]), n


# ================================================================== 划分
def make_splits(n_splits=N_SPLITS, per=N_TEST_PER_SEASON, seed0=SEED0):
    rng = np.random.default_rng(seed0)
    pools = {s: np.array(season_codes(s), dtype=np.int32) for s in SEASONS}
    splits, seen = [], set()
    guard = 0
    while len(splits) < n_splits:
        guard += 1
        if guard > 200000:
            raise RuntimeError("划分去重失败")
        test = set()
        for s in SEASONS:
            pick = rng.choice(pools[s], size=per[s], replace=False)
            test.update(int(x) for x in pick)
        key = frozenset(test)
        if key in seen:
            continue
        seen.add(key)
        splits.append({"split": len(splits) + 1, "test_codes": sorted(test),
                       "train_codes": sorted(set(ALL_CODES.tolist()) - test)})
    return splits


def write_split_defs(splits):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, meta = [], []
    for sp in splits:
        cnt = {s: 0 for s in SEASONS}
        for c in sp["test_codes"]:
            cnt[PERIOD_SEASON[PERIOD_OF_CODE[c]]] += 1
        for c in sp["test_codes"]:
            p = PERIOD_OF_CODE[c]
            rows.append(dict(split=sp["split"], sample="test", year=p[0],
                             period=("1-2月" if p[1] == 0 else f"{p[1]}月"),
                             season=PERIOD_SEASON[p]))
        meta.append(dict(split=sp["split"], n_test_periods=len(sp["test_codes"]),
                         n_train_periods=len(sp["train_codes"]),
                         test_spring=cnt["spring"], test_summer=cnt["summer"],
                         test_autumn=cnt["autumn"], test_winter=cnt["winter"],
                         pct_test=round(100.0 * len(sp["test_codes"]) / len(GRID), 4)))
    pd.DataFrame(rows).to_csv(OUT / "split_def.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(meta).to_csv(OUT / "split_meta.csv", index=False, encoding="utf-8-sig")
    return meta


# ================================================================== 路径缓存（一次）
def prep_path_one(job):
    prov, season = job
    t0 = time.time()
    dst = WORK / "paths" / f"{prov}_{season}.npz"
    if dst.exists():
        return (prov, season, "skip", 0, 0.0)
    try:
        ws, nper = wind_surface_fixed(prov, season)
        if ws is None or not np.isfinite(ws.values).any():
            return (prov, season, "no_wind_data", nper, time.time() - t0)
        wm = fit_poly(ws, POLY_DEGREE)
        if wm is None:
            return (prov, season, "wind_fit_fail", nper, time.time() - t0)
        path = optimal_path(wm)
        if path is None:
            return (prov, season, "path_fail", nper, time.time() - t0)
        (WORK / "paths").mkdir(parents=True, exist_ok=True)
        np.savez(dst, k=path["k"], b=path["b"],
                 wind_at_path=wm.eval(path["k"], path["b"]),
                 n_wind_periods=np.int64(nper),
                 wind_r2=np.float64(wm.r2 if np.isfinite(wm.r2) else -1.0))
        return (prov, season, "ok", nper, time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        return (prov, season, "ERR:" + str(exc)[:180], 0, time.time() - t0)


