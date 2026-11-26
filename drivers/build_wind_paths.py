#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the wind-valley path files used by the M4 selector.

For each province/season wind-surface file (``<prov>_<season>.npz`` with
``periods``, ``acc`` and ``cnt`` arrays, as produced by the wind-speed
comparison stage), this script builds the smooth wind-SMAPE surface,
follows the one-dimensional valley and writes::

    <prov>_<season>.npz
        k, b, wind_at_path, n_wind_periods, wind_r2

Those files are the input of ``drivers/cv_m4.py``.  The M4 selector
itself reads the wind-SMAPE CSV surfaces and does not need the npz
paths.  The generation error is not used here.

Usage::

    python drivers/build_wind_paths.py \
        --surf-dir data/wind_surfaces \
        --out data/wind_valley_paths
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv_base as base  # noqa: E402


def build_one(surf_file: Path, out_dir: Path, overwrite: bool = False) -> Path:
    prov, season = surf_file.stem.rsplit("_", 1)
    if season not in base.SEASONS:
        raise ValueError(f"unrecognised season in {surf_file.name}")
    dst = out_dir / f"{prov}_{season}.npz"
    if dst.exists() and not overwrite:
        return dst
    surf, nper = base.wind_surface_fixed(prov, season)
    if surf is None or not np.isfinite(surf.values).any():
        raise RuntimeError(f"no finite wind surface for {prov}/{season}")
    model = base.fit_poly(surf, base.POLY_DEGREE)
    if model is None:
        raise RuntimeError(f"wind-surface polynomial fit failed for {prov}/{season}")
    path = base.optimal_path(model)
    if path is None:
        raise RuntimeError(f"wind-valley path failed for {prov}/{season}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst,
        k=path["k"],
        b=path["b"],
        wind_at_path=model.eval(path["k"], path["b"]),
        n_wind_periods=np.int64(nper),
        wind_r2=np.float64(model.r2 if np.isfinite(model.r2) else -1.0),
    )
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wind-valley paths for M4")
    parser.add_argument("--surf-dir", type=Path, required=True,
                        help="directory containing <prov>_<season>.npz surf files")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for <prov>_<season>.npz paths")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base.SURF = args.surf_dir.resolve()
    surf_files = sorted(args.surf_dir.glob("*.npz"))
    if not surf_files:
        raise FileNotFoundError(f"no *.npz files in {args.surf_dir}")
    failures = []
    for surf_file in surf_files:
        try:
            dst = build_one(surf_file, args.out, overwrite=args.overwrite)
            print("ok", dst.name)
        except Exception as exc:  # noqa: BLE001
            failures.append((surf_file.name, str(exc)))
    if failures:
        raise SystemExit(f"{len(failures)} path build(s) failed: {failures[:3]}")
    print(f"wrote {len(surf_files)} path files to {args.out}")


if __name__ == "__main__":
    main()
