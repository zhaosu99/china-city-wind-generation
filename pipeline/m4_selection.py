#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4 wind-speed calibration: wind-only selection of (k, b).

This module implements the calibration that produced
``parameters/optimal_kb_M4.csv`` in the manuscript.

Method (M4)
-----------
For each province and season:

1. Build a smooth wind-speed error surface W(k, b) against the
   CDMet-constrained benchmark.
2. Follow the one-dimensional valley b*(k) = argmin_b W(k, b).
3. Select the point on that valley with the lowest *wind-speed* SMAPE.

The generation data are deliberately **not** read in this module.  The
legacy M2 workflow used the generation-error surface to choose the
candidate path; that code is kept for transparency in
``legacy/m2_find_optimal_kb_full_data.py`` and is not part of the final
M4 calibration.

Input naming
------------
``wind_dir``:
    ``<prov>_<season>_w_smape_enums.csv`` with index b and columns k.

``input_dir``:
    ``<prov>_<season>_full_gen_smape.csv`` files are used **only** to
    enumerate the province/season pairs, so the CLI is compatible with
    the production pipeline.  Their contents are not read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wind_valley import (
    MIN_OPTIMAL_K,
    get_optimal_path_from_dataframe,
    interpolate_smape_value,
    load_smape_csv,
)

SEASONS = ("spring", "summer", "autumn", "winter")
K_MIN, K_MAX = MIN_OPTIMAL_K, 3.0
B_MIN, B_MAX = -2.0, 2.0


def select_m4_from_path(
    k_path,
    b_path,
    wind_smape,
    *,
    k_min: float = K_MIN,
    k_max: float = K_MAX,
    b_min: float = B_MIN,
    b_max: float = B_MAX,
) -> dict:
    """Select the minimum wind error along the wind-valley path.

    Parameters are plain arrays, which makes the core selection testable
    without any generation data.  The selected point is the minimum
    wind-SMAPE point on the wind-valley path inside the calibration box.
    """
    k_path = np.asarray(k_path, dtype=float).ravel()
    b_path = np.asarray(b_path, dtype=float).ravel()
    wind_smape = np.asarray(wind_smape, dtype=float).ravel()
    if not (k_path.size == b_path.size == wind_smape.size):
        raise ValueError("k_path, b_path and wind_smape must have the same length")
    if k_path.size == 0:
        raise ValueError("empty wind-valley path")

    in_box = (
        (k_path >= k_min - 1e-9)
        & (k_path <= k_max + 1e-9)
        & (b_path >= b_min - 1e-9)
        & (b_path <= b_max + 1e-9)
    )
    if not np.any(in_box):
        raise ValueError("no path point inside the calibration box")
    valid = in_box & np.isfinite(wind_smape)
    if not valid.any():
        raise ValueError("no finite wind-SMAPE value inside the calibration box")
    values = np.where(valid, wind_smape, np.inf)

    idx = int(np.argmin(values))
    k = float(k_path[idx])
    b = float(b_path[idx])
    on_boundary = bool(
        np.isclose(k, k_min, atol=1e-9)
        or np.isclose(k, k_max, atol=1e-9)
        or np.isclose(b, b_min, atol=1e-9)
        or np.isclose(b, b_max, atol=1e-9)
    )
    return {
        "k": k,
        "b": b,
        "w_smape": float(wind_smape[idx]),
        "candidate_point_count": int(in_box.sum()),
        "on_boundary": on_boundary,
        "selection_mode": "M4_wind_only",
    }


def select_m4_from_surface(wind_df: pd.DataFrame, poly_degree: int = 9) -> dict:
    """Build the wind valley from a wind-SMAPE surface and select M4."""
    path = get_optimal_path_from_dataframe(
        wind_df,
        poly_degree=poly_degree,
        k_min=K_MIN,
        k_max=K_MAX,
        b_min=B_MIN,
        b_max=B_MAX,
        return_boundary_flag=True,
    )
    (
        k_path,
        b_path,
        model,
        poly,
        _k_fine,
        _b_fine,
        _s_fine,
        _fit_r2,
        _path_on_boundary,
    ) = path
    if k_path is None or model is None:
        raise RuntimeError("could not build a wind-valley path from the wind surface")
    wind_on_path = np.array(
        [interpolate_smape_value(model, poly, k, b) for k, b in zip(k_path, b_path)],
        dtype=float,
    )
    return select_m4_from_path(k_path, b_path, wind_on_path)


def _parse_pair_name(path: Path) -> tuple[str, str] | None:
    stem = path.name
    suffix = "_full_gen_smape.csv"
    if not stem.endswith(suffix):
        return None
    body = stem[: -len(suffix)]
    for season in SEASONS:
        marker = f"_{season}"
        if body.endswith(marker):
            prov = body[: -len(marker)]
            if prov:
                return prov, season
    return None


def _plot_selection(wind_df: pd.DataFrame, result: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=160)
    image = ax.imshow(
        np.asarray(wind_df.values, dtype=float),
        origin="lower",
        aspect="auto",
        extent=[
            float(wind_df.columns.min()),
            float(wind_df.columns.max()),
            float(wind_df.index.min()),
            float(wind_df.index.max()),
        ],
        cmap="viridis_r",
    )
    ax.scatter([result["k"]], [result["b"]], c="gold", edgecolors="black", s=80, zorder=5)
    ax.set_xlabel("k (wind-speed multiplier)")
    ax.set_ylabel("b (wind-speed offset, m s$^{-1}$)")
    ax.set_title(f"M4 wind-only selection (SMAPE={result['w_smape']:.3f})")
    fig.colorbar(image, ax=ax, label="wind-speed SMAPE")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def find_optimal_kb_m4(
    input_dir: Path,
    wind_dir: Path,
    output_dir: Path,
    optimal_kb_csv: Path | None = None,
    compute: bool = True,
    plot: bool = False,
    overwrite: bool = False,
    poly_degree: int = 9,
    max_workers: int | None = None,  # accepted for pipeline compatibility
) -> dict[str, object]:
    """Run the M4 wind-only selection over all province/season pairs."""
    input_dir = Path(input_dir).resolve()
    wind_dir = Path(wind_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_csv = (
        Path(optimal_kb_csv).resolve()
        if optimal_kb_csv is not None
        else output_dir / "optimal_kb.csv"
    )
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not wind_dir.exists():
        raise FileNotFoundError(f"wind_dir not found: {wind_dir}")
    if poly_degree < 1:
        raise ValueError("poly_degree must be >= 1")
    if not compute and not plot:
        raise ValueError("compute and plot cannot both be False")

    gen_csvs = sorted(input_dir.glob("*_full_gen_smape.csv"))
    if not gen_csvs:
        raise FileNotFoundError(f"no *_full_gen_smape.csv found in {input_dir}")

    rows: list[dict] = []
    skipped: list[dict] = []
    failures: list[dict] = []
    for gen_csv in gen_csvs:
        pair = _parse_pair_name(gen_csv)
        if pair is None:
            skipped.append({"file": str(gen_csv), "reason": "unrecognised filename"})
            continue
        prov, season = pair
        wind_csv = wind_dir / f"{prov}_{season}_w_smape_enums.csv"
        if not wind_csv.exists():
            skipped.append({"file": str(gen_csv), "wind_csv": str(wind_csv), "reason": "missing wind CSV"})
            continue
        try:
            wind_df = load_smape_csv(wind_csv)
            result = select_m4_from_surface(wind_df, poly_degree=poly_degree)
            rows.append(
                {
                    "prov": prov,
                    "season": season,
                    "w_smape": result["w_smape"],
                    "k": result["k"],
                    "b": result["b"],
                    # M4 does not use generation data; leave the diagnostic column empty.
                    "gen_smape": np.nan,
                    "candidate_point_count": result["candidate_point_count"],
                    "selection_mode": result["selection_mode"],
                    "on_boundary": result["on_boundary"],
                }
            )
            if plot:
                _plot_selection(wind_df, result, output_dir / "plots" / f"{prov}_{season}_m4_selection.jpg")
        except Exception as exc:  # noqa: BLE001
            failures.append({"prov": prov, "season": season, "error": str(exc)[:300]})

    if not rows:
        raise RuntimeError("M4 selection produced no rows")
    if failures:
        raise RuntimeError(f"M4 selection failed for {len(failures)} province/season pair(s): {failures[:3]}")

    df = (
        pd.DataFrame(rows)
        .sort_values(["prov", "season"], kind="stable")
        .reset_index(drop=True)
    )
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"{output_csv} already exists; pass overwrite=True to replace it"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return {
        "output_csv": str(output_csv),
        "n_rows": int(len(df)),
        "n_failures": len(failures),
        "failure_count": len(failures),  # panel.py failure-gate key
        "n_skipped": len(skipped),
        "selection_mode": "M4_wind_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 wind-only (k, b) calibration")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="directory containing *_full_gen_smape.csv (names only)")
    parser.add_argument("--wind-dir", type=Path, required=True,
                        help="directory containing *_w_smape_enums.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("results/optimal_kb"))
    parser.add_argument("--optimal-kb-csv", type=Path, default=None)
    parser.add_argument("--poly-degree", type=int, default=9)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = find_optimal_kb_m4(
        input_dir=args.input_dir,
        wind_dir=args.wind_dir,
        output_dir=args.output_dir,
        optimal_kb_csv=args.optimal_kb_csv,
        compute=True,
        plot=args.plot,
        overwrite=args.overwrite,
        poly_degree=args.poly_degree,
    )
    print(summary)


if __name__ == "__main__":
    main()
