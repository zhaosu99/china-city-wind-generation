from __future__ import annotations

import argparse
import math
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from indices import smape
from utils import safe_read_csv

SEASONS = ("spring", "summer", "autumn", "winter")
FILENAME_RE = re.compile(r"^(?P<prov>.+)_(?P<season>spring|summer|autumn|winter)_w_compare\.csv$")


def _float_range_inclusive(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")

    count = int(math.floor((end - start) / step + 1e-12)) + 1
    values: list[float] = []
    for idx in range(count):
        values.append(round(start + idx * step, 10))

    if not math.isclose(values[-1], end, rel_tol=0.0, abs_tol=1e-9):
        values.append(round(end, 10))

    return np.array(values, dtype=float)


def _parse_province_season(file_path: Path) -> tuple[str, str]:
    match = FILENAME_RE.match(file_path.name)
    if not match:
        raise ValueError(
            f"Unexpected filename format: {file_path.name}. "
            f"Expected '<prov_short>_<season>_w_compare.csv'"
        )
    prov_short = match.group("prov").strip()
    season = match.group("season").strip()
    if season not in SEASONS:
        raise ValueError(f"Invalid season in filename: {file_path.name}")
    return prov_short, season


def _build_task_list(input_dir: Path) -> list[dict[str, Any]]:
    csv_files = sorted(input_dir.glob("*_w_compare.csv"))
    tasks: list[dict[str, Any]] = []
    for csv_file in csv_files:
        prov_short, season = _parse_province_season(csv_file)
        tasks.append(
            {
                "input_csv": str(csv_file),
                "prov_short": prov_short,
                "season": season,
            }
        )
    return tasks


def _compute_single_prov_season(task: dict[str, Any]) -> dict[str, Any]:
    input_csv = Path(task["input_csv"])
    prov_short = str(task["prov_short"])
    season = str(task["season"])
    output_dir = Path(task["output_dir"])
    era5_col_name = str(task["era5_col_name"])
    cdmet_col_name = str(task["cdmet_col_name"])
    overwrite = bool(task["overwrite"])

    k_values = np.array(task["k_values"], dtype=float)
    b_values = np.array(task["b_values"], dtype=float)
    output_csv = output_dir / f"{prov_short}_{season}_w_smape_enums.csv"

    if output_csv.exists() and not overwrite:
        return {
            "prov_short": prov_short,
            "season": season,
            "output_csv": str(output_csv),
            "rows": int(len(b_values)),
            "cols": int(len(k_values)),
            "written": False,
            "skipped_existing": True,
        }

    df = safe_read_csv(input_csv)
    missing_cols = [c for c in [era5_col_name, cdmet_col_name] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns {missing_cols}: {input_csv}")

    era5_w = df[era5_col_name].to_numpy(dtype=float)
    cdmet_w = df[cdmet_col_name].to_numpy(dtype=float)

    if len(era5_w) != len(cdmet_w):
        raise ValueError(
            f"Input length mismatch in {input_csv}: "
            f"{era5_col_name}={len(era5_w)}, {cdmet_col_name}={len(cdmet_w)}"
        )
    if len(era5_w) == 0:
        raise ValueError(f"Input CSV contains no rows: {input_csv}")

    result_df = pd.DataFrame(
        np.nan,
        index=b_values,
        columns=k_values,
        dtype=float,
    )

    for row_idx, b in enumerate(b_values):
        for col_idx, k in enumerate(k_values):
            transformed = era5_w * k + b
            result_df.iloc[row_idx, col_idx] = float(smape(cdmet_w, transformed))

    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=True, header=True, encoding="utf-8-sig")

    return {
        "prov_short": prov_short,
        "season": season,
        "output_csv": str(output_csv),
        "rows": int(result_df.shape[0]),
        "cols": int(result_df.shape[1]),
        "written": True,
        "skipped_existing": False,
    }


def prov_w_smape(
    input_dir: Path,
    output_dir: Path,
    era5_col_name: str = "era5_w",
    cdmet_col_name: str = "cdmet_w",
    k_start: float = 0.1,
    k_end: float = 3.0,
    k_step: float = 0.1,
    b_start: float = -2.0,
    b_end: float = 2.0,
    b_step: float = 0.1,
    max_workers: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    For each (prov_short, season) CSV, compute SMAPE grid of (k, b).

    Input CSV naming convention:
    - <prov_short>_<season>_w_compare.csv

    Output layout:
    - output_dir/<prov_short>_<season>_w_smape_enums.csv

    Output CSV shape:
    - index: b values
    - columns: k values
    - values: smape(cdmet_w, era5_w * k + b)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = _float_range_inclusive(k_start, k_end, k_step)
    b_values = _float_range_inclusive(b_start, b_end, b_step)

    tasks = _build_task_list(input_dir)
    if not tasks:
        raise ValueError(f"No '*_w_compare.csv' files found in input_dir: {input_dir}")

    for task in tasks:
        task["output_dir"] = str(output_dir)
        task["era5_col_name"] = era5_col_name
        task["cdmet_col_name"] = cdmet_col_name
        task["k_values"] = k_values.tolist()
        task["b_values"] = b_values.tolist()
        task["overwrite"] = overwrite

    max_workers_final = len(tasks) if max_workers is None else max(1, min(int(max_workers), len(tasks)))
    print(
        f"Found {len(tasks)} (province, season) input CSV files. "
        "Each pair reads CSV once and computes full (k,b) grid in-process. "
        f"Using ProcessPoolExecutor with {max_workers_final} workers."
    )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers_final) as executor:
        future_map = {executor.submit(_compute_single_prov_season, task): task for task in tasks}
        for idx, future in enumerate(as_completed(future_map), 1):
            task = future_map[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{idx}/{len(tasks)}] done {result['prov_short']}/{result['season']} -> "
                    f"{result['output_csv']} | skipped_existing={result['skipped_existing']}"
                )
            except Exception as exc:
                executor.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(
                    "SMAPE enumeration failed for task: "
                    f"prov={task['prov_short']}, season={task['season']}, input={task['input_csv']}"
                ) from exc

    return {
        "task_count": len(results),
        "k_count": int(len(k_values)),
        "b_count": int(len(b_values)),
        "total_existing_skips": sum(1 for item in results if item["skipped_existing"]),
        "results": results,
    }


def parse_arguments() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Compute (k,b) SMAPE enums for province-season wind comparison CSV files",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"/geogfs1/groups/geog_liulb/hongyu/results/prov_w_compare"),
        help="Directory containing '<prov_short>_<season>_w_compare.csv' files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"/geogfs1/groups/geog_liulb/hongyu/results/prov_w_smape_enums"),
        help="Output directory for '<prov_short>_<season>_w_smape_enums.csv' files.",
    )

    parser.add_argument("--era5-col-name", type=str, default="era5_w")
    parser.add_argument("--cdmet-col-name", type=str, default="cdmet_w")

    parser.add_argument("--k-start", type=float, default=0.1)
    parser.add_argument("--k-end", type=float, default=3.0)
    parser.add_argument("--k-step", type=float, default=0.1)

    parser.add_argument("--b-start", type=float, default=-2.0)
    parser.add_argument("--b-end", type=float, default=2.0)
    parser.add_argument("--b-step", type=float, default=0.1)

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Process count for (province, season) tasks. Default: number of tasks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV files.")

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    try:
        print("Initializing command line argument parser...")
        run_kwargs = parse_arguments()

        print("Starting prov_cdmet_w_smape with configuration:")
        for key, value in run_kwargs.items():
            print(f"  {key}: {value}")
        print()

        summary = prov_w_smape(**run_kwargs)
        print("\n--- Execution Complete ---")
        print(
            "Summary: "
            f"tasks={summary['task_count']}, "
            f"k_count={summary['k_count']}, "
            f"b_count={summary['b_count']}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[CRITICAL] Script failed: {str(e)}")
        traceback.print_exc()
