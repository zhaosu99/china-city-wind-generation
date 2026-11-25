from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from indices import smape
from utils import safe_read_csv


KB_PATTERN = re.compile(r"^k(-?\d+(?:\.\d+)?)b(-?\d+(?:\.\d+)?)$")


def _parse_kb_columns(columns) -> list[tuple[str, float, float]]:
    items: list[tuple[str, float, float]] = []
    for col in columns:
        match = KB_PATTERN.match(str(col).strip())
        if not match:
            continue
        items.append((str(col), float(match.group(1)), float(match.group(2))))
    items.sort(key=lambda item: (item[2], item[1]))
    return items


def _parse_prov_and_season(csv_path: Path) -> tuple[str, str]:
    parts = csv_path.stem.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return csv_path.stem, "unknown"


def _compute_subset_smape_grid(df: pd.DataFrame, subset_times: tuple[str, ...]) -> pd.DataFrame:
    time_col = df.columns[0]
    official_col = next((c for c in df.columns if str(c).strip().lower() == "official"), None)
    if official_col is None:
        raise KeyError("未找到 official 列")

    indexed_df = df.copy()
    indexed_df[time_col] = indexed_df[time_col].astype(str).str.strip()
    indexed_df = indexed_df.set_index(time_col)

    kb_items = _parse_kb_columns(indexed_df.columns)
    if not kb_items:
        raise ValueError("未找到任何 kb 列")

    all_k = sorted({k for _, k, _ in kb_items})
    all_b = sorted({b for _, _, b in kb_items})
    subset_df = indexed_df.loc[indexed_df.index.intersection(subset_times)]
    grid = pd.DataFrame(index=all_b, columns=all_k, dtype=float)

    for col, k_val, b_val in kb_items:
        pair = subset_df[[official_col, col]].dropna()
        if pair.empty:
            grid.loc[b_val, k_val] = np.nan
            continue
        y_true = pd.to_numeric(pair[official_col], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(pair[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            grid.loc[b_val, k_val] = np.nan
            continue
        grid.loc[b_val, k_val] = float(smape(y_true[mask], y_pred[mask]))

    return grid


def _get_valid_intersection_times(df) -> list[str]:
    time_col = df.columns[0]
    official_col = next((c for c in df.columns if str(c).strip().lower() == "official"), None)
    if official_col is None:
        raise KeyError("未找到 official 列")

    kb_columns = [col for col in df.columns if KB_PATTERN.match(str(col).strip())]
    if not kb_columns:
        raise KeyError("未找到任何 kb 列")

    required_columns = [official_col, *kb_columns]
    tmp = df.loc[df[required_columns].notna().all(axis=1), [time_col]].copy()
    tmp[time_col] = tmp[time_col].astype(str).str.strip()
    tmp = tmp[tmp[time_col] != ""]
    return tmp[time_col].drop_duplicates().tolist()


def _build_output_csv_path(output_dir: Path, source_csv: Path) -> Path:
    prov_short, season = _parse_prov_and_season(source_csv)
    return output_dir / f"{prov_short}_{season}_full_gen_smape.csv"


def _normalize_excluded_provinces(excluded_provinces: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    if excluded_provinces is None:
        return {"西藏", "香港", "澳门"}
    return {str(item).strip() for item in excluded_provinces if str(item).strip()}


def _run_single_full_smape_task(task: tuple[str, str, bool]) -> dict[str, Any]:
    csv_path_str, output_csv_str, overwrite = task
    csv_path = Path(csv_path_str)
    output_csv = Path(output_csv_str)

    if output_csv.exists() and not overwrite:
        return {
            "csv": csv_path.name,
            "output_csv": str(output_csv),
            "written": False,
            "skipped_existing": True,
        }

    df = safe_read_csv(csv_path)
    if df is None or df.empty:
        raise ValueError(f"输入 CSV 为空: {csv_path}")

    all_times = tuple(_get_valid_intersection_times(df))
    if not all_times:
        raise ValueError(f"未找到基于 official 与全部 kb 列共同有效的时间交集: {csv_path}")

    grid = _compute_subset_smape_grid(df, all_times)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_csv, encoding="utf-8-sig")

    return {
        "csv": csv_path.name,
        "output_csv": str(output_csv),
        "written": True,
        "skipped_existing": False,
        "time_count": len(all_times),
    }


def prov_gen_full_smape(
    input_dir: str | Path,
    output_dir: str | Path,
    max_workers: int | None = None,
    file_pattern: str = "*_monthly_gen_enums*.csv",
    excluded_provinces: list[str] | tuple[str, ...] | set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"input_dir 不存在或不是目录: {input_root}")

    csv_files = sorted(input_root.glob(file_pattern), key=lambda path: path.name)
    if not csv_files:
        raise FileNotFoundError(f"目录下未匹配到文件: {input_root} / {file_pattern}")

    excluded_province_set = _normalize_excluded_provinces(excluded_provinces)
    target_csv_files: list[Path] = []
    excluded_files: list[str] = []
    for csv_path in csv_files:
        prov_short, _ = _parse_prov_and_season(csv_path)
        if prov_short in excluded_province_set:
            excluded_files.append(str(csv_path))
            continue
        target_csv_files.append(csv_path)

    output_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(csv_path), str(_build_output_csv_path(output_root, csv_path)), overwrite)
        for csv_path in target_csv_files
    ]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_full_smape_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["csv"])
    written_files = [item["output_csv"] for item in results if item["written"]]
    skipped_files = [item["output_csv"] for item in results if item["skipped_existing"]]

    return {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "file_pattern": file_pattern,
        "excluded_provinces": sorted(excluded_province_set),
        "excluded_file_count": len(excluded_files),
        "excluded_files": excluded_files,
        "task_count": len(tasks),
        "written_count": len(written_files),
        "written_files": written_files,
        "skipped_existing_count": len(skipped_files),
        "skipped_existing_files": skipped_files,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为每个省季 monthly_gen_enums CSV 直接生成一张全量 generation SMAPE surface。",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="输入目录，读取 *_monthly_gen_enums*.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--max-workers", type=int, default=None, help="并行 worker 数")
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*_monthly_gen_enums*.csv",
        help="输入文件匹配模式",
    )
    parser.add_argument(
        "--excluded-provinces",
        type=str,
        nargs="*",
        default=["西藏", "香港", "澳门"],
        help="这些省份/地区不参与 step7 计算。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出 CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prov_gen_full_smape(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        file_pattern=args.file_pattern,
        excluded_provinces=args.excluded_provinces,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()