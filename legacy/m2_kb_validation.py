from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from aggr_farm import aggr_farm_output
from find_optimal_kb import (
    find_min_wind_on_gen_op,
    fit_surface_from_dataframe,
    get_optimal_path_from_dataframe,
    interpolate_smape_value,
    load_smape_csv,
)
from indices import smape
from utils import chinese_to_pinyi, open_dataset_with_fallback, safe_read_csv, sanitize_filename_part


MONTHLY_ENUM_PATTERN = re.compile(r"^(?P<prov>.+?)_(?P<season>[^_]+)_monthly_gen_enums.*\.csv$")
DETAIL_CSV_NAME = "kb_validation_detail.csv"
SUMMARY_CSV_NAME = "kb_validation_summary.csv"
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


def _parse_monthly_enum_filename(csv_path: Path) -> dict[str, str] | None:
    match = MONTHLY_ENUM_PATTERN.match(csv_path.name)
    if match is None:
        return None
    return {
        "prov": match.group("prov"),
        "season": match.group("season"),
    }


def _parse_year_month_label(label: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d{4})年(\d{1,2})月", str(label).strip())
    if match is None:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return year, month


def _parse_jan_feb_label(label: str) -> int | None:
    match = re.fullmatch(r"(\d{4})年1-2月", str(label).strip())
    if match is None:
        return None
    return int(match.group(1))


def _month_label(year: int, month: int) -> str:
    return f"{year}年{month}月"


def _jan_feb_label(year: int) -> str:
    return f"{year}年1-2月"


def _label_sort_key(label: str) -> tuple[int, int, str]:
    jan_feb_year = _parse_jan_feb_label(label)
    if jan_feb_year is not None:
        return (jan_feb_year, 2, str(label))

    parsed = _parse_year_month_label(label)
    if parsed is None:
        return (9999, 99, str(label))
    return (parsed[0], parsed[1], str(label))


def _augment_with_winter_jan_feb_labels(series_map: dict[str, float]) -> dict[str, float]:
    augmented = dict(series_map)
    winter_monthly_values: dict[int, dict[int, float]] = {}

    for label, value in series_map.items():
        parsed = _parse_year_month_label(label)
        if parsed is None:
            continue

        year, month = parsed
        if month not in (1, 2):
            continue
        winter_monthly_values.setdefault(year, {})[month] = float(value)

    for year, month_values in winter_monthly_values.items():
        jan_value = month_values.get(1)
        feb_value = month_values.get(2)
        if jan_value is None and feb_value is None:
            continue
        augmented.setdefault(_jan_feb_label(year), float((jan_value or 0.0) + (feb_value or 0.0)))

    return augmented


def _get_valid_intersection_times(df: pd.DataFrame) -> list[str]:
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


def _build_official_map(df: pd.DataFrame) -> dict[str, float]:
    time_col = df.columns[0]
    official_col = next((c for c in df.columns if str(c).strip().lower() == "official"), None)
    if official_col is None:
        raise KeyError("未找到 official 列")

    tmp = df[[time_col, official_col]].copy()
    tmp[time_col] = tmp[time_col].astype(str).str.strip()
    tmp[official_col] = pd.to_numeric(tmp[official_col], errors="coerce")
    tmp = tmp.dropna(subset=[official_col])
    official_map = {
        str(row[time_col]): float(row[official_col])
        for _, row in tmp.iterrows()
        if str(row[time_col]).strip() != ""
    }
    return _augment_with_winter_jan_feb_labels(official_map)


def _prepare_repeated_kfold_tasks(
    labels: list[str],
    n_splits: int,
    n_repeats: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    if n_splits < 2:
        raise ValueError(f"n_splits 必须 >= 2, 当前为: {n_splits}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats 必须 >= 1, 当前为: {n_repeats}")
    if len(labels) < n_splits:
        raise ValueError(f"样本数不足以做 {n_splits} 折交叉验证: n={len(labels)}")

    ordered_labels = sorted(labels, key=_label_sort_key)
    tasks: list[dict[str, Any]] = []
    for repeat_id in range(1, n_repeats + 1):
        rng = random.Random(random_seed + repeat_id - 1)
        shuffled = ordered_labels.copy()
        rng.shuffle(shuffled)

        fold_sizes = [len(shuffled) // n_splits] * n_splits
        for idx in range(len(shuffled) % n_splits):
            fold_sizes[idx] += 1

        start = 0
        for fold_id, fold_size in enumerate(fold_sizes, start=1):
            stop = start + fold_size
            holdout_labels = shuffled[start:stop]
            train_labels = shuffled[:start] + shuffled[stop:]
            start = stop

            tasks.append(
                {
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "random_seed": random_seed + repeat_id - 1,
                    "train_labels": sorted(train_labels, key=_label_sort_key),
                    "holdout_labels": sorted(holdout_labels, key=_label_sort_key),
                    "all_labels": ordered_labels,
                }
            )

    return tasks


def _build_task_id(prov: str, season: str, repeat_id: int, fold_id: int) -> str:
    prov_slug = sanitize_filename_part(chinese_to_pinyi(str(prov))) or sanitize_filename_part(str(prov)) or "prov"
    season_slug = sanitize_filename_part(str(season)) or "season"
    return f"{prov_slug}_{season_slug}_r{repeat_id:02d}_f{fold_id:02d}"


def _extract_monthly_gen_map(nc_path: Path) -> dict[str, float]:
    ds = open_dataset_with_fallback(nc_path)
    monthly_var_name = "total_monthly_gen"
    if monthly_var_name not in ds.data_vars:
        available_vars = sorted(str(name) for name in ds.data_vars)
        raise KeyError(f"未找到 total_monthly_gen 变量: {nc_path}, data_vars={available_vars}")

    da = ds[monthly_var_name]
    if "date" not in da.dims:
        raise KeyError(f"{monthly_var_name} 缺少 date 维度: {nc_path}")

    dates = pd.to_datetime(da["date"].values)
    values = np.asarray(da.values, dtype=float).reshape(-1)
    result: dict[str, float] = {}
    for date_value, raw_value in zip(dates, values):
        if not np.isfinite(raw_value):
            continue
        label = _month_label(int(date_value.year), int(date_value.month))
        result[label] = float(raw_value)
    return _augment_with_winter_jan_feb_labels(result)


def _compute_smape_for_labels(pred_map: dict[str, float], official_map: dict[str, float], labels: list[str]) -> float:
    pred_values: list[float] = []
    official_values: list[float] = []
    missing_labels: list[str] = []

    for label in labels:
        if label not in pred_map or label not in official_map:
            missing_labels.append(label)
            continue
        pred_values.append(float(pred_map[label]))
        official_values.append(float(official_map[label]))

    if missing_labels:
        raise KeyError(f"以下月份在重算结果或 official 中缺失: {missing_labels}")
    if not pred_values:
        raise ValueError("没有可用于计算 SMAPE 的月份")

    return float(smape(np.asarray(official_values, dtype=float), np.asarray(pred_values, dtype=float)))


def _build_point_payload(labels: list[str]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for label in labels:
        parsed = _parse_year_month_label(label)
        jan_feb_year = _parse_jan_feb_label(label)
        payload.append(
            {
                "label": label,
                "year": parsed[0] if parsed is not None else jan_feb_year,
                "month": parsed[1] if parsed is not None else (2 if jan_feb_year is not None else None),
            }
        )
    return payload


def _write_single_row_csv(row_csv: Path, row: dict[str, Any]) -> None:
    row_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(row_csv, index=False, encoding="utf-8-sig")


def _write_meta_json(meta_json: Path, payload: dict[str, Any]) -> None:
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_full_data_kb_lookup(optimal_kb_csv: Path | None) -> dict[tuple[str, str], dict[str, float]]:
    if optimal_kb_csv is None or not optimal_kb_csv.exists():
        return {}

    df = safe_read_csv(optimal_kb_csv)
    required_columns = {"prov", "season", "k", "b"}
    if not required_columns.issubset(df.columns):
        return {}

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in df.iterrows():
        prov = str(row["prov"])
        season = str(row["season"])
        lookup[(prov, season)] = {
            "k": float(row["k"]),
            "b": float(row["b"]),
        }
    return lookup


def _get_successful_task_ids(df: pd.DataFrame) -> set[str]:
    if "task_id" not in df.columns or df.empty:
        return set()
    if "status" not in df.columns:
        return set(df["task_id"].dropna().astype(str).tolist())

    success_mask = df["status"].astype(str).str.strip().str.lower() == "success"
    return set(df.loc[success_mask, "task_id"].dropna().astype(str).tolist())


def _load_existing_task_ids(detail_csv: Path, rows_dir: Path) -> set[str]:
    task_ids: set[str] = set()

    if detail_csv.exists():
        detail_df = safe_read_csv(detail_csv)
        task_ids.update(_get_successful_task_ids(detail_df))

    if rows_dir.exists():
        for row_csv in rows_dir.glob("*.csv"):
            row_df = safe_read_csv(row_csv)
            task_ids.update(_get_successful_task_ids(row_df))

    return task_ids


def _collect_row_frames(rows_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if not rows_dir.exists():
        return frames
    for row_csv in sorted(rows_dir.glob("*.csv"), key=lambda path: path.name):
        frames.append(safe_read_csv(row_csv))
    return frames


def _build_detail_csv(detail_csv: Path, rows_dir: Path, overwrite: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if detail_csv.exists() and not overwrite:
        frames.append(safe_read_csv(detail_csv))
    frames.extend(_collect_row_frames(rows_dir))

    if frames:
        detail_df = pd.concat(frames, ignore_index=True)
        if "task_id" in detail_df.columns:
            detail_df = detail_df.drop_duplicates(subset=["task_id"], keep="last")
        detail_df = detail_df.sort_values(by=["prov", "season", "repeat_id", "fold_id"], kind="stable").reset_index(drop=True)
    else:
        detail_df = pd.DataFrame()

    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    return detail_df


def _build_summary_csv(summary_csv: Path, detail_df: pd.DataFrame) -> pd.DataFrame:
    success_df = detail_df.loc[detail_df.get("status", pd.Series(dtype=str)) == "success"].copy()
    if success_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "prov",
                "season",
                "successful_task_count",
                "holdout_smape_mean",
                "holdout_smape_std",
                "holdout_smape_median",
                "train_smape_mean",
                "train_smape_std",
                "generalization_gap_mean",
                "generalization_gap_std",
                "selected_k_mean",
                "selected_k_std",
                "selected_b_mean",
                "selected_b_std",
                "wind_smape_mean",
            ]
        )
    else:
        grouped = success_df.groupby(["prov", "season"], as_index=False)
        summary_df = grouped.agg(
            successful_task_count=("task_id", "count"),
            holdout_smape_mean=("holdout_smape", "mean"),
            holdout_smape_std=("holdout_smape", "std"),
            holdout_smape_median=("holdout_smape", "median"),
            train_smape_mean=("train_smape", "mean"),
            train_smape_std=("train_smape", "std"),
            generalization_gap_mean=("generalization_gap", "mean"),
            generalization_gap_std=("generalization_gap", "std"),
            selected_k_mean=("selected_k", "mean"),
            selected_k_std=("selected_k", "std"),
            selected_b_mean=("selected_b", "mean"),
            selected_b_std=("selected_b", "std"),
            wind_smape_mean=("wind_smape_at_selected_kb", "mean"),
        )
        summary_df = summary_df.sort_values(by=["prov", "season"], kind="stable").reset_index(drop=True)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    return summary_df


def _cleanup_row_csvs(rows_dir: Path) -> None:
    if not rows_dir.exists():
        return
    for row_csv in rows_dir.glob("*.csv"):
        row_csv.unlink(missing_ok=True)


def _run_single_validation_task(task: dict[str, Any]) -> dict[str, Any]:
    start_time = time.perf_counter()

    task_id = task["task_id"]
    meta_json = Path(task["meta_json"])
    row_csv = Path(task["row_csv"])
    tmp_dir = Path(task["tmp_dir"])

    row: dict[str, Any] = {
        "task_id": task_id,
        "prov": task["prov"],
        "season": task["season"],
        "repeat_id": task["repeat_id"],
        "fold_id": task["fold_id"],
        "random_seed": task["random_seed"],
        "n_total": len(task["all_labels"]),
        "n_train": len(task["train_labels"]),
        "n_holdout": len(task["holdout_labels"]),
        "selected_k": np.nan,
        "selected_b": np.nan,
        "wind_smape_at_selected_kb": np.nan,
        "train_surface_smape": np.nan,
        "train_smape": np.nan,
        "holdout_smape": np.nan,
        "generalization_gap": np.nan,
        "candidate_point_count": np.nan,
        "full_data_k": task.get("full_data_k", np.nan),
        "full_data_b": task.get("full_data_b", np.nan),
        "meta_json": str(meta_json),
        "runtime_seconds": np.nan,
        "status": "failed",
        "error_message": None,
    }

    meta_payload: dict[str, Any] = {
        "task_id": task_id,
        "prov": task["prov"],
        "season": task["season"],
        "repeat_id": task["repeat_id"],
        "fold_id": task["fold_id"],
        "random_seed": task["random_seed"],
        "n_splits": task["n_splits"],
        "n_repeats": task["n_repeats"],
        "all_labels": task["all_labels"],
        "train_labels": task["train_labels"],
        "holdout_labels": task["holdout_labels"],
        "all_points": _build_point_payload(task["all_labels"]),
        "train_points": _build_point_payload(task["train_labels"]),
        "holdout_points": _build_point_payload(task["holdout_labels"]),
        "full_data_k": task.get("full_data_k"),
        "full_data_b": task.get("full_data_b"),
        "status": "failed",
        "error_message": None,
    }

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            gen_df = safe_read_csv(task["gen_csv"])
            official_map = _build_official_map(gen_df)
            train_grid = _compute_subset_smape_grid(gen_df, tuple(task["train_labels"]))
            wind_df = load_smape_csv(task["wind_csv"])

            gen_k, gen_b, train_model, train_poly, _, _, _, _ = get_optimal_path_from_dataframe(
                train_grid,
                poly_degree=task["poly_degree"],
            )
            if gen_k is None or train_model is None or train_poly is None:
                raise RuntimeError(f"无法从训练集 generation surface 中提取有效 optimal path: {task['gen_csv']}")

            wind_model, wind_poly, _, _, _, _ = fit_surface_from_dataframe(
                wind_df,
                poly_degree=task["poly_degree"],
            )
            if wind_model is None or wind_poly is None:
                raise RuntimeError(f"无法从 wind surface 中拟合有效模型: {task['wind_csv']}")

            min_result = find_min_wind_on_gen_op(gen_k, gen_b, wind_model, wind_poly)
            selected_k = float(min_result["k"])
            selected_b = float(min_result["b"])
            wind_smape = float(min_result["smape"])
            train_surface_smape = float(interpolate_smape_value(train_model, train_poly, selected_k, selected_b))

            rerun_result = aggr_farm_output(
                era5_dir=task["era5_dir"],
                cdmet_dir=task["cdmet_dir"],
                inventory_csv_path=task["inventory_csv_path"],
                farm_power_curve_dir=task["farm_power_curve_dir"],
                farm_adjusted_w_dir=task["farm_adjusted_w_dir"],
                farm_adjusted_w_varname=task["farm_adjusted_w_varname"],
                output_dir=tmp_dir,
                output_hourly_gen=False,
                output_monthly_gen=True,
                output_hourly_cf=False,
                output_w_compare=False,
                output_raw_w=False,
                select_period=task["season"],
                period_mode="season",
                prov_short=[task["prov"]],
                prov_long=None,
                city_short=None,
                city_long=None,
                kb_info=(selected_k, selected_b),
                gen_adjust_info=None,
                city_capacity_adjust_csv_path=task["city_capacity_adjust_csv_path"],
                prov_utility_rate_adjust_csv_path=task["prov_utility_rate_adjust_csv_path"],
                overwrite=True,
                aggr_level="prov",
                include_kb_in_gen_filenames=False,
                horizon_interp_method=task["horizon_interp_method"],
                max_workers=1,
                verbose=False,
                show_progress=False,
            )

            province_nc = Path(rerun_result["province_generation_files"][task["season"]][task["prov"]])
            monthly_gen_map = _extract_monthly_gen_map(province_nc)
            train_smape = _compute_smape_for_labels(monthly_gen_map, official_map, task["train_labels"])
            holdout_smape = _compute_smape_for_labels(monthly_gen_map, official_map, task["holdout_labels"])

            row.update(
                {
                    "selected_k": selected_k,
                    "selected_b": selected_b,
                    "wind_smape_at_selected_kb": wind_smape,
                    "train_surface_smape": train_surface_smape,
                    "train_smape": train_smape,
                    "holdout_smape": holdout_smape,
                    "generalization_gap": float(holdout_smape - train_smape),
                    "candidate_point_count": int(len(gen_k)),
                    "status": "success",
                }
            )
            meta_payload.update(
                {
                    "selected_k": selected_k,
                    "selected_b": selected_b,
                    "wind_smape_at_selected_kb": wind_smape,
                    "train_surface_smape": train_surface_smape,
                    "train_smape": train_smape,
                    "holdout_smape": holdout_smape,
                    "generalization_gap": float(holdout_smape - train_smape),
                    "candidate_point_count": int(len(gen_k)),
                    "province_generation_nc": str(province_nc),
                    "status": "success",
                }
            )
    except Exception as exc:
        row["error_message"] = str(exc)
        meta_payload["error_message"] = str(exc)
    finally:
        row["runtime_seconds"] = float(time.perf_counter() - start_time)
        meta_payload["runtime_seconds"] = row["runtime_seconds"]
        _write_meta_json(meta_json, meta_payload)
        _write_single_row_csv(row_csv, row)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return row


def kb_validation(
    prov_gen_enums_dir: str | Path,
    wind_dir: str | Path,
    optimal_kb_csv: str | Path | None,
    output_dir: str | Path,
    era5_dir: str | Path,
    cdmet_dir: str | Path | None,
    inventory_csv_path: str | Path,
    farm_power_curve_dir: str | Path | None,
    farm_adjusted_w_dir: str | Path | None,
    farm_adjusted_w_varname: str = "raw_w",
    city_capacity_adjust_csv_path: str | Path | None = None,
    prov_utility_rate_adjust_csv_path: str | Path | None = None,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_seed: int = 42,
    poly_degree: int = 9,
    horizon_interp_method: str = "loess",
    max_workers: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    prov_gen_enums_root = Path(prov_gen_enums_dir).resolve()
    wind_root = Path(wind_dir).resolve()
    output_root = Path(output_dir).resolve()
    optimal_kb_path = Path(optimal_kb_csv).resolve() if optimal_kb_csv is not None else None

    if not prov_gen_enums_root.exists():
        raise FileNotFoundError(f"未找到 prov_gen_enums_dir: {prov_gen_enums_root}")
    if not wind_root.exists():
        raise FileNotFoundError(f"未找到 wind_dir: {wind_root}")
    if n_splits < 2:
        raise ValueError(f"n_splits 必须 >= 2, 当前为: {n_splits}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats 必须 >= 1, 当前为: {n_repeats}")

    output_root.mkdir(parents=True, exist_ok=True)
    rows_dir = output_root / "rows"
    meta_dir = output_root / "meta"
    tmp_runs_dir = output_root / "_tmp_runs"
    detail_csv = output_root / DETAIL_CSV_NAME
    summary_csv = output_root / SUMMARY_CSV_NAME

    if overwrite:
        shutil.rmtree(rows_dir, ignore_errors=True)
        shutil.rmtree(tmp_runs_dir, ignore_errors=True)

    existing_task_ids = set() if overwrite else _load_existing_task_ids(detail_csv, rows_dir)
    full_data_lookup = _load_full_data_kb_lookup(optimal_kb_path)

    tasks: list[dict[str, Any]] = []
    skipped_items: list[dict[str, Any]] = []
    gen_csv_list = sorted(prov_gen_enums_root.glob("*_monthly_gen_enums*.csv"), key=lambda path: path.name)
    if not gen_csv_list:
        raise FileNotFoundError(f"未找到任何 *_monthly_gen_enums*.csv: {prov_gen_enums_root}")

    for gen_csv in gen_csv_list:
        meta = _parse_monthly_enum_filename(gen_csv)
        if meta is None:
            skipped_items.append({"gen_csv": str(gen_csv), "reason": "unrecognized_filename"})
            continue

        prov = meta["prov"]
        season = meta["season"]
        wind_csv = wind_root / f"{prov}_{season}_w_smape_enums.csv"
        if not wind_csv.exists():
            skipped_items.append({"gen_csv": str(gen_csv), "wind_csv": str(wind_csv), "reason": "missing_wind_csv"})
            continue

        gen_df = safe_read_csv(gen_csv)
        valid_labels = _get_valid_intersection_times(gen_df)
        if len(valid_labels) < n_splits:
            skipped_items.append(
                {
                    "gen_csv": str(gen_csv),
                    "reason": "insufficient_valid_months",
                    "valid_month_count": len(valid_labels),
                }
            )
            continue

        split_tasks = _prepare_repeated_kfold_tasks(
            valid_labels,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_seed=random_seed,
        )
        full_data_item = full_data_lookup.get((prov, season), {})
        for split_task in split_tasks:
            task_id = _build_task_id(prov, season, split_task["repeat_id"], split_task["fold_id"])
            if task_id in existing_task_ids:
                skipped_items.append({"task_id": task_id, "reason": "already_completed"})
                continue

            tasks.append(
                {
                    "task_id": task_id,
                    "prov": prov,
                    "season": season,
                    "repeat_id": split_task["repeat_id"],
                    "fold_id": split_task["fold_id"],
                    "random_seed": split_task["random_seed"],
                    "n_splits": n_splits,
                    "n_repeats": n_repeats,
                    "all_labels": split_task["all_labels"],
                    "train_labels": split_task["train_labels"],
                    "holdout_labels": split_task["holdout_labels"],
                    "gen_csv": str(gen_csv),
                    "wind_csv": str(wind_csv),
                    "row_csv": str(rows_dir / f"{task_id}.csv"),
                    "meta_json": str(meta_dir / f"{task_id}.json"),
                    "tmp_dir": str(tmp_runs_dir / task_id),
                    "era5_dir": str(era5_dir),
                    "cdmet_dir": str(cdmet_dir) if cdmet_dir is not None else None,
                    "inventory_csv_path": str(inventory_csv_path),
                    "farm_power_curve_dir": str(farm_power_curve_dir) if farm_power_curve_dir is not None else None,
                    "farm_adjusted_w_dir": str(farm_adjusted_w_dir) if farm_adjusted_w_dir is not None else None,
                    "farm_adjusted_w_varname": farm_adjusted_w_varname,
                    "city_capacity_adjust_csv_path": str(city_capacity_adjust_csv_path) if city_capacity_adjust_csv_path is not None else None,
                    "prov_utility_rate_adjust_csv_path": str(prov_utility_rate_adjust_csv_path) if prov_utility_rate_adjust_csv_path is not None else None,
                    "poly_degree": poly_degree,
                    "horizon_interp_method": horizon_interp_method,
                    "full_data_k": full_data_item.get("k", np.nan),
                    "full_data_b": full_data_item.get("b", np.nan),
                }
            )

    completed_results: list[dict[str, Any]] = []
    if tasks:
        worker_count = len(tasks) if max_workers is None else max(1, min(int(max_workers), len(tasks)))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_single_validation_task, task) for task in tasks]
            with tqdm(
                total=len(futures),
                desc="Step12 validation",
                unit="task",
                dynamic_ncols=True,
            ) as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
                    completed_results.append(future.result())

    detail_df = _build_detail_csv(detail_csv, rows_dir, overwrite=overwrite)
    summary_df = _build_summary_csv(summary_csv, detail_df)
    _cleanup_row_csvs(rows_dir)

    success_count = int((detail_df.get("status", pd.Series(dtype=str)) == "success").sum()) if not detail_df.empty else 0
    failure_count = int((detail_df.get("status", pd.Series(dtype=str)) == "failed").sum()) if not detail_df.empty else 0

    return {
        "prov_gen_enums_dir": str(prov_gen_enums_root),
        "wind_dir": str(wind_root),
        "output_dir": str(output_root),
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "task_count": len(tasks),
        "success_count": success_count,
        "failure_count": failure_count,
        "skipped_count": len(skipped_items),
        "skipped_items": skipped_items,
        "meta_dir": str(meta_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="固定 wind surface，对 monthly generation 侧执行 repeated K-fold held-out validation。",
    )
    parser.add_argument("--prov-gen-enums-dir", type=Path, required=True)
    parser.add_argument("--wind-dir", type=Path, required=True)
    parser.add_argument("--optimal-kb-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--era5-dir", type=Path, required=True)
    parser.add_argument("--cdmet-dir", type=Path, default=None)
    parser.add_argument("--inventory-csv-path", type=Path, required=True)
    parser.add_argument("--farm-power-curve-dir", type=Path, default=None)
    parser.add_argument("--farm-adjusted-w-dir", type=Path, default=None)
    parser.add_argument("--farm-adjusted-w-varname", type=str, default="raw_w")
    parser.add_argument("--city-capacity-adjust-csv-path", type=Path, default=None)
    parser.add_argument("--prov-utility-rate-adjust-csv-path", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--poly-degree", type=int, default=9)
    parser.add_argument("--horizon-interp-method", type=str, default="loess")
    parser.add_argument("--max-workers", type=int, default=max((os.cpu_count() or 1) // 2, 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kb_validation(
        prov_gen_enums_dir=args.prov_gen_enums_dir,
        wind_dir=args.wind_dir,
        optimal_kb_csv=args.optimal_kb_csv,
        output_dir=args.output_dir,
        era5_dir=args.era5_dir,
        cdmet_dir=args.cdmet_dir,
        inventory_csv_path=args.inventory_csv_path,
        farm_power_curve_dir=args.farm_power_curve_dir,
        farm_adjusted_w_dir=args.farm_adjusted_w_dir,
        farm_adjusted_w_varname=args.farm_adjusted_w_varname,
        city_capacity_adjust_csv_path=args.city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=args.prov_utility_rate_adjust_csv_path,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_seed=args.random_seed,
        poly_degree=args.poly_degree,
        horizon_interp_method=args.horizon_interp_method,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()