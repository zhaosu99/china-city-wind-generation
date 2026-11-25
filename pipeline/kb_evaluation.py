from __future__ import annotations

import argparse
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from aggr_farm import DEFAULT_INVENTORY_CSV_PATH, aggr_farm_output
from indices import smape
from prov_gen_enums import _extract_model_series_from_nc, _infer_winter_plan_from_labels, _label_sort_key
from prov_gen_smape import get_intersection_times
from utils import resolve_existing_column, safe_read_csv


logger = logging.getLogger(__name__)


SUMMARY_OUTPUT_COLUMNS = [
    "prov",
    "season",
    "ratio",
    "index",
    "w_smape",
    "k",
    "b",
    "train_gen_smape",
    "test_gen_smape",
    "full_gen_smape",
    "train_n",
    "test_n",
    "full_n",
]


def _result_key(prov: str, season: str, ratio: int, index: int) -> tuple[str, str, int, int]:
    return (str(prov).strip(), str(season).strip().lower(), int(ratio), int(index))


def _normalize_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if "w_smape" not in df.columns:
        if "smape" not in df.columns:
            raise ValueError("summary_smape_kb.csv 缺少 w_smape/smape 列")
        df = df.rename(columns={"smape": "w_smape"})

    required_columns = ["prov", "season", "ratio", "index", "w_smape", "k", "b"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"summary_smape_kb.csv 缺少必要列: {missing_columns}，实际列为: {list(df.columns)}"
        )

    df["prov"] = df["prov"].astype(str).str.strip()
    df["season"] = df["season"].astype(str).str.strip().str.lower()
    for column in ("ratio", "index", "w_smape", "k", "b"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["ratio", "index", "w_smape", "k", "b"]].isna().any().any():
        bad_rows = df[df[["ratio", "index", "w_smape", "k", "b"]].isna().any(axis=1)]
        bad_records = bad_rows[["prov", "season", "ratio", "index", "w_smape", "k", "b"]].to_dict(
            orient="records"
        )
        raise ValueError(
            "summary_smape_kb.csv 存在无法解析的数值行: "
            f"{bad_records}"
        )

    df["ratio"] = df["ratio"].round().astype(int)
    df["index"] = df["index"].round().astype(int)
    return df


def _build_sampling_json_path(samplings_dir: Path, prov: str, season: str, ratio: int) -> Path:
    csv_stem = f"{prov}_{season}_monthly_gen_enums"
    return samplings_dir / str(ratio) / f"{csv_stem}_train{ratio}.json"


def _load_existing_output(output_csv: Path) -> pd.DataFrame:
    existing_df = pd.read_csv(output_csv, encoding="utf-8-sig")
    missing_columns = [column for column in SUMMARY_OUTPUT_COLUMNS if column not in existing_df.columns]
    if missing_columns:
        raise ValueError(
            f"已有 kb_evaluation.csv 缺少必要列: {missing_columns}，实际列为: {list(existing_df.columns)}"
        )

    existing_df = existing_df.copy()
    existing_df["prov"] = existing_df["prov"].astype(str).str.strip()
    existing_df["season"] = existing_df["season"].astype(str).str.strip().str.lower()
    existing_df["ratio"] = pd.to_numeric(existing_df["ratio"], errors="coerce").round().astype("Int64")
    existing_df["index"] = pd.to_numeric(existing_df["index"], errors="coerce").round().astype("Int64")
    existing_df = existing_df.dropna(subset=["ratio", "index"]).copy()
    existing_df["ratio"] = existing_df["ratio"].astype(int)
    existing_df["index"] = existing_df["index"].astype(int)
    return existing_df[SUMMARY_OUTPUT_COLUMNS].copy()


def _load_training_labels(
    samplings_dir: Path,
    prov: str,
    season: str,
    ratio: int,
    combo_index: int,
) -> list[str]:
    json_path = _build_sampling_json_path(samplings_dir, prov, season, ratio)
    if not json_path.exists():
        raise FileNotFoundError(f"未找到 sampling json: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"sampling json 不是 dict: {json_path}")

    combo_key = str(int(combo_index))
    if combo_key not in payload:
        raise KeyError(f"sampling json 中未找到 index={combo_index}: {json_path}")

    labels = [str(item).strip() for item in payload[combo_key] if str(item).strip()]
    if not labels:
        raise ValueError(f"sampling json 中 index={combo_index} 为空: {json_path}")
    return labels


def _load_official_series(prov_gen_enums_dir: Path, prov: str, season: str) -> tuple[dict[str, float], list[str]]:
    csv_path = prov_gen_enums_dir / f"{prov}_{season}_monthly_gen_enums.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 prov_gen_enums 输出文件: {csv_path}")

    df = safe_read_csv(csv_path)
    if df is None or df.empty:
        raise ValueError(f"prov_gen_enums 输出为空: {csv_path}")

    time_col = df.columns[0]
    official_col = resolve_existing_column(df.columns, ["official"], "official")
    valid_times = get_intersection_times(df)

    indexed_df = df.copy()
    indexed_df[time_col] = indexed_df[time_col].astype(str).str.strip()
    indexed_df = indexed_df[indexed_df[time_col] != ""]
    indexed_df = indexed_df.set_index(time_col)

    official_series = pd.to_numeric(indexed_df[official_col], errors="coerce")
    official_series = official_series[~official_series.index.duplicated(keep="last")]

    output = {
        label: float(official_series.loc[label])
        for label in valid_times
        if label in official_series.index and pd.notna(official_series.loc[label])
    }
    winter_plan = _infer_winter_plan_from_labels(output.keys()) if season == "winter" else {}
    return output, valid_times, winter_plan


def _job_key(prov: str, season: str, k: float, b: float) -> tuple[str, str, float, float]:
    return (prov, season, round(float(k), 10), round(float(b), 10))


def _build_job_output_dir(tmp_root: Path, prov: str, season: str, k: float, b: float) -> Path:
    job_token = hashlib.sha1(f"{prov}|{season}|{k:.10f}|{b:.10f}".encode("utf-8")).hexdigest()[:12]
    return tmp_root / f"{season}_{job_token}"


def _run_generation_evaluation_job(
    *,
    prov: str,
    season: str,
    k: float,
    b: float,
    era5_dir: str,
    cdmet_dir: Optional[str],
    inventory_csv_path: str | Path,
    farm_power_curve_dir: str | Path | None,
    farm_adjusted_w_dir: str | Path,
    farm_adjusted_w_varname: str,
    city_capacity_adjust_csv_path: str | Path | None,
    prov_utility_rate_adjust_csv_path: str | Path | None,
    gen_adjust_info: Any,
    horizon_interp_method: str,
    inner_max_workers: Optional[int],
    verbose: bool,
    keep_intermediate: bool,
    tmp_root: Path,
    winter_plan: Optional[dict[int, str]] = None,
) -> dict[str, float]:
    output_dir = _build_job_output_dir(tmp_root, prov, season, k, b)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 本步骤按逐场 CF csv 评估，必须给出曲线目录。原先是靠 use_cf_csv=True + aggr_farm_aux 的
    # 守卫间接保证，现改为在此**显式断言**（行为等价：dir=None 仍然 fail-loud，只是错误位置提前）。
    if farm_power_curve_dir is None:
        raise ValueError(
            "kb_evaluation 需要逐场功率曲线目录 farm_power_curve_dir（用于读取 farm_<id>.csv）；"
            "该参数不能为 None。"
        )

    try:
        result = aggr_farm_output(
            era5_dir=era5_dir,
            cdmet_dir=cdmet_dir,
            inventory_csv_path=inventory_csv_path,
            farm_power_curve_dir=farm_power_curve_dir,
            farm_adjusted_w_dir=farm_adjusted_w_dir,
            farm_adjusted_w_varname=farm_adjusted_w_varname,
            output_dir=output_dir,
            output_hourly_gen=False,
            output_monthly_gen=True,
            output_hourly_cf=False,
            output_w_compare=False,
            output_raw_w=False,
            select_period=season,
            period_mode="season",
            prov_short=[prov],
            prov_long=None,
            city_short=None,
            city_long=None,
            kb_info=(float(k), float(b)),
            gen_adjust_info=gen_adjust_info,
            city_capacity_adjust_csv_path=city_capacity_adjust_csv_path,
            prov_utility_rate_adjust_csv_path=prov_utility_rate_adjust_csv_path,
            overwrite=True,
            aggr_level="prov",
            include_kb_in_gen_filenames=False,
            horizon_interp_method=horizon_interp_method,
            max_workers=inner_max_workers,
            verbose=verbose,
            show_progress=False,
        )

        period_map = result.get("province_generation_files", {}).get(season, {})
        nc_path_str = period_map.get(prov)
        if nc_path_str is None:
            raise KeyError(
                f"aggr_farm_output 未返回省级 monthly generation 文件: prov={prov}, season={season}, result_keys={list(period_map.keys())}"
            )

        return _extract_model_series_from_nc(Path(nc_path_str), season, winter_plan=winter_plan)
    finally:
        if not keep_intermediate and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)


def _run_generation_job_task(task: dict[str, Any]) -> dict[str, Any]:
    prov = str(task["prov"])
    season = str(task["season"])
    k = float(task["k"])
    b = float(task["b"])
    prediction_map = _run_generation_evaluation_job(
        prov=prov,
        season=season,
        k=k,
        b=b,
        era5_dir=task["era5_dir"],
        cdmet_dir=task["cdmet_dir"],
        inventory_csv_path=task["inventory_csv_path"],
        farm_power_curve_dir=task["farm_power_curve_dir"],
        farm_adjusted_w_dir=task["farm_adjusted_w_dir"],
        farm_adjusted_w_varname=task["farm_adjusted_w_varname"],
        city_capacity_adjust_csv_path=task["city_capacity_adjust_csv_path"],
        prov_utility_rate_adjust_csv_path=task["prov_utility_rate_adjust_csv_path"],
        gen_adjust_info=task["gen_adjust_info"],
        horizon_interp_method=task["horizon_interp_method"],
        inner_max_workers=task["inner_max_workers"],
        verbose=task["verbose"],
        keep_intermediate=task["keep_intermediate"],
        tmp_root=task["tmp_root"],
        winter_plan=task.get("winter_plan"),
    )
    return {
        "job_key": _job_key(prov, season, k, b),
        "prediction_map": prediction_map,
    }


def _calculate_subset_smape(
    prediction_map: dict[str, float],
    official_map: dict[str, float],
    labels: list[str],
) -> tuple[float, int]:
    ordered_labels = sorted(set(labels), key=_label_sort_key)
    y_true: list[float] = []
    y_pred: list[float] = []

    for label in ordered_labels:
        if label not in prediction_map or label not in official_map:
            continue
        pred_value = pd.to_numeric(prediction_map[label], errors="coerce")
        true_value = pd.to_numeric(official_map[label], errors="coerce")
        if pd.isna(pred_value) or pd.isna(true_value):
            continue
        y_pred.append(float(pred_value))
        y_true.append(float(true_value))

    if not y_true:
        return float("nan"), 0

    return float(smape(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))), len(y_true)


def _build_output_df(
    summary_result_keys: list[tuple[str, str, int, int]],
    existing_result_map: dict[tuple[str, str, int, int], dict[str, Any]],
    new_result_map: dict[tuple[str, str, int, int], dict[str, Any]],
) -> pd.DataFrame:
    final_rows: list[dict[str, Any]] = []
    for result_key in summary_result_keys:
        if result_key in existing_result_map:
            final_rows.append(existing_result_map[result_key])
            continue
        if result_key in new_result_map:
            final_rows.append(new_result_map[result_key])
    return pd.DataFrame(final_rows, columns=SUMMARY_OUTPUT_COLUMNS)


def _write_output_snapshot(
    summary_result_keys: list[tuple[str, str, int, int]],
    existing_result_map: dict[tuple[str, str, int, int], dict[str, Any]],
    new_result_map: dict[tuple[str, str, int, int], dict[str, Any]],
    output_csv: Path,
) -> pd.DataFrame:
    output_df = _build_output_df(summary_result_keys, existing_result_map, new_result_map)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_df


def kb_evaluation(
    summary_csv: str | Path,
    output_csv: str | Path | None,
    samplings_dir: str | Path,
    prov_gen_enums_dir: str | Path,
    era5_dir: str,
    cdmet_dir: Optional[str] = None,
    inventory_csv_path: str | Path = DEFAULT_INVENTORY_CSV_PATH,
    farm_power_curve_dir: str | Path | None = None,
    farm_adjusted_w_dir: str | Path | None = None,
    farm_adjusted_w_varname: str = "raw_w",
    city_capacity_adjust_csv_path: str | Path | None = None,
    prov_utility_rate_adjust_csv_path: str | Path | None = None,
    gen_adjust_info: Any = None,
    horizon_interp_method: str = "loess",
    outer_max_workers: Optional[int] = None,
    inner_max_workers: Optional[int] = 4,
    verbose: bool = False,
    keep_intermediate: bool = True,
    intermediate_dir: str | Path | None = None,
    flush_every_rows: Optional[int] = 100,
    overwrite: bool = False,
) -> dict[str, Any]:
    """基于最优 k/b 汇总结果重算省级发电，并输出 train/test/full SMAPE 评估表。

    该函数会读取 `find_optimal_kb.py` 产出的 `summary_smape_kb.csv`，针对其中每一条
    `(prov, season, ratio, index, k, b)` 组合，重新调用省级聚合发电流程生成预测月度发电序列，
    再与 `prov_gen_enums` 中的官方发电序列进行对比，分别计算训练集、测试集和全样本上的
    generation SMAPE，最终写出 `kb_evaluation.csv`。

    处理逻辑上会：
    1. 规范化 summary 表字段，并跳过 `ratio == 100` 的行；
    2. 以 `(prov, season, k, b)` 去重，只对唯一参数组合执行一次发电重算；
    3. 如果输出文件已存在且 `overwrite=False`，则跳过已完成的结果行；
    4. 按 summary 原始顺序回填结果并写出最终评估表。

    Args:
        summary_csv: `summary_smape_kb.csv` 路径，至少需包含 `prov`、`season`、`ratio`、
            `index`、`w_smape`/`smape`、`k`、`b` 等列。
        output_csv: 输出评估 CSV 路径。若为 `None`，默认写到 `summary_csv` 同目录下的
            `kb_evaluation.csv`。
        samplings_dir: `prov_gen_smape.py` 输出的 `samplings` 目录，用于读取每个 ratio/index
            对应的训练样本时间标签。
        prov_gen_enums_dir: `prov_gen_enums.py` 输出目录，用于读取各省各季节的官方月度发电序列。
        era5_dir: ERA5 数据目录，传给底层 `aggr_farm_output`。
        cdmet_dir: CDMet 数据目录，可为 `None`。
        inventory_csv_path: 风电场清单 CSV 路径。
        farm_power_curve_dir: 风场功率曲线 CSV 目录，可为 `None`。
        farm_adjusted_w_dir: 风场订正风速目录，必须显式提供。
        farm_adjusted_w_varname: `farm_adjusted_w_dir` 中使用的风速变量名。
        city_capacity_adjust_csv_path: 城市容量修正表路径，可为 `None`。
        prov_utility_rate_adjust_csv_path: 省级利用率修正表路径，可为 `None`。
        gen_adjust_info: 额外的发电修正信息，原样传给 `aggr_farm_output`。
        horizon_interp_method: 水平插值方法，支持 `closest`、`idw`、`loess`。
        outer_max_workers: 外层并发数，即唯一 `(prov, season, k, b)` 任务的最大进程数。
            为 `None` 时默认使用全部唯一任务数。
        inner_max_workers: 传给 `aggr_farm_output` 的内部 farm 级并发数。
        verbose: 是否输出底层聚合流程的详细日志。
        keep_intermediate: 是否保留中间生成的省级发电文件。
        intermediate_dir: 中间目录根路径；为 `None` 时默认写到输出目录下的
            `_tmp_kb_evaluation`。
        flush_every_rows: 每累计计算多少条结果行就增量写一次 `kb_evaluation.csv`。
            设为 `0` 或 `None` 时仅在最后统一写出。
        overwrite: 是否忽略已有 `kb_evaluation.csv` 并强制重算全部结果。

    Returns:
        一个字典，包含输入输出路径、实际评估行数、唯一任务数、内外层并发配置，以及跳过的
        `ratio == 100` 和已存在结果的统计信息。
    """
    summary_csv = Path(summary_csv).resolve()
    samplings_dir = Path(samplings_dir).resolve()
    prov_gen_enums_dir = Path(prov_gen_enums_dir).resolve()
    if output_csv is None:
        output_csv = summary_csv.with_name("kb_evaluation.csv")
    output_csv = Path(output_csv).resolve()

    if farm_adjusted_w_dir is None:
        raise ValueError("kb_evaluation 需要显式提供 farm_adjusted_w_dir")
    farm_adjusted_w_dir = Path(farm_adjusted_w_dir).resolve()

    if flush_every_rows is not None:
        flush_every_rows = int(flush_every_rows)
        if flush_every_rows < 0:
            raise ValueError(f"flush_every_rows 不能小于 0，收到: {flush_every_rows}")
        if flush_every_rows == 0:
            flush_every_rows = None

    if not summary_csv.exists():
        raise FileNotFoundError(f"未找到 summary_csv: {summary_csv}")
    if not samplings_dir.exists():
        raise FileNotFoundError(f"未找到 samplings_dir: {samplings_dir}")
    if not prov_gen_enums_dir.exists():
        raise FileNotFoundError(f"未找到 prov_gen_enums_dir: {prov_gen_enums_dir}")
    if not farm_adjusted_w_dir.exists():
        raise FileNotFoundError(f"未找到 farm_adjusted_w_dir: {farm_adjusted_w_dir}")

    tmp_root = Path(intermediate_dir).resolve() if intermediate_dir is not None else output_csv.parent / "_tmp_kb_evaluation"
    tmp_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    summary_df = _normalize_summary_df(summary_df)
    skipped_ratio_100_count = int((summary_df["ratio"] == 100).sum())
    summary_df = summary_df[summary_df["ratio"] != 100].copy().reset_index(drop=True)

    existing_output_df: Optional[pd.DataFrame] = None
    existing_result_map: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    if output_csv.exists() and not overwrite:
        existing_output_df = _load_existing_output(output_csv)
        existing_result_map = {
            _result_key(row.prov, row.season, row.ratio, row.index): row._asdict()
            for row in existing_output_df.itertuples(index=False)
        }

    summary_df["result_key"] = [
        _result_key(row.prov, row.season, row.ratio, row.index)
        for row in summary_df.itertuples(index=False)
    ]
    summary_result_keys = summary_df["result_key"].tolist()
    pending_df = summary_df[~summary_df["result_key"].isin(existing_result_map)].copy().reset_index(drop=True)

    if summary_df.empty:
        empty_df = pd.DataFrame(columns=SUMMARY_OUTPUT_COLUMNS)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return {
            "summary_csv": str(summary_csv),
            "output_csv": str(output_csv),
            "evaluated_row_count": 0,
            "unique_job_count": 0,
            "outer_max_workers": 0,
            "inner_max_workers": inner_max_workers,
            "skipped_ratio_100_count": skipped_ratio_100_count,
            "skipped_existing_count": 0,
        }

    if pending_df.empty and existing_result_map:
        final_df = _write_output_snapshot(summary_result_keys, existing_result_map, {}, output_csv)
        return {
            "summary_csv": str(summary_csv),
            "output_csv": str(output_csv),
            "evaluated_row_count": len(final_df),
            "unique_job_count": 0,
            "outer_max_workers": 0,
            "inner_max_workers": inner_max_workers,
            "skipped_ratio_100_count": skipped_ratio_100_count,
            "skipped_existing_count": len(final_df),
        }

    official_cache: dict[tuple[str, str], tuple[dict[str, float], list[str], dict[int, str]]] = {}
    combo_cache: dict[tuple[str, str, int, int], list[str]] = {}
    unique_jobs: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    pending_rows_by_job: dict[tuple[str, str, float, float], list[Any]] = {}
    rows_per_job: dict[tuple[str, str, float, float], int] = {}
    for row in pending_df.itertuples(index=False):
        key = _job_key(row.prov, row.season, row.k, row.b)
        if key not in unique_jobs:
            unique_jobs[key] = {
                "prov": row.prov,
                "season": row.season,
                "k": float(row.k),
                "b": float(row.b),
            }
        pending_rows_by_job.setdefault(key, []).append(row)
        rows_per_job[key] = rows_per_job.get(key, 0) + 1

    logger.info(
        "kb_evaluation starting | rows=%s | pending_rows=%s | skipped_existing=%s | unique_jobs=%s",
        len(summary_df),
        len(pending_df),
        len(summary_df) - len(pending_df),
        len(unique_jobs),
    )

    effective_outer_max_workers = (
        len(unique_jobs)
        if outer_max_workers is None
        else max(1, min(int(outer_max_workers), len(unique_jobs)))
    )

    for job in unique_jobs.values():
        official_key = (str(job["prov"]).strip(), str(job["season"]).strip().lower())
        if official_key not in official_cache:
            official_cache[official_key] = _load_official_series(prov_gen_enums_dir, official_key[0], official_key[1])

    job_tasks = [
        {
            "prov": job["prov"],
            "season": job["season"],
            "k": job["k"],
            "b": job["b"],
            "era5_dir": era5_dir,
            "cdmet_dir": cdmet_dir,
            "inventory_csv_path": inventory_csv_path,
            "farm_power_curve_dir": farm_power_curve_dir,
            "farm_adjusted_w_dir": farm_adjusted_w_dir,
            "farm_adjusted_w_varname": farm_adjusted_w_varname,
            "city_capacity_adjust_csv_path": city_capacity_adjust_csv_path,
            "prov_utility_rate_adjust_csv_path": prov_utility_rate_adjust_csv_path,
            "gen_adjust_info": gen_adjust_info,
            "horizon_interp_method": horizon_interp_method,
            "inner_max_workers": inner_max_workers,
            "verbose": verbose,
            "keep_intermediate": keep_intermediate,
            "tmp_root": tmp_root,
            "winter_plan": official_cache[(str(job["prov"]).strip(), str(job["season"]).strip().lower())][2]
            if str(job["season"]).strip().lower() == "winter"
            else None,
        }
        for job in unique_jobs.values()
    ]

    new_result_map: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    rows_since_flush = 0

    def _process_task_result(task_result: dict[str, Any]) -> None:
        nonlocal rows_since_flush

        job_key = task_result["job_key"]
        prediction_map = task_result["prediction_map"]

        for row in pending_rows_by_job[job_key]:
            prov = str(row.prov)
            season = str(row.season)
            ratio = int(row.ratio)
            combo_index = int(row.index)

            official_key = (prov, season)
            if official_key not in official_cache:
                official_cache[official_key] = _load_official_series(prov_gen_enums_dir, prov, season)
            official_map, valid_times, _ = official_cache[official_key]

            combo_key = (prov, season, ratio, combo_index)
            if combo_key not in combo_cache:
                combo_cache[combo_key] = _load_training_labels(samplings_dir, prov, season, ratio, combo_index)
            train_labels = combo_cache[combo_key]
            train_label_set = set(train_labels)
            test_labels = [label for label in valid_times if label not in train_label_set]

            train_gen_smape, train_n = _calculate_subset_smape(prediction_map, official_map, train_labels)
            test_gen_smape, test_n = _calculate_subset_smape(prediction_map, official_map, test_labels)
            full_gen_smape, full_n = _calculate_subset_smape(prediction_map, official_map, valid_times)

            new_result_map[_result_key(prov, season, ratio, combo_index)] = {
                "prov": prov,
                "season": season,
                "ratio": ratio,
                "index": combo_index,
                "w_smape": float(row.w_smape),
                "k": float(row.k),
                "b": float(row.b),
                "train_gen_smape": train_gen_smape,
                "test_gen_smape": test_gen_smape,
                "full_gen_smape": full_gen_smape,
                "train_n": int(train_n),
                "test_n": int(test_n),
                "full_n": int(full_n),
            }

        rows_since_flush += len(pending_rows_by_job[job_key])
        if flush_every_rows is not None and rows_since_flush >= flush_every_rows:
            snapshot_df = _write_output_snapshot(summary_result_keys, existing_result_map, new_result_map, output_csv)
            logger.info(
                "kb_evaluation flushed partial results | output_rows=%s | new_rows_since_last_flush=%s",
                len(snapshot_df),
                rows_since_flush,
            )
            rows_since_flush = 0

    with tqdm(total=len(pending_df), desc="Evaluating kb rows", unit="row", dynamic_ncols=True) as pbar:
        if len(job_tasks) == 1 or effective_outer_max_workers == 1:
            for task in job_tasks:
                task_result = _run_generation_job_task(task)
                _process_task_result(task_result)
                pbar.update(rows_per_job.get(task_result["job_key"], 0))
        else:
            with ProcessPoolExecutor(max_workers=effective_outer_max_workers) as executor:
                futures = [executor.submit(_run_generation_job_task, task) for task in job_tasks]
                for future in as_completed(futures):
                    task_result = future.result()
                    _process_task_result(task_result)
                    pbar.update(rows_per_job.get(task_result["job_key"], 0))

    output_df = _write_output_snapshot(summary_result_keys, existing_result_map, new_result_map, output_csv)

    if not keep_intermediate and tmp_root.exists() and not any(tmp_root.iterdir()):
        tmp_root.rmdir()

    return {
        "summary_csv": str(summary_csv),
        "output_csv": str(output_csv),
        "evaluated_row_count": len(output_df),
        "unique_job_count": len(unique_jobs),
        "outer_max_workers": effective_outer_max_workers,
        "inner_max_workers": inner_max_workers,
        "skipped_ratio_100_count": skipped_ratio_100_count,
        "skipped_existing_count": len(summary_df) - len(pending_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 summary_smape_kb.csv，重算省级月度发电并输出 train/test/full generation SMAPE 评估表。"
    )
    parser.add_argument("--summary-csv", type=Path, required=True, help="find_optimal_kb.py 输出的 summary_smape_kb.csv")
    parser.add_argument("--output-csv", type=Path, default=None, help="评估结果 CSV 路径，默认与 summary 同目录下的 kb_evaluation.csv")
    parser.add_argument("--samplings-dir", type=Path, required=True, help="prov_gen_smape.py 输出目录下的 samplings 子目录")
    parser.add_argument("--prov-gen-enums-dir", type=Path, required=True, help="prov_gen_enums.py 输出目录")
    parser.add_argument("--era5-dir", type=str, required=True, help="ERA5 数据目录")
    parser.add_argument("--cdmet-dir", type=str, default=None, help="CDMet 数据目录")
    parser.add_argument("--inventory-csv-path", type=Path, default=Path(DEFAULT_INVENTORY_CSV_PATH), help="风电场清单 CSV")
    parser.add_argument("--farm-power-curve-dir", type=Path, default=None, help="功率曲线 CSV 目录")
    parser.add_argument("--farm-adjusted-w-dir", type=Path, required=True, help="step2 生成的 full_year 原始风速目录")
    parser.add_argument("--farm-adjusted-w-varname", type=str, default="raw_w", help="复用原始风速变量名")
    parser.add_argument("--city-capacity-adjust-csv-path", type=Path, default=None, help="城市容量修正表")
    parser.add_argument("--prov-utility-rate-adjust-csv-path", type=Path, default=None, help="省利用率修正表")
    parser.add_argument("--horizon-interp-method", choices=["closest", "idw", "loess"], default="loess")
    parser.add_argument("--outer-max-workers", type=int, default=None, help="外层唯一 (prov, season, k, b) 任务并发数")
    parser.add_argument(
        "--inner-max-workers",
        "--aggr-max-workers",
        dest="inner_max_workers",
        type=int,
        default=4,
        help="传给 aggr_farm_output 的 farm 并发数",
    )
    parser.add_argument("--verbose", action="store_true", help="输出底层 aggr_farm 详细日志")
    parser.add_argument("--keep-intermediate", action="store_true", help="保留中间 province generation 文件")
    parser.add_argument("--intermediate-dir", type=Path, default=None, help="中间目录根路径")
    parser.add_argument(
        "--flush-every-rows",
        type=int,
        default=100,
        help="每累计计算多少行就增量写一次 kb_evaluation.csv；设为 0 表示仅在最后统一写出",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 kb_evaluation.csv 结果，不跳过已存在行")
    return parser.parse_args()


def cli_main() -> dict[str, Any]:
    args = parse_args()
    return kb_evaluation(
        summary_csv=args.summary_csv,
        output_csv=args.output_csv,
        samplings_dir=args.samplings_dir,
        prov_gen_enums_dir=args.prov_gen_enums_dir,
        era5_dir=args.era5_dir,
        cdmet_dir=args.cdmet_dir,
        inventory_csv_path=args.inventory_csv_path,
        farm_power_curve_dir=args.farm_power_curve_dir,
        farm_adjusted_w_dir=args.farm_adjusted_w_dir,
        farm_adjusted_w_varname=args.farm_adjusted_w_varname,
        city_capacity_adjust_csv_path=args.city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=args.prov_utility_rate_adjust_csv_path,
        gen_adjust_info=None,
        horizon_interp_method=args.horizon_interp_method,
        outer_max_workers=args.outer_max_workers,
        inner_max_workers=args.inner_max_workers,
        verbose=args.verbose,
        keep_intermediate=args.keep_intermediate,
        intermediate_dir=args.intermediate_dir,
        flush_every_rows=args.flush_every_rows,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cli_main()