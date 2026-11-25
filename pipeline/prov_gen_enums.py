from __future__ import annotations

import argparse
import os
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from utils import chinese_to_pinyi, normalize_text, open_dataset_with_fallback, resolve_existing_column, safe_read_csv

logger = logging.getLogger(__name__)


SEASON_MONTHS: dict[str, set[int]] = {
	"spring": {3, 4, 5},
	"summer": {6, 7, 8},
	"autumn": {9, 10, 11},
	"winter": {12, 1, 2},
}


def _parse_year_month_header(text: Any) -> Optional[tuple[int, int]]:
	normalized = normalize_text(text)
	if normalized is None:
		return None

	match = re.fullmatch(r"(\d{4})年(\d{1,2})月", normalized)
	if match is None:
		return None

	year = int(match.group(1))
	month = int(match.group(2))
	if month < 1 or month > 12:
		return None
	return (year, month)


def _month_label(year: int, month: int) -> str:
	return f"{year}年{month}月"


def _jan_feb_label(year: int) -> str:
	return f"{year}年1-2月"


def _parse_official_value(raw_value: Any, unit_to_mwh_factor: float) -> Optional[float]:
	value = pd.to_numeric(raw_value, errors="coerce")
	if pd.isna(value):
		return None

	value = float(value)
	if value <= 0:
		return None
	return value * unit_to_mwh_factor


def _label_sort_key(label: str) -> tuple[int, int]:
	jan_feb_match = re.fullmatch(r"(\d{4})年1-2月", label)
	if jan_feb_match is not None:
		return (int(jan_feb_match.group(1)), 2)

	month_match = re.fullmatch(r"(\d{4})年(\d{1,2})月", label)
	if month_match is not None:
		return (int(month_match.group(1)), int(month_match.group(2)))

	return (9999, 99)


def _infer_winter_plan_from_labels(labels: Iterable[str]) -> dict[int, str]:
	plan: dict[int, str] = {}
	for label in labels:
		jan_feb_match = re.fullmatch(r"(\d{4})年1-2月", str(label).strip())
		if jan_feb_match is not None:
			year = int(jan_feb_match.group(1))
			plan.setdefault(year, "jan_feb")
			continue

		ym = _parse_year_month_header(label)
		if ym is None:
			continue
		year, month = ym
		if month in (1, 2):
			plan[year] = "monthly"

	return plan


def _build_prov_name_map(admin_csv_path: Path) -> tuple[dict[str, str], set[str]]:
	admin_df = safe_read_csv(admin_csv_path)
	if admin_df is None or admin_df.empty:
		raise ValueError(f"省市名映射 CSV 为空: {admin_csv_path}")

	full_col = resolve_existing_column(admin_df.columns, ["省全称", "province_full", "prov_full"], "省全称")
	short_col = resolve_existing_column(admin_df.columns, ["省简称", "province_short", "prov_short"], "省简称")

	full_to_short: dict[str, str] = {}
	short_set: set[str] = set()
	for _, row in admin_df.iterrows():
		full_name = normalize_text(row.get(full_col))
		short_name = normalize_text(row.get(short_col))
		if full_name is None or short_name is None:
			continue
		full_to_short[full_name] = short_name
		short_set.add(short_name)

	if not full_to_short:
		raise ValueError(f"省市名映射 CSV 未解析到有效省份映射: {admin_csv_path}")

	return full_to_short, short_set


def _resolve_prov_short_name(raw_name: Any, full_to_short: dict[str, str], short_set: set[str]) -> Optional[str]:
	normalized = normalize_text(raw_name)
	if normalized is None:
		return None

	if normalized in full_to_short:
		return full_to_short[normalized]
	if normalized in short_set:
		return normalized

	suffix_patterns = ["维吾尔自治区", "壮族自治区", "回族自治区", "特别行政区", "自治区", "省", "市"]
	for suffix in suffix_patterns:
		if normalized.endswith(suffix):
			trimmed = normalized[: -len(suffix)]
			if trimmed in short_set:
				return trimmed

	return None


def _build_official_details(
	current_df: pd.DataFrame,
	cumulative_df: pd.DataFrame,
	current_region_col: str,
	cumulative_region_col: str,
	full_to_short: dict[str, str],
	short_set: set[str],
	unit_to_mwh_factor: float,
) -> dict[str, dict[str, Any]]:
	current_columns: list[tuple[str, int, int]] = []
	for col in current_df.columns:
		parsed = _parse_year_month_header(col)
		if parsed is None:
			continue
		year, month = parsed
		current_columns.append((str(col), year, month))

	cumulative_columns: list[tuple[str, int]] = []
	for col in cumulative_df.columns:
		parsed = _parse_year_month_header(col)
		if parsed is None:
			continue
		year, month = parsed
		if month == 2:
			cumulative_columns.append((str(col), year))

	current_values_by_prov: dict[str, dict[tuple[int, int], float]] = {}
	for _, row in current_df.iterrows():
		short_name = _resolve_prov_short_name(row.get(current_region_col), full_to_short, short_set)
		if short_name is None:
			continue

		province_values = current_values_by_prov.setdefault(short_name, {})
		for col_name, year, month in current_columns:
			value = _parse_official_value(row.get(col_name), unit_to_mwh_factor)
			if value is None:
				continue
			province_values[(year, month)] = value

	cumulative_feb_by_prov: dict[str, dict[int, float]] = {}
	for _, row in cumulative_df.iterrows():
		short_name = _resolve_prov_short_name(row.get(cumulative_region_col), full_to_short, short_set)
		if short_name is None:
			continue

		province_values = cumulative_feb_by_prov.setdefault(short_name, {})
		for col_name, year in cumulative_columns:
			value = _parse_official_value(row.get(col_name), unit_to_mwh_factor)
			if value is None:
				continue
			province_values[year] = value

	all_provinces = set(current_values_by_prov.keys()) | set(cumulative_feb_by_prov.keys())
	official_details: dict[str, dict[str, Any]] = {}
	for short_name in all_provinces:
		current_values = current_values_by_prov.get(short_name, {})
		cumulative_feb_values = cumulative_feb_by_prov.get(short_name, {})

		series: dict[str, float] = {}
		winter_plan: dict[int, str] = {}
		winter_source_notes: list[str] = []

		for (year, month), value in sorted(current_values.items()):
			if month not in (1, 2):
				series[_month_label(year, month)] = value

		winter_years = sorted(
			{year for year, month in current_values.keys() if month in (1, 2)} | set(cumulative_feb_values.keys())
		)
		for year in winter_years:
			jan_current = current_values.get((year, 1))
			feb_current = current_values.get((year, 2))
			feb_cumulative = cumulative_feb_values.get(year)

			if jan_current is not None and feb_current is not None:
				series[_month_label(year, 1)] = jan_current
				series[_month_label(year, 2)] = feb_current
				winter_plan[year] = "monthly"
				winter_source_notes.append(f"{year}:monthly_direct")
				continue

			if jan_current is None and feb_current is not None and feb_cumulative is not None:
				jan_recovered = feb_cumulative - feb_current
				if jan_recovered > 0:
					series[_month_label(year, 1)] = jan_recovered
					series[_month_label(year, 2)] = feb_current
					winter_plan[year] = "monthly"
					winter_source_notes.append(f"{year}:jan_recovered_from_cumulative_minus_feb")
					continue

			if feb_cumulative is not None:
				series[_jan_feb_label(year)] = feb_cumulative
				winter_plan[year] = "jan_feb"
				if jan_current is None and feb_current is None:
					source = "jan_feb_cumulative_only"
				elif jan_current is None and feb_current is not None:
					source = "jan_feb_cumulative_fallback_missing_jan_current"
				elif jan_current is not None and feb_current is None:
					source = "jan_feb_cumulative_fallback_missing_feb_current"
				else:
					source = "jan_feb_cumulative_fallback_nonpositive_jan_recovery"
				winter_source_notes.append(f"{year}:{source}")

		official_details[short_name] = {
			"series": series,
			"winter_plan": winter_plan,
			"winter_source_notes": winter_source_notes,
		}

	return official_details


def _extract_official_details_by_province(
	monthly_current_csv_path: Path,
	monthly_cumulative_csv_path: Path,
	admin_province_csv_path: Path,
	unit_to_mwh_factor: float,
) -> dict[str, dict[str, Any]]:
	full_to_short, short_set = _build_prov_name_map(admin_province_csv_path)

	current_df = safe_read_csv(monthly_current_csv_path)
	if current_df is None or current_df.empty:
		raise ValueError(f"当期值 CSV 为空: {monthly_current_csv_path}")

	cumulative_df = safe_read_csv(monthly_cumulative_csv_path)
	if cumulative_df is None or cumulative_df.empty:
		raise ValueError(f"累计值 CSV 为空: {monthly_cumulative_csv_path}")

	current_region_col = resolve_existing_column(current_df.columns, ["地区", "region", "省", "province"], "地区")
	cumulative_region_col = resolve_existing_column(cumulative_df.columns, ["地区", "region", "省", "province"], "地区")

	return _build_official_details(
		current_df=current_df,
		cumulative_df=cumulative_df,
		current_region_col=current_region_col,
		cumulative_region_col=cumulative_region_col,
		full_to_short=full_to_short,
		short_set=short_set,
		unit_to_mwh_factor=unit_to_mwh_factor,
	)


def _extract_official_by_province(
	monthly_current_csv_path: Path,
	monthly_cumulative_csv_path: Path,
	admin_province_csv_path: Path,
	unit_to_mwh_factor: float,
) -> dict[str, dict[str, float]]:
	official_details = _extract_official_details_by_province(
		monthly_current_csv_path=monthly_current_csv_path,
		monthly_cumulative_csv_path=monthly_cumulative_csv_path,
		admin_province_csv_path=admin_province_csv_path,
		unit_to_mwh_factor=unit_to_mwh_factor,
	)
	return {
		prov_short: dict(detail.get("series", {}))
		for prov_short, detail in official_details.items()
	}


def _extract_model_series_from_nc(
	nc_path: Path,
	season: str,
	winter_plan: Optional[dict[int, str]] = None,
) -> dict[str, float]:
	ds = open_dataset_with_fallback(nc_path)

	if "total_monthly_gen" not in ds.data_vars:
		raise KeyError(f"{nc_path} 中缺少 total_monthly_gen")
	if "date" not in ds.coords:
		raise KeyError(f"{nc_path} 中缺少 date 坐标")

	series = ds["total_monthly_gen"].to_series()
	if series.empty:
		return {}

	season_lower = season.lower()
	if season_lower not in SEASON_MONTHS:
		raise ValueError(f"不支持的 season: {season}")

	output: dict[str, float] = {}

	if season_lower != "winter":
		valid_months = SEASON_MONTHS[season_lower]
		for dt, value in series.items():
			if pd.isna(value):
				continue
			ts = pd.Timestamp(dt)
			if ts.month not in valid_months:
				continue
			output[_month_label(ts.year, ts.month)] = float(value)
		return output

	winter_monthly_values: dict[tuple[int, int], float] = {}
	for dt, value in series.items():
		if pd.isna(value):
			continue
		ts = pd.Timestamp(dt)

		if ts.month == 12:
			output[_month_label(ts.year, 12)] = float(value)
		elif ts.month in (1, 2):
			winter_monthly_values[(ts.year, ts.month)] = float(value)

	winter_plan = winter_plan or {}
	for year in sorted({year for year, _ in winter_monthly_values.keys()}):
		mode = winter_plan.get(year, "jan_feb")
		if mode not in {"monthly", "jan_feb"}:
			raise ValueError(f"不支持的 winter_plan: year={year}, mode={mode}")

		jan_value = winter_monthly_values.get((year, 1))
		feb_value = winter_monthly_values.get((year, 2))
		if mode == "monthly":
			if jan_value is not None:
				output[_month_label(year, 1)] = jan_value
			if feb_value is not None:
				output[_month_label(year, 2)] = feb_value
			continue

		if jan_value is None and feb_value is None:
			continue
		output[_jan_feb_label(year)] = float((jan_value or 0.0) + (feb_value or 0.0))

	return output


def _parse_kb_column_from_filename(filename: str) -> Optional[tuple[str, float, float]]:
	match = re.fullmatch(r"k(-?\d+(?:\.\d+)?)b(-?\d+(?:\.\d+)?)_gen\.nc", filename)
	if match is None:
		return None

	k_token = match.group(1)
	b_token = match.group(2)
	return (f"k{k_token}b{b_token}", float(k_token), float(b_token))


def _official_for_season(official_series_map: dict[str, float], season: str) -> dict[str, float]:
	if season == "winter":
		winter_output: dict[str, float] = {}
		for label, value in official_series_map.items():
			if re.fullmatch(r"(\d{4})年1-2月", label) is not None:
				winter_output[label] = value
				continue

			ym = _parse_year_month_header(label)
			if ym is None:
				continue
			_, month = ym
			if month in (1, 2, 12):
				winter_output[label] = value
		return winter_output

	valid_months = SEASON_MONTHS[season]
	valid_labels: set[str] = set()
	for label in official_series_map.keys():
		ym = _parse_year_month_header(label)
		if ym is None:
			continue
		_, month = ym
		if month in valid_months:
			valid_labels.add(label)

	return {k: official_series_map[k] for k in valid_labels if k in official_series_map}


def _process_single_prov_season_task(task: dict[str, Any]) -> dict[str, Any]:
	"""Worker: process one (prov, season) pair with isolated in-process I/O only."""
	prov_short = str(task["prov_short"])
	prov_pinyin = str(task["prov_pinyin"])
	season = str(task["season"])
	season_dir = Path(task["season_dir"])
	output_root = Path(task["output_dir"])
	overwrite = bool(task["overwrite"])
	output_csv_path = output_root / f"{prov_short}_{season}_monthly_gen_enums.csv"

	if output_csv_path.exists() and not overwrite:
		return {
			"status": "skipped_existing",
			"prov_short": prov_short,
			"prov_pinyin": prov_pinyin,
			"season": season,
			"output_csv_path": str(output_csv_path),
			"reason": "output_exists",
		}

	official_details_map = _extract_official_details_by_province(
		monthly_current_csv_path=Path(task["monthly_current_csv_path"]),
		monthly_cumulative_csv_path=Path(task["monthly_cumulative_csv_path"]),
		admin_province_csv_path=Path(task["admin_province_csv_path"]),
		unit_to_mwh_factor=float(task["unit_to_mwh_factor"]),
	)

	official_detail = official_details_map.get(prov_short, {})
	official_series_map = dict(official_detail.get("series", {}))
	if not official_series_map:
		return {
			"status": "skipped",
			"prov_short": prov_short,
			"prov_pinyin": prov_pinyin,
			"season": season,
			"reason": "official_missing",
		}

	kb_items: list[tuple[str, float, float, Path]] = []
	for nc_path in sorted(season_dir.glob("*.nc"), key=lambda p: p.name):
		parsed = _parse_kb_column_from_filename(nc_path.name)
		if parsed is None:
			continue
		col_name, k_value, b_value = parsed
		kb_items.append((col_name, k_value, b_value, nc_path))

	if not kb_items:
		return {
			"status": "skipped",
			"prov_short": prov_short,
			"prov_pinyin": prov_pinyin,
			"season": season,
			"reason": "no_kb_files",
		}

	kb_items.sort(key=lambda item: (item[1], item[2]))

	kb_series_map: dict[str, dict[str, float]] = {}
	read_errors: list[str] = []
	winter_plan = official_detail.get("winter_plan", {}) if season == "winter" else None
	winter_source_notes = list(official_detail.get("winter_source_notes", [])) if season == "winter" else []
	for col_name, _, _, nc_path in kb_items:
		try:
			kb_series_map[col_name] = _extract_model_series_from_nc(nc_path, season, winter_plan=winter_plan)
		except Exception as exc:  # noqa: BLE001 - per-file isolation in worker
			read_errors.append(f"{nc_path.name}: {exc}")

	if not kb_series_map:
		return {
			"status": "skipped",
			"prov_short": prov_short,
			"prov_pinyin": prov_pinyin,
			"season": season,
			"reason": "all_kb_failed",
			"errors": read_errors,
		}

	official_for_season = _official_for_season(official_series_map, season)

	union_index = set(official_for_season.keys())
	for series_map in kb_series_map.values():
		union_index.update(series_map.keys())

	if not union_index:
		return {
			"status": "skipped",
			"prov_short": prov_short,
			"prov_pinyin": prov_pinyin,
			"season": season,
			"reason": "empty_union_index",
		}

	sorted_index = sorted(union_index, key=_label_sort_key)
	ordered_kb_columns = [item[0] for item in kb_items if item[0] in kb_series_map]
	ordered_columns = ["official", *ordered_kb_columns]

	series_by_column: dict[str, pd.Series] = {
		"official": pd.Series(official_for_season)
	}
	for col_name in ordered_kb_columns:
		series_by_column[col_name] = pd.Series(kb_series_map[col_name])

	# Build all columns at once to avoid DataFrame fragmentation from repeated inserts.
	result_df = pd.DataFrame(series_by_column)
	result_df = result_df.reindex(index=sorted_index, columns=ordered_columns)
	result_df.index.name = "time"

	output_root.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(output_csv_path, encoding="utf-8-sig")

	return {
		"status": "ok",
		"prov_short": prov_short,
		"prov_pinyin": prov_pinyin,
		"season": season,
		"output_csv_path": str(output_csv_path),
		"warning_errors": read_errors,
		"winter_source_notes": winter_source_notes,
	}


def prov_gen_enums(
	input_dir: str | Path,
	output_dir: str | Path,
	monthly_current_csv_path: str | Path = r"D:\Data\国家数据网站\分省月度数据-风电-当期值(亿千瓦时).csv",
	monthly_cumulative_csv_path: str | Path = r"D:\Data\国家数据网站\分省月度数据-风电-当年累积值(亿千瓦时).csv",
	admin_province_csv_path: str | Path = r"D:\Data\admin\省市名全简称.csv",
	max_workers: Optional[int] = None,
	overwrite: bool = False,
) -> dict[str, Any]:
	"""
	对 input_dir 下每个(省,季节)组合输出一份 CSV：
	{prov_short}_{season}_monthly_gen_enums.csv

	规则：
	1. official 来自两份国家数据 CSV，统一换算到 MWh。
	2. 空值和非正数视为缺失；春夏秋保留对应月份。
	3. 冬季按年混合处理：若可恢复或直接取得 1/2 月，则保留 12 月、1 月、2 月；否则回退保留同年 12 月和同年 1-2 月。
	3. CSV index 为时间标签，取 official 与所有 kb 列时间索引并集。
	4. province 在 official / model 任一侧缺失时，整省跳过并记录日志。
	"""
	input_root = Path(input_dir)
	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	if not input_root.exists() or not input_root.is_dir():
		raise FileNotFoundError(f"input_dir 不存在或不是目录: {input_root}")

	unit_to_mwh_factor = 100_000.0
	official_map = _extract_official_by_province(
		monthly_current_csv_path=Path(monthly_current_csv_path),
		monthly_cumulative_csv_path=Path(monthly_cumulative_csv_path),
		admin_province_csv_path=Path(admin_province_csv_path),
		unit_to_mwh_factor=unit_to_mwh_factor,
	)

	pinyin_to_short = {chinese_to_pinyi(prov_short): prov_short for prov_short in official_map.keys()}

	created_files: list[str] = []
	skipped_provinces: list[str] = []
	skipped_existing_files: list[str] = []
	tasks: list[dict[str, Any]] = []

	model_prov_short_set: set[str] = set()
	for prov_dir in sorted([p for p in input_root.iterdir() if p.is_dir()], key=lambda p: p.name):
		prov_pinyin = prov_dir.name
		prov_short = pinyin_to_short.get(prov_pinyin)
		if prov_short is None:
			logger.warning("跳过省份: 模型目录存在但 official 中无对应省简称, prov_pinyin=%s", prov_pinyin)
			skipped_provinces.append(prov_pinyin)
			continue

		if not official_map.get(prov_short):
			logger.warning("跳过省份: official 数据为空, prov_short=%s", prov_short)
			skipped_provinces.append(prov_short)
			continue

		model_prov_short_set.add(prov_short)
		for season_dir in sorted([p for p in prov_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
			season = season_dir.name.lower()
			if season not in SEASON_MONTHS:
				continue
			tasks.append(
				{
					"prov_short": prov_short,
					"prov_pinyin": prov_pinyin,
					"season": season,
					"season_dir": str(season_dir),
					"output_dir": str(output_root),
					"monthly_current_csv_path": str(Path(monthly_current_csv_path)),
					"monthly_cumulative_csv_path": str(Path(monthly_cumulative_csv_path)),
					"admin_province_csv_path": str(Path(admin_province_csv_path)),
					"unit_to_mwh_factor": unit_to_mwh_factor,
					"overwrite": overwrite,
				}
			)

	for official_prov_short in sorted(official_map.keys()):
		if official_prov_short not in model_prov_short_set:
			logger.warning("跳过省份: official 中存在但模型目录缺失, prov_short=%s", official_prov_short)
			skipped_provinces.append(official_prov_short)

	if not tasks:
		return {
			"input_dir": str(input_root),
			"output_dir": str(output_root),
			"created_file_count": 0,
			"created_files": [],
			"skipped_provinces": sorted(set(skipped_provinces)),
			"skipped_existing_file_count": 0,
			"skipped_existing_files": [],
			"task_count": 0,
		}

	if max_workers is None:
		cpu = os.cpu_count() or 1
		worker_count = max(1, min(len(tasks), cpu - 1 if cpu > 1 else 1))
	else:
		worker_count = max(1, min(int(max_workers), len(tasks)))

	with ProcessPoolExecutor(max_workers=worker_count) as executor:
		futures = [executor.submit(_process_single_prov_season_task, task) for task in tasks]
		for future in as_completed(futures):
			result = future.result()
			status = result.get("status")
			if status == "ok":
				created_files.append(str(result["output_csv_path"]))
				winter_source_notes = result.get("winter_source_notes", [])
				if result.get("season") == "winter" and winter_source_notes:
					logger.info(
						"冬季 official 来源: prov=%s, details=%s",
						result.get("prov_short"),
						"; ".join(winter_source_notes),
					)
				for err in result.get("warning_errors", []):
					logger.warning(
						"单文件读取失败但任务已完成: prov=%s, season=%s, detail=%s",
						result.get("prov_short"),
						result.get("season"),
						err,
					)
			elif status == "skipped_existing":
				skipped_existing_files.append(str(result["output_csv_path"]))
				logger.info(
					"跳过已存在输出: prov=%s, season=%s, output=%s",
					result.get("prov_short", result.get("prov_pinyin")),
					result.get("season"),
					result.get("output_csv_path"),
				)
			else:
				reason = result.get("reason", "unknown")
				logger.info(
					"跳过组合: prov=%s, season=%s, reason=%s",
					result.get("prov_short", result.get("prov_pinyin")),
					result.get("season"),
					reason,
				)
				if reason == "official_missing":
					skipped_provinces.append(str(result.get("prov_short", result.get("prov_pinyin"))))
				for err in result.get("errors", []):
					logger.warning(
						"任务失败详情: prov=%s, season=%s, detail=%s",
						result.get("prov_short", result.get("prov_pinyin")),
						result.get("season"),
						err,
					)

	return {
		"input_dir": str(input_root),
		"output_dir": str(output_root),
		"created_file_count": len(created_files),
		"created_files": created_files,
		"skipped_provinces": sorted(set(skipped_provinces)),
		"skipped_existing_file_count": len(skipped_existing_files),
		"skipped_existing_files": skipped_existing_files,
		"task_count": len(tasks),
		"max_workers": worker_count,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate monthly generation enums CSVs by (province, season) in multiprocessing mode")
	parser.add_argument("--input_dir", type=str, help="Input directory like .../gen/prov", default='/geogfs1/groups/geog_liulb/hongyu/results/kb_batch_aggr/gen/prov')
	parser.add_argument("--output_dir", type=str, help="Output CSV directory", default="/geogfs1/groups/geog_liulb/hongyu/results/prov_gen_enums")
	parser.add_argument(
		"--monthly_current_csv_path",
		type=str,
		default="/geogfs1/groups/geog_liulb/hongyu/data/分省月度数据-风电-当期值(亿千瓦时).csv",
		help="Official monthly current-value CSV (unit: 亿千瓦时)",
	)
	parser.add_argument(
		"--monthly_cumulative_csv_path",
		type=str,
		default="/geogfs1/groups/geog_liulb/hongyu/data/分省月度数据-风电-当年累积值(亿千瓦时).csv",
		help="Official monthly cumulative-value CSV (unit: 亿千瓦时)",
	)
	parser.add_argument(
		"--admin_province_csv_path",
		type=str,
		default="/geogfs1/groups/geog_liulb/hongyu/data/省市名全简称.csv",
		help="Province full/short name mapping CSV",
	)
	parser.add_argument(
		"--max_workers",
		type=int,
		default=None,
		help="Process worker count, default auto",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing output CSV files",
	)
	parser.add_argument(
		"--log_level",
		type=str,
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
		help="Logging level",
	)

	args = parser.parse_args()
	logging.basicConfig(
		level=getattr(logging, args.log_level.upper(), logging.INFO),
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)

	result = prov_gen_enums(
		input_dir=args.input_dir,
		output_dir=args.output_dir,
		monthly_current_csv_path=args.monthly_current_csv_path,
		monthly_cumulative_csv_path=args.monthly_cumulative_csv_path,
		admin_province_csv_path=args.admin_province_csv_path,
		max_workers=args.max_workers,
		overwrite=args.overwrite,
	)
	logger.info(
		"Done | tasks=%s | created=%s | skipped_provinces=%s",
		result.get("task_count"),
		result.get("created_file_count"),
		len(result.get("skipped_provinces", [])),
	)


if __name__ == "__main__":
	main()
