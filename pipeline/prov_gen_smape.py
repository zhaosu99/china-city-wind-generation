from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from indices import smape
from utils import safe_read_csv


KB_PATTERN = re.compile(r"^k(-?\d+(?:\.\d+)?)b(-?\d+(?:\.\d+)?)$")


def get_intersection_times(df: pd.DataFrame) -> list[str]:
	time_col = df.columns[0]
	official_col = next((c for c in df.columns if str(c).strip().lower() == "official"), None)
	if official_col is None:
		raise KeyError("未找到 official 列")
	if "k0.1b-2" not in df.columns:
		raise KeyError("未找到 k0.1b-2 列")

	tmp = df.loc[df[official_col].notna() & df["k0.1b-2"].notna(), [time_col]].copy()
	tmp[time_col] = tmp[time_col].astype(str).str.strip()
	tmp = tmp[tmp[time_col] != ""]
	return tmp[time_col].drop_duplicates().tolist()


def load_existing_combos(json_path: Path, valid_times: set[str], train_size: int) -> list[tuple[str, ...]]:
	if not json_path.exists():
		return []

	try:
		payload = json.loads(json_path.read_text(encoding="utf-8"))
		if not isinstance(payload, dict):
			raise ValueError("json 不是 dict")

		combos: list[tuple[str, ...]] = []
		seen: set[tuple[str, ...]] = set()
		for key in sorted(payload.keys(), key=lambda item: int(item)):
			arr = payload[key]
			if not isinstance(arr, list) or len(arr) != train_size:
				raise ValueError("元素不是目标长度 list")
			arr = [str(item).strip() for item in arr]
			if len(set(arr)) != train_size or any(item not in valid_times for item in arr):
				raise ValueError("元素有重复或越界")
			combo = tuple(arr)
			if combo in seen:
				raise ValueError("组合重复")
			seen.add(combo)
			combos.append(combo)
		return combos
	except Exception:
		json_path.unlink(missing_ok=True)
		return []


def generate_combos(
	times: list[str],
	train_size: int,
	combi_num: int,
	seed: int,
	existing: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
	if not times or train_size <= 0:
		return []

	max_possible = math.comb(len(times), train_size)
	target = min(combi_num, max_possible)

	rng = random.Random(seed)
	idx_map = {time_label: idx for idx, time_label in enumerate(times)}
	chosen: list[tuple[str, ...]] = []
	chosen_set: set[tuple[str, ...]] = set()

	for combo in existing:
		ordered = tuple(sorted(combo, key=lambda item: idx_map[item]))
		if ordered not in chosen_set:
			chosen.append(ordered)
			chosen_set.add(ordered)
		if len(chosen) >= target:
			return chosen[:target]

	for _ in range(max(1000, target * 40)):
		if len(chosen) >= target:
			break
		combo = tuple(sorted(rng.sample(times, train_size), key=lambda item: idx_map[item]))
		if combo not in chosen_set:
			chosen.append(combo)
			chosen_set.add(combo)

	if len(chosen) < target:
		for comb_idx in itertools.combinations(range(len(times)), train_size):
			combo = tuple(times[idx] for idx in comb_idx)
			if combo in chosen_set:
				continue
			chosen.append(combo)
			chosen_set.add(combo)
			if len(chosen) >= target:
				break

	return chosen[:target]


def build_one_csv(
	csv_path: Path,
	output_dir: Path,
	ratio: float,
	combi_num: int,
	seed: int,
	overwrite: bool = False,
) -> dict[str, Any]:
	df = safe_read_csv(csv_path)
	times = get_intersection_times(df)
	train_size = max(1, math.floor(len(times) * ratio)) if times else 0

	output_dir.mkdir(parents=True, exist_ok=True)
	ratio_tag = int(round(ratio * 100))
	json_path = output_dir / "samplings" / str(ratio_tag) / f"{csv_path.stem}_train{ratio_tag}.json"
	json_path.parent.mkdir(parents=True, exist_ok=True)

	existing = load_existing_combos(json_path, set(times), train_size)
	max_possible = math.comb(len(times), train_size) if times and train_size > 0 else 0
	target = min(combi_num, max_possible) if max_possible > 0 else 0
	json_written = True
	if json_path.exists() and not overwrite and len(existing) >= target:
		combos = existing[:target]
		json_written = False
	else:
		combos = generate_combos(times, train_size, combi_num, seed, existing)
		payload = {idx + 1: list(combo) for idx, combo in enumerate(combos)}
		json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

	return {
		"combos": combos,
		"csv": csv_path.name,
		"ratio": ratio,
		"time_total": len(times),
		"time_train": train_size,
		"time_test": max(0, len(times) - train_size),
		"combo_count": len(combos),
		"reused_count": min(len(existing), len(combos)),
		"json": str(json_path),
		"json_written": json_written,
		"json_skipped_existing": not json_written,
	}


def parse_kb_columns(columns) -> list[tuple[str, float, float]]:
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


def _build_surface_rel_path(
	*,
	surface_type: str,
	prov_short: str,
	season: str,
	ratio_tag: int,
	combo_id: int,
) -> str:
	return str(
		Path("surfaces")
		/ surface_type
		/ prov_short
		/ season
		/ str(ratio_tag)
		/ f"{prov_short}_{season}_{ratio_tag}_Np{combo_id}_{surface_type}_gen_smape.csv"
	)


def _compute_subset_smape_grid(df: pd.DataFrame, subset_times: tuple[str, ...]) -> pd.DataFrame:
	time_col = df.columns[0]
	official_col = next((c for c in df.columns if str(c).strip().lower() == "official"), None)
	if official_col is None:
		raise KeyError("未找到 official 列")

	indexed_df = df.copy()
	indexed_df[time_col] = indexed_df[time_col].astype(str).str.strip()
	indexed_df = indexed_df.set_index(time_col)

	kb_items = parse_kb_columns(indexed_df.columns)
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


def _run_single_subset_smape_task(task: tuple[str, tuple[str, ...], str, bool]) -> dict[str, Any]:
	csv_path_str, subset_times, output_path_str, overwrite = task
	csv_path = Path(csv_path_str)
	output_path = Path(output_path_str)
	if output_path.exists() and not overwrite:
		return {
			"output_path": str(output_path),
			"written": False,
			"skipped_existing": True,
		}

	df = safe_read_csv(csv_path)
	grid = _compute_subset_smape_grid(df, subset_times)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	grid.to_csv(output_path, encoding="utf-8-sig")
	return {
		"output_path": str(output_path),
		"written": True,
		"skipped_existing": False,
	}


def save_smape_grids_for_combos(
	csv_path: Path,
	output_dir: Path,
	ratio: float,
	combos: list[tuple[str, ...]],
	max_workers: int | None = None,
	overwrite: bool = False,
) -> dict[str, Any]:
	if not combos:
		return {
			"train_paths": [],
			"test_paths": [],
			"train_written_count": 0,
			"train_skipped_existing_count": 0,
			"test_written_count": 0,
			"test_skipped_existing_count": 0,
		}

	df = safe_read_csv(csv_path)
	all_times = tuple(get_intersection_times(df))
	prov_short, season = _parse_prov_and_season(csv_path)
	ratio_tag = int(round(ratio * 100))
	tasks: list[tuple[str, tuple[str, ...], str, bool]] = []
	train_paths: list[str] = []
	test_paths: list[str] = []

	for combo_id, combo_times in enumerate(combos, start=1):
		train_times = tuple(combo_times)
		train_set = set(train_times)
		test_times = tuple(time_label for time_label in all_times if time_label not in train_set)

		train_rel_path = _build_surface_rel_path(
			surface_type="train",
			prov_short=prov_short,
			season=season,
			ratio_tag=ratio_tag,
			combo_id=combo_id,
		)
		test_rel_path = _build_surface_rel_path(
			surface_type="test",
			prov_short=prov_short,
			season=season,
			ratio_tag=ratio_tag,
			combo_id=combo_id,
		)
		train_output_path = str(output_dir / train_rel_path)
		test_output_path = str(output_dir / test_rel_path)
		train_paths.append(train_output_path)
		test_paths.append(test_output_path)
		tasks.append((str(csv_path), train_times, train_output_path, overwrite))
		tasks.append((str(csv_path), test_times, test_output_path, overwrite))

	results: list[dict[str, Any]] = []
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(_run_single_subset_smape_task, task) for task in tasks]
		for future in as_completed(futures):
			results.append(future.result())

	train_written_count = 0
	train_skipped_existing_count = 0
	test_written_count = 0
	test_skipped_existing_count = 0
	for item in results:
		output_path = str(item["output_path"])
		if "_train_gen_smape.csv" in output_path:
			if item["written"]:
				train_written_count += 1
			if item["skipped_existing"]:
				train_skipped_existing_count += 1
		else:
			if item["written"]:
				test_written_count += 1
			if item["skipped_existing"]:
				test_skipped_existing_count += 1

	return {
		"train_paths": train_paths,
		"test_paths": test_paths,
		"train_written_count": train_written_count,
		"train_skipped_existing_count": train_skipped_existing_count,
		"test_written_count": test_written_count,
		"test_skipped_existing_count": test_skipped_existing_count,
	}


def _load_smape_grid(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
	df.index = pd.to_numeric(df.index, errors="coerce")
	df.columns = pd.to_numeric(df.columns, errors="coerce")
	df = df.loc[~df.index.isna(), ~pd.isna(df.columns)]
	return df.sort_index(axis=0).sort_index(axis=1)


def _aggregate_surface_paths(surface_paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
	if not surface_paths:
		raise ValueError("surface_paths 不能为空")

	reference_df = _load_smape_grid(Path(surface_paths[0]))
	index = reference_df.index
	columns = reference_df.columns
	stacked_values: list[np.ndarray] = []
	for path_str in surface_paths:
		cur_df = _load_smape_grid(Path(path_str)).reindex(index=index, columns=columns)
		stacked_values.append(cur_df.to_numpy(dtype=float))

	stacked = np.stack(stacked_values, axis=0)
	mean_df = pd.DataFrame(np.nanmean(stacked, axis=0), index=index, columns=columns)
	std_df = pd.DataFrame(np.nanstd(stacked, axis=0), index=index, columns=columns)
	return mean_df, std_df


def build_aggregate_smape_grids(
	*,
	csv_path: Path,
	output_dir: Path,
	train_paths: list[str],
	test_paths: list[str],
) -> dict[str, Any]:
	if not train_paths or not test_paths:
		raise ValueError("构建 aggregate 底图需要非空 train/test surface 路径")

	prov_short, season = _parse_prov_and_season(csv_path)
	aggregate_dir = output_dir / "aggregate"
	aggregate_dir.mkdir(parents=True, exist_ok=True)

	train_mean_df, train_std_df = _aggregate_surface_paths(train_paths)
	test_mean_df, test_std_df = _aggregate_surface_paths(test_paths)

	train_mean_path = aggregate_dir / f"{prov_short}_{season}_mean_train_gen_smape.csv"
	train_std_path = aggregate_dir / f"{prov_short}_{season}_std_train_gen_smape.csv"
	test_mean_path = aggregate_dir / f"{prov_short}_{season}_mean_test_gen_smape.csv"
	test_std_path = aggregate_dir / f"{prov_short}_{season}_std_test_gen_smape.csv"

	train_mean_df.to_csv(train_mean_path, encoding="utf-8-sig")
	train_std_df.to_csv(train_std_path, encoding="utf-8-sig")
	test_mean_df.to_csv(test_mean_path, encoding="utf-8-sig")
	test_std_df.to_csv(test_std_path, encoding="utf-8-sig")

	meta_path = aggregate_dir / f"{prov_short}_{season}_aggregate_meta.json"
	meta_path.write_text(
		json.dumps(
			{
				"prov": prov_short,
				"season": season,
				"train_surface_count": len(train_paths),
				"test_surface_count": len(test_paths),
				"train_surface_paths": train_paths,
				"test_surface_paths": test_paths,
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)

	return {
		"train_mean_path": str(train_mean_path),
		"train_std_path": str(train_std_path),
		"test_mean_path": str(test_mean_path),
		"test_std_path": str(test_std_path),
		"meta_path": str(meta_path),
	}


def load_combos_for_ratio(csv_path: Path, output_dir: Path, ratio: float, combi_num: int) -> list[tuple[str, ...]]:
	ratio_tag = int(round(ratio * 100))
	json_path = output_dir / "samplings" / str(ratio_tag) / f"{csv_path.stem}_train{ratio_tag}.json"
	if not json_path.exists():
		raise FileNotFoundError(f"未找到 combo json: {json_path}")

	df = safe_read_csv(csv_path)
	times = get_intersection_times(df)
	train_size = max(1, math.floor(len(times) * ratio)) if times else 0
	combos = load_existing_combos(json_path, set(times), train_size)
	if not combos:
		raise ValueError(f"json 无有效组合: {json_path}")
	return combos[:combi_num]


def process_single_csv_task(
	task: tuple[str, str, dict[float, int], int, bool, bool, int | None, bool],
) -> dict[str, Any]:
	csv_path_str, output_dir_str, ratio_combi_map, seed, gen_combo, cal_combo, inner_max_workers, overwrite = task
	csv_path = Path(csv_path_str)
	output_dir = Path(output_dir_str)

	reports: list[dict[str, Any]] = []
	all_train_paths: list[str] = []
	all_test_paths: list[str] = []
	total_train_written = 0
	total_train_skipped_existing = 0
	total_test_written = 0
	total_test_skipped_existing = 0
	for ratio, combi_num in ratio_combi_map.items():
		if gen_combo:
			report = build_one_csv(csv_path, output_dir, ratio, combi_num, seed, overwrite=overwrite)
			combos = report["combos"]
		else:
			combos = load_combos_for_ratio(csv_path, output_dir, ratio, combi_num)
			report = {
				"combos": combos,
				"csv": csv_path.name,
				"ratio": ratio,
				"combo_count": len(combos),
				"reused_count": 0,
				"json": "skip_gen_combo",
				"json_written": False,
				"json_skipped_existing": False,
			}

		if cal_combo:
			surface_summary = save_smape_grids_for_combos(
				csv_path,
				output_dir,
				ratio,
				combos,
				max_workers=inner_max_workers,
				overwrite=overwrite,
			)
			report.update(surface_summary)
			report["train_surface_count"] = len(surface_summary["train_paths"])
			report["test_surface_count"] = len(surface_summary["test_paths"])
			all_train_paths.extend(surface_summary["train_paths"])
			all_test_paths.extend(surface_summary["test_paths"])
			total_train_written += surface_summary["train_written_count"]
			total_train_skipped_existing += surface_summary["train_skipped_existing_count"]
			total_test_written += surface_summary["test_written_count"]
			total_test_skipped_existing += surface_summary["test_skipped_existing_count"]
		else:
			report["train_paths"] = []
			report["test_paths"] = []
			report["train_surface_count"] = 0
			report["test_surface_count"] = 0

		reports.append(report)

	aggregate_summary = None
	if cal_combo and all_train_paths and all_test_paths:
		aggregate_summary = build_aggregate_smape_grids(
			csv_path=csv_path,
			output_dir=output_dir,
			train_paths=all_train_paths,
			test_paths=all_test_paths,
		)

	return {
		"csv": csv_path.name,
		"reports": reports,
		"aggregate_summary": aggregate_summary,
		"train_surface_written_count": total_train_written,
		"train_surface_skipped_existing_count": total_train_skipped_existing,
		"test_surface_written_count": total_test_written,
		"test_surface_skipped_existing_count": total_test_skipped_existing,
	}


def parse_ratio_combi_map(items: list[str]) -> dict[float, int]:
	ratio_combi_map: dict[float, int] = {}
	for item in items:
		if ":" not in item:
			raise ValueError(f"ratio_combi 格式错误: {item}, 期望 ratio:combi_num")
		ratio_str, combi_str = item.split(":", 1)
		try:
			ratio = float(ratio_str)
		except ValueError as exc:
			raise ValueError(f"ratio 不是数字: {ratio_str}") from exc
		try:
			combi_num = int(combi_str)
		except ValueError as exc:
			raise ValueError(f"combi_num 不是整数: {combi_str}") from exc
		ratio_combi_map[ratio] = combi_num

	if not ratio_combi_map:
		raise ValueError("ratio_combi 不能为空")
	return ratio_combi_map


def _normalize_ratio_combi_map(ratio_combi_map: dict[float, int] | list[str]) -> dict[float, int]:
	if isinstance(ratio_combi_map, dict):
		return dict(ratio_combi_map)
	return parse_ratio_combi_map(ratio_combi_map)


def _validate_inputs(
	input_dir: Path,
	ratio_combi_map: dict[float, int],
	file_pattern: str,
) -> list[Path]:
	for ratio, combi_num in ratio_combi_map.items():
		if ratio <= 0 or ratio >= 1:
			raise ValueError(f"training_ratio 超出范围: {ratio}")
		if combi_num <= 0:
			raise ValueError(f"combi_num 必须是正整数, ratio={ratio}, combi_num={combi_num}")
	if not input_dir.exists():
		raise FileNotFoundError(f"未找到输入目录: {input_dir}")

	csv_files = sorted(input_dir.glob(file_pattern), key=lambda path: path.name)
	if not csv_files:
		raise FileNotFoundError(f"目录下未匹配到文件: {input_dir} / {file_pattern}")
	return csv_files


def print_run_summary(results: list[dict[str, Any]]) -> None:
	print(f"生成完成，共处理 {len(results)} 个 CSV。")
	for result in sorted(results, key=lambda item: item["csv"]):
		print(f"\n[CSV] {result['csv']}")
		for report in result["reports"]:
			print(
				f"- ratio={report['ratio']}, "
				f"time_total={report.get('time_total', 'NA')}, "
				f"time_train={report.get('time_train', 'NA')}, "
				f"time_test={report.get('time_test', 'NA')}, "
				f"combo_count={report['combo_count']}, "
				f"train_surfaces={report.get('train_surface_count', 0)}, "
				f"test_surfaces={report.get('test_surface_count', 0)}, "
				f"json={report['json']}"
			)
		aggregate_summary = result.get("aggregate_summary")
		if aggregate_summary:
			print(f"- aggregate_train_mean={aggregate_summary['train_mean_path']}")
			print(f"- aggregate_test_mean={aggregate_summary['test_mean_path']}")


def prov_gen_smape(
	input_dir: Path,
	output_dir: Path,
	ratio_combi_map: dict[float, int] | list[str],
	seed: int = 42,
	gen_combo: bool = True,
	cal_combo: bool = True,
	outer_max_workers: int | None = None,
	inner_max_workers: int | None = None,
	file_pattern: str = "*_monthly_gen_enums*.csv",
	overwrite: bool = False,
) -> list[dict[str, Any]]:
	input_dir = Path(input_dir).resolve()
	output_dir = Path(output_dir).resolve()
	ratio_combi_map = _normalize_ratio_combi_map(ratio_combi_map)

	output_dir.mkdir(parents=True, exist_ok=True)
	csv_files = _validate_inputs(input_dir, ratio_combi_map, file_pattern)

	tasks = [
		(
			str(csv_path),
			str(output_dir),
			dict(ratio_combi_map),
			seed,
			gen_combo,
			cal_combo,
			inner_max_workers,
			overwrite,
		)
		for csv_path in csv_files
	]

	results: list[dict[str, Any]] = []
	with ProcessPoolExecutor(max_workers=outer_max_workers) as executor:
		futures = [executor.submit(process_single_csv_task, task) for task in tasks]
		for future in as_completed(futures):
			results.append(future.result())

	return sorted(results, key=lambda item: item["csv"])


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="批量生成 train/test generation SMAPE 网格，并聚合为平均/标准差底图。",
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path(r"D:\HKU PhD\wind data paper\try_intput"),
		help="输入目录，读取 *_monthly_gen_enums*.csv",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(r"D:\HKU PhD\wind data paper\try\combos"),
		help="输出目录",
	)
	parser.add_argument(
		"--ratio-combi",
		nargs="+",
		default=["0.8:100"],
		help="训练比例与组合数映射，格式 ratio:combi_num，例如 0.8:100 0.9:120",
	)
	parser.add_argument("--seed", type=int, default=42, help="随机种子")
	parser.add_argument(
		"--gen-combo",
		dest="gen_combo",
		action="store_true",
		help="生成/刷新 combo json（默认开启）",
	)
	parser.add_argument(
		"--no-gen-combo",
		dest="gen_combo",
		action="store_false",
		help="不生成 combo json，改为读取已有 json",
	)
	parser.set_defaults(gen_combo=True)
	parser.add_argument(
		"--cal-combo",
		dest="cal_combo",
		action="store_true",
		help="计算每个 combo 的 train/test SMAPE 网格并输出 aggregate 底图（默认开启）",
	)
	parser.add_argument(
		"--no-cal-combo",
		dest="cal_combo",
		action="store_false",
		help="不计算 combo 底图",
	)
	parser.set_defaults(cal_combo=True)
	parser.add_argument(
		"--outer-max-workers",
		type=int,
		default=None,
		help="外层并发 worker 数（每个 CSV 一个任务），默认 None",
	)
	parser.add_argument(
		"--inner-max-workers",
		type=int,
		default=None,
		help="内层并发 worker 数（每个 combo 两个 train/test surface 任务），默认 None",
	)
	parser.add_argument(
		"--file-pattern",
		type=str,
		default="*_monthly_gen_enums*.csv",
		help="输入文件匹配模式",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="覆盖已存在的 combo json 和 SMAPE 输出文件",
	)
	return parser.parse_args()


def cli_main() -> None:
	args = parse_args()
	results = prov_gen_smape(
		input_dir=args.input_dir,
		output_dir=args.output_dir,
		ratio_combi_map=args.ratio_combi,
		seed=args.seed,
		gen_combo=args.gen_combo,
		cal_combo=args.cal_combo,
		outer_max_workers=args.outer_max_workers,
		inner_max_workers=args.inner_max_workers,
		file_pattern=args.file_pattern,
		overwrite=args.overwrite,
	)
	print_run_summary(results)


if __name__ == "__main__":
	cli_main()