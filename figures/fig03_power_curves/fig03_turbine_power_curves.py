import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from difflib import SequenceMatcher


# --- centralized paths (added by plot_paper_v2 reorganization) --------------
# Original hard-coded values:
#   CSV_PATH           = D:\Data\turbines\wind_turbines.csv
#   INVENTORY_CSV_PATH = D:\HKU PhD\wind data paper\wind_farms_inventory_till_2023.csv
#   OUTPUT_PATH        = D:\HKU PhD\wind data paper\plot\fig4_turbine_power_curves.png
import sys as _sys

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in _sys.path:
	_sys.path.insert(0, str(_HERE.parent))
import config_paths as _cfg  # noqa: E402

CSV_PATH = _cfg.resolve_first_existing(
	_cfg.TURBINE_CSV_CANDIDATES,
	kind="turbine parameter CSV",
	hint=f"set {_cfg.WINDPAPER_TURBINE_CSV} to the path of wind_turbines.csv",
)
INVENTORY_CSV_PATH = _cfg.resolve_first_existing(
	_cfg.INVENTORY_CSV_CANDIDATES,
	kind="wind farm inventory CSV",
	hint=f"set {_cfg.WINDPAPER_PAPER_ROOT} to the directory holding wind_farms_inventory_till_2023.csv",
)
OUTPUT_PATH = _HERE / "output" / "fig03_turbine_power_curves.png"
SELECTION_CONFIG_PATH = Path(__file__).with_suffix(".json")

REQUIRED_COLUMNS = [
	"Company",
	"Model",
	"Rated power",
	"Cut-in wind speed",
	"Rated wind speed",
	"Cut-out wind speed",
	"Hub height",
]

INVENTORY_MODEL_COLUMN = "机型编码"
INVENTORY_CAPACITY_COLUMN = "风电场装机MW"
TOP_MODEL_COUNT = 10
CANDIDATE_COUNT = 10
MATCH_LABEL_MAX_WIDTH = 44
Y_MINOR_STEP = 0.1
CURVE_W_STEP = 0.01

PALETTE = [
	"#D81B60",
	"#1E88E5",
	"#FFC107",
	"#004D40",
	"#8E24AA",
	"#43A047",
	"#FB8C00",
	"#3949AB",
	"#E53935",
	"#00897B",
	"#6D4C41",
	"#546E7A",
]

LINESTYLES = ["solid"]

X_MAJOR_STEP = 5
X_MINOR_STEP = 1
AXIS_LABEL_FONT_SIZE = 15
TICK_LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 12
PLOT_CURVE_SMOOTH_SIGMA_POINTS = 2.0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def safe_read_csv(file_path: Union[str, Path]) -> pd.DataFrame:
	file_path = Path(file_path)
	fallback_encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
	last_error: Optional[Exception] = None

	for encoding in fallback_encodings:
		try:
			return pd.read_csv(file_path, encoding=encoding)
		except UnicodeDecodeError as exc:
			last_error = exc
			continue

	if last_error is not None:
		raise last_error
	return pd.read_csv(file_path)


def generate_power_curve(
	cut_in_v: float = 3.0,
	rated_v: float = 12.0,
	cut_out_v: float = 25.0,
	v_slope: float = 1.0,
	v_intercept: float = 0.0,
	power_order: int = 3,
	smooth: bool = True,
	fixed_sigma: Union[bool, float] = False,
	sigma_slope: float = 0.2,
	sigma_intercept: float = 0.6,
	sigma_truncate: float = 4.0,
	w_step: float = 0.05,
	w_max: float = 40.0,
	plot: bool = False,
	ax: Optional[plt.Axes] = None,
	plot_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	if not (cut_in_v < rated_v < cut_out_v):
		raise ValueError(
			f"风速约束条件失败: 需要 cut_in_v < rated_v < cut_out_v，"
			f"实际值为 cut_in_v={cut_in_v}, rated_v={rated_v}, cut_out_v={cut_out_v}"
		)
	if w_step <= 0 or w_max <= cut_out_v:
		raise ValueError(
			f"网格参数无效: 需要 w_step > 0 且 w_max > cut_out_v，"
			f"实际值为 w_step={w_step}, w_max={w_max}"
		)
	if power_order <= 0:
		raise ValueError(f"power_order 必须为正整数，实际值为 {power_order}")
	if plot and not isinstance(plot_kwargs, (dict, type(None))):
		raise TypeError(f"plot_kwargs 必须为字典或 None，实际类型为 {type(plot_kwargs)}")

	w_grid = np.arange(0, w_max + w_step, w_step, dtype=np.float64)
	dw = w_step

	def _cf_raw(v: float) -> float:
		v_adj = v_slope * v + v_intercept
		if v_adj < cut_in_v:
			return 0.0
		if v_adj < rated_v:
			numerator = v_adj ** power_order - cut_in_v ** power_order
			denominator = rated_v ** power_order - cut_in_v ** power_order
			return numerator / denominator if denominator != 0 else 0.0
		if v_adj < cut_out_v:
			return 1.0
		return 0.0

	cf_raw = np.array([_cf_raw(w) for w in w_grid], dtype=np.float64)

	if not smooth:
		cf_grid = cf_raw
	else:
		if fixed_sigma is not False and isinstance(fixed_sigma, (int, float)):
			if fixed_sigma <= 0:
				raise ValueError(f"fixed_sigma 必须为正数，实际值为 {fixed_sigma}")

			sigma_w = float(fixed_sigma)
			sigma_pix = sigma_w / dw
			cf_grid = gaussian_filter1d(
				cf_raw,
				sigma=sigma_pix,
				mode="nearest",
				truncate=sigma_truncate,
				output=np.float64,
			)
		elif fixed_sigma is False:
			sigma_pix_grid = (sigma_intercept + sigma_slope * w_grid) / dw
			cf_grid = np.zeros_like(cf_raw, dtype=np.float64)
			n = len(w_grid)

			kernels = []
			for i in range(n):
				sigma_pix = sigma_pix_grid[i]
				radius = int(np.ceil(sigma_truncate * sigma_pix))
				x = np.arange(-radius, radius + 1)
				kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
				kernel = kernel / kernel.sum()
				start = max(0, i - radius)
				end = min(n, i + radius + 1)
				kernels.append((start, end, kernel[: end - start]))

			for i, (start, end, kernel) in enumerate(kernels):
				cf_grid[i] = np.sum(cf_raw[start:end] * kernel)
		else:
			raise ValueError(f"fixed_sigma 必须为 False 或正数，实际值为 {fixed_sigma}")

	if plot:
		if ax is None:
			_, ax = plt.subplots(figsize=(8, 4.5))
			own_fig = True
		else:
			own_fig = False

		kw = {"label": "Capacity Factor"}
		if plot_kwargs:
			kw.update(plot_kwargs)

		ax.plot(w_grid, cf_grid, antialiased=True, **kw)
		ax.set_xlabel("风速 (m/s)")
		ax.set_ylabel("容量因子")
		ax.set_title("风电功率曲线")
		ax.grid(True, linestyle="--", alpha=0.6)
		ax.set_xlim(0, w_max)
		ax.set_ylim(-0.05, 1.05)
		ax.legend()

		if own_fig:
			plt.tight_layout()
			plt.show()

	return w_grid, cf_grid


def farm_cf(
	w_hub: xr.DataArray,
	w_grid: Optional[np.ndarray] = None,
	cf_grid: Optional[np.ndarray] = None,
	time_name: str = "time",
	cut_in_v: float = 3.0,
	rated_v: float = 12.0,
	cut_out_v: float = 25.0,
	v_slope: float = 1.0,
	v_intercept: float = 0.0,
	power_order: int = 3,
	smooth: bool = True,
	fixed_sigma: Union[bool, float] = False,
	sigma_slope: float = 0.2,
	sigma_intercept: float = 0.6,
	sigma_truncate: float = 4.0,
	w_step: float = 0.05,
	w_max: float = 40.0,
) -> xr.DataArray:
	if time_name not in w_hub.dims:
		raise ValueError(f"w_hub 必须包含时间维度: {time_name}")

	if (w_grid is None) != (cf_grid is None):
		raise ValueError("w_grid 和 cf_grid 必须同时提供，或同时为 None")

	if w_grid is None and cf_grid is None:
		w_grid_used, cf_grid_used = generate_power_curve(
			cut_in_v=cut_in_v,
			rated_v=rated_v,
			cut_out_v=cut_out_v,
			v_slope=v_slope,
			v_intercept=v_intercept,
			power_order=power_order,
			smooth=smooth,
			fixed_sigma=fixed_sigma,
			sigma_slope=sigma_slope,
			sigma_intercept=sigma_intercept,
			sigma_truncate=sigma_truncate,
			w_step=w_step,
			w_max=w_max,
			plot=False,
		)
		curve_source = "generated"
	else:
		w_grid_used = np.asarray(w_grid, dtype=np.float64)
		cf_grid_used = np.asarray(cf_grid, dtype=np.float64)
		if w_grid_used.ndim != 1:
			raise ValueError(f"w_grid 必须为一维数组，得到 ndim={w_grid_used.ndim}")
		if cf_grid_used.ndim != 1:
			raise ValueError(f"cf_grid 必须为一维数组，得到 ndim={cf_grid_used.ndim}")
		if len(w_grid_used) != len(cf_grid_used):
			raise ValueError(
				f"w_grid 与 cf_grid 长度必须一致，得到 {len(w_grid_used)} vs {len(cf_grid_used)}"
			)
		curve_source = "provided"

	speed_values = np.asarray(w_hub.values, dtype=np.float64)
	cf_values = np.interp(
		x=speed_values,
		xp=w_grid_used,
		fp=cf_grid_used,
		left=0.0,
		right=float(cf_grid_used[-1]),
	).astype(np.float64)

	return xr.DataArray(
		cf_values,
		coords=w_hub.coords,
		dims=w_hub.dims,
		name="farm_cf",
		attrs={
			"long_name": "Wind farm capacity factor",
			"units": "1",
			"source_w_hub_name": w_hub.name if w_hub.name is not None else "w_hub",
			"cf_curve_source": curve_source,
			"interpolation_method": "numpy.interp",
			"interpolation_left_fill": 0.0,
			"interpolation_right_fill": float(cf_grid_used[-1]),
			"power_curve_point_count": int(len(cf_grid_used)),
			"power_curve_w_min": float(w_grid_used[0]),
			"power_curve_w_max": float(w_grid_used[-1]),
		},
	)


def _choose_y_tick_steps(y_max: float) -> Tuple[float, float]:
	major_candidates = [0.5, 1.0, 2.0, 5.0, 10.0]
	for major_step in major_candidates:
		if y_max / major_step <= 6:
			return major_step, major_step / 2.0
	major_step = max(10.0, np.ceil(y_max / 6.0 / 5.0) * 5.0)
	return float(major_step), float(major_step / 2.0)


def _build_speed_da(w_max: float = 40.0, w_step: float = CURVE_W_STEP) -> xr.DataArray:
	wind_speeds = np.arange(0.0, w_max + w_step, w_step, dtype=np.float64)
	return xr.DataArray(
		wind_speeds,
		dims=["time"],
		coords={"time": np.arange(wind_speeds.size, dtype=np.int32)},
		name="w_hub",
	)


def _compute_power_curve_mw(row: pd.Series, speed_da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
	cf_da = farm_cf(
		w_hub=speed_da,
		cut_in_v=float(row["Cut-in wind speed"]),
		rated_v=float(row["Rated wind speed"]),
		cut_out_v=float(row["Cut-out wind speed"]),
		w_step=CURVE_W_STEP,
	)
	power_mw = np.asarray(cf_da.values, dtype=np.float64) * float(row["Rated power"])
	return np.asarray(speed_da.values, dtype=np.float64), power_mw



def _clean_model_string(value: str) -> str:
	return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def _parse_inventory_model_code(model_code: str) -> Tuple[float | None, float | None]:
	match = re.search(r"(\d+(?:\.\d+)?)\s*[-_/]\s*(\d+(?:\.\d+)?)", str(model_code))
	if match is None:
		return None, None
	return float(match.group(1)), float(match.group(2))


def _parse_reference_rotor_diameter(model: str, rated_power_mw: float) -> float | None:
	numbers = [float(number) for number in re.findall(r"\d+(?:\.\d+)?", str(model))]
	if not numbers:
		return None

	rated_power_kw = float(rated_power_mw) * 1000.0
	tolerance_kw = max(5.0, rated_power_kw * 0.03)
	remaining_numbers: List[float] = []
	removed_rated_power = False
	for number in numbers:
		if not removed_rated_power and abs(number - rated_power_kw) <= tolerance_kw:
			removed_rated_power = True
			continue
		remaining_numbers.append(number)

	diameter_candidates = [number for number in remaining_numbers if 20.0 <= number <= 250.0]
	if diameter_candidates:
		return float(diameter_candidates[0])

	diameter_candidates = [number for number in numbers if 20.0 <= number <= 250.0]
	if diameter_candidates:
		return float(diameter_candidates[0])
	return None


def _score_reference_match(inventory_model: str, reference_row: pd.Series) -> Tuple[float, float | None, float]:
	rotor_diameter_m, rated_power_kw = _parse_inventory_model_code(inventory_model)
	reference_rated_power_mw = float(reference_row["Rated power"])
	reference_rated_power_kw = reference_rated_power_mw * 1000.0
	reference_rotor_diameter = _parse_reference_rotor_diameter(
		str(reference_row["Model"]),
		reference_rated_power_mw,
	)
	clean_inventory_model = _clean_model_string(inventory_model)
	clean_reference_model = _clean_model_string(str(reference_row["Model"]))
	string_similarity = SequenceMatcher(None, clean_inventory_model, clean_reference_model).ratio()

	power_score = 1.0
	if rated_power_kw is not None:
		power_score = abs(reference_rated_power_kw - rated_power_kw) / max(rated_power_kw, 1.0)

	diameter_score = 1.5
	if rotor_diameter_m is not None and reference_rotor_diameter is not None:
		diameter_score = abs(reference_rotor_diameter - rotor_diameter_m) / max(rotor_diameter_m, 1.0)

	total_score = power_score + diameter_score + (1.0 - string_similarity) * 0.25
	return total_score, reference_rotor_diameter, string_similarity


def _load_top_inventory_models(inventory_csv_path: Path, top_n: int = TOP_MODEL_COUNT) -> pd.DataFrame:
	inventory_df = safe_read_csv(inventory_csv_path)
	if INVENTORY_MODEL_COLUMN not in inventory_df.columns:
		raise KeyError(f"Missing required column in inventory CSV: {INVENTORY_MODEL_COLUMN}")
	if INVENTORY_CAPACITY_COLUMN not in inventory_df.columns:
		raise KeyError(f"Missing required column in inventory CSV: {INVENTORY_CAPACITY_COLUMN}")

	inventory_df = inventory_df[[INVENTORY_MODEL_COLUMN, INVENTORY_CAPACITY_COLUMN]].copy()
	inventory_df[INVENTORY_MODEL_COLUMN] = inventory_df[INVENTORY_MODEL_COLUMN].fillna("").astype(str).str.strip()
	inventory_df[INVENTORY_CAPACITY_COLUMN] = pd.to_numeric(inventory_df[INVENTORY_CAPACITY_COLUMN], errors="coerce")
	inventory_df = inventory_df.loc[
		(inventory_df[INVENTORY_MODEL_COLUMN] != "")
		& inventory_df[INVENTORY_CAPACITY_COLUMN].notna()
	].copy()

	model_counts = (
		inventory_df.groupby(INVENTORY_MODEL_COLUMN, as_index=False)
		.agg(
			inventory_count=(INVENTORY_MODEL_COLUMN, "size"),
			inventory_capacity_mw=(INVENTORY_CAPACITY_COLUMN, "sum"),
		)
		.sort_values(by=["inventory_capacity_mw", "inventory_count"], ascending=[False, False])
		.head(top_n)
		.rename(columns={INVENTORY_MODEL_COLUMN: "inventory_model"})
		.reset_index(drop=True)
	)
	return model_counts


def _prepare_reference_turbines(csv_path: Path) -> pd.DataFrame:
	reference_df = safe_read_csv(csv_path)
	missing_columns = [column for column in REQUIRED_COLUMNS if column not in reference_df.columns]
	if missing_columns:
		raise KeyError(f"Missing required columns in turbine CSV: {missing_columns}")

	reference_df = reference_df.dropna(subset=REQUIRED_COLUMNS).copy()
	numeric_columns = [
		"Rated power",
		"Cut-in wind speed",
		"Rated wind speed",
		"Cut-out wind speed",
		"Hub height",
	]
	for column in numeric_columns:
		reference_df[column] = pd.to_numeric(reference_df[column], errors="coerce")

	reference_df = reference_df.dropna(subset=numeric_columns).copy()
	reference_df = reference_df.loc[
		(reference_df["Rated power"] > 0)
		& (reference_df["Cut-in wind speed"] > 0)
		& (reference_df["Rated wind speed"] > reference_df["Cut-in wind speed"])
		& (reference_df["Cut-out wind speed"] > reference_df["Rated wind speed"])
		& (reference_df["Hub height"] > 0)
	].copy()
	return reference_df.reset_index(drop=True)


def _wrap_match_label(reference_company: str, reference_model: str) -> str:
	label = f"{reference_company} | {reference_model}"
	if len(label) <= MATCH_LABEL_MAX_WIDTH:
		return label
	return label[: MATCH_LABEL_MAX_WIDTH - 1] + "…"


def _order_plot_dataframe(plot_df: pd.DataFrame) -> pd.DataFrame:
	return plot_df.sort_values(
		by=["reference_company", "inventory_capacity_mw", "reference_model"],
		ascending=[True, False, True],
	).reset_index(drop=True)



def _build_name_similarity_candidates(
	inventory_model: str,
	reference_df: pd.DataFrame,
	candidate_count: int = CANDIDATE_COUNT,
) -> list[dict]:
	candidates: list[dict] = []
	clean_inventory_model = _clean_model_string(inventory_model)

	for _, reference_row in reference_df.iterrows():
		reference_company = str(reference_row["Company"])
		reference_model = str(reference_row["Model"])
		string_similarity = SequenceMatcher(
			None,
			clean_inventory_model,
			_clean_model_string(reference_model),
		).ratio()
		candidates.append(
			{
				"reference_company": reference_company,
				"reference_model": reference_model,
				"string_similarity": float(string_similarity),
				"reference_row": reference_row,
			}
		)

	candidates.sort(
		key=lambda candidate: (
			-candidate["string_similarity"],
			candidate["reference_company"],
			candidate["reference_model"],
		)
	)

	unique_candidates: list[dict] = []
	seen_keys: set[tuple[str, str]] = set()
	for candidate in candidates:
		candidate_key = (candidate["reference_company"], candidate["reference_model"])
		if candidate_key in seen_keys:
			continue
		seen_keys.add(candidate_key)
		unique_candidates.append(candidate)
		if len(unique_candidates) >= candidate_count:
			break

	return unique_candidates


def _print_selection_intro(top_inventory_df: pd.DataFrame) -> None:
	print("Top-10 selection logic:")
	print("1. Group the inventory by 机型编码.")
	print("2. Sum 风电场装机MW within each 机型编码.")
	print("3. Rank by total installed capacity descending.")
	print("4. Break ties by the row count of that 机型编码 descending.")
	print("5. Keep the top 10 models.")
	print()
	print(f"Selection cache path: {SELECTION_CONFIG_PATH}")
	print("If this JSON exists and is valid, the script will reuse it and skip prompting.")
	print()
	print("Top 10 inventory models by installed capacity:")
	for rank, inventory_row in enumerate(top_inventory_df.itertuples(index=False), start=1):
		print(
			f"  {rank}. {inventory_row.inventory_model} | "
			f"capacity_mw={float(inventory_row.inventory_capacity_mw):.2f} | "
			f"count={int(inventory_row.inventory_count)}"
		)
	print()


def _prompt_for_candidate_selection(inventory_row: pd.Series, candidates: list[dict]) -> dict:
	inventory_model = str(inventory_row["inventory_model"])
	print("-" * 80)
	print(
		f"Select a reference turbine for {inventory_model} "
		f"(capacity_mw={float(inventory_row['inventory_capacity_mw']):.2f}, "
		f"count={int(inventory_row['inventory_count'])})"
	)
	print(f"Top {len(candidates)} candidates by model-name similarity:")
	for index, candidate in enumerate(candidates, start=1):
		reference_row = candidate["reference_row"]
		print(
			f"  {index}. {candidate['reference_company']} | {candidate['reference_model']} | "
			f"rated_power={float(reference_row['Rated power']):.3f} MW | "
			f"similarity={float(candidate['string_similarity']):.4f}"
		)

	while True:
		selection = input(f"Enter your choice [1-{len(candidates)}]: ").strip()
		if selection.isdigit():
			selected_index = int(selection)
			if 1 <= selected_index <= len(candidates):
				return candidates[selected_index - 1]
		print(f"Invalid choice: {selection}. Please enter an integer from 1 to {len(candidates)}.")


def _serialize_selection_config(top_inventory_df: pd.DataFrame, selected_df: pd.DataFrame) -> dict:
	selection_rows = []
	selected_lookup = selected_df.set_index("inventory_model")
	for inventory_row in top_inventory_df.itertuples(index=False):
		inventory_model = str(inventory_row.inventory_model)
		selected_row = selected_lookup.loc[inventory_model]
		selection_rows.append(
			{
				"inventory_model": inventory_model,
				"inventory_count": int(inventory_row.inventory_count),
				"inventory_capacity_mw": float(inventory_row.inventory_capacity_mw),
				"reference_company": str(selected_row["reference_company"]),
				"reference_model": str(selected_row["reference_model"]),
			}
		)

	return {
		"top_n": int(TOP_MODEL_COUNT),
		"candidate_count": int(CANDIDATE_COUNT),
		"selection_logic": [
			"Group the inventory by 机型编码.",
			"Sum 风电场装机MW within each 机型编码.",
			"Rank by total installed capacity descending.",
			"Break ties by the row count of that 机型编码 descending.",
			"Keep the top 10 models.",
		],
		"selections": selection_rows,
	}


def _save_selection_config(config_path: Path, top_inventory_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
	config = _serialize_selection_config(top_inventory_df, selected_df)
	config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_selection_config(config_path: Path) -> dict | None:
	if not config_path.exists():
		return None
	return json.loads(config_path.read_text(encoding="utf-8"))


def _build_selected_row(inventory_row: pd.Series, reference_row: pd.Series, string_similarity: float) -> dict:
	return {
		"inventory_model": str(inventory_row["inventory_model"]),
		"inventory_count": int(inventory_row["inventory_count"]),
		"inventory_capacity_mw": float(inventory_row["inventory_capacity_mw"]),
		"reference_company": str(reference_row["Company"]),
		"reference_model": str(reference_row["Model"]),
		"string_similarity": float(string_similarity),
		"curve_label": _wrap_match_label(str(reference_row["Company"]), str(reference_row["Model"])),
		**reference_row.to_dict(),
	}


def _build_selection_dataframe_from_config(
	config: dict,
	top_inventory_df: pd.DataFrame,
	reference_df: pd.DataFrame,
) -> pd.DataFrame | None:
	selection_rows = config.get("selections")
	if not isinstance(selection_rows, list):
		return None

	selected_rows: list[dict] = []
	for inventory_row in top_inventory_df.itertuples(index=False):
		inventory_model = str(inventory_row.inventory_model)
		matching_selection = next(
			(selection for selection in selection_rows if selection.get("inventory_model") == inventory_model),
			None,
		)
		if matching_selection is None:
			return None

		reference_company = str(matching_selection.get("reference_company", "")).strip()
		reference_model = str(matching_selection.get("reference_model", "")).strip()
		matched_reference = reference_df.loc[
			(reference_df["Company"].astype(str).str.strip() == reference_company)
			& (reference_df["Model"].astype(str).str.strip() == reference_model)
		]
		if matched_reference.empty:
			return None

		reference_row = matched_reference.iloc[0].copy()
		string_similarity = SequenceMatcher(
			None,
			_clean_model_string(inventory_model),
			_clean_model_string(str(reference_row["Model"])),
		).ratio()
		inventory_series = pd.Series(
			{
				"inventory_model": inventory_model,
				"inventory_count": int(inventory_row.inventory_count),
				"inventory_capacity_mw": float(inventory_row.inventory_capacity_mw),
			}
		)
		selected_rows.append(_build_selected_row(inventory_series, reference_row, string_similarity))

	selected_df = pd.DataFrame(selected_rows)
	return selected_df.sort_values(by=["inventory_capacity_mw", "inventory_count"], ascending=[False, False]).reset_index(drop=True)


def _select_top_inventory_models_interactively(
	top_inventory_df: pd.DataFrame,
	reference_df: pd.DataFrame,
	candidate_count: int = CANDIDATE_COUNT,
) -> pd.DataFrame:
	_print_selection_intro(top_inventory_df)
	selected_rows: list[dict] = []

	for _, inventory_row in top_inventory_df.iterrows():
		inventory_row = inventory_row.copy()
		candidates = _build_name_similarity_candidates(
			str(inventory_row["inventory_model"]),
			reference_df,
			candidate_count=candidate_count,
		)
		selected_candidate = _prompt_for_candidate_selection(inventory_row, candidates)
		selected_reference_row = selected_candidate["reference_row"].copy()
		selected_rows.append(_build_selected_row(inventory_row, selected_reference_row, float(selected_candidate["string_similarity"])))
		print(
			f"Selected: {selected_candidate['reference_company']} | {selected_candidate['reference_model']}"
		)
		print()

	selected_df = pd.DataFrame(selected_rows)
	return selected_df.sort_values(
		by=["inventory_capacity_mw", "inventory_count"],
		ascending=[False, False],
	).reset_index(drop=True)


def _select_top_inventory_models(
	top_inventory_df: pd.DataFrame,
	reference_df: pd.DataFrame,
	config_path: Path = SELECTION_CONFIG_PATH,
	candidate_count: int = CANDIDATE_COUNT,
) -> pd.DataFrame:
	config = _load_selection_config(config_path)
	if config is not None:
		selected_df = _build_selection_dataframe_from_config(config, top_inventory_df, reference_df)
		if selected_df is not None:
			print(f"Loaded selection cache from: {config_path}")
			return selected_df
		print(f"Selection cache is invalid or out of date: {config_path}")
		print("Falling back to interactive selection.")
		print()

	selected_df = _select_top_inventory_models_interactively(
		top_inventory_df,
		reference_df,
		candidate_count=candidate_count,
	)
	_save_selection_config(config_path, top_inventory_df, selected_df)
	print(f"Saved selection cache to: {config_path}")
	print()
	return selected_df


def plot_selected_turbine_power_curves(
	inventory_csv_path: Path = INVENTORY_CSV_PATH,
	csv_path: Path = CSV_PATH,
	output_path: Path = OUTPUT_PATH,
	selection_config_path: Path = SELECTION_CONFIG_PATH,
	top_n: int = TOP_MODEL_COUNT,
) -> Tuple[Path, pd.DataFrame]:
	top_inventory_df = _load_top_inventory_models(inventory_csv_path, top_n=top_n)
	reference_df = _prepare_reference_turbines(csv_path)
	plot_df = _select_top_inventory_models(
		top_inventory_df,
		reference_df,
		config_path=selection_config_path,
	)
	plot_df = _order_plot_dataframe(plot_df)
	speed_da = _build_speed_da()

	fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
	max_power = 0.0

	for row_index, (_, row) in enumerate(plot_df.iterrows()):
		wind_speeds, power_curve_mw = _compute_power_curve_mw(row, speed_da)
		plot_power_curve_mw = power_curve_mw
		if PLOT_CURVE_SMOOTH_SIGMA_POINTS > 0:
			plot_power_curve_mw = gaussian_filter1d(
				power_curve_mw,
				sigma=PLOT_CURVE_SMOOTH_SIGMA_POINTS,
				mode="nearest",
			)
		color = PALETTE[row_index % len(PALETTE)]
		max_power = max(max_power, float(np.nanmax(plot_power_curve_mw)))
		ax.plot(
			wind_speeds,
			plot_power_curve_mw,
			label=str(row["curve_label"]),
			color=color,
			linestyle="solid",
			linewidth=2.1,
			alpha=0.95,
			antialiased=True,
		)

	y_max = max(max_power * 1.05, 0.5)
	y_major_step, y_minor_step = _choose_y_tick_steps(y_max)
	ax.set_xlim(0, 40)
	ax.set_ylim(0, y_max)
	ax.xaxis.set_major_locator(MultipleLocator(X_MAJOR_STEP))
	ax.xaxis.set_minor_locator(MultipleLocator(X_MINOR_STEP))
	ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
	ax.yaxis.set_minor_locator(MultipleLocator(Y_MINOR_STEP))
	ax.grid(which="major", linestyle="--", linewidth=0.55, alpha=0.35, color="#8f8f8f")
	ax.grid(which="minor", linestyle="--", linewidth=0.55, alpha=0.22, color="#b8b8b8")
	ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)
	ax.tick_params(axis="x", which="minor", labelbottom=False)
	ax.tick_params(axis="y", which="minor", labelleft=False)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.set_xlabel("Wind speed (m/s)", fontsize=AXIS_LABEL_FONT_SIZE)
	ax.set_ylabel("Power (MW)", fontsize=AXIS_LABEL_FONT_SIZE)
	legend = ax.legend(
		loc="upper right",
		fontsize=LEGEND_FONT_SIZE,
		frameon=True,
		handlelength=2.8,
		labelspacing=0.6,
	)
	legend.get_frame().set_edgecolor("#C8C8C8")
	legend.get_frame().set_linewidth(0.8)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return output_path, plot_df


def main() -> None:
	output_path, plot_df = plot_selected_turbine_power_curves()
	print(f"Saved figure to: {output_path}")
	print(f"Selection JSON: {SELECTION_CONFIG_PATH}")
	print(f"Curves plotted: {len(plot_df)}")
	print("Selected turbine model matches:")
	for _, row in plot_df.iterrows():
		print(
			f"  {row['inventory_model']} ({float(row['inventory_capacity_mw']):.2f} MW; n={int(row['inventory_count'])}) -> "
			f"{row['reference_company']} | {row['reference_model']} | "
			f"similarity={float(row['string_similarity']):.4f}"
		)


if __name__ == "__main__":
	main()
