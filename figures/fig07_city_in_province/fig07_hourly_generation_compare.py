from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import xarray as xr


_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
	sys.path.insert(0, str(_HERE.parent))
import config_paths as _cfg  # noqa: E402

# Single source of truth for the 蒙东/蒙西 split and the study-wide 香港 exclusion.
# Do NOT re-declare these sets here (2026-10 audit: 47 hand-copied variants, 36 wrong).
_LIB = _HERE.parent / "_lib"
if str(_LIB) not in sys.path:
	sys.path.insert(0, str(_LIB))
import grid_map as _grid_map  # noqa: E402

PROJECT_ROOT = _HERE.parent

# Original defaults (paper_root = D:\HKU PhD\wind data paper):
#   no_kb      = <paper_root>/no_kb_gen/city
#   kb_adjust  = <paper_root>/farm_aggr_with_kb/city
#   final      = <paper_root>/final_gen/city
#
# plot_paper_v2 note: ``farm_aggr_with_kb/city`` was a cluster-only intermediate
# that is *not* part of the downloadable pipeline products, so ``kb_adjust`` now
# resolves through ``config_paths`` to its semantically equivalent successor,
# the KB-adjusted generation cube ``calibrated_gen/gen/city``.  Override with
# ``WINDPAPER_KB_ADJUST_ROOT`` (or ``--kb-adjust-root``).
DEFAULT_SOURCE_ROOTS = {
	"no_kb": _cfg.NO_KB_GEN_DIR,
	"kb_adjust": _cfg.KB_ADJUST_GEN_DIR,
	"final": _cfg.FINAL_GEN_DIR,
}
DEFAULT_OUTPUT_ROOT = _HERE / "output"
DEFAULT_HOURLY_START_YEAR = 2011
PLOT_FONT_SIZE = 16
SOURCE_ORDER = ["no_kb", "kb_adjust", "final"]
SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
SEASON_LABELS = {
	"spring": "Spring",
	"summer": "Summer",
	"autumn": "Autumn",
	"winter": "Winter",
}
SOURCE_LABELS = {
	"no_kb": "no_kb",
	"kb_adjust": "kb_adjust",
	"final": "final",
}
SOURCE_COLORS = {
	"no_kb": "#4E79A7",
	"kb_adjust": "#F28E2B",
	"final": "#59A14F",
}
PROVINCE_EARLIEST_NONZERO_START = {
	"新疆": pd.Timestamp("1989-10-01"),
	"内蒙古": pd.Timestamp("1990-01-01"),
}
# 蒙东电网口径（2026-10 用户修正）: only 赤峰市 / 通辽市 / 呼伦贝尔市 / 兴安盟.
# 锡林郭勒盟 (xilinguolei) is NOT part of 蒙东电网; it falls through to 蒙西
# (-> 华北电网) via ``_split_neimenggu``.  Do not add it back.
# Imported from the single source of truth ``_lib/grid_map.py``.
NEIMENGGU_EAST_CITY_KEYS = set(_grid_map.INNER_MONGOLIA_EAST_CITY_KEYS)

# 研究范围排除地区（2026-10 用户要求）: 香港 is not studied at all — no monthly or
# hourly panel, no legend entry, no ranking row.  Applied once, at load time, in
# ``_load_source_province_generation`` (and again when intersecting the city
# directories across sources).  Keys are *normalised* province labels
# (``_normalize_label`` strips 特别行政区, so "香港特别行政区" -> "香港").
# ``EXCLUDED_CITY_KEYS`` is the same set expressed as data-directory province
# keys (``<province_key>__<city_key>``), used to skip a directory *before* its
# netCDF is even opened.  Both come from ``_lib/grid_map.py``.
EXCLUDED_REGIONS = set(_grid_map.EXCLUDED_REGIONS)
EXCLUDED_CITY_KEYS = set(_grid_map.EXCLUDED_REGION_KEYS)

PROVINCE_PLOT_ORDER = [
	"北京",
	"天津",
	"河北",
	"山西",
	"山东",
	"辽宁",
	"吉林",
	"黑龙江",
	"蒙东",
	"上海",
	"江苏",
	"浙江",
	"安徽",
	"福建",
	"河南",
	"湖北",
	"湖南",
	"江西",
	"陕西",
	"甘肃",
	"青海",
	"宁夏",
	"新疆",
	"四川",
	"重庆",
	"西藏",
	"广东",
	"广西",
	"云南",
	"贵州",
	"海南",
	# "香港": 已按 EXCLUDED_REGIONS 整体排除出本研究，不再出图。
	"蒙西",
]


def _open_dataset_with_fallback(path: str | Path, retries: int = 2, sleep_seconds: float = 0.6) -> xr.Dataset:
	resolved_path = Path(path)
	last_error: Exception | None = None
	for attempt in range(retries + 1):
		try:
			with xr.open_dataset(str(resolved_path)) as ds:
				return ds.load()
		except Exception as exc:  # noqa: BLE001
			last_error = exc
			if attempt < retries:
				time.sleep(sleep_seconds * (attempt + 1))

	try:
		with xr.open_dataset(str(resolved_path), engine="scipy") as ds:
			return ds.load()
	except Exception as exc:  # noqa: BLE001
		raise OSError(f"Failed to open NetCDF file: {resolved_path}; last_error={last_error}; scipy_error={exc}") from exc


def _normalize_label(value: Any) -> str:
	text = str(value).strip()
	for suffix in ("维吾尔自治区", "壮族自治区", "回族自治区", "自治区", "特别行政区", "自治州", "地区", "盟", "省", "市"):
		if text.endswith(suffix) and len(text) > len(suffix):
			text = text[: -len(suffix)]
			break
	return text.strip()


def _safe_slug(text: str) -> str:
	return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _month_to_season(month: int) -> str:
	if 3 <= month <= 5:
		return "spring"
	if 6 <= month <= 8:
		return "summer"
	if 9 <= month <= 11:
		return "autumn"
	return "winter"


def _season_year(timestamp: pd.Timestamp) -> int:
	return int(timestamp.year) - 1 if int(timestamp.month) in {1, 2} else int(timestamp.year)


def _discover_city_nc_files(city_root: str | Path) -> list[Path]:
	city_root = Path(city_root)
	if not city_root.exists():
		raise FileNotFoundError(f"city_root does not exist: {city_root}")
	paths = sorted(city_root.glob("*/full_year/gen.nc"))
	if not paths:
		raise FileNotFoundError(f"No city full_year gen.nc files found under: {city_root}")
	return paths


def _discover_city_dir_names(city_root: str | Path) -> set[str]:
	city_root = Path(city_root)
	if not city_root.exists():
		# Optional source: absent directories are skipped by the caller rather
		# than aborting the figure.  Never raise from here.
		return set()
	return {
		path.name
		for path in city_root.iterdir()
		if path.is_dir()
		and (path / "full_year" / "gen.nc").exists()
		and _normalize_label(path.name.split("__", maxsplit=1)[0]) not in (EXCLUDED_REGIONS | EXCLUDED_CITY_KEYS)
	}


def _split_neimenggu(city_key: str) -> tuple[str, str]:
	if city_key in NEIMENGGU_EAST_CITY_KEYS:
		return "mengdong", "蒙东"
	return "mengxi", "蒙西"


def _resolve_province_group(province_key: str, province_label: str, city_key: str) -> tuple[str, str]:
	if province_key == "neimenggu" or province_label == "内蒙古":
		return _split_neimenggu(city_key)
	return province_key, province_label


def _province_earliest_nonzero_start(province_label: str) -> pd.Timestamp | None:
	normalized = _normalize_label(province_label)
	return PROVINCE_EARLIEST_NONZERO_START.get(normalized)


def _sort_province_labels(labels: list[str]) -> list[str]:
	order_map = {label: index for index, label in enumerate(PROVINCE_PLOT_ORDER)}
	return sorted(labels, key=lambda label: (order_map.get(label, len(order_map)), label))


def _empty_generation_df() -> pd.DataFrame:
	return pd.DataFrame(columns=["source", "province_label", "time", "generation_mwh"])


def _finalize_generation_df(source_key: str, province_series_map: dict[str, pd.Series]) -> pd.DataFrame:
	if not province_series_map:
		return _empty_generation_df()

	records: list[pd.DataFrame] = []
	for province_label in _sort_province_labels(list(province_series_map)):
		province_series = province_series_map[province_label].sort_index()
		frame = province_series.rename("generation_mwh").reset_index()
		frame.columns = ["time", "generation_mwh"]
		frame["province_label"] = province_label
		frame["source"] = source_key
		records.append(frame)

	result = pd.concat(records, ignore_index=True)
	result["time"] = pd.to_datetime(result["time"])
	result["generation_mwh"] = pd.to_numeric(result["generation_mwh"], errors="coerce").fillna(0.0)
	return result[["source", "province_label", "time", "generation_mwh"]]


def _load_source_province_generation(
	source_key: str,
	city_root: str | Path,
	*,
	allowed_city_dirs: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	province_hourly_series_map: dict[str, pd.Series] = {}
	province_monthly_series_map: dict[str, pd.Series] = {}

	for nc_path in _discover_city_nc_files(city_root):
		city_dir = nc_path.parent.parent.name
		if allowed_city_dirs is not None and city_dir not in allowed_city_dirs:
			continue
		province_key, city_key = city_dir.split("__", maxsplit=1)
		ds = _open_dataset_with_fallback(nc_path)
		if "total_gen" not in ds.data_vars or "time" not in ds["total_gen"].dims:
			raise KeyError(f"dataset missing hourly total_gen on time dimension: {nc_path}")

		province_label_raw = ds.attrs.get("province_short_name", province_key)
		province_label = _normalize_label(province_label_raw)
		# 研究范围过滤（唯一入口）: 香港 整体不入研究，不出任何月/小时面板与图例。
		# Checked both on the normalised Chinese label and on the directory key so
		# the exclusion cannot be bypassed by a relabelled file.
		if province_label in EXCLUDED_REGIONS or province_key in EXCLUDED_CITY_KEYS:
			continue
		_province_group_key, province_group_label = _resolve_province_group(
			province_key=province_key,
			province_label=province_label,
			city_key=city_key,
		)

		earliest_ts = _province_earliest_nonzero_start(province_label)
		hourly_series = ds["total_gen"].to_series()
		hourly_series.index = pd.to_datetime(hourly_series.index)
		hourly_series = pd.to_numeric(hourly_series, errors="coerce").fillna(0.0)
		if earliest_ts is not None:
			hourly_series.loc[hourly_series.index < earliest_ts] = 0.0

		existing_hourly = province_hourly_series_map.get(province_group_label)
		province_hourly_series_map[province_group_label] = hourly_series if existing_hourly is None else existing_hourly.add(hourly_series, fill_value=0.0)

		monthly_series = hourly_series.resample("MS").sum(min_count=1).fillna(0.0)

		existing_monthly = province_monthly_series_map.get(province_group_label)
		province_monthly_series_map[province_group_label] = monthly_series if existing_monthly is None else existing_monthly.add(monthly_series, fill_value=0.0)

	if not province_monthly_series_map:
		raise ValueError(f"No generation records were loaded for source: {source_key}")

	hourly_df = _finalize_generation_df(source_key, province_hourly_series_map)
	monthly_df = _finalize_generation_df(source_key, province_monthly_series_map)
	return hourly_df, monthly_df


def _write_figure(figure: go.Figure, output_path: str | Path, figsize: tuple[float, float]) -> None:
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	width = int(figsize[0] * 100)
	height = int(figsize[1] * 100)
	html_path = output_path.with_suffix(".html")
	png_path = output_path.with_suffix(".png")
	figure.write_html(str(html_path), include_plotlyjs="cdn")
	figure.write_image(str(png_path), width=width, height=height, scale=2)


def _base_axis_style() -> dict[str, Any]:
	return {
		"showline": True,
		"linewidth": 1.0,
		"linecolor": "#444444",
		"ticks": "outside",
		"ticklen": 5,
		"tickwidth": 1,
		"tickcolor": "#444444",
		"showticklabels": True,
		"automargin": True,
		"zeroline": False,
	}


def _legend_config() -> dict[str, Any]:
	return {
		"orientation": "h",
		"yanchor": "bottom",
		"y": 1.02,
		"xanchor": "left",
		"x": 0.0,
		"font": {"size": PLOT_FONT_SIZE - 1},
	}


def _build_time_axis(index: pd.Index, tick_format: str) -> dict[str, Any]:
	return {
		"title": "Time",
		"range": [pd.Timestamp(index.min()), pd.Timestamp(index.max())],
		"tickformat": tick_format,
		"showgrid": True,
		"gridcolor": "#ececec",
		"gridwidth": 0.4,
		"griddash": "dash",
		**_base_axis_style(),
	}


def _build_generation_axis(df: pd.DataFrame, value_col: str = "generation_gwh") -> dict[str, Any]:
	max_value = float(df[value_col].max()) if not df.empty else 0.0
	return {
		"title": "Generation (GWh)",
		"rangemode": "tozero",
		"range": [0, max_value * 1.08 if max_value > 0 else 1],
		"showgrid": True,
		"gridcolor": "#e9e9e9",
		"gridwidth": 0.45,
		"griddash": "dash",
		**_base_axis_style(),
	}


def _plot_source_compare(
	plot_df: pd.DataFrame,
	*,
	province_label: str,
	plot_title: str,
	output_path: str | Path,
	x_tick_format: str,
	figsize: tuple[float, float],
) -> None:
	if plot_df.empty:
		return

	figure = go.Figure()
	for source_key in SOURCE_ORDER:
		source_df = plot_df[plot_df["source"] == source_key].sort_values("time")
		if source_df.empty:
			continue
		figure.add_trace(
			go.Scatter(
				x=source_df["time"],
				y=source_df["generation_gwh"],
				mode="lines",
				name=SOURCE_LABELS[source_key],
				line={"color": SOURCE_COLORS[source_key], "width": 2.0},
				hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + SOURCE_LABELS[source_key] + ": %{y:,.2f} GWh<extra></extra>",
			)
		)

	figure.update_layout(
		template="plotly_white",
		width=int(figsize[0] * 100),
		height=int(figsize[1] * 100),
		hovermode="x unified",
		font={"family": "Arial, Microsoft YaHei, SimHei", "size": PLOT_FONT_SIZE},
		margin={"l": 90, "r": 40, "t": 70, "b": 80},
		legend=_legend_config(),
		title={"text": f"{province_label}: {plot_title}", "x": 0.01, "xanchor": "left"},
		xaxis=_build_time_axis(plot_df["time"], x_tick_format),
		yaxis=_build_generation_axis(plot_df),
	)
	_write_figure(figure, output_path, figsize)


def generate_hourly_generation_compare_plots(
	*,
	source_roots: dict[str, str | Path] | None = None,
	output_root: str | Path = DEFAULT_OUTPUT_ROOT,
	hourly_start_year: int = DEFAULT_HOURLY_START_YEAR,
) -> dict[str, Any]:
	resolved_source_roots = dict(source_roots if source_roots is not None else DEFAULT_SOURCE_ROOTS)
	output_root = Path(output_root)
	output_root.mkdir(parents=True, exist_ok=True)

	# Robustness: a source whose directory is absent is *skipped* with a warning.
	# Every comparison source except the first is optional, so a missing one must
	# never abort the figure (the original code raised FileNotFoundError here).
	resolved_source_roots, missing_source_roots = _cfg.select_existing_roots(resolved_source_roots)
	if missing_source_roots:
		print("[fig07] WARNING: skipping source(s) whose directory does not exist:")
		for source_key, city_root in missing_source_roots.items():
			print(f"[fig07]   - {source_key}: {city_root}")
	if not resolved_source_roots:
		raise FileNotFoundError(
			"None of the requested generation sources exist on disk: "
			+ ", ".join(f"{key}={path}" for key, path in missing_source_roots.items())
		)

	monthly_output_root = output_root / "monthly_source_compare"
	monthly_output_root.mkdir(parents=True, exist_ok=True)
	hourly_output_root = output_root / "hourly_province_year_season"
	hourly_output_root.mkdir(parents=True, exist_ok=True)

	hourly_frames: list[pd.DataFrame] = []
	monthly_frames: list[pd.DataFrame] = []
	hourly_available_sources: list[str] = []
	city_dir_names_by_source = {
		source_key: _discover_city_dir_names(city_root)
		for source_key, city_root in resolved_source_roots.items()
	}
	common_city_dirs = set.intersection(*city_dir_names_by_source.values()) if city_dir_names_by_source else set()
	if not common_city_dirs:
		raise ValueError("No common city full_year generation outputs were found across the selected sources")

	for source_key, city_root in resolved_source_roots.items():
		hourly_df, monthly_df = _load_source_province_generation(
			source_key,
			city_root,
			allowed_city_dirs=common_city_dirs,
		)
		monthly_frames.append(monthly_df)
		if not hourly_df.empty:
			hourly_frames.append(hourly_df)
			hourly_available_sources.append(source_key)

	all_hourly = pd.concat(hourly_frames, ignore_index=True) if hourly_frames else _empty_generation_df()
	all_hourly["generation_gwh"] = all_hourly["generation_mwh"] / 1000.0
	monthly_df = pd.concat(monthly_frames, ignore_index=True)
	monthly_df["generation_gwh"] = monthly_df["generation_mwh"] / 1000.0

	# 口径安全网（不改变任何计算，仅断言排除生效）：香港 一旦泄漏进来就立刻失败。
	_leaked = sorted(set(monthly_df["province_label"].astype(str)) & EXCLUDED_REGIONS)
	if _leaked:
		raise AssertionError(f"EXCLUDED_REGIONS leaked into the dataset: {_leaked}")
	_leaked_dirs = sorted(d for d in common_city_dirs if d.split("__", maxsplit=1)[0] in EXCLUDED_CITY_KEYS)
	if _leaked_dirs:
		raise AssertionError(f"EXCLUDED_CITY_KEYS leaked into the common city dirs: {_leaked_dirs}")

	provinces = _sort_province_labels(sorted(monthly_df["province_label"].dropna().astype(str).unique().tolist()))
	monthly_plot_count = 0
	hourly_plot_count = 0

	for province_label in provinces:
		province_monthly_df = monthly_df[monthly_df["province_label"] == province_label].copy()
		_plot_source_compare(
			province_monthly_df,
			province_label=province_label,
			plot_title="Monthly Generation Comparison",
			output_path=monthly_output_root / f"{_safe_slug(province_label)}_monthly_generation_compare",
			x_tick_format="%Y-%m",
			figsize=(15, 7),
		)
		if not province_monthly_df.empty:
			monthly_plot_count += 1

		if all_hourly.empty:
			continue
		province_hourly_df = all_hourly[all_hourly["province_label"] == province_label].copy()
		if province_hourly_df.empty:
			continue
		province_hourly_df["year"] = province_hourly_df["time"].map(_season_year)
		province_hourly_df["season"] = province_hourly_df["time"].dt.month.map(_month_to_season)
		province_hourly_df = province_hourly_df[province_hourly_df["year"] >= hourly_start_year].copy()
		if province_hourly_df.empty:
			continue

		province_dir = hourly_output_root / _safe_slug(province_label)
		province_dir.mkdir(parents=True, exist_ok=True)
		for year in sorted(province_hourly_df["year"].unique().tolist()):
			for season in SEASON_ORDER:
				season_df = province_hourly_df[
					(province_hourly_df["year"] == year) & (province_hourly_df["season"] == season)
				].copy()
				if season_df.empty:
					continue
			season_df = season_df.copy().sort_values("time")
			_plot_source_compare(
				season_df,
				province_label=province_label,
				plot_title=f"Hourly Generation Comparison ({year} {SEASON_LABELS[season]})",
				output_path=province_dir / f"{year}_{season}_hourly_generation_compare",
				x_tick_format="%m-%d",
				figsize=(16, 7),
			)
			hourly_plot_count += 1

	return {
		"output_root": str(output_root),
		"monthly_output_root": str(monthly_output_root),
		"hourly_output_root": str(hourly_output_root),
		"source_roots": {key: str(value) for key, value in resolved_source_roots.items()},
		"skipped_source_roots": {key: str(value) for key, value in missing_source_roots.items()},
		"common_city_dir_count": len(common_city_dirs),
		"excluded_city_dirs_by_source": {
			key: sorted(city_dir_names_by_source[key] - common_city_dirs)
			for key in resolved_source_roots
		},
		"province_count": len(provinces),
		"monthly_plot_count": monthly_plot_count,
		"hourly_plot_count": hourly_plot_count,
		"hourly_available_sources": hourly_available_sources,
		"hourly_missing_sources": [key for key in SOURCE_ORDER if key in resolved_source_roots and key not in hourly_available_sources],
		"hourly_start_year": int(hourly_start_year),
	}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compare provincial monthly and hourly generation across no_kb, kb_adjust, and final outputs")
	parser.add_argument("--no-kb-root", type=Path, default=DEFAULT_SOURCE_ROOTS["no_kb"], help="Path to no_kb_gen/city")
	parser.add_argument(
		"--kb-adjust-root",
		type=Path,
		default=DEFAULT_SOURCE_ROOTS["kb_adjust"],
		help="Path to the KB-adjusted generation cube; defaults to calibrated_gen/gen/city "
		f"(the successor of the retired farm_aggr_with_kb/city, override with {_cfg.WINDPAPER_KB_ADJUST_ROOT})",
	)
	parser.add_argument("--final-root", type=Path, default=DEFAULT_SOURCE_ROOTS["final"], help="Path to final_gen/city")
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory to store outputs")
	parser.add_argument("--hourly-start-year", type=int, default=DEFAULT_HOURLY_START_YEAR, help="Start year for yearly hourly plots")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	result = generate_hourly_generation_compare_plots(
		source_roots={
			"no_kb": args.no_kb_root,
			"kb_adjust": args.kb_adjust_root,
			"final": args.final_root,
		},
		output_root=args.output_root,
		hourly_start_year=args.hourly_start_year,
	)
	print(result)


if __name__ == "__main__":
	main()