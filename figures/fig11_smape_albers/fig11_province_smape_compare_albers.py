from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer
import pandas as pd
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.patheffects as patheffects


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "_lib") not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT / "_lib"))

import config_paths as _cfg  # noqa: E402

# Vendored copy of `code.indices.smape` (byte-identical to
# D:\HKU PhD\wind data paper\CalculationCode\indices.py; the `code` package did
# not ship with the plot directory).
from indices import smape  # noqa: E402


# Original defaults:
#   DEFAULT_NO_KB_ROOT   = <paper_root>/no_kb_gen/city
#   DEFAULT_CALIBRATED_ROOT = <paper_root>/calibrated_gen/city
#   DEFAULT_FINAL_ROOT   = <paper_root>/final_gen/city
DEFAULT_NO_KB_ROOT = _cfg.NO_KB_GEN_DIR
DEFAULT_CALIBRATED_ROOT = _cfg.CALIBRATED_GEN_DIR
DEFAULT_FINAL_ROOT = _cfg.FINAL_GEN_DIR
DEFAULT_PROVINCE_GEOJSON = _cfg.resolve_admin_geojson("中国_省.geojson")
DEFAULT_CITY_GEOJSON = _cfg.resolve_admin_geojson("中国_市.geojson")
DEFAULT_NINE_DOTTED_LINE_PATH = _cfg.resolve_nine_dotted_line()
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "output" / "fig11_province_smape_compare_albers.png"
DEFAULT_TABLE_PATH = SCRIPT_DIR / "output" / "fig11_province_smape_compare_albers.csv"
MISSING_COLOR = "#D9D9D9"

ALBERS_PROJ4 = (
	"+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
	"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
ALBERS_CRS = CRS.from_proj4(ALBERS_PROJ4)
LL_TO_ALBERS = Transformer.from_crs("EPSG:4326", ALBERS_CRS, always_xy=True)

MAINLAND_BOUNDS_LL = (73.0, 136.0, 17.0, 54.5)
PANEL_BOUNDS_LL = (73.0, 137.0, 17.0, 54.5)
SOUTH_CHINA_SEA_BOUNDS_LL = (105.0, 125.0, 3.0, 25.5)


def _project_bounds(min_lon: float, max_lon: float, min_lat: float, max_lat: float, samples: int = 361) -> tuple[float, float, float, float]:
	lon_edge = np.linspace(min_lon, max_lon, samples)
	lat_edge = np.linspace(min_lat, max_lat, samples)
	edge_lon = np.concatenate([lon_edge, lon_edge, np.full(samples, min_lon), np.full(samples, max_lon)])
	edge_lat = np.concatenate([np.full(samples, min_lat), np.full(samples, max_lat), lat_edge, lat_edge])
	xs, ys = LL_TO_ALBERS.transform(edge_lon, edge_lat)
	return float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys))


MAINLAND_BOUNDS = _project_bounds(*MAINLAND_BOUNDS_LL)

# 面板的纵向范围：实测中国版图在 MAINLAND 框内的可见 y 为 1740529–5921486，
# 而投影外接矩形上界到 6250118，导致上方多出 329 km 空白、下方为 0。
# 这里保持框高不变、整体平移，使版图在子图内上下留白相等（各约 164 km）。
PANEL_YLIM = (1576213.0, 5951486.0)
PANEL_BOUNDS = _project_bounds(*PANEL_BOUNDS_LL)
SOUTH_CHINA_SEA_BOUNDS = _project_bounds(*SOUTH_CHINA_SEA_BOUNDS_LL)
SOUTH_CHINA_SEA_INSET_RECT = (0.80, 0.05, 0.16, 0.26)
NO_CALIBRATION_PROVINCES = {"香港", "澳门", "台湾", "西藏"}
NO_CALIBRATION_PROVINCE_CODES = {"54", "71", "81", "82"}
_LON_DEG_IN_M = 85270.0
_LAT_DEG_IN_M = 110950.0
LABEL_OFFSETS = {
	"北京": (-0.6 * _LON_DEG_IN_M, 0.45 * _LAT_DEG_IN_M),
	"天津": (0.55 * _LON_DEG_IN_M, 0.0),
	"河北": (0.0, -1.0 * _LAT_DEG_IN_M),
}


plt.rcParams["font.family"] = ["Arial", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14.0
plt.rcParams["font.weight"] = "normal"

PANEL_TITLE_FONTSIZE = 18.5
COLORBAR_LABEL_FONTSIZE = 16.0
COLORBAR_TICK_FONTSIZE = 14.0
MAP_VALUE_FONTSIZE = 13.6
LEGEND_FONTSIZE = 13.0


def _build_smape_colormap() -> LinearSegmentedColormap:
	# Single-hue sequential ramp with monotonic lightness: lower SMAPE always looks lighter.
	return LinearSegmentedColormap.from_list(
		"smape_single_hue",
		["#fffaf0", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"],
	)


def _build_delta_colormap() -> LinearSegmentedColormap:
	return LinearSegmentedColormap.from_list(
		"smape_delta",
		["#f7fcf5", "#d4efcf", "#92d28b", "#45ad5c", "#127a39", "#00441b"],
	)


def _build_bias_colormap() -> LinearSegmentedColormap:
	return LinearSegmentedColormap.from_list(
		"bias_diverging",
		["#7f2704", "#fdd49e", "#f7f7f7", "#c7e9c0", "#238b45"],
	)


def _open_dataset_with_fallback(path: str | Path, retries: int = 2, sleep_seconds: float = 0.6) -> xr.Dataset:
	resolved_path = Path(path)
	last_error: Exception | None = None
	for attempt in range(retries + 1):
		try:
			with xr.open_dataset(str(resolved_path)) as dataset:
				return dataset.load()
		except Exception as exc:  # noqa: BLE001
			last_error = exc
			if attempt < retries:
				time.sleep(sleep_seconds * (attempt + 1))

	try:
		with xr.open_dataset(str(resolved_path), engine="scipy") as dataset:
			return dataset.load()
	except Exception as exc:  # noqa: BLE001
		raise OSError(f"Failed to open NetCDF file: {resolved_path}; last_error={last_error}; scipy_error={exc}") from exc


def _normalize_label(value: object) -> str:
	text = str(value).strip()
	for suffix in ("维吾尔自治区", "壮族自治区", "回族自治区", "自治区", "特别行政区", "自治州", "地区", "盟", "省", "市"):
		if text.endswith(suffix) and len(text) > len(suffix):
			text = text[: -len(suffix)]
			break
	return text.strip()


def _province_code_from_gb(value: object) -> str:
	text = str(value).strip()
	if len(text) >= 5:
		return text[3:5]
	return ""


def _discover_city_nc_files(city_root: str | Path) -> list[Path]:
	city_root = Path(city_root)
	if not city_root.exists():
		raise FileNotFoundError(f"city_root does not exist: {city_root}")
	paths = sorted(city_root.glob("*__*/full_year/gen.nc"))
	if not paths:
		raise FileNotFoundError(f"No city full_year gen.nc files found under: {city_root}")
	return paths


def _monthly_series_from_dataset(dataset: xr.Dataset, source_path: Path) -> pd.Series:
	if "total_monthly_gen" in dataset.data_vars:
		monthly = dataset["total_monthly_gen"].to_series()
		monthly.index = pd.to_datetime(monthly.index)
		return monthly.sort_index().fillna(0.0)

	if "total_gen" not in dataset.data_vars or "time" not in dataset["total_gen"].dims:
		raise KeyError(f"dataset missing total_monthly_gen and hourly total_gen: {source_path}")

	hourly = dataset["total_gen"].to_series()
	hourly.index = pd.to_datetime(hourly.index)
	return hourly.resample("MS").sum(min_count=1).fillna(0.0)


def _load_province_monthly(root: str | Path) -> pd.DataFrame:
	records: list[pd.DataFrame] = []
	for nc_path in _discover_city_nc_files(root):
		dataset = _open_dataset_with_fallback(nc_path)
		province_label = _normalize_label(dataset.attrs.get("province_short_name", nc_path.parent.parent.name.split("__", maxsplit=1)[0]))
		monthly_series = _monthly_series_from_dataset(dataset, nc_path)
		frame = monthly_series.rename("generation_mwh").reset_index()
		frame.columns = ["time", "generation_mwh"]
		frame["province_label"] = province_label
		records.append(frame)

	if not records:
		raise ValueError(f"No monthly records loaded from {root}")

	monthly_long = pd.concat(records, ignore_index=True)
	monthly_long["time"] = pd.to_datetime(monthly_long["time"])
	monthly_long["generation_mwh"] = pd.to_numeric(monthly_long["generation_mwh"], errors="coerce").fillna(0.0)
	province_monthly = (
		monthly_long.groupby(["province_label", "time"], as_index=False)["generation_mwh"]
		.sum()
		.sort_values(["province_label", "time"])
		.reset_index(drop=True)
	)
	return province_monthly


def _compute_province_smape(pred_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	ref_groups = {province: frame[["time", "generation_mwh"]].copy() for province, frame in ref_df.groupby("province_label", sort=True)}
	for province_label, pred_group in pred_df.groupby("province_label", sort=True):
		ref_group = ref_groups.get(province_label)
		if ref_group is None:
			rows.append({"province_label": province_label, "smape": pd.NA})
			continue
		aligned = pred_group[["time", "generation_mwh"]].merge(
			ref_group,
			on="time",
			how="inner",
			suffixes=("_pred", "_ref"),
		)
		if aligned.empty:
			rows.append({"province_label": province_label, "smape": pd.NA})
			continue
		rows.append(
			{
				"province_label": province_label,
				"smape": float(smape(aligned["generation_mwh_ref"].to_numpy(), aligned["generation_mwh_pred"].to_numpy())),
			}
		)

	ref_only_provinces = sorted(set(ref_groups) - set(pred_df["province_label"].unique()))
	for province_label in ref_only_provinces:
		rows.append({"province_label": province_label, "smape": pd.NA})

	return pd.DataFrame(rows).drop_duplicates(subset=["province_label"]).sort_values("province_label").reset_index(drop=True)


def _compute_province_bias_ratio(pred_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	ref_groups = {province: frame[["time", "generation_mwh"]].copy() for province, frame in ref_df.groupby("province_label", sort=True)}
	for province_label, pred_group in pred_df.groupby("province_label", sort=True):
		ref_group = ref_groups.get(province_label)
		if ref_group is None:
			rows.append({"province_label": province_label, "bias_ratio": pd.NA})
			continue
		aligned = pred_group[["time", "generation_mwh"]].merge(
			ref_group,
			on="time",
			how="inner",
			suffixes=("_pred", "_ref"),
		)
		if aligned.empty:
			rows.append({"province_label": province_label, "bias_ratio": pd.NA})
			continue
		total_pred = float(aligned["generation_mwh_pred"].sum())
		total_ref = float(aligned["generation_mwh_ref"].sum())
		if abs(total_ref) < 1e-12:
			bias_ratio = 0.0 if abs(total_pred) < 1e-12 else pd.NA
		else:
			bias_ratio = float(100.0 * total_pred / total_ref)
		rows.append({"province_label": province_label, "bias_ratio": bias_ratio})

	ref_only_provinces = sorted(set(ref_groups) - set(pred_df["province_label"].unique()))
	for province_label in ref_only_provinces:
		rows.append({"province_label": province_label, "bias_ratio": pd.NA})

	return pd.DataFrame(rows).drop_duplicates(subset=["province_label"]).sort_values("province_label").reset_index(drop=True)


def _read_province_map(geojson_path: str | Path) -> gpd.GeoDataFrame:
	province_map = gpd.read_file(geojson_path)
	if province_map.crs is None:
		province_map = province_map.set_crs("EPSG:4326")
	province_map = province_map.to_crs(ALBERS_CRS)
	province_map["name"] = province_map["name"].fillna("").astype(str).str.strip()
	province_map = province_map.loc[province_map["name"].ne("境界线")].copy()
	province_map["province_label"] = province_map["name"].map(_normalize_label)
	return province_map


def _read_city_map(geojson_path: str | Path) -> gpd.GeoDataFrame:
	city_map = gpd.read_file(geojson_path)
	city_map["name"] = city_map["name"].fillna("").astype(str).str.strip()
	city_map = city_map.loc[city_map["name"].ne("境界线")].copy()
	city_map["province_code"] = city_map["gb"].map(_province_code_from_gb)
	city_map = city_map.loc[~city_map["province_code"].isin(NO_CALIBRATION_PROVINCE_CODES)].copy()
	if city_map.crs is None:
		city_map = city_map.set_crs("EPSG:4326")
	return city_map.to_crs(ALBERS_CRS)


def _read_nine_dotted_line(shp_path: str | Path) -> gpd.GeoDataFrame | None:
	shp_path = Path(shp_path)
	if not shp_path.exists():
		return None
	nine_dotted_line = gpd.read_file(shp_path)
	if nine_dotted_line.crs is None:
		nine_dotted_line = nine_dotted_line.set_crs("EPSG:4326")
	return nine_dotted_line.to_crs(ALBERS_CRS)


def _build_export_table(
	province_map: gpd.GeoDataFrame,
	before_df: pd.DataFrame,
	after_df: pd.DataFrame,
	bias_before_df: pd.DataFrame,
	bias_after_df: pd.DataFrame,
) -> pd.DataFrame:
	table_df = province_map[["name", "province_label"]].drop_duplicates().copy()
	table_df = table_df.merge(before_df.rename(columns={"smape": "smape_before"}), on="province_label", how="left")
	table_df = table_df.merge(after_df.rename(columns={"smape": "smape_after"}), on="province_label", how="left")
	table_df = table_df.merge(bias_before_df.rename(columns={"bias_ratio": "bias_before"}), on="province_label", how="left")
	table_df = table_df.merge(bias_after_df.rename(columns={"bias_ratio": "bias_after"}), on="province_label", how="left")
	masked = table_df["province_label"].isin(NO_CALIBRATION_PROVINCES)
	for column_name in ("smape_before", "smape_after", "bias_before", "bias_after"):
		table_df.loc[masked, column_name] = pd.NA
	table_df["smape_delta"] = pd.to_numeric(table_df["smape_before"], errors="coerce") - pd.to_numeric(table_df["smape_after"], errors="coerce")
	table_df["bias_delta"] = (pd.to_numeric(table_df["bias_after"], errors="coerce") - pd.to_numeric(table_df["bias_before"], errors="coerce")).abs()
	table_df["has_before_value"] = table_df["smape_before"].notna()
	table_df["has_after_value"] = table_df["smape_after"].notna()
	return table_df.sort_values("province_label").reset_index(drop=True)


def _write_export_table(table_df: pd.DataFrame, table_path: str | Path) -> None:
	table_path = Path(table_path)
	table_path.parent.mkdir(parents=True, exist_ok=True)
	table_df.to_csv(table_path, index=False, encoding="utf-8-sig")


def _nice_upper_bound(value: float) -> float:
	if value <= 0:
		return 1.0
	if value <= 10:
		return float(int(value + 0.9999))
	exponent = len(str(int(value))) - 1
	base = 10 ** max(exponent, 0)
	fraction = value / base
	if fraction <= 1:
		nice_fraction = 1
	elif fraction <= 2:
		nice_fraction = 2
	elif fraction <= 5:
		nice_fraction = 5
	else:
		nice_fraction = 10
	return float(nice_fraction * base)


def _build_colorbar_config(value_series: pd.Series) -> tuple[mcolors.Normalize, list[float], list[float], float]:
	combined = pd.to_numeric(value_series, errors="coerce").dropna()
	vmax = _nice_upper_bound(float(combined.max())) if not combined.empty else 10.0
	if vmax <= 20:
		step = 5.0
	elif vmax <= 60:
		step = 10.0
	elif vmax <= 120:
		step = 20.0
	elif vmax <= 300:
		step = 50.0
	else:
		step = 100.0
	ticks: list[float] = []
	current = 0.0
	while current <= vmax + 1e-9:
		ticks.append(float(current))
		current += step
	if abs(ticks[-1] - vmax) > 1e-9:
		ticks.append(vmax)
	bounds = ticks.copy()
	return mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax), ticks, bounds, vmax


def _build_diverging_colorbar_config(value_series: pd.Series) -> tuple[mcolors.Normalize, list[float], float]:
	valid_values = pd.to_numeric(value_series, errors="coerce").dropna()
	max_abs = _nice_upper_bound(float(valid_values.abs().max())) if not valid_values.empty else 10.0
	if max_abs <= 20:
		step = 5.0
	elif max_abs <= 50:
		step = 10.0
	else:
		step = 20.0
	ticks: list[float] = []
	current = -max_abs
	while current <= max_abs + 1e-9:
		ticks.append(float(current))
		current += step
	if 0.0 not in ticks:
		ticks.append(0.0)
	return mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs), sorted(set(ticks)), max_abs


def _build_fixed_ratio_colorbar_config() -> tuple[mcolors.Normalize, list[float], float]:
	return mcolors.TwoSlopeNorm(vmin=0.0, vcenter=100.0, vmax=300.0), [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0], 300.0


def _build_fixed_delta200_colorbar_config() -> tuple[mcolors.Normalize, list[float], float]:
	return mcolors.TwoSlopeNorm(vmin=-150.0, vcenter=0.0, vmax=150.0), [-150.0, -75.0, 0.0, 75.0, 150.0], 150.0


def _subset_by_bounds(gdf: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
	min_lon, max_lon, min_lat, max_lat = bounds
	geom_bounds = gdf.geometry.bounds
	mask = (
		(geom_bounds["maxx"] >= min_lon)
		& (geom_bounds["minx"] <= max_lon)
		& (geom_bounds["maxy"] >= min_lat)
		& (geom_bounds["miny"] <= max_lat)
	)
	return gdf.loc[mask].copy()


def _build_facecolors(values: pd.Series, colormap: mcolors.Colormap, norm: mcolors.Normalize) -> list[str]:
	facecolors: list[str] = []
	for value in values:
		if pd.isna(value):
			facecolors.append(MISSING_COLOR)
		else:
			facecolors.append(mcolors.to_hex(colormap(norm(float(value)))))
	return facecolors


def _add_city_boundaries(axis: plt.Axes, city_map: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> None:
	city_subset = _subset_by_bounds(city_map, bounds)
	if city_subset.empty:
		return
	city_subset.boundary.plot(
		ax=axis,
		color="#8A8A8A",
		linewidth=0.25,
		linestyle=(0, (2, 2)),
		zorder=1.05,
	)


def _attach_no_data_legend(axis: plt.Axes) -> None:
	legend_patch = Patch(facecolor=MISSING_COLOR, edgecolor="#6E6E6E", linewidth=0.5, label="Data area without calibration")
	axis.legend(
		handles=[legend_patch],
		loc="lower left",
		bbox_to_anchor=(0.002, 0.002),
		frameon=False,
		fontsize=LEGEND_FONTSIZE,
		handlelength=1.15,
		handletextpad=0.3,
		borderaxespad=0.0,
		borderpad=0.15,
	)


def _add_panel_colorbar(
	figure: plt.Figure,
	axis: plt.Axes,
	norm: mcolors.Normalize,
	colormap: mcolors.Colormap,
	ticks: list[float],
	label: str,
) -> None:
	axis_pos = axis.get_position()
	colorbar_ax = figure.add_axes([
		axis_pos.x1 + 0.0012,
		axis_pos.y0 + 0.05 * axis_pos.height,
		0.012,
		0.87 * axis_pos.height,
	])
	scalar_mappable = ScalarMappable(norm=norm, cmap=colormap)
	scalar_mappable.set_array([])
	colorbar = figure.colorbar(scalar_mappable, cax=colorbar_ax, orientation="vertical")
	colorbar.set_ticks(ticks)
	colorbar.set_label(label, fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=8)
	colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
	colorbar.outline.set_linewidth(0.6)


PANEL_LABEL_FONTSIZE = 9.0


def _add_panel_label(axis: plt.Axes, label: str) -> None:
	"""ESSD style panel label: Arial 9 pt, "(a)" in the panel corner."""
	axis.text(
		0.012,
		0.988,
		label,
		transform=axis.transAxes,
		ha="left",
		va="top",
		fontsize=PANEL_LABEL_FONTSIZE,
		fontfamily="Arial",
		color="black",
		zorder=10,
		clip_on=False,
	)


def _add_panel_title(axis: plt.Axes, title: str) -> None:
	if not title:
		return
	axis.text(
		0.5,
		1.004,
		title,
		transform=axis.transAxes,
		ha="center",
		va="bottom",
		fontsize=PANEL_TITLE_FONTSIZE,
		fontweight="normal",
		clip_on=False,
	)


def _add_south_china_sea_inset(
	figure: plt.Figure,
	axis: plt.Axes,
	plot_df: gpd.GeoDataFrame,
	city_map: gpd.GeoDataFrame,
	column_name: str,
	colormap: mcolors.Colormap,
	norm: mcolors.Normalize,
	nine_dotted_line: gpd.GeoDataFrame | None,
) -> None:
	inset = axis.inset_axes(SOUTH_CHINA_SEA_INSET_RECT)
	inset.set_facecolor("white")

	plot_subset = _subset_by_bounds(plot_df, SOUTH_CHINA_SEA_BOUNDS)
	plot_subset = plot_subset.copy()
	plot_subset["__panel_color__"] = _build_facecolors(plot_subset[column_name], colormap, norm)
	plot_subset.plot(
		color=plot_subset["__panel_color__"],
		linewidth=0.35,
		edgecolor="#6E6E6E",
		ax=inset,
	)
	_add_city_boundaries(inset, city_map, SOUTH_CHINA_SEA_BOUNDS)
	if not plot_subset.empty:
		plot_subset.boundary.plot(ax=inset, color="#7A7A7A", linewidth=0.4, zorder=1.2)

	if nine_dotted_line is not None:
		nine_dotted_subset = _subset_by_bounds(nine_dotted_line, SOUTH_CHINA_SEA_BOUNDS)
		if not nine_dotted_subset.empty:
			nine_dotted_subset.plot(
				ax=inset,
				color="black",
				linewidth=0.6,
				linestyle=(0, (4, 3)),
				zorder=2,
			)

	inset.set_xlim(SOUTH_CHINA_SEA_BOUNDS[0], SOUTH_CHINA_SEA_BOUNDS[1])
	inset.set_ylim(SOUTH_CHINA_SEA_BOUNDS[2], SOUTH_CHINA_SEA_BOUNDS[3])
	inset.set_aspect("equal")
	inset.set_xticks([])
	inset.set_yticks([])
	inset.set_anchor("SE")
	for spine in inset.spines.values():
		spine.set_visible(True)
		spine.set_linewidth(0.7)
		spine.set_edgecolor("black")


def _annotate_panel_values(axis: plt.Axes, plot_df: gpd.GeoDataFrame, value_col: str) -> None:
	label_df = plot_df.loc[plot_df[value_col].notna(), ["province_label", "geometry", value_col]].copy()
	if label_df.empty:
		return
	label_points = label_df.geometry.representative_point()
	for (_, row), point in zip(label_df.iterrows(), label_points):
		dx, dy = LABEL_OFFSETS.get(str(row["province_label"]), (0.0, 0.0))
		text = axis.text(
			point.x + dx,
			point.y + dy,
			f"{float(row[value_col]):.1f}",
			ha="center",
			va="center",
			fontsize=MAP_VALUE_FONTSIZE,
			color="#1f1f1f",
			zorder=3,
		)
		text.set_path_effects([
			patheffects.withStroke(linewidth=1.6, foreground="white", alpha=0.95),
		])


def _plot_smape_compare(
	province_map: gpd.GeoDataFrame,
	city_map: gpd.GeoDataFrame,
	before_df: pd.DataFrame,
	after_df: pd.DataFrame,
	bias_before_df: pd.DataFrame,
	bias_after_df: pd.DataFrame,
	output_path: str | Path,
	nine_dotted_line: gpd.GeoDataFrame | None,
) -> None:
	plot_df = province_map.merge(before_df.rename(columns={"smape": "smape_before"}), on="province_label", how="left")
	plot_df = plot_df.merge(after_df.rename(columns={"smape": "smape_after"}), on="province_label", how="left")
	plot_df = plot_df.merge(bias_before_df.rename(columns={"bias_ratio": "bias_before"}), on="province_label", how="left")
	plot_df = plot_df.merge(bias_after_df.rename(columns={"bias_ratio": "bias_after"}), on="province_label", how="left")
	masked = plot_df["province_label"].isin(NO_CALIBRATION_PROVINCES)
	for column_name in ("smape_before", "smape_after", "bias_before", "bias_after"):
		plot_df.loc[masked, column_name] = pd.NA
	plot_df["smape_delta"] = pd.to_numeric(plot_df["smape_before"], errors="coerce") - pd.to_numeric(plot_df["smape_after"], errors="coerce")
	plot_df["bias_delta"] = (pd.to_numeric(plot_df["bias_after"], errors="coerce") - pd.to_numeric(plot_df["bias_before"], errors="coerce")).abs()
	shared_values = pd.concat([plot_df["smape_before"], plot_df["smape_after"]], ignore_index=True)
	ratio_shared_values = pd.concat([plot_df["bias_before"], plot_df["bias_after"]], ignore_index=True)
	smape_delta_norm, smape_delta_ticks, smape_delta_vmax = _build_diverging_colorbar_config(plot_df["smape_delta"])

	figure = plt.figure(figsize=(20.2, 20.2))
	grid = figure.add_gridspec(
		nrows=3,
		ncols=2,
		left=0.035,
		right=0.955,
		top=0.962,
		bottom=0.035,
		hspace=0.16,
		wspace=0.105,
	)
	axes = [[figure.add_subplot(grid[row_index, column_index]) for column_index in range(2)] for row_index in range(3)]
	shared_colormap = _build_delta_colormap()
	shared_norm, shared_colorbar_ticks, _, shared_vmax = _build_colorbar_config(shared_values)
	ratio_colormap = _build_bias_colormap()
	ratio_norm, ratio_colorbar_ticks, ratio_vmax = _build_fixed_ratio_colorbar_config()
	smape_delta_colormap = _build_bias_colormap()
	bias_delta_colormap = _build_bias_colormap()
	bias_delta_norm, bias_delta_ticks, bias_delta_vmax = _build_fixed_delta200_colorbar_config()
	before_mean = float(plot_df["smape_before"].dropna().mean())
	after_mean = float(plot_df["smape_after"].dropna().mean())
	delta_mean = float(plot_df["smape_delta"].dropna().mean())
	bias_before_mean = float(plot_df["bias_before"].dropna().mean())
	bias_after_mean = float(plot_df["bias_after"].dropna().mean())
	bias_delta_mean = float(plot_df["bias_delta"].dropna().mean())
	reduction_pct = (1.0 - (after_mean / before_mean)) * 100 if before_mean > 0 else 0.0
	improved_count = int(
		plot_df.loc[plot_df["smape_before"].notna() & plot_df["smape_after"].notna(), ["province_label", "smape_before", "smape_after"]]
		.assign(improved=lambda frame: frame["smape_after"] < frame["smape_before"])
		["improved"]
		.sum()
	)

	panel_specs = [
		(
			("smape_before", f"(a) SMAPE of pre-calibration generation against reference generation\nMean = {before_mean:.1f}%", shared_colormap, shared_norm, shared_colorbar_ticks, "SMAPE (%)"),
			("bias_before", f"(b) Ratio of pre-calibration generation to reference generation\nMean = {bias_before_mean:.1f}%", ratio_colormap, ratio_norm, ratio_colorbar_ticks, "Ratio (%)"),
		),
		(
			("smape_after", f"(c) SMAPE of post-calibration generation against reference generation\nMean = {after_mean:.1f}%", shared_colormap, shared_norm, shared_colorbar_ticks, "SMAPE (%)"),
			("bias_after", f"(d) Ratio of post-calibration generation to reference generation\nMean = {bias_after_mean:.1f}%", ratio_colormap, ratio_norm, ratio_colorbar_ticks, "Ratio (%)"),
		),
		(
			("smape_delta", f"(e) Improvement in SMAPE after calibration against reference generation\nMean = {delta_mean:.1f} pp", smape_delta_colormap, smape_delta_norm, smape_delta_ticks, "SMAPE improvement (pp)"),
			("bias_delta", f"(f) Change in the ratio of post-calibration generation to reference generation\nMean = {bias_delta_mean:.1f}%", bias_delta_colormap, bias_delta_norm, bias_delta_ticks, "Absolute delta (%)"),
		),
	]

	for row_axes, row_specs in zip(axes, panel_specs):
		for axis, (column_name, title, colormap, norm, ticks, colorbar_label) in zip(row_axes, row_specs):
			panel_df = plot_df.copy()
			panel_df["__panel_color__"] = _build_facecolors(panel_df[column_name], colormap, norm)
			panel_df.plot(
				color=panel_df["__panel_color__"],
				linewidth=0.6,
				edgecolor="#6E6E6E",
				ax=axis,
			)
			_add_city_boundaries(axis, city_map, MAINLAND_BOUNDS)
			panel_df.boundary.plot(ax=axis, color="#6E6E6E", linewidth=0.52, zorder=1.2)
			axis.set_xlim(PANEL_BOUNDS[0], PANEL_BOUNDS[1])
			axis.set_ylim(PANEL_YLIM[0], PANEL_YLIM[1])
			axis.set_aspect("equal")
			axis.set_axis_off()
			_annotate_panel_values(axis, panel_df, column_name)
			_attach_no_data_legend(axis)
			_add_south_china_sea_inset(figure, axis, panel_df, city_map, column_name, colormap, norm, nine_dotted_line)
			_add_panel_colorbar(figure, axis, norm, colormap, ticks, colorbar_label)
			_add_panel_title(axis, title)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
	plt.close(figure)


def generate_province_smape_compare(
	no_kb_root: str | Path = DEFAULT_NO_KB_ROOT,
	calibrated_root: str | Path = DEFAULT_CALIBRATED_ROOT,
	final_root: str | Path = DEFAULT_FINAL_ROOT,
	province_geojson: str | Path = DEFAULT_PROVINCE_GEOJSON,
	city_geojson: str | Path = DEFAULT_CITY_GEOJSON,
	output_path: str | Path = DEFAULT_OUTPUT_PATH,
	table_path: str | Path = DEFAULT_TABLE_PATH,
) -> dict[str, object]:
	final_monthly = _load_province_monthly(final_root)
	no_kb_monthly = _load_province_monthly(no_kb_root)
	calibrated_monthly = _load_province_monthly(calibrated_root)

	before_df = _compute_province_smape(no_kb_monthly, final_monthly)
	after_df = _compute_province_smape(calibrated_monthly, final_monthly)
	bias_before_df = _compute_province_bias_ratio(no_kb_monthly, final_monthly)
	bias_after_df = _compute_province_bias_ratio(calibrated_monthly, final_monthly)
	province_map = _read_province_map(province_geojson)
	city_map = _read_city_map(city_geojson)
	nine_dotted_line = _read_nine_dotted_line(DEFAULT_NINE_DOTTED_LINE_PATH)
	table_df = _build_export_table(province_map, before_df, after_df, bias_before_df, bias_after_df)
	_plot_smape_compare(province_map, city_map, before_df, after_df, bias_before_df, bias_after_df, output_path, nine_dotted_line)
	_write_export_table(table_df, table_path)

	return {
		"output_path": str(output_path),
		"table_path": str(table_path),
		"province_geojson": str(province_geojson),
		"city_geojson": str(city_geojson),
		"before_available": int(before_df["smape"].notna().sum()),
		"after_available": int(after_df["smape"].notna().sum()),
		"map_province_count": int(province_map["province_label"].nunique()),
		"smape_colorbar_vmax": float(_build_colorbar_config(pd.concat([table_df["smape_before"], table_df["smape_after"]], ignore_index=True))[3]),
		"smape_delta_colorbar_vmax": float(_build_diverging_colorbar_config(table_df["smape_delta"])[2]),
		"bias_colorbar_vmax": float(_build_fixed_ratio_colorbar_config()[2]),
		"bias_delta_colorbar_vmax": float(_build_fixed_delta200_colorbar_config()[2]),
	}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot province-level SMAPE maps before and after calibration")
	parser.add_argument("--no-kb-root", type=Path, default=DEFAULT_NO_KB_ROOT, help="Path to no_kb_gen/city")
	parser.add_argument("--calibrated-root", type=Path, default=DEFAULT_CALIBRATED_ROOT, help="Path to calibrated_gen/city")
	parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT, help="Path to final_gen/city")
	parser.add_argument("--province-geojson", type=Path, default=DEFAULT_PROVINCE_GEOJSON, help="Path to province geojson")
	parser.add_argument("--city-geojson", type=Path, default=DEFAULT_CITY_GEOJSON, help="Path to city geojson")
	parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="PNG output path")
	parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH, help="CSV output path")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	result = generate_province_smape_compare(
		no_kb_root=args.no_kb_root,
		calibrated_root=args.calibrated_root,
		final_root=args.final_root,
		province_geojson=args.province_geojson,
		city_geojson=args.city_geojson,
		output_path=args.output_path,
		table_path=args.table_path,
	)
	print(result)


if __name__ == "__main__":
	main()