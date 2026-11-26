import math
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box


REPO_ROOT = Path(os.environ.get("WINDPAPER_ROOT", Path(__file__).resolve().parents[2]))
data_dir = Path(os.environ.get("WINDPAPER_PAPER_ROOT", REPO_ROOT))
out_dir = Path(os.environ.get("WINDPAPER_FIG_OUT", Path(__file__).resolve().parent / "output"))
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = data_dir / "wind_farms_inventory_till_2023.csv"
ADMIN_DIR = Path(os.environ.get("WINDPAPER_ADMIN_DATA", REPO_ROOT / "data" / "admin"))
geojson_path = ADMIN_DIR / "中国_市.geojson"
nine_dotted_line_path = ADMIN_DIR / "九段线" / "china_nine_dotted_line.shp"
geojson_debug_html_path = out_dir / "geojson_debug.html"

GEO_CRS = "EPSG:4326"

PROJECTION_DEFS = [
	{
		"key": "albers",
		"label": "Projection: Albers equal-area conic  |  105°E, 25°N/47°N  |  WGS84",
		"proj4": "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
		"filename": "fig1_wind_farm_map_albers.png",
	},
	{
		"key": "lambert",
		"label": "Projection: Lambert conformal conic  |  105°E, 25°N/47°N  |  WGS84",
		"proj4": "+proj=lcc +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
		"filename": "fig1_wind_farm_map_lambert.png",
	},
	{
		"key": "cgcs2000_albers",
		"label": "Projection: CGCS2000 / Albers equal-area conic  |  105°E, 25°N/47°N  |  GRS80",
		"proj4": "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs",
		"filename": "fig1_wind_farm_map_cgcs2000_albers.png",
	},
]

LON_COL = "经度"
LAT_COL = "纬度"
TYPE_COL = "陆海1-2"
CAPACITY_COL = "风电场装机MW"
CITY_COL = "市全称"
MAP_MIN_LAT = 17.0
NO_DATA_CODES = {"156710000"}
INNER_MONGOLIA_EAST_CODES = {"156150400", "156150500", "156150700", "156152200"}
GRID_PROVINCE_GROUPS = {
	"Northeast China Grid": {"21", "22", "23", "15E"},
	"North China Grid": {"11", "12", "13", "14", "15W", "37"},
	"Northwest China Grid": {"61", "62", "63", "64", "65"},
	"Southwest China Grid": {"50", "51", "54"},
	"Central China Grid": {"36", "41", "42", "43"},
	"East China Grid": {"31", "32", "33", "34", "35"},
	"China Southern Power Grid": {"44", "45", "46", "52", "53"},
}

NO_DATA_COLOR = "#F5F5F5"
CITY_ZERO_CAPACITY_COLOR = "#FFFFFF"
CAPACITY_COLORS = ["#F7F7F2", "#DCEFE8", "#A9D7D0", "#67B8C6", "#3C87B8", "#2457A6", "#162B6F"]

MAP_X_BUFFER_RATIO = 0.03
MAP_Y_BUFFER_RATIO = 0.04
MAIN_PAD_LEFT_RATIO = 0.08
MAIN_PAD_RIGHT_RATIO = 0.19
MAIN_PAD_Y_RATIO = 0.04

COLORBAR_MAX_MW = 20000.0
COLORBAR_TICK_STEP_MW = 2000.0
COLORBAR_LABEL_PAD = 18

FIGURE_SIZE = (13.5, 10.0)
MAIN_AX_RECT = [0.04, 0.06, 0.78, 0.88]
COLORBAR_X = 0.845
COLORBAR_WIDTH = 0.022
INSET_WIDTH_FIG = 0.145
INSET_TITLE_FONTSIZE = 11

SOUTH_CHINA_SEA_BOUNDS_LL = (105.0, 3.0, 125.0, 26.5)
SOUTH_CHINA_SEA_LABEL = "South China Sea Islands"

GRATICULE_LON_RANGE = (70, 140)
GRATICULE_LAT_RANGE = (0, 60)
GRATICULE_LON_STEP = 10
GRATICULE_LAT_STEP = 10
INSET_GRATICULE_STEP = 5
GRATICULE_SAMPLES = 361
GRATICULE_COLOR = "#CCCCCC"
GRATICULE_WIDTH = 0.4
GRATICULE_ZORDER = 0.4
GRATICULE_ALPHA = 0.75
INSET_GRATICULE_COLOR = "#BFBFBF"
INSET_GRATICULE_WIDTH = 0.35

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def make_projection(definition: dict) -> dict:
	crs = CRS.from_proj4(definition["proj4"])
	return {
		"key": definition["key"],
		"label": definition["label"],
		"proj4": definition["proj4"],
		"filename": definition["filename"],
		"crs": crs,
		"to_map": Transformer.from_crs(GEO_CRS, crs, always_xy=True),
		"to_geo": Transformer.from_crs(crs, GEO_CRS, always_xy=True),
	}


def read_wind_farm_csv(csv_file: Path) -> pd.DataFrame:
	encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"]
	last_error: UnicodeDecodeError | None = None

	for encoding in encodings:
		try:
			return pd.read_csv(csv_file, encoding=encoding)
		except UnicodeDecodeError as error:
			last_error = error

	if last_error is not None:
		raise last_error
	raise ValueError(f"Unable to read CSV file: {csv_file}")


def read_base_map(geojson_file: Path) -> gpd.GeoDataFrame:
	gdf = gpd.read_file(geojson_file)
	name_series = gdf["name"].fillna("").astype(str).str.strip()
	return gdf.loc[name_series.ne("境界线")].copy()


def read_nine_dotted_line(shp_file: Path) -> gpd.GeoDataFrame:
	gdf = gpd.read_file(shp_file)
	if gdf.crs is None:
		return gdf.set_crs(GEO_CRS)
	return gdf.to_crs(GEO_CRS)


def normalize_city_name(series: pd.Series) -> pd.Series:
	return series.fillna("").astype(str).str.strip()


def inspect_geojson_interactive(geojson_file: Path, output_html: Path) -> None:
	geo_df = read_base_map(geojson_file)
	if geo_df.crs is None:
		geo_df = geo_df.set_crs(GEO_CRS)
	else:
		geo_df = geo_df.to_crs(GEO_CRS)

	geo_df = geo_df.reset_index(drop=True).copy()
	geo_df["feature_id"] = geo_df.index.astype(str)
	geo_json = geo_df.__geo_interface__

	fig = px.choropleth(
		geo_df,
		geojson=geo_json,
		locations="feature_id",
		featureidkey="properties.feature_id",
		color_discrete_sequence=["#D9D9D9"],
		hover_data={"feature_id": True, "name": True, "gb": True},
	)
	fig.update_traces(
		marker_line_color="#666666",
		marker_line_width=0.7,
		hovertemplate="feature_id=%{location}<br>name=%{customdata[1]}<br>gb=%{customdata[2]}<extra></extra>",
	)
	fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
	fig.update_layout(title="GeoJSON inspection", margin={"l": 0, "r": 0, "t": 40, "b": 0}, showlegend=False)
	fig.write_html(output_html, include_plotlyjs="cdn")
	fig.show()


def assign_province_boundary_group(gb_code: str) -> str:
	if gb_code.startswith("15615"):
		return "15E" if gb_code in INNER_MONGOLIA_EAST_CODES else "15W"
	return gb_code[3:5]


def assign_grid_boundary_group(gb_code: str) -> str | None:
	province_code = assign_province_boundary_group(gb_code)
	for grid_name, province_codes in GRID_PROVINCE_GROUPS.items():
		if province_code in province_codes:
			return grid_name
	return None


def compute_display_bounds_ll(base_map_ll: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
	min_lon, min_lat, max_lon, max_lat = base_map_ll.total_bounds
	min_lat = max(min_lat, MAP_MIN_LAT)
	lon_buffer = (max_lon - min_lon) * MAP_X_BUFFER_RATIO
	lat_buffer = (max_lat - min_lat) * MAP_Y_BUFFER_RATIO
	min_lon -= lon_buffer
	max_lon += lon_buffer
	min_lat = max(MAP_MIN_LAT, min_lat - lat_buffer)
	max_lat += lat_buffer
	return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def project_bbox(bbox_ll: tuple[float, float, float, float], to_map: Transformer, samples: int = 361) -> tuple[float, float, float, float]:
	min_lon, min_lat, max_lon, max_lat = bbox_ll
	lon_edge = np.linspace(min_lon, max_lon, samples)
	lat_edge = np.linspace(min_lat, max_lat, samples)
	edge_lon = np.concatenate([lon_edge, lon_edge, np.full(samples, min_lon), np.full(samples, max_lon)])
	edge_lat = np.concatenate([np.full(samples, min_lat), np.full(samples, max_lat), lat_edge, lat_edge])
	xs, ys = to_map.transform(edge_lon, edge_lat)
	return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))


def combine_projected_bounds(
	*layers,
	pad_left: float = 0.0,
	pad_right: float = 0.0,
	pad_y: float = 0.0,
) -> tuple[float, float, float, float]:
	bounds = [layer.total_bounds for layer in layers if layer is not None and len(layer) > 0]
	if not bounds:
		raise ValueError("Cannot derive display bounds: all layers are empty")
	arr = np.asarray(bounds, dtype=float)
	min_x = float(arr[:, 0].min())
	min_y = float(arr[:, 1].min())
	max_x = float(arr[:, 2].max())
	max_y = float(arr[:, 3].max())
	span_x = max_x - min_x
	span_y = max_y - min_y
	# 横向左右分别留白：右侧宽留白是为了把南海附图收进主图右下角而不压到台湾/钓鱼岛
	return (
		min_x - span_x * pad_left,
		min_y - span_y * pad_y,
		max_x + span_x * pad_right,
		max_y + span_y * pad_y,
	)


def inset_rect_for_main(fig: plt.Figure, main_pos, inset_bounds: tuple[float, float, float, float], width_fig: float) -> list[float]:
	data_ratio = (inset_bounds[2] - inset_bounds[0]) / (inset_bounds[3] - inset_bounds[1])
	fig_width, fig_height = fig.get_size_inches()
	width_inches = width_fig * fig_width
	height_fig = (width_inches / data_ratio) / fig_height
	return [main_pos.x1 - width_fig, main_pos.y0, width_fig, height_fig]


def build_graticule_lines(
	to_map: Transformer,
	lon_step: int = GRATICULE_LON_STEP,
	lat_step: int = GRATICULE_LAT_STEP,
) -> tuple[list[tuple[int, LineString]], list[tuple[int, LineString]]]:
	lon_lines: list[tuple[int, LineString]] = []
	for lon in range(GRATICULE_LON_RANGE[0], GRATICULE_LON_RANGE[1] + 1, lon_step):
		lats = np.linspace(GRATICULE_LAT_RANGE[0], GRATICULE_LAT_RANGE[1], GRATICULE_SAMPLES)
		xs, ys = to_map.transform(np.full(lats.size, float(lon)), lats)
		lon_lines.append((lon, LineString(np.column_stack([xs, ys]).tolist())))

	lat_lines: list[tuple[int, LineString]] = []
	for lat in range(GRATICULE_LAT_RANGE[0], GRATICULE_LAT_RANGE[1] + 1, lat_step):
		lons = np.linspace(GRATICULE_LON_RANGE[0], GRATICULE_LON_RANGE[1], GRATICULE_SAMPLES)
		xs, ys = to_map.transform(lons, np.full(lons.size, float(lat)))
		lat_lines.append((lat, LineString(np.column_stack([xs, ys]).tolist())))

	return lon_lines, lat_lines


def _first_point(geometry):
	if geometry is None or geometry.is_empty:
		return None
	if geometry.geom_type == "Point":
		return geometry
	if geometry.geom_type in ("MultiPoint", "GeometryCollection"):
		for part in geometry.geoms:
			if part.geom_type == "Point":
				return part
	return None


def draw_graticule(
	ax: plt.Axes,
	xlim: tuple[float, float],
	ylim: tuple[float, float],
	to_map: Transformer,
	zorder: float = 1.5,
	show_tick_labels: bool = True,
	lon_step: int = GRATICULE_LON_STEP,
	lat_step: int = GRATICULE_LAT_STEP,
	color: str = GRATICULE_COLOR,
	linewidth: float = GRATICULE_WIDTH,
	alpha: float = GRATICULE_ALPHA,
) -> None:
	lon_lines, lat_lines = build_graticule_lines(to_map, lon_step, lat_step)

	for _, geometry in lon_lines + lat_lines:
		xs, ys = geometry.xy
		ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=zorder, alpha=alpha)

	if not show_tick_labels:
		return

	bottom_edge = LineString([(xlim[0], ylim[0]), (xlim[1], ylim[0])])
	left_edge = LineString([(xlim[0], ylim[0]), (xlim[0], ylim[1])])

	xticks: list[float] = []
	xlabels: list[str] = []
	for lon, geometry in lon_lines:
		point = _first_point(geometry.intersection(bottom_edge))
		if point is None or not (xlim[0] <= point.x <= xlim[1]):
			continue
		xticks.append(point.x)
		xlabels.append(f"{lon}°E")

	yticks: list[float] = []
	ylabels: list[str] = []
	for lat, geometry in lat_lines:
		point = _first_point(geometry.intersection(left_edge))
		if point is None or not (ylim[0] <= point.y <= ylim[1]):
			continue
		yticks.append(point.y)
		ylabels.append(f"{lat}°N")

	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabels, fontsize=8)
	ax.set_yticks(yticks)
	ax.set_yticklabels(ylabels, fontsize=8)


def prepare_city_capacity(wind_df: pd.DataFrame) -> pd.DataFrame:
	wind_df = wind_df.copy()
	wind_df[LON_COL] = pd.to_numeric(wind_df[LON_COL], errors="coerce")
	wind_df[LAT_COL] = pd.to_numeric(wind_df[LAT_COL], errors="coerce")
	wind_df[TYPE_COL] = pd.to_numeric(wind_df[TYPE_COL], errors="coerce")
	wind_df[CAPACITY_COL] = pd.to_numeric(wind_df[CAPACITY_COL], errors="coerce")
	wind_df[CITY_COL] = normalize_city_name(wind_df[CITY_COL])

	wind_df = wind_df.dropna(subset=[LON_COL, LAT_COL, TYPE_COL, CAPACITY_COL]).copy()
	wind_df = wind_df[wind_df[CITY_COL].ne("")].copy()
	wind_df = wind_df[wind_df[TYPE_COL].isin([1, 2])].copy()
	wind_df = wind_df[wind_df[LAT_COL] >= MAP_MIN_LAT].copy()

	wind_df["onshore_capacity_mw"] = np.where(wind_df[TYPE_COL] == 1, wind_df[CAPACITY_COL], 0.0)
	wind_df["offshore_capacity_mw"] = np.where(wind_df[TYPE_COL] == 2, wind_df[CAPACITY_COL], 0.0)

	city_capacity = (
		wind_df.groupby(CITY_COL, as_index=False)
		.agg(
			total_capacity_mw=(CAPACITY_COL, "sum"),
			farm_count=(CAPACITY_COL, "size"),
			onshore_capacity_mw=("onshore_capacity_mw", "sum"),
			offshore_capacity_mw=("offshore_capacity_mw", "sum"),
		)
	)
	city_capacity["total_capacity_gw"] = city_capacity["total_capacity_mw"] / 1000.0
	return city_capacity


def prepare_base_layers(base_map: gpd.GeoDataFrame, city_capacity: pd.DataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoSeries, gpd.GeoSeries, gpd.GeoSeries]:
	base_map = base_map.copy()
	base_map["city_name"] = normalize_city_name(base_map["name"])

	no_data_mask = base_map["gb"].astype(str).isin(NO_DATA_CODES)
	no_data_gdf = base_map[no_data_mask].copy()
	base_map_main = base_map[~no_data_mask].copy()
	base_map_main = base_map_main.merge(city_capacity, left_on="city_name", right_on=CITY_COL, how="left")

	for column in ["total_capacity_mw", "total_capacity_gw", "farm_count", "onshore_capacity_mw", "offshore_capacity_mw"]:
		base_map_main[column] = pd.to_numeric(base_map_main[column], errors="coerce").fillna(0)

	base_map_main["has_wind_farm"] = base_map_main["farm_count"] > 0
	province_base = base_map_main[base_map_main["gb"].astype(str).str.match(r"^156\d{6}$", na=False)].copy()
	province_base["province_code"] = province_base["gb"].astype(str).map(assign_province_boundary_group)
	province_boundaries = province_base.dissolve(by="province_code").boundary
	grid_base = province_base.copy()
	grid_base["grid_code"] = grid_base["gb"].astype(str).map(assign_grid_boundary_group)
	grid_base = grid_base.dropna(subset=["grid_code"])
	grid_boundaries = grid_base.dissolve(by="grid_code").boundary
	national_outline = base_map_main.dissolve().boundary
	return base_map_main, no_data_gdf, province_boundaries, grid_boundaries, national_outline


def make_capacity_norm() -> mcolors.Normalize:
	return mcolors.Normalize(vmin=0.0, vmax=COLORBAR_MAX_MW)


def make_capacity_cmap() -> mcolors.LinearSegmentedColormap:
	return mcolors.LinearSegmentedColormap.from_list("wind_capacity_multihue", CAPACITY_COLORS)


def build_colorbar_ticks(vmax: float) -> list[float]:
	if vmax <= 0:
		return [0.0]
	return list(np.arange(0.0, vmax + COLORBAR_TICK_STEP_MW, COLORBAR_TICK_STEP_MW))


def _plot_capacity_layers(ax: plt.Axes, city_map: gpd.GeoDataFrame, cmap, norm) -> None:
	city_map[city_map["total_capacity_mw"] <= 0].plot(
		ax=ax, color=CITY_ZERO_CAPACITY_COLOR, edgecolor="none", linewidth=0, zorder=1
	)
	city_map[city_map["total_capacity_mw"] > 0].plot(
		ax=ax, column="total_capacity_mw", cmap=cmap, norm=norm, edgecolor="none", linewidth=0, zorder=1.05
	)


def _plot_boundary_layers(
	ax: plt.Axes,
	city_map: gpd.GeoDataFrame,
	no_data_gdf: gpd.GeoDataFrame,
	province_boundaries: gpd.GeoSeries,
	grid_boundaries: gpd.GeoSeries,
	nine_dotted_line: gpd.GeoDataFrame,
	scale: float = 1.0,
) -> None:
	city_map.boundary.plot(
		ax=ax, color="#B8B8B8", linewidth=0.35 * scale, linestyle=(0, (3, 3)), zorder=2
	)
	province_boundaries.plot(
		ax=ax, color="#7F7F7F", linewidth=0.6 * scale, linestyle=(0, (4, 3)), zorder=2.2
	)
	grid_boundaries.plot(ax=ax, color="#2F2F2F", linewidth=0.7 * scale, linestyle="solid", zorder=2.4)

	if not no_data_gdf.empty:
		no_data_gdf.plot(
			ax=ax,
			color=NO_DATA_COLOR,
			edgecolor="#A0A0A0",
			linewidth=0.35 * scale,
			hatch="////",
			zorder=2.5,
		)

	nine_dotted_line.plot(
		ax=ax, color="black", linewidth=0.75 * scale, linestyle=(0, (4, 3)), zorder=3
	)


def plot_south_china_sea_inset(
	fig: plt.Figure,
	main_ax: plt.Axes,
	city_map: gpd.GeoDataFrame,
	no_data_gdf: gpd.GeoDataFrame,
	province_boundaries: gpd.GeoSeries,
	grid_boundaries: gpd.GeoSeries,
	nine_dotted_line: gpd.GeoDataFrame,
	norm,
	cmap,
	bounds: tuple[float, float, float, float],
	to_map: Transformer,
) -> None:
	main_pos = main_ax.get_position()
	inset_ax = fig.add_axes(inset_rect_for_main(fig, main_pos, bounds, INSET_WIDTH_FIG))
	inset_ax.set_facecolor("white")

	min_x, min_y, max_x, max_y = bounds
	inset_ax.set_xlim(min_x, max_x)
	inset_ax.set_ylim(min_y, max_y)
	inset_ax.set_aspect("equal")

	_plot_capacity_layers(inset_ax, city_map, cmap, norm)
	draw_graticule(
		inset_ax,
		(min_x, max_x),
		(min_y, max_y),
		to_map,
		zorder=GRATICULE_ZORDER,
		show_tick_labels=False,
		lon_step=INSET_GRATICULE_STEP,
		lat_step=INSET_GRATICULE_STEP,
		color=INSET_GRATICULE_COLOR,
		linewidth=INSET_GRATICULE_WIDTH,
	)
	_plot_boundary_layers(
		inset_ax, city_map, no_data_gdf, province_boundaries, grid_boundaries, nine_dotted_line, scale=0.85
	)

	inset_ax.set_xticks([])
	inset_ax.set_yticks([])
	inset_title = inset_ax.text(
		0.5,
		1.045,
		SOUTH_CHINA_SEA_LABEL,
		transform=inset_ax.transAxes,
		ha="center",
		va="bottom",
		fontsize=INSET_TITLE_FONTSIZE,
		color="black",
		clip_on=False,
	)
	fig.canvas.draw()
	_renderer = fig.canvas.get_renderer()
	_title_box = inset_title.get_window_extent(renderer=_renderer)
	_inset_box = inset_ax.get_window_extent(renderer=_renderer)
	_fits = _title_box.width <= _inset_box.width
	print(f"  inset title     : {_title_box.width:.0f} px wide  vs inset {_inset_box.width:.0f} px  -> {'fits' if _fits else 'OVERFLOW!'}")
	if not _fits:
		print("  [WARN] inset title overflows the inset frame: increase INSET_WIDTH_FIG or reduce title fontsize")
	for spine in inset_ax.spines.values():
		spine.set_visible(True)
		spine.set_color("black")
		spine.set_linewidth(0.7)



def plot_city_capacity_2d(
	city_map: gpd.GeoDataFrame,
	nine_dotted_line: gpd.GeoDataFrame,
	no_data_gdf: gpd.GeoDataFrame,
	province_boundaries: gpd.GeoSeries,
	grid_boundaries: gpd.GeoSeries,
	bounds: tuple[float, float, float, float],
	inset_bounds: tuple[float, float, float, float],
	projection: dict,
	output_path: Path,
) -> None:
	norm = make_capacity_norm()
	cmap = make_capacity_cmap()
	to_map = projection["to_map"]

	fig = plt.figure(figsize=FIGURE_SIZE, dpi=300)
	ax = fig.add_axes(MAIN_AX_RECT)

	min_x, min_y, max_x, max_y = bounds
	ax.set_xlim(min_x, max_x)
	ax.set_ylim(min_y, max_y)
	ax.set_aspect("equal")

	_plot_capacity_layers(ax, city_map, cmap, norm)
	draw_graticule(ax, (min_x, max_x), (min_y, max_y), to_map, zorder=GRATICULE_ZORDER)
	_plot_boundary_layers(ax, city_map, no_data_gdf, province_boundaries, grid_boundaries, nine_dotted_line)

	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.grid(False)

	active_pos = ax.get_position()
	colorbar_ax = fig.add_axes(
		[COLORBAR_X, active_pos.y0, COLORBAR_WIDTH, active_pos.height]
	)

	legend_handles = [
		Line2D([0], [0], color="#B8B8B8", linewidth=0.35, linestyle=(0, (3, 3)), label="City boundary"),
		Line2D([0], [0], color="#7F7F7F", linewidth=0.7, linestyle=(0, (4, 3)), label="Province boundary"),
		Line2D([0], [0], color="#2F2F2F", linewidth=1.0, linestyle="solid", label="Grid boundary"),
		Line2D([0], [0], color="black", linewidth=0.75, linestyle=(0, (4, 3)), label="South China Sea dashed line"),
		Patch(facecolor=CITY_ZERO_CAPACITY_COLOR, edgecolor="#D0D0D0", label="City with no wind farm till 2023"),
	]
	if not no_data_gdf.empty:
		legend_handles.append(Patch(facecolor=NO_DATA_COLOR, edgecolor="#A0A0A0", hatch="////", label="No data area"))

	base_legend = ax.legend(handles=legend_handles, loc="lower left", frameon=True)
	base_legend.get_frame().set_edgecolor("#C8C8C8")
	base_legend.get_frame().set_linewidth(0.8)

	scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
	scalar_mappable.set_array([])
	colorbar = fig.colorbar(scalar_mappable, cax=colorbar_ax, ticks=build_colorbar_ticks(norm.vmax))
	colorbar.set_label("Cumulative Installed Capacity by City (MW)", labelpad=COLORBAR_LABEL_PAD)

	plot_south_china_sea_inset(
		fig,
		ax,
		city_map,
		no_data_gdf,
		province_boundaries,
		grid_boundaries,
		nine_dotted_line,
		norm,
		cmap,
		inset_bounds,
		to_map,
	)

	verify_graticule_below(ax, to_map)
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def build_city_capacity() -> pd.DataFrame:
	wind_df = read_wind_farm_csv(csv_path)
	return prepare_city_capacity(wind_df)


def inset_occlusion_report(
	fig: plt.Figure,
	main_pos,
	main_bounds: tuple[float, float, float, float],
	inset_bounds: tuple[float, float, float, float],
	to_geo: Transformer,
) -> None:
	"""Report the lon/lat window the inset panel covers on the main map (must contain no land)."""
	data_ratio = (inset_bounds[2] - inset_bounds[0]) / (inset_bounds[3] - inset_bounds[1])
	fig_width, fig_height = fig.get_size_inches()
	height_fig = (INSET_WIDTH_FIG * fig_width / data_ratio) / fig_height
	left_ratio = (main_pos.x1 - INSET_WIDTH_FIG - main_pos.x0) / main_pos.width
	right_ratio = (main_pos.x1 - main_pos.x0) / main_pos.width
	bottom_ratio = (main_pos.y0 - main_pos.y0) / main_pos.height
	top_ratio = (INSET_WIDTH_FIG * fig_width / data_ratio / fig_height) / main_pos.height
	x0 = main_bounds[0] + left_ratio * (main_bounds[2] - main_bounds[0])
	x1 = main_bounds[0] + right_ratio * (main_bounds[2] - main_bounds[0])
	y0 = main_bounds[1] + bottom_ratio * (main_bounds[3] - main_bounds[1])
	y1 = main_bounds[1] + top_ratio * (main_bounds[3] - main_bounds[1])
	lon0, lat0 = to_geo.transform(x0, y0)
	lon1, lat1 = to_geo.transform(x1, y1)
	print(f"  inset covers main map : x {x0/1000:.0f}->{x1/1000:.0f} km  ({lon0:.2f}E -> {lon1:.2f}E)")
	print(f"                          y {y0/1000:.0f}->{y1/1000:.0f} km  ({lat0:.2f}N -> {lat1:.2f}N)")


def verify_graticule_below(ax: plt.Axes, to_map: Transformer) -> None:
	"""采样像素，确认经纬网没有压在城市色块之上。"""
	ax.figure.canvas.draw()
	buffer = np.asarray(ax.figure.canvas.buffer_rgba())
	height = buffer.shape[0]
	probes = [("on graticule 110E/35N", 110.0, 35.0), ("inland 103E/33N", 103.0, 33.0),
	          ("inland 116E/40N", 116.0, 40.0), ("offshore 126E/30N", 126.0, 30.0)]
	for name, lon, lat in probes:
		x, y = to_map.transform(lon, lat)
		px, py = ax.transData.transform((x, y))
		rgb = tuple(int(v) for v in buffer[height - int(round(py)), int(round(px))][:3])
		tag = "GRATICULE ON TOP!" if rgb in {(204, 204, 204), (217, 217, 217)} else "ok (grid hidden)"
		print(f"  pixel {name:22s} -> RGB {rgb}  [{tag}]")


def main() -> None:
	base_map_ll = read_base_map(geojson_path)
	if base_map_ll.crs is None:
		base_map_ll = base_map_ll.set_crs(GEO_CRS)
	else:
		base_map_ll = base_map_ll.to_crs(GEO_CRS)

	nine_dotted_line_ll = read_nine_dotted_line(nine_dotted_line_path)
	city_capacity = build_city_capacity()

	display_bounds_ll = compute_display_bounds_ll(base_map_ll)
	base_map_ll = gpd.clip(base_map_ll, box(*display_bounds_ll))

	print(f"lon/lat bbox     : lon {display_bounds_ll[0]:.3f} -> {display_bounds_ll[2]:.3f}, lat {display_bounds_ll[1]:.3f} -> {display_bounds_ll[3]:.3f}")
	print()

	for definition in PROJECTION_DEFS:
		projection = make_projection(definition)
		crs = projection["crs"]
		to_map = projection["to_map"]
		to_geo = projection["to_geo"]

		base_map = base_map_ll.to_crs(crs)
		nine_dotted_line = nine_dotted_line_ll.to_crs(crs)
		city_map, no_data_gdf, province_boundaries, grid_boundaries, _ = prepare_base_layers(base_map, city_capacity)

		main_bounds = combine_projected_bounds(
			city_map,
			no_data_gdf,
			province_boundaries,
			grid_boundaries,
			pad_left=MAIN_PAD_LEFT_RATIO,
			pad_right=MAIN_PAD_RIGHT_RATIO,
			pad_y=MAIN_PAD_Y_RATIO,
		)
		inset_bounds = project_bbox(SOUTH_CHINA_SEA_BOUNDS_LL, to_map)

		main_frame = box(*main_bounds)
		clipped = nine_dotted_line.geometry.intersection(main_frame)
		visible = int((~clipped.is_empty).sum())
		visible_km = float(sum(g.length for g in clipped if not g.is_empty) / 1000.0)
		inset_frame = box(*inset_bounds)
		inset_segments = int((~nine_dotted_line.geometry.intersection(inset_frame).is_empty).sum())

		aspect = (main_bounds[2] - main_bounds[0]) / (main_bounds[3] - main_bounds[1])
		ax_phys = (MAIN_AX_RECT[2] * FIGURE_SIZE[0]) / (MAIN_AX_RECT[3] * FIGURE_SIZE[1])

		print(f"[{projection['key']}]")
		print(f"  main bounds (km) : x {main_bounds[0]/1000:.0f} -> {main_bounds[2]/1000:.0f}, y {main_bounds[1]/1000:.0f} -> {main_bounds[3]/1000:.0f}")
		print(f"  main aspect      : {aspect:.4f}  (axes physical {ax_phys:.4f})  -> axes 高 {'收窄' if aspect > ax_phys else '不变'}")
		print(f"  dashed line      : main {visible}/{len(nine_dotted_line)} seg ({visible_km:.1f} km) | inset {inset_segments}/{len(nine_dotted_line)} seg")

		fig_probe = plt.figure(figsize=FIGURE_SIZE)
		probe_ax = fig_probe.add_axes(MAIN_AX_RECT)
		probe_ax.set_xlim(main_bounds[0], main_bounds[2])
		probe_ax.set_ylim(main_bounds[1], main_bounds[3])
		probe_ax.set_aspect("equal")
		inset_occlusion_report(fig_probe, probe_ax.get_position(), main_bounds, inset_bounds, to_geo)
		plt.close(fig_probe)

		print(f"  output           : {projection['filename']}")
		plot_city_capacity_2d(
			city_map,
			nine_dotted_line,
			no_data_gdf,
			province_boundaries,
			grid_boundaries,
			main_bounds,
			inset_bounds,
			projection,
			out_dir / projection["filename"],
		)
		print()


if __name__ == "__main__":
	main()
