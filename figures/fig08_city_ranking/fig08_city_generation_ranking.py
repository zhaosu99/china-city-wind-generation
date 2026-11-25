"""Annual city generation ranking bump chart (Arial 8 pt, 300 dpi)."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import math
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr
from PIL import Image


_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import config_paths as _cfg  # noqa: E402


DEFAULT_TOP_N = 20
DEFAULT_FIG_BASENAME = "fig08_city_generation_ranking_standalone"
FONT_FAMILY = "Arial"
FONT_SIZE_PT = 8
PX_PER_INCH = 100
TEXT_SIZE = round(FONT_SIZE_PT * PX_PER_INCH / 72.0, 2)
EXPORT_DPI = 300
LAYOUT_SCALE_X = 0.57
LAYOUT_SCALE_Y = 0.78
FIG_WIDTH = int(1400 * LAYOUT_SCALE_X)
FIG_HEIGHT = int(800 * LAYOUT_SCALE_Y)
# 边距由字号决定（不随画布缩放）：左边距要给 y 刻度数字 + "Rank" 标题留位
MARGIN = {"l": 60, "r": 45, "t": 20, "b": 45}
INNER_WIDTH = FIG_WIDTH - MARGIN["l"] - MARGIN["r"]
# 右侧城市标签列不随画布缩放（字号固定），按固定像素留位：最长名 "Bayannur (Inner Mongolia)" 实测 131 px
LABEL_PAD_RIGHT = 16
LABEL_MAX_WIDTH = 132
LABEL_GAP = 20
LABEL_X = (FIG_WIDTH - LABEL_PAD_RIGHT - LABEL_MAX_WIDTH - MARGIN["l"]) / INNER_WIDTH
PLOT_DOMAIN_RIGHT = LABEL_X - LABEL_GAP / INNER_WIDTH
TITLE_STANDOFF = 10
BACKGROUND_LINE_COLOR = "#d2d7de"
BACKGROUND_MARKER_COLOR = "#e2e6eb"
OVERFLOW_TICKS = [30, 40, 60, 80, 120]
OVERFLOW_POSITIONS = [21.2, 22.4, 23.8, 25.2, 27]
MUTED_COLORS = [
	"#1F4E79",
	"#C84C09",
	"#2A7F62",
	"#7A3E9D",
	"#B33C3C",
	"#008B8B",
	"#8A6E00",
	"#D1497A",
	"#4C6A92",
	"#5B8C2A",
	"#A05A2C",
	"#006D77",
	"#6C4E9B",
	"#C06C00",
	"#3F7D58",
	"#A23B72",
	"#7B6D5D",
	"#2F6690",
	"#6B8E23",
	"#B56576",
]

PROVINCE_LABEL_OVERRIDES = {
	"neimenggu": "Inner Mongolia",
}

CITY_LABEL_OVERRIDES = {
	"neimenggu__alashan": "Alxa",
	"neimenggu__bayannaoer": "Bayannur",
	"neimenggu__eerduosi": "Ordos",
	"neimenggu__huhehaote": "Hohhot",
	"neimenggu__hulunbeier": "Hulunbuir",
	"neimenggu__wulanchabu": "Ulaanqab",
	"neimenggu__xilinguolei": "Xilingol",
	"neimenggu__xingan": "Hinggan",
}


def open_dataset_with_fallback(path: Path) -> xr.Dataset:
	try:
		return xr.open_dataset(path)
	except Exception:
		return xr.open_dataset(path, engine="scipy")


def slug_to_label(city_slug: str) -> str:
	prov_slug, city_slug_only = city_slug.split("__", maxsplit=1)
	province = PROVINCE_LABEL_OVERRIDES.get(prov_slug, prov_slug.replace("_", " ").title())
	city = CITY_LABEL_OVERRIDES.get(city_slug, city_slug_only.replace("_", " ").title())
	if province == city:
		return city
	return f"{city} ({province})"


def load_annual_city_generation(city_root: Path) -> pd.DataFrame:
	city_series: dict[str, pd.Series] = {}

	for nc_path in sorted(city_root.glob("*/full_year/gen.nc")):
		city_slug = nc_path.parents[1].name
		ds = open_dataset_with_fallback(nc_path)
		try:
			if "total_monthly_gen" in ds.data_vars and "date" in ds.coords:
				series = ds["total_monthly_gen"].to_series().resample("YE").sum()
			elif "total_gen" in ds.data_vars and "time" in ds.coords:
				series = ds["total_gen"].resample(time="YE").sum(skipna=True).to_series()
			else:
				raise KeyError(
					f"dataset does not contain monthly or hourly total generation: {nc_path}"
				)
		finally:
			ds.close()

		series.index = pd.to_datetime(series.index)
		series.index = series.index.year
		series = pd.to_numeric(series, errors="coerce").sort_index()
		city_series[slug_to_label(city_slug)] = series

	if not city_series:
		raise FileNotFoundError(f"no city generation files found under: {city_root}")

	annual_df = pd.DataFrame(city_series).sort_index()
	annual_df.index.name = "year"
	return annual_df


def select_top_cities(annual_df: pd.DataFrame, top_n: int) -> list[str]:
	annual_df = annual_df.dropna(how="all")
	if annual_df.empty:
		raise ValueError("annual city generation table is empty")

	last_year = annual_df.index.max()
	last_year_ranking = annual_df.loc[last_year].dropna().sort_values(ascending=False)
	if last_year_ranking.empty:
		raise ValueError(f"annual city generation table is empty for final year: {last_year}")
	return last_year_ranking.head(top_n).index.tolist()


def build_ranking_table(annual_df: pd.DataFrame) -> pd.DataFrame:
	selected_df = annual_df.loc[annual_df.notna().any(axis=1)].copy()
	rank_df = selected_df.rank(axis=1, ascending=False, method="first")

	ranking_long = (
		rank_df.stack(future_stack=True)
		.rename("rank")
		.to_frame()
		.join(selected_df.stack(future_stack=True).rename("generation_mwh"))
		.dropna(subset=["rank", "generation_mwh"])
		.reset_index()
		.rename(columns={"level_1": "city"})
	)
	ranking_long["rank"] = ranking_long["rank"].astype(int)
	ranking_long["generation_gwh"] = ranking_long["generation_mwh"] / 1_000.0
	return ranking_long


def rank_to_plot_position(rank: float, top_n: int) -> float:
	if rank <= top_n:
		return float(rank)
	anchors_rank = [top_n, *OVERFLOW_TICKS]
	anchors_pos = [float(top_n), *OVERFLOW_POSITIONS]
	if rank <= anchors_rank[1]:
		left_rank, right_rank = anchors_rank[0], anchors_rank[1]
		left_pos, right_pos = anchors_pos[0], anchors_pos[1]
		fraction = (float(rank) - float(left_rank)) / float(right_rank - left_rank)
		return left_pos + fraction * (right_pos - left_pos)

	if rank >= anchors_rank[-1]:
		return anchors_pos[-1]

	insert_at = bisect_right(anchors_rank, float(rank))
	left_rank, right_rank = anchors_rank[insert_at - 1], anchors_rank[insert_at]
	left_pos, right_pos = anchors_pos[insert_at - 1], anchors_pos[insert_at]
	fraction = (float(rank) - float(left_rank)) / float(right_rank - left_rank)
	return left_pos + fraction * (right_pos - left_pos)


def build_rank_ticks(top_n: int, max_rank: int) -> tuple[list[float], list[str]]:
	tick_ranks = list(range(1, top_n + 1))
	overflow_candidates = OVERFLOW_TICKS
	tick_ranks.extend(rank for rank in overflow_candidates if top_n < rank <= max_rank)
	if max_rank > top_n and max_rank not in tick_ranks:
		tick_ranks.append(max_rank)

	tick_vals = [rank_to_plot_position(rank, top_n) for rank in tick_ranks]
	tick_text = [str(rank) for rank in tick_ranks]
	return tick_vals, tick_text


def build_figure(ranking_df: pd.DataFrame, top_cities: list[str], top_n: int) -> go.Figure:
	color_map = {city: MUTED_COLORS[idx % len(MUTED_COLORS)] for idx, city in enumerate(top_cities)}
	background_cities = sorted(
		city for city in ranking_df.loc[ranking_df["rank"] <= top_n, "city"].unique() if city not in top_cities
	)
	year_start = ranking_df["year"].min()
	year_end = ranking_df["year"].max()
	year_ticks = list(range(int(year_start), int(year_end) + 1))
	year_tick_labels = [
		str(year) if (year - int(year_start)) % 5 == 0 or year == int(year_end) else ""
		for year in year_ticks
	]

	fig = go.Figure()
	for city in background_cities:
		city_df = ranking_df.loc[ranking_df["city"] == city].sort_values("year")
		background_y = city_df["rank"].map(lambda rank: rank_to_plot_position(rank, top_n))
		fig.add_trace(
			go.Scatter(
				x=city_df["year"],
				y=background_y,
				mode="lines+markers",
				showlegend=False,
				hoverinfo="skip",
				line={"color": BACKGROUND_LINE_COLOR, "width": 1.0},
				marker={
					"size": 2.8,
					"color": BACKGROUND_MARKER_COLOR,
					"line": {"color": BACKGROUND_LINE_COLOR, "width": 0.5},
				},
				opacity=0.95,
			)
		)

	for city in top_cities:
		city_df = ranking_df.loc[ranking_df["city"] == city].sort_values("year")
		plot_y = city_df["rank"].map(lambda rank: rank_to_plot_position(rank, top_n))
		fig.add_trace(
			go.Scatter(
				x=city_df["year"],
				y=plot_y,
				mode="lines+markers",
				name=city,
				line={"color": color_map[city], "width": 1.65},
				marker={
					"size": 3.6,
					"color": color_map[city],
					"line": {"color": color_map[city], "width": 0},
				},
				opacity=0.9,
				hovertemplate=(
					"<b>%{fullData.name}</b><br>"
					"Year: %{x}<br>"
					"Rank: %{customdata[0]}<br>"
					"Generation: %{customdata[1]:.2f} GWh<extra></extra>"
				),
				customdata=list(zip(city_df["rank"], city_df["generation_gwh"])),
			)
		)

		last_row = city_df.iloc[-1]
		fig.add_annotation(
			xref="paper",
			yref="y",
			x=LABEL_X,
			y=float(rank_to_plot_position(float(last_row["rank"]), top_n)),
			text=city,
			showarrow=False,
			xanchor="left",
			font={"size": TEXT_SIZE, "color": color_map[city]},
		)

	max_visible_rank = int(ranking_df["rank"].max())
	max_plot_rank = ranking_df.loc[ranking_df["city"].isin(top_cities), "rank"].map(
		lambda rank: rank_to_plot_position(rank, top_n)
	).max()
	tick_vals, tick_text = build_rank_ticks(top_n, max_visible_rank)

	fig.update_layout(
		template="simple_white",
		width=FIG_WIDTH,
		height=FIG_HEIGHT,
		margin=MARGIN,
		hovermode="x unified",
		plot_bgcolor="#fbfbfd",
		paper_bgcolor="white",
		showlegend=False,
		font={"family": FONT_FAMILY, "size": TEXT_SIZE, "color": "#1f2937"},
	)
	fig.update_xaxes(
		title_text="Year",
		domain=[0.0, PLOT_DOMAIN_RIGHT],
		tickangle=0,
		tickmode="array",
		tickvals=year_ticks,
		ticktext=year_tick_labels,
		tickfont={"size": TEXT_SIZE},
		title_font={"size": TEXT_SIZE},
		ticks="outside",
		ticklen=5,
		tickwidth=0.8,
		showgrid=True,
		gridcolor="#aab7c8",
		gridwidth=0.5,
		griddash="dot",
		zeroline=False,
		range=[year_start - 0.3, year_end + 0.3],
	)
	fig.update_yaxes(
		title_text="Rank",
		title_font={"size": TEXT_SIZE},
		title_standoff=TITLE_STANDOFF,
		tickmode="array",
		tickvals=tick_vals,
		ticktext=tick_text,
		tickfont={"size": TEXT_SIZE},
		range=[max_plot_rank + 0.5, 0.5],
		showgrid=True,
		gridcolor="#b4c0ce",
		gridwidth=0.5,
		griddash="dot",
		zeroline=False,
	)
	return fig


def export_figure(fig: go.Figure, output_dir: Path, file_basename: str) -> tuple[Path, Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	html_path = output_dir / f"{file_basename}.html"
	png_path = output_dir / f"{file_basename}.png"
	fig.write_html(html_path, include_plotlyjs="cdn")
	png_width = fig.layout.width if fig.layout.width is not None else 700
	png_height = fig.layout.height if fig.layout.height is not None else 500
	fig.write_image(
		png_path,
		width=png_width,
		height=png_height,
		scale=EXPORT_DPI / PX_PER_INCH,
	)
	with Image.open(png_path) as png:
		image = png.copy()
	image.save(png_path, dpi=(EXPORT_DPI, EXPORT_DPI))
	return html_path, png_path


def parse_args() -> argparse.Namespace:
	script_path = Path(__file__).resolve()
	default_city_root = _cfg.FINAL_GEN_DIR
	default_output_dir = script_path.parent / "output"

	parser = argparse.ArgumentParser(
		description="Create an annual top-city wind generation ranking figure."
	)
	parser.add_argument(
		"--city-root",
		type=Path,
		default=default_city_root,
		help="Directory containing city generation folders.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=default_output_dir,
		help="Directory for the exported HTML and PNG files.",
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=DEFAULT_TOP_N,
		help="Number of cities to include in the ranking figure.",
	)
	parser.add_argument(
		"--file-basename",
		default=DEFAULT_FIG_BASENAME,
		help="Base name used for exported files.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	annual_df = load_annual_city_generation(args.city_root)
	top_cities = select_top_cities(annual_df, top_n=args.top_n)
	ranking_df = build_ranking_table(annual_df)
	fig = build_figure(ranking_df, top_cities, top_n=args.top_n)
	html_path, png_path = export_figure(fig, args.output_dir, args.file_basename)
	print(f"Saved HTML to: {html_path}")
	print(f"Saved PNG to: {png_path}")


if __name__ == "__main__":
	main()
