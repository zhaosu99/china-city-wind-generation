"""Figure 9 (a) 当年发电量排名 + (b) 1989 年起累计发电量排名（同款 bump chart，只是数值口径不同）。

新脚本：不改动 fig08_city_generation_ranking.py（只 import 它的数据加载、配色与坐标压缩函数）。
输出 fig08_city_generation_ranking_ab.{png,html}，不覆盖原图。

------------------------------------------------------------------------------
本文件 = `fig08_city_generation_ranking_ab.py` 的副本（runner 实际使用的那个脚本保持
一字不动）。唯一新增的是**可选**开关 `--drop-gray`，用来去掉背景城市的浅灰 bump 线：

    --drop-gray none   默认值，与 `fig08_city_generation_ranking_ab.py` 完全一致
    --drop-gray both   两个 panel 都不画灰色背景线
    --drop-gray b      只有下面那个 panel (b) 不画灰色背景线

除该开关外，本副本与原脚本的绘图代码逐行相同。
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import config_paths as _cfg  # noqa: E402

import fig08_city_generation_ranking as base

TOP_N_A = 10  # (a) 当年发电量排名 Top 10
TOP_N_B = 10  # (b) 1989~2023 累计排名 Top 10
CUM_UNIT = 1e6  # MWh -> TWh
# 只在 (b) 出现的城市用，与 (a) 用到的颜色都不重复
EXTRA_COLORS = ["#D9A400", "#6E1423"]

# 排名 -> 图上位置的压缩刻度：1~10 名等距一格，10 名之后改紧凑（log 式）间距，
# 间距量级沿用老代码 OVERFLOW_POSITIONS（30/40/60 之间约 1.1~1.4 格）。
RANK_ANCHORS = [1, 10, 20, 30, 40, 60, 70, 100, 300]
POS_ANCHORS = [1.0, 10.0, 11.5, 12.8, 13.9, 15.3, 16.3, 17.5, 19.0]
TICK_ANCHOR_RANKS = [20, 30, 40, 60, 70, 80, 120]

PANEL_TITLE_A = "(a) Top 10 cities by annual wind power generation"
PANEL_TITLE_B = "(b) Top 10 cities by cumulative wind power generation, 1989~2023"
TITLE_GAP_PX = 10  # 子图标题底边与 panel 顶边之间的间距

FIG_WIDTH = 900
PANEL_A_PLOT_H = 340
GAP_PX = 60
PANEL_B_PLOT_H = PANEL_A_PLOT_H  # 两个 panel 等高，样式保持一致
MARGIN = {"l": 150, "r": 45, "t": 40, "b": 45}
FIG_HEIGHT = MARGIN["t"] + PANEL_A_PLOT_H + GAP_PX + PANEL_B_PLOT_H + MARGIN["b"]
INNER_HEIGHT = FIG_HEIGHT - MARGIN["t"] - MARGIN["b"]
INNER_WIDTH = FIG_WIDTH - MARGIN["l"] - MARGIN["r"]
Y_A_BOTTOM = 1.0 - PANEL_A_PLOT_H / INNER_HEIGHT
Y_B_TOP = PANEL_B_PLOT_H / INNER_HEIGHT
# 城市标签列：两个 panel 共用同一列，按固定像素预留（字号不随画布缩放）
LABEL_PAD_RIGHT = 16
LABEL_MAX_WIDTH = 132
LABEL_X = (FIG_WIDTH - LABEL_PAD_RIGHT - LABEL_MAX_WIDTH - MARGIN["l"]) / INNER_WIDTH
PLOT_DOMAIN_RIGHT = LABEL_X - 20 / INNER_WIDTH


def rank_to_position(rank: float) -> float:
	"""分段线性插值：1~10 名一格一档，10 名之后压缩（20/30/40/60/70… 都挨在下面）。"""
	if rank <= RANK_ANCHORS[0]:
		return POS_ANCHORS[0]
	if rank >= RANK_ANCHORS[-1]:
		return POS_ANCHORS[-1]
	for rank0, rank1, pos0, pos1 in zip(RANK_ANCHORS, RANK_ANCHORS[1:], POS_ANCHORS, POS_ANCHORS[1:]):
		if rank <= rank1:
			return pos0 + (float(rank) - rank0) / (rank1 - rank0) * (pos1 - pos0)
	return POS_ANCHORS[-1]


def build_rank_ticks(top_n: int, axis_top_rank: int) -> tuple[list[float], list[str]]:
	ticks = list(range(1, top_n + 1)) + [
		r for r in TICK_ANCHOR_RANKS if top_n < r <= axis_top_rank
	]
	if axis_top_rank not in ticks:
		ticks.append(axis_top_rank)
	return [rank_to_position(r) for r in ticks], [str(r) for r in ticks]



def build_color_map(alpha_cities: list[str], beta_cities: list[str]) -> dict[str, str]:
	"""每个城市一个 unique 颜色：(a) 的 20 城沿用原色序，只在 (b) 出现的城市接在池子后面。"""
	pool = list(base.MUTED_COLORS) + EXTRA_COLORS
	ordered = list(alpha_cities) + [city for city in beta_cities if city not in alpha_cities]
	if len(ordered) > len(pool):
		raise ValueError(f"color pool too small: need {len(ordered)}, have {len(pool)}")
	color_map = {city: pool[idx] for idx, city in enumerate(ordered)}
	assert len(set(color_map.values())) == len(color_map), "city colors are not unique"
	return color_map


def add_bump_panel(
	fig: go.Figure,
	values_df: pd.DataFrame,
	top_n: int,
	color_map: dict[str, str],
	yref: str,
	xaxis_id: str | None,
	yaxis_id: str | None,
	value_divisor: float,
	value_unit: str,
	draw_background: bool = True,
) -> tuple[list[str], pd.DataFrame]:
	"""一个 bump panel：灰线 = 曾进过前 top_n 名、最终没排进名的城市。

	draw_background=False 时不画这些浅灰背景线（其余一切不变）。
	"""
	ranking_df = base.build_ranking_table(values_df)
	top_cities = base.select_top_cities(values_df, top_n)
	background_cities = sorted(
		city
		for city in ranking_df.loc[ranking_df["rank"] <= top_n, "city"].unique()
		if city not in top_cities
	)
	axis_kwargs = {} if xaxis_id is None else {"xaxis": xaxis_id, "yaxis": yaxis_id}

	for city in (background_cities if draw_background else []):
		city_df = ranking_df.loc[ranking_df["city"] == city].sort_values("year")
		fig.add_trace(
			go.Scatter(
				x=city_df["year"],
				y=city_df["rank"].map(rank_to_position),
				mode="lines+markers",
				showlegend=False,
				hoverinfo="skip",
				line={"color": base.BACKGROUND_LINE_COLOR, "width": 1.0},
				marker={
					"size": 2.8,
					"color": base.BACKGROUND_MARKER_COLOR,
					"line": {"color": base.BACKGROUND_LINE_COLOR, "width": 0.5},
				},
				opacity=0.95,
				**axis_kwargs,
			)
		)

	for city in top_cities:
		city_df = ranking_df.loc[ranking_df["city"] == city].sort_values("year")
		values = city_df["generation_gwh"] / value_divisor
		fig.add_trace(
			go.Scatter(
				x=city_df["year"],
				y=city_df["rank"].map(rank_to_position),
				mode="lines+markers",
				name=city,
				line={"color": color_map[city], "width": 1.65},
				marker={"size": 3.6, "color": color_map[city], "line": {"color": color_map[city], "width": 0}},
				opacity=0.9,
				hovertemplate=(
					"<b>%{fullData.name}</b><br>Year: %{x}<br>Rank: %{customdata[0]}<br>"
					f"Generation: %{{customdata[1]:.2f}} {value_unit}<extra></extra>"
				),
				customdata=list(zip(city_df["rank"], values)),
				**axis_kwargs,
			)
		)
		last_row = city_df.iloc[-1]
		fig.add_annotation(
			xref="paper",
			yref=yref,
			x=LABEL_X,
			y=float(rank_to_position(float(last_row["rank"]))),
			text=city,
			showarrow=False,
			xanchor="left",
			font={"size": base.TEXT_SIZE, "color": color_map[city]},
		)

	worst_rank = int(ranking_df.loc[ranking_df["city"].isin(top_cities), "rank"].max())
	axis_top_rank = -(-worst_rank // 10) * 10  # 前十名历史最差成绩向上取整到整十，y 轴只到这里
	tick_vals, tick_text = build_rank_ticks(top_n, axis_top_rank)
	range_top = rank_to_position(axis_top_rank) + 0.5
	return top_cities, tick_vals, tick_text, range_top, ranking_df["year"].min(), ranking_df["year"].max()


def build_figure(annual_df: pd.DataFrame, drop_gray: str = "none") -> go.Figure:
	cum_df = annual_df.cumsum()  # 各城市上线前的缺行保持 NaN，不参与当年累计排名
	top_a = base.select_top_cities(annual_df, top_n=TOP_N_A)
	top_b = base.select_top_cities(cum_df, top_n=TOP_N_B)
	color_map = build_color_map(top_a, top_b)

	fig = go.Figure()
	_, tick_vals_a, tick_text_a, range_top_a, year_start, year_end = add_bump_panel(
		fig, annual_df, TOP_N_A, color_map, "y", None, None, 1.0, "GWh",
		draw_background=drop_gray not in ("both",),
	)
	_, tick_vals_b, tick_text_b, range_top_b, _, _ = add_bump_panel(
		fig, cum_df, TOP_N_B, color_map, "y2", "x2", "y2", 1000.0, "TWh",
		draw_background=drop_gray not in ("both", "b"),
	)

	year_start, year_end = int(year_start), int(year_end)
	year_ticks = list(range(year_start, year_end + 1))
	year_tick_labels = [
		str(y) if (y - year_start) % 5 == 0 or y == year_end else "" for y in year_ticks
	]
	range_top = max(range_top_a, range_top_b)

	fig.update_layout(
		template="simple_white",
		width=FIG_WIDTH,
		height=FIG_HEIGHT,
		margin=MARGIN,
		hovermode="closest",
		plot_bgcolor="#fbfbfd",
		paper_bgcolor="white",
		showlegend=False,
		font={"family": base.FONT_FAMILY, "size": base.TEXT_SIZE, "color": "#1f2937"},
	)
	common_x = {
		"domain": [0.0, PLOT_DOMAIN_RIGHT],
		"range": [year_start - 0.3, year_end + 0.3],
		"tickmode": "array",
		"tickvals": year_ticks,
		"ticktext": year_tick_labels,
		"ticks": "outside",
		"ticklen": 5,
		"tickwidth": 0.8,
		"tickfont": {"size": base.TEXT_SIZE},
		"showgrid": True,
		"gridcolor": "#aab7c8",
		"gridwidth": 0.5,
		"griddash": "dot",
		"zeroline": False,
	}
	common_y = {
		"range": [range_top, 0.5],
		"tickfont": {"size": base.TEXT_SIZE},
		"title": {"text": "Rank", "font": {"size": base.TEXT_SIZE}},
		"title_standoff": 10,
		"showgrid": True,
		"gridcolor": "#b4c0ce",
		"gridwidth": 0.5,
		"griddash": "dot",
		"zeroline": False,
	}
	fig.update_layout(
		xaxis={**common_x},
		yaxis={
			**common_y,
			"domain": [Y_A_BOTTOM, 1.0],
			"tickmode": "array",
			"tickvals": tick_vals_a,
			"ticktext": tick_text_a,
		},
		# 横坐标标题两个 panel 只写一次，放在最下面
		xaxis2={
			**common_x,
			"anchor": "y2",
			# Year 标题与年份刻度贴紧一点
			"title": {"text": "Year", "font": {"size": base.TEXT_SIZE}, "standoff": 2},
		},
		yaxis2={
			**common_y,
			"anchor": "x2",
			"domain": [0.0, Y_B_TOP],
			"tickmode": "array",
			"tickvals": tick_vals_b,
			"ticktext": tick_text_b,
		},
	)
	# 两个 panel 用同一个 yshift（实测标题到 panel 顶边都是 12 px）；改动 GAP_PX / 画布后要复测这一条
	for title, yref_panel in ((PANEL_TITLE_A, "y domain"), (PANEL_TITLE_B, "y2 domain")):
		fig.add_annotation(
			xref="paper", yref=yref_panel,
			x=PLOT_DOMAIN_RIGHT / 2,  # 子图标题相对绘图区居中
			y=1.0,
			yshift=TITLE_GAP_PX,  # 从 panel 顶边再向上抬固定像素
			text=title, showarrow=False,
			xanchor="center", yanchor="bottom",
			font={"size": base.TEXT_SIZE, "color": "#1f2937"},
		)
	return fig, top_a, top_b, color_map


def parse_args() -> argparse.Namespace:
	script_path = Path(__file__).resolve()
	parser = argparse.ArgumentParser(description="Figure 9 (a) annual ranking + (b) cumulative ranking.")
	parser.add_argument("--city-root", type=Path, default=_cfg.FINAL_GEN_DIR)
	parser.add_argument("--output-dir", type=Path, default=script_path.parent / "output")
	parser.add_argument("--file-basename", default="fig08_city_generation_ranking_ab")
	parser.add_argument(
		"--drop-gray",
		choices=["none", "both", "b"],
		default="none",
		help="去掉浅灰背景城市 bump 线：none=原样(默认)；both=两个 panel 都去掉；b=只去掉下面 (b) panel。",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	annual_df = base.load_annual_city_generation(args.city_root)
	fig, top_a, top_b, color_map = build_figure(annual_df, drop_gray=args.drop_gray)
	html_path, png_path = base.export_figure(fig, args.output_dir, args.file_basename)
	print(f"Saved HTML to: {html_path}")
	print(f"Saved PNG to: {png_path}")

	cum_twh = (annual_df.sum() / CUM_UNIT).sort_values(ascending=False)
	total = cum_twh.sum()
	print(f"Cumulative 1989~2023, top {TOP_N_B}:")
	for rank, city in enumerate(top_b, 1):
		print(f"  {rank:2d}. {city:34s} {cum_twh[city]:8.1f} TWh  color {color_map[city]}")
	print(
		f"Check: top{TOP_N_B} {cum_twh.loc[top_b].sum():,.1f} TWh / {annual_df.shape[1]} cities {total:,.1f} TWh"
		f" = {cum_twh.loc[top_b].sum() / total * 100:.2f}%"
	)
	print(f"Color check: {len(color_map)} cities, {len(set(color_map.values()))} unique colors")


if __name__ == "__main__":
	main()
