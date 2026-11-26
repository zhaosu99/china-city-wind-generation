from __future__ import annotations

import argparse
import os
import re
import warnings
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


THREAD_ENV_DEFAULTS = {
	'OMP_NUM_THREADS': '1',
	'OPENBLAS_NUM_THREADS': '1',
	'MKL_NUM_THREADS': '1',
	'NUMEXPR_NUM_THREADS': '1',
}

for env_name, env_value in THREAD_ENV_DEFAULTS.items():
	os.environ.setdefault(env_name, env_value)

import numpy as np
import pandas as pd
import matplotlib

if not os.environ.get('MPLBACKEND'):
	matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


AGG_TRAIN_PATTERN = re.compile(
	r'^(?P<prov>.+?)_(?P<season>[^_]+)_mean_train_gen_smape\.csv$'
)
MIN_OPTIMAL_K = 0.5
PLOT_MODES = ('all', 'failed', 'none')
RUN_MODES = ('compute', 'plot')
DEFAULT_TEST_ABS_TOL = 0.6
DEFAULT_TEST_REL_TOL = 0.10
SEASON_ORDER = ('spring', 'summer', 'autumn', 'winter')
SEASON_LABELS = {
	'spring': 'Spring',
	'summer': 'Summer',
	'autumn': 'Autumn',
	'winter': 'Winter',
}
PANEL_TITLES = (
	'Mean Train Generation SMAPE Surface',
	'Mean Test Generation SMAPE Surface',
	'Wind Speed SMAPE Surface & Optimal (k, b)',
)
PLOT_TEXT_SIZE = 14
CANDIDATE_PATH_COLOR = 'deepskyblue'
RETAINED_PATH_COLOR = 'red'
OMITTED_PATH_COLOR = 'dimgray'


def format_plot_failure_message(exc: Exception) -> str:
	message = str(exc)
	replacements = {
		f'在 k ∈ [{MIN_OPTIMAL_K:.2f}, 3.00] 范围内不存在有效的 generation optimal path': (
			f'No valid generation optimal path exists in the k range [{MIN_OPTIMAL_K:.2f}, 3.00]'
		),
		'无法从 train csv 中提取有效 optimal path': 'Unable to extract a valid optimal path from mean train surface',
		'无法从 test csv 中拟合有效模型': 'Unable to fit a valid model from mean test surface',
		'无法从 wind csv 中拟合有效模型': 'Unable to fit a valid model from wind surface',
		'经过 testing 筛选后未保留任何 generation path 点': 'No generation path points were retained after test validation',
	}
	for source, target in replacements.items():
		message = message.replace(source, target)
	return message


def should_force_spawn(force_spawn: bool | None) -> bool:
	if force_spawn is not None:
		return force_spawn
	return sys.platform.startswith('win')


def get_thread_env_config() -> dict[str, str]:
	return {
		env_name: os.environ.get(env_name, '')
		for env_name in THREAD_ENV_DEFAULTS
	}


def load_smape_csv(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path, index_col=0, encoding='utf-8-sig')
	df.index = pd.to_numeric(df.index, errors='coerce')
	df.columns = pd.to_numeric(df.columns, errors='coerce')
	df = df.loc[~df.index.isna(), ~pd.isna(df.columns)]
	return df.sort_index(axis=0).sort_index(axis=1)


def fit_poly_surface(df: pd.DataFrame, poly_degree: int = 7):
	if poly_degree < 1:
		raise ValueError(f'poly_degree 必须 >= 1, 当前为: {poly_degree}')

	df_clean = df.fillna(np.nan).replace([np.inf, -np.inf], np.nan)
	if df_clean.isnull().all().all():
		return None, None, None

	k_values = df_clean.columns.values.astype(float)
	b_values = df_clean.index.values.astype(float)
	B, K = np.meshgrid(b_values, k_values, indexing='ij')

	X_original = np.column_stack([K.ravel(), B.ravel()])
	y_raw = df_clean.values.ravel()
	mask = ~np.isnan(y_raw)
	if not np.any(mask):
		return None, None, None

	X_filtered = X_original[mask]
	y_filtered = y_raw[mask]
	if len(y_filtered) < 10:
		return None, None, None

	poly_features = PolynomialFeatures(degree=poly_degree, include_bias=True)
	X_poly = poly_features.fit_transform(X_filtered)
	model = LinearRegression()
	model.fit(X_poly, y_filtered)
	fit_r2 = model.score(X_poly, y_filtered)
	return model, poly_features, fit_r2


def build_fine_smape_map(model, poly_features, k_fine: np.ndarray, b_fine: np.ndarray) -> np.ndarray:
	B_fine, K_fine = np.meshgrid(b_fine, k_fine, indexing='ij')
	X_query = np.column_stack([K_fine.ravel(), B_fine.ravel()])
	X_poly = poly_features.transform(X_query)
	y_pred = model.predict(X_poly)
	return y_pred.reshape(len(b_fine), len(k_fine))


def fit_surface_from_dataframe(
	df: pd.DataFrame,
	poly_degree: int = 7,
	fine_step: float = 0.01,
	k_min: float = MIN_OPTIMAL_K,
	k_max: float = 3.0,
	b_min: float = -2.0,
	b_max: float = 2.0,
):
	model, poly_features, fit_r2 = fit_poly_surface(df, poly_degree=poly_degree)
	if model is None:
		return None, None, None, None, None, None

	k_fine = np.arange(k_min, k_max + fine_step / 2, fine_step)
	b_fine = np.arange(b_min, b_max + fine_step / 2, fine_step)
	if len(k_fine) < 2 or len(b_fine) < 2:
		return None, None, None, None, None, None

	S_fine_pred = build_fine_smape_map(model, poly_features, k_fine, b_fine)
	return model, poly_features, k_fine, b_fine, S_fine_pred, fit_r2


def get_optimal_path_from_dataframe(
	df: pd.DataFrame,
	poly_degree: int = 7,
	fine_step: float = 0.01,
	k_min: float = MIN_OPTIMAL_K,
	k_max: float = 3.0,
	b_min: float = -2.0,
	b_max: float = 2.0,
	op_b_min: float = -2.0,
	op_b_max: float = 2.0,
	edge_tol: float = 1e-9,
	allow_boundary_fallback: bool = True,
	return_boundary_flag: bool = False,
):
	def _none_result():
		base = (None, None, None, None, None, None, None, None)
		return base + (False,) if return_boundary_flag else base

	model, poly_features, k_fine, b_fine, S_fine_pred, fit_r2 = fit_surface_from_dataframe(
		df,
		poly_degree=poly_degree,
		fine_step=fine_step,
		k_min=k_min,
		k_max=k_max,
		b_min=b_min,
		b_max=b_max,
	)
	if model is None:
		return _none_result()

	optimal_b_indices = np.argmin(S_fine_pred, axis=0)
	optimal_b_values = b_fine[optimal_b_indices]
	k_values_for_path = k_fine

	op_range_mask = (optimal_b_values >= op_b_min - edge_tol) & (optimal_b_values <= op_b_max + edge_tol)
	if not np.any(op_range_mask):
		return _none_result()

	k_values_for_path = k_values_for_path[op_range_mask]
	optimal_b_values = optimal_b_values[op_range_mask]

	boundary_mask = (
		np.isclose(optimal_b_values, op_b_min, atol=edge_tol)
		| np.isclose(optimal_b_values, op_b_max, atol=edge_tol)
	)

	start_idx = None
	end_idx = None
	started = False
	n_points = len(k_values_for_path)

	for idx, is_boundary in enumerate(boundary_mask):
		next_is_boundary = boundary_mask[idx + 1] if idx + 1 < n_points else None
		if not started:
			if is_boundary:
				if next_is_boundary is False:
					start_idx = idx
					started = True
			else:
				start_idx = idx
				started = True
			continue
		if idx != start_idx and is_boundary:
			end_idx = idx
			break

	on_boundary = False
	if start_idx is None:
		if not allow_boundary_fallback:
			return _none_result()
		# 用户裁定：找不到内点最优解时，允许最优点落在 k/b 定义域边界上（边界次优解），
		# 但必须**告警 + 标注**，不能静默当作正常内点解。
		on_boundary = True
		warnings.warn(
			"最优 b(k) 曲线整条贴 k/b 定义域边界（k∈[%.2f, %.2f], b∈[%.2f, %.2f]）："
			"无内点最优解 ⇒ 采用边界次优解（selection_mode=boundary_fallback）"
			% (k_min, k_max, b_min, b_max),
			UserWarning,
			stacklevel=2,
		)
		k_values_final = k_values_for_path
		optimal_b_final = optimal_b_values
	elif end_idx is None:
		k_values_final = k_values_for_path[start_idx:]
		optimal_b_final = optimal_b_values[start_idx:]
	else:
		k_values_final = k_values_for_path[start_idx:end_idx + 1]
		optimal_b_final = optimal_b_values[start_idx:end_idx + 1]

	if len(k_values_final) > 0 and len(optimal_b_final) > 0:
		result = (k_values_final, optimal_b_final, model, poly_features,
		          k_fine, b_fine, S_fine_pred, fit_r2)
		return result + (on_boundary,) if return_boundary_flag else result
	return _none_result()


def interpolate_smape_value(model, poly_features, k_val: float, b_val: float) -> float:
	X_input = np.array([[k_val, b_val]])
	X_poly = poly_features.transform(X_input)
	return float(model.predict(X_poly)[0])


def find_min_wind_on_gen_op(gen_k, gen_b, wind_model, wind_poly):
	gen_k = np.asarray(gen_k, dtype=float)
	gen_b = np.asarray(gen_b, dtype=float)
	if gen_k.size == 0 or gen_b.size == 0:
		raise RuntimeError('经过 testing 筛选后未保留任何 generation path 点')

	op_wind_smape = np.array([
		interpolate_smape_value(wind_model, wind_poly, k_val, b_val)
		for k_val, b_val in zip(gen_k, gen_b)
	])
	min_idx = int(np.argmin(op_wind_smape))
	return {
		'smape': float(op_wind_smape[min_idx]),
		'k': float(gen_k[min_idx]),
		'b': float(gen_b[min_idx]),
		'all_op_wind_smape': op_wind_smape,
	}


def parse_aggregate_train_filename(train_csv: Path) -> dict[str, str] | None:
	match = AGG_TRAIN_PATTERN.match(train_csv.name)
	if match is None:
		return None
	return {
		'prov': match.group('prov'),
		'season': match.group('season'),
	}


def _match_k_index(k_values: np.ndarray, k_val: float) -> int:
	idx = int(np.searchsorted(k_values, k_val))
	if idx == 0:
		return 0
	if idx >= len(k_values):
		return len(k_values) - 1
	if abs(k_values[idx] - k_val) < abs(k_values[idx - 1] - k_val):
		return idx
	return idx - 1


def build_test_band(
	test_path_k: np.ndarray,
	*,
	test_model,
	test_poly,
	test_k_fine: np.ndarray,
	test_b_fine: np.ndarray,
	test_S_pred: np.ndarray,
	test_abs_tol: float,
	test_rel_tol: float,
) -> dict[str, np.ndarray]:
	band_k = np.asarray(test_path_k, dtype=float)
	lower_b = np.full(len(band_k), np.nan, dtype=float)
	upper_b = np.full(len(band_k), np.nan, dtype=float)
	test_col_min = np.full(len(band_k), np.nan, dtype=float)
	test_tol = np.full(len(band_k), np.nan, dtype=float)

	for idx, k_val in enumerate(band_k):
		col_idx = _match_k_index(test_k_fine, k_val)
		col_values = test_S_pred[:, col_idx]
		finite_mask = np.isfinite(col_values)
		if not np.any(finite_mask):
			continue

		col_min = float(np.nanmin(col_values))
		tol = max(float(test_abs_tol), abs(col_min) * float(test_rel_tol))
		acceptable_mask = finite_mask & (col_values <= col_min + tol)
		if np.any(acceptable_mask):
			acceptable_b = test_b_fine[acceptable_mask]
			lower_b[idx] = float(np.min(acceptable_b))
			upper_b[idx] = float(np.max(acceptable_b))
		test_col_min[idx] = col_min
		test_tol[idx] = tol

	return {
		'band_k': band_k,
		'lower_b': lower_b,
		'upper_b': upper_b,
		'test_col_min': test_col_min,
		'test_tol': test_tol,
	}


def filter_train_path_by_test_band(
	train_k: np.ndarray,
	train_b: np.ndarray,
	*,
	band_k: np.ndarray,
	lower_b: np.ndarray,
	upper_b: np.ndarray,
	test_model,
	test_poly,
	test_k_fine: np.ndarray,
	test_S_pred: np.ndarray,
	test_abs_tol: float,
	test_rel_tol: float,
	edge_tol: float = 1e-9,
) -> dict[str, np.ndarray]:
	train_k = np.asarray(train_k, dtype=float)
	train_b = np.asarray(train_b, dtype=float)
	band_k = np.asarray(band_k, dtype=float)
	lower_b = np.asarray(lower_b, dtype=float)
	upper_b = np.asarray(upper_b, dtype=float)

	test_smape = np.full(len(train_k), np.nan, dtype=float)
	test_col_min = np.full(len(train_k), np.nan, dtype=float)
	test_tol = np.full(len(train_k), np.nan, dtype=float)
	test_delta = np.full(len(train_k), np.nan, dtype=float)
	retained_mask = np.zeros(len(train_k), dtype=bool)

	if band_k.size == 0:
		return {
			'test_smape': test_smape,
			'test_col_min': test_col_min,
			'test_tol': test_tol,
			'test_delta': test_delta,
			'retained_mask': retained_mask,
		}

	band_k_min = float(np.nanmin(band_k))
	band_k_max = float(np.nanmax(band_k))

	for idx, (k_val, b_val) in enumerate(zip(train_k, train_b)):
		point_value = float(interpolate_smape_value(test_model, test_poly, float(k_val), float(b_val)))
		test_smape[idx] = point_value

		col_idx = _match_k_index(test_k_fine, k_val)
		col_values = test_S_pred[:, col_idx]
		finite_mask = np.isfinite(col_values)
		if np.any(finite_mask):
			col_min = float(np.nanmin(col_values))
			tol = max(float(test_abs_tol), abs(col_min) * float(test_rel_tol))
			test_col_min[idx] = col_min
			test_tol[idx] = tol
			test_delta[idx] = point_value - col_min

		if k_val < band_k_min - edge_tol or k_val > band_k_max + edge_tol:
			continue

		band_idx = _match_k_index(band_k, k_val)
		lower = lower_b[band_idx]
		upper = upper_b[band_idx]
		if not (np.isfinite(lower) and np.isfinite(upper)):
			continue

		retained_mask[idx] = lower - edge_tol <= b_val <= upper + edge_tol

	return {
		'test_smape': test_smape,
		'test_col_min': test_col_min,
		'test_tol': test_tol,
		'test_delta': test_delta,
		'retained_mask': retained_mask,
	}


def _draw_path(ax, path_k, path_b, *, color: str, label: str, linewidth: float = 2.2, zorder: int = 6):
	path_k = np.asarray(path_k, dtype=float)
	path_b = np.asarray(path_b, dtype=float)
	if path_k.size == 0:
		return
	if path_k.size == 1:
		ax.scatter(path_k, path_b, color=color, s=26, zorder=zorder, label=label)
		return
	ax.plot(path_k, path_b, color=color, linewidth=linewidth, zorder=zorder + 1, label=label)


def _draw_masked_path(
	ax,
	full_k,
	full_b,
	mask,
	*,
	color: str,
	label: str,
	linewidth: float = 2.2,
	zorder: int = 6,
):
	full_k = np.asarray(full_k, dtype=float)
	full_b = np.asarray(full_b, dtype=float)
	mask = np.asarray(mask, dtype=bool)
	if full_k.size == 0 or mask.size != full_k.size or not np.any(mask):
		return

	if int(np.count_nonzero(mask)) == 1:
		ax.scatter(full_k[mask], full_b[mask], color=color, s=26, zorder=zorder, label=label)
		return

	path_k = full_k.copy()
	path_b = full_b.copy()
	path_k[~mask] = np.nan
	path_b[~mask] = np.nan
	ax.plot(path_k, path_b, color=color, linewidth=linewidth, zorder=zorder + 1, label=label)


def _place_legend_above(ax, handles=None, labels=None, ncol: int = 1):
	if handles is None or labels is None:
		handles, labels = ax.get_legend_handles_labels()
	if not handles:
		return
	legend = ax.legend(
		handles,
		labels,
		loc='upper right',
		ncol=ncol,
		frameon=True,
		facecolor='white',
		framealpha=1.0,
		prop={'size': PLOT_TEXT_SIZE},
	)
	legend.set_zorder(30)


def _style_axis(ax, *, title: str | None = None, season_label: str | None = None):
	if title is not None:
		ax.set_title(title, fontsize=PLOT_TEXT_SIZE)
	ax.set_xlabel('k', fontsize=PLOT_TEXT_SIZE)
	ax.set_ylabel('b', fontsize=PLOT_TEXT_SIZE)
	ax.tick_params(axis='both', labelsize=PLOT_TEXT_SIZE)
	if season_label is not None:
		ax.text(
			-0.36,
			0.5,
			season_label,
			transform=ax.transAxes,
			fontsize=PLOT_TEXT_SIZE,
			va='center',
			ha='center',
			zorder=30,
		)


def _draw_unavailable_row(axes_row, *, show_titles: bool, season_label: str, message: str) -> None:
	for idx, ax in enumerate(axes_row):
		if show_titles:
			ax.set_title(PANEL_TITLES[idx], fontsize=PLOT_TEXT_SIZE)
		if idx == 0:
			ax.text(
				-0.36,
				0.5,
				season_label,
				transform=ax.transAxes,
				fontsize=PLOT_TEXT_SIZE,
				va='center',
				ha='center',
				zorder=30,
			)
		ax.set_axis_off()
		ax.text(
			0.5,
			0.5,
			message,
			transform=ax.transAxes,
			fontsize=PLOT_TEXT_SIZE,
			va='center',
			ha='center',
		)


def _draw_selection_row(
	axes_row,
	selection: dict[str, object],
	*,
	show_titles: bool,
	season_label: str | None = None,
	failure_message: str | None = None,
):
	ax1, ax2, ax3 = axes_row
	titles = PANEL_TITLES if show_titles else (None, None, None)

	train_k = np.asarray(selection['train_k'], dtype=float)
	train_b = np.asarray(selection['train_b'], dtype=float)
	band_k = np.asarray(selection['band_k'], dtype=float)
	retained_mask = np.asarray(selection['retained_mask'], dtype=bool)
	band_lower_b = np.asarray(selection['band_lower_b'], dtype=float)
	band_upper_b = np.asarray(selection['band_upper_b'], dtype=float)
	train_k_fine = np.asarray(selection['train_k_fine'], dtype=float)
	train_b_fine = np.asarray(selection['train_b_fine'], dtype=float)
	train_S_pred = np.asarray(selection['train_S_pred'], dtype=float)
	test_k_fine = np.asarray(selection['test_k_fine'], dtype=float)
	test_b_fine = np.asarray(selection['test_b_fine'], dtype=float)
	test_S_pred = np.asarray(selection['test_S_pred'], dtype=float)
	wind_k_fine = np.asarray(selection['wind_k_fine'], dtype=float)
	wind_b_fine = np.asarray(selection['wind_b_fine'], dtype=float)
	wind_S_pred = np.asarray(selection['wind_S_pred'], dtype=float)
	train_r2 = float(selection['train_r2'])
	test_r2 = float(selection['test_r2'])
	wind_r2 = float(selection['wind_r2'])
	min_result = selection['min_result']

	train_display = np.clip(train_S_pred, 0, 100)
	im1 = ax1.imshow(
		train_display,
		cmap='viridis_r',
		origin='lower',
		extent=[train_k_fine.min(), train_k_fine.max(), train_b_fine.min(), train_b_fine.max()],
		aspect='auto',
		vmin=0,
		vmax=100,
	)
	_style_axis(ax1, title=titles[0], season_label=season_label)
	ax1.text(
		0.02,
		0.96,
		f'R2={train_r2:.4f}',
		transform=ax1.transAxes,
		fontsize=PLOT_TEXT_SIZE,
		va='top',
		bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9),
		zorder=30,
	)
	_draw_path(ax1, train_k, train_b, color=CANDIDATE_PATH_COLOR, label='Candidate Path')
	_place_legend_above(
		ax1,
		handles=[Line2D([0], [0], color=CANDIDATE_PATH_COLOR, linewidth=2.2)],
		labels=['Candidate Path'],
		ncol=1,
	)

	test_display = np.clip(test_S_pred, 0, 100)
	ax2.imshow(
		test_display,
		cmap='viridis_r',
		origin='lower',
		extent=[test_k_fine.min(), test_k_fine.max(), test_b_fine.min(), test_b_fine.max()],
		aspect='auto',
		vmin=0,
		vmax=100,
	)
	_style_axis(ax2, title=titles[1])
	ax2.text(
		0.02,
		0.96,
		f'R2={test_r2:.4f}',
		transform=ax2.transAxes,
		fontsize=PLOT_TEXT_SIZE,
		va='top',
		bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9),
		zorder=30,
	)
	valid_band = np.isfinite(band_lower_b) & np.isfinite(band_upper_b)
	if np.any(valid_band):
		ax2.fill_between(
			band_k[valid_band],
			band_lower_b[valid_band],
			band_upper_b[valid_band],
			color='white',
			alpha=0.28,
			label='Acceptable Band',
		)
	omitted_mask = ~retained_mask
	_draw_masked_path(ax2, train_k, train_b, omitted_mask, color=OMITTED_PATH_COLOR, label='Omitted Path', linewidth=2.0, zorder=7)
	_draw_masked_path(ax2, train_k, train_b, retained_mask, color=RETAINED_PATH_COLOR, label='Retained Path', linewidth=2.0, zorder=8)
	_place_legend_above(
		ax2,
		handles=[
			Patch(facecolor='white', edgecolor='black', linewidth=1.0, alpha=0.28),
			Line2D([0], [0], color=RETAINED_PATH_COLOR, linewidth=2.0),
			Line2D([0], [0], color=OMITTED_PATH_COLOR, linewidth=2.0),
		],
		labels=['Acceptable Band', 'Retained Path', 'Omitted Path'],
		ncol=1,
	)

	wind_display = np.clip(wind_S_pred, 0, 100)
	ax3.imshow(
		wind_display,
		cmap='viridis_r',
		origin='lower',
		extent=[wind_k_fine.min(), wind_k_fine.max(), wind_b_fine.min(), wind_b_fine.max()],
		aspect='auto',
		vmin=0,
		vmax=100,
	)
	_style_axis(ax3, title=titles[2])
	ax3.text(
		0.02,
		0.96,
		f'R2={wind_r2:.4f}',
		transform=ax3.transAxes,
		fontsize=PLOT_TEXT_SIZE,
		va='top',
		bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9),
		zorder=30,
	)
	_draw_masked_path(ax3, train_k, train_b, retained_mask, color=RETAINED_PATH_COLOR, label='Retained Path', linewidth=2.0, zorder=8)
	if min_result is not None:
		ax3.scatter(
			min_result['k'],
			min_result['b'],
			color='gold',
			edgecolors='black',
			linewidth=1.2,
			s=96,
			zorder=10,
			label=f"Optimal (k, b):\n({min_result['k']:.2f}, {min_result['b']:.2f})",
		)
	third_handles = [Line2D([0], [0], color=RETAINED_PATH_COLOR, linewidth=2.0)]
	third_labels = ['Retained Path']
	if min_result is not None:
		third_handles.append(
			Line2D(
				[0],
				[0],
				marker='o',
				markersize=9,
				linestyle='None',
				markerfacecolor='gold',
				markeredgecolor='black',
			)
		)
		third_labels.append(f"Optimal (k, b):\n({min_result['k']:.2f}, {min_result['b']:.2f})")
	_place_legend_above(ax3, handles=third_handles, labels=third_labels, ncol=1)
	if failure_message is not None:
		ax3.text(
			0.02,
			0.04,
			failure_message,
			transform=ax3.transAxes,
			fontsize=PLOT_TEXT_SIZE,
			va='bottom',
			ha='left',
			bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85),
		)

	return im1


def plot_selection_figure(*, selection: dict[str, object], output_plot: Path, failure_message: str | None = None):
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	im1 = _draw_selection_row(axes, selection, show_titles=True, failure_message=failure_message)
	fig.subplots_adjust(left=0.06, right=0.93, wspace=0.24)
	cbar_ax = fig.add_axes([0.955, 0.16, 0.012, 0.68])
	cbar = fig.colorbar(im1, cax=cbar_ax)
	cbar.ax.tick_params(labelsize=PLOT_TEXT_SIZE)
	plt.savefig(output_plot, dpi=300, bbox_inches='tight', format='jpg')
	plt.close()


def plot_province_selection_figure(
	*,
	prov: str,
	season_entries: dict[str, dict[str, object]],
	output_plot: Path,
):
	fig, axes = plt.subplots(4, 3, figsize=(18, 22))
	im_for_cbar = None

	for row_idx, season in enumerate(SEASON_ORDER):
		row_axes = axes[row_idx]
		entry = season_entries.get(season)
		season_label = SEASON_LABELS.get(season, season)
		show_titles = row_idx == 0
		if entry is None or entry.get('selection') is None:
			message = 'Unavailable'
			if entry is not None and entry.get('failure_message'):
				message = str(entry['failure_message'])
			_draw_unavailable_row(row_axes, show_titles=show_titles, season_label=season_label, message=message)
			continue

		im_current = _draw_selection_row(
			row_axes,
			entry['selection'],
			show_titles=show_titles,
			season_label=season_label,
			failure_message=entry.get('failure_message'),
		)
		if im_for_cbar is None:
			im_for_cbar = im_current

	fig.subplots_adjust(left=0.14, right=0.93, top=0.98, bottom=0.05, wspace=0.24, hspace=0.34)
	if im_for_cbar is not None:
		cbar_ax = fig.add_axes([0.955, 0.12, 0.012, 0.76])
		cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
		cbar.ax.tick_params(labelsize=PLOT_TEXT_SIZE)
	output_plot.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_plot, dpi=300, bbox_inches='tight', format='jpg')
	plt.close()


def _compute_selection(
	*,
	train_csv: Path,
	test_csv: Path,
	wind_csv: Path,
	poly_degree: int,
	test_abs_tol: float,
	test_rel_tol: float,
) -> dict[str, object]:
	train_df = load_smape_csv(train_csv)
	test_df = load_smape_csv(test_csv)
	wind_df = load_smape_csv(wind_csv)

	train_k, train_b, train_model, train_poly, train_k_fine, train_b_fine, train_S_pred, train_r2 = get_optimal_path_from_dataframe(
		train_df,
		poly_degree=poly_degree,
	)
	if train_k is None or train_model is None:
		raise RuntimeError(f'无法从 train csv 中提取有效 optimal path: {train_csv}')

	test_model, test_poly, test_k_fine, test_b_fine, test_S_pred, test_r2 = fit_surface_from_dataframe(
		test_df,
		poly_degree=poly_degree,
	)
	if test_model is None:
		raise RuntimeError(f'无法从 test csv 中拟合有效模型: {test_csv}')
	test_path_k, test_path_b, _, _, _, _, _, _ = get_optimal_path_from_dataframe(
		test_df,
		poly_degree=poly_degree,
	)
	if test_path_k is None or test_path_b is None:
		raise RuntimeError(f'无法从 test csv 中提取有效 optimal path: {test_csv}')

	wind_model, wind_poly, wind_k_fine, wind_b_fine, wind_S_pred, wind_r2 = fit_surface_from_dataframe(
		wind_df,
		poly_degree=poly_degree,
	)
	if wind_model is None:
		raise RuntimeError(f'无法从 wind csv 中拟合有效模型: {wind_csv}')

	band_info = build_test_band(
			test_path_k,
		test_model=test_model,
		test_poly=test_poly,
		test_k_fine=test_k_fine,
		test_b_fine=test_b_fine,
		test_S_pred=test_S_pred,
		test_abs_tol=test_abs_tol,
		test_rel_tol=test_rel_tol,
	)
	filter_info = filter_train_path_by_test_band(
		train_k,
		train_b,
		band_k=band_info['band_k'],
		lower_b=band_info['lower_b'],
		upper_b=band_info['upper_b'],
		test_model=test_model,
		test_poly=test_poly,
		test_k_fine=test_k_fine,
		test_S_pred=test_S_pred,
		test_abs_tol=test_abs_tol,
		test_rel_tol=test_rel_tol,
	)
	retained_mask = filter_info['retained_mask']
	retained_k = np.asarray(train_k)[retained_mask]
	retained_b = np.asarray(train_b)[retained_mask]
	if retained_k.size == 0:
		raise RuntimeError('经过 testing 筛选后未保留任何 generation path 点')

	min_result = find_min_wind_on_gen_op(retained_k, retained_b, wind_model, wind_poly)
	final_k = float(min_result['k'])
	final_b = float(min_result['b'])
	final_train_smape = float(interpolate_smape_value(train_model, train_poly, final_k, final_b))
	final_test_smape = float(interpolate_smape_value(test_model, test_poly, final_k, final_b))
	final_k_idx = _match_k_index(np.asarray(train_k, dtype=float), final_k)
	final_test_delta = float(filter_info['test_delta'][final_k_idx]) if np.isfinite(filter_info['test_delta'][final_k_idx]) else np.nan

	return {
		'train_k': np.asarray(train_k, dtype=float),
		'train_b': np.asarray(train_b, dtype=float),
		'band_k': np.asarray(band_info['band_k'], dtype=float),
		'retained_mask': np.asarray(retained_mask, dtype=bool),
		'retained_k': retained_k,
		'retained_b': retained_b,
		'band_lower_b': band_info['lower_b'],
		'band_upper_b': band_info['upper_b'],
		'train_k_fine': np.asarray(train_k_fine, dtype=float),
		'train_b_fine': np.asarray(train_b_fine, dtype=float),
		'train_S_pred': np.asarray(train_S_pred, dtype=float),
		'train_r2': float(train_r2),
		'test_k_fine': np.asarray(test_k_fine, dtype=float),
		'test_b_fine': np.asarray(test_b_fine, dtype=float),
		'test_S_pred': np.asarray(test_S_pred, dtype=float),
		'test_r2': float(test_r2),
		'wind_k_fine': np.asarray(wind_k_fine, dtype=float),
		'wind_b_fine': np.asarray(wind_b_fine, dtype=float),
		'wind_S_pred': np.asarray(wind_S_pred, dtype=float),
		'wind_r2': float(wind_r2),
		'min_result': min_result,
		'final_row': {
			'w_smape': float(min_result['smape']),
			'k': final_k,
			'b': final_b,
			'train_smape': final_train_smape,
			'test_smape': final_test_smape,
			'test_delta': final_test_delta,
			'retained_point_count': int(retained_k.size),
			'candidate_point_count': int(len(train_k)),
			'test_abs_tol': float(test_abs_tol),
			'test_rel_tol': float(test_rel_tol),
		},
	}


def process_one_combo(
	train_csv: Path,
	test_csv: Path,
	wind_csv: Path,
	output_plot: Path | None,
	plot_mode: str,
	poly_degree: int = 7,
	test_abs_tol: float = DEFAULT_TEST_ABS_TOL,
	test_rel_tol: float = DEFAULT_TEST_REL_TOL,
) -> dict[str, object]:
	if plot_mode not in PLOT_MODES:
		raise ValueError(f'plot_mode 必须是 {PLOT_MODES} 之一，当前为: {plot_mode}')

	meta = parse_aggregate_train_filename(train_csv)
	selection = _compute_selection(
		train_csv=train_csv,
		test_csv=test_csv,
		wind_csv=wind_csv,
		poly_degree=poly_degree,
		test_abs_tol=test_abs_tol,
		test_rel_tol=test_rel_tol,
	)
	final_row = selection['final_row']

	return {
		'train_csv': str(train_csv),
		'test_csv': str(test_csv),
		'wind_csv': str(wind_csv),
		'prov': meta['prov'] if meta is not None else None,
		'season': meta['season'] if meta is not None else None,
		'output_plot': None,
		'selection': selection if plot_mode == 'all' else None,
		**final_row,
	}


def _process_task(task: tuple[str, str, str, str, str, int, float, float]) -> dict[str, object]:
	train_csv_str, test_csv_str, wind_csv_str, output_plot_str, plot_mode, poly_degree, test_abs_tol, test_rel_tol = task
	train_csv = Path(train_csv_str)
	test_csv = Path(test_csv_str)
	wind_csv = Path(wind_csv_str)
	output_plot = Path(output_plot_str) if output_plot_str else None
	meta = parse_aggregate_train_filename(train_csv)
	try:
		result = process_one_combo(
			train_csv=train_csv,
			test_csv=test_csv,
			wind_csv=wind_csv,
			output_plot=output_plot,
			plot_mode=plot_mode,
			poly_degree=poly_degree,
			test_abs_tol=test_abs_tol,
			test_rel_tol=test_rel_tol,
		)
		result['ok'] = True
		return result
	except Exception as exc:
		return {
			'ok': False,
			'train_csv': str(train_csv),
			'test_csv': str(test_csv),
			'wind_csv': str(wind_csv),
			'prov': meta['prov'] if meta is not None else None,
			'season': meta['season'] if meta is not None else None,
			'error': str(exc),
		}


def build_province_plot_figures(
	*,
	success_results: list[dict[str, object]],
	failed_results: list[dict[str, object]],
	output_dir: Path,
) -> list[str]:
	province_entries: dict[str, dict[str, dict[str, object]]] = {}

	for item in success_results:
		prov = item.get('prov')
		season = item.get('season')
		selection = item.get('selection')
		if not prov or not season or selection is None:
			continue
		province_entries.setdefault(str(prov), {})[str(season)] = {
			'selection': selection,
			'failure_message': None,
		}

	for item in failed_results:
		prov = item.get('prov')
		season = item.get('season')
		if not prov or not season:
			continue
		province_entries.setdefault(str(prov), {})[str(season)] = {
			'selection': None,
			'failure_message': format_plot_failure_message(RuntimeError(str(item.get('error', 'Unavailable')))),
		}

	plot_paths: list[str] = []
	for prov in sorted(province_entries):
		output_plot = output_dir / 'plots' / f'{prov}_optimal_kb.jpg'
		plot_province_selection_figure(
			prov=prov,
			season_entries=province_entries[prov],
			output_plot=output_plot,
		)
		plot_paths.append(str(output_plot))
		for item in success_results:
			if item.get('prov') == prov:
				item['output_plot'] = str(output_plot)

	return plot_paths


def build_summary_csv(success_results: list[dict[str, object]], output_csv: Path) -> None:
	rows: list[dict[str, object]] = []
	for item in success_results:
		train_csv = Path(str(item['train_csv']))
		meta = parse_aggregate_train_filename(train_csv)
		if meta is None:
			continue
		rows.append(
			{
				'prov': meta['prov'],
				'season': meta['season'],
				'w_smape': float(item['w_smape']),
				'k': float(item['k']),
				'b': float(item['b']),
				'train_smape': float(item['train_smape']),
				'test_smape': float(item['test_smape']),
				'test_delta': float(item['test_delta']),
				'retained_point_count': int(item['retained_point_count']),
				'candidate_point_count': int(item['candidate_point_count']),
				'test_abs_tol': float(item['test_abs_tol']),
				'test_rel_tol': float(item['test_rel_tol']),
			}
		)

	df = pd.DataFrame(
		rows,
		columns=[
			'prov', 'season', 'w_smape', 'k', 'b',
			'train_smape', 'test_smape', 'test_delta',
			'retained_point_count', 'candidate_point_count', 'test_abs_tol', 'test_rel_tol',
		],
	)
	if not df.empty:
		df = df.sort_values(by=['prov', 'season'], kind='stable').reset_index(drop=True)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_csv, index=False, encoding='utf-8-sig')


def find_optimal_kb(
	input_dir: Path,
	wind_dir: Path,
	output_dir: Path,
	optimal_kb_csv: Path | None = None,
	run_mode: str = 'compute',
	plot_mode: str = 'failed',
	poly_degree: int = 7,
	max_workers: int | None = 8,
	force_spawn: bool | None = None,
	test_abs_tol: float = DEFAULT_TEST_ABS_TOL,
	test_rel_tol: float = DEFAULT_TEST_REL_TOL,
) -> dict[str, object]:
	if run_mode not in RUN_MODES:
		raise ValueError(f'run_mode 必须是 {RUN_MODES} 之一，当前为: {run_mode}')
	if plot_mode not in PLOT_MODES:
		raise ValueError(f'plot_mode 必须是 {PLOT_MODES} 之一，当前为: {plot_mode}')

	effective_force_spawn = should_force_spawn(force_spawn)
	if effective_force_spawn:
		try:
			multiprocessing.set_start_method('spawn')
		except RuntimeError:
			pass

	input_dir = Path(input_dir).resolve()
	wind_dir = Path(wind_dir).resolve()
	output_dir = Path(output_dir).resolve()
	optimal_kb_csv = optimal_kb_csv.resolve() if optimal_kb_csv is not None else output_dir / 'optimal_kb.csv'

	if poly_degree < 1:
		raise ValueError(f'poly_degree 必须 >= 1, 当前为: {poly_degree}')
	if test_abs_tol < 0:
		raise ValueError(f'test_abs_tol 不能小于 0，当前为: {test_abs_tol}')
	if test_rel_tol < 0:
		raise ValueError(f'test_rel_tol 不能小于 0，当前为: {test_rel_tol}')

	if not input_dir.exists():
		raise FileNotFoundError(f'未找到 input_dir: {input_dir}')
	if not wind_dir.exists():
		raise FileNotFoundError(f'未找到 wind_dir: {wind_dir}')
	output_dir.mkdir(parents=True, exist_ok=True)

	aggregate_dir = input_dir / 'aggregate' if (input_dir / 'aggregate').exists() else input_dir
	train_csv_list = sorted(aggregate_dir.glob('*_mean_train_gen_smape.csv'))
	if not train_csv_list:
		raise FileNotFoundError(f'未找到任何 *_mean_train_gen_smape.csv: {aggregate_dir}')

	ok_count = 0
	skip_count = 0
	fail_count = 0
	success_results: list[dict[str, object]] = []
	failed_results: list[dict[str, object]] = []
	skipped_items: list[dict[str, object]] = []
	tasks: list[tuple[str, str, str, str, str, int, float, float]] = []

	for train_csv in train_csv_list:
		meta = parse_aggregate_train_filename(train_csv)
		if meta is None:
			skip_count += 1
			skipped_items.append({'train_csv': str(train_csv), 'reason': 'unrecognized_filename'})
			continue

		prov = meta['prov']
		season = meta['season']
		test_csv = aggregate_dir / f'{prov}_{season}_mean_test_gen_smape.csv'
		wind_csv = wind_dir / f'{prov}_{season}_w_smape_enums.csv'
		if not test_csv.exists():
			skip_count += 1
			skipped_items.append({'train_csv': str(train_csv), 'test_csv': str(test_csv), 'reason': 'missing_test_csv'})
			continue
		if not wind_csv.exists():
			skip_count += 1
			skipped_items.append({'train_csv': str(train_csv), 'wind_csv': str(wind_csv), 'reason': 'missing_wind_csv'})
			continue

		output_plot = output_dir / 'plots' / f'{prov}_{season}_optimal_kb.jpg'
		tasks.append(
			(
				str(train_csv),
				str(test_csv),
				str(wind_csv),
				str(output_plot),
				plot_mode,
				poly_degree,
				test_abs_tol,
				test_rel_tol,
			)
		)

	if tasks:
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			futures = [executor.submit(_process_task, task) for task in tasks]
			for future in as_completed(futures):
				result = future.result()
				if result['ok']:
					ok_count += 1
					success_results.append(result)
				else:
					fail_count += 1
					failed_results.append(result)

	province_plot_paths: list[str] = []
	if plot_mode == 'all':
		province_plot_paths = build_province_plot_figures(
			success_results=success_results,
			failed_results=failed_results,
			output_dir=output_dir,
		)

	if run_mode == 'compute':
		build_summary_csv(success_results, optimal_kb_csv)


	print('\n处理完成:', flush=True)
	print(f'- 提交任务: {len(tasks)}', flush=True)
	print(f'- 成功: {ok_count}', flush=True)
	print(f'- 跳过: {skip_count}', flush=True)
	print(f'- 失败: {fail_count}', flush=True)

	return {
		'input_dir': str(input_dir),
		'aggregate_dir': str(aggregate_dir),
		'wind_dir': str(wind_dir),
		'output_dir': str(output_dir),
		'optimal_kb_csv': str(optimal_kb_csv),
		'submitted_task_count': len(tasks),
		'success_count': ok_count,
		'skip_count': skip_count,
		'failure_count': fail_count,
		'test_abs_tol': float(test_abs_tol),
		'test_rel_tol': float(test_rel_tol),
		'plot_paths': province_plot_paths,
		'success_results': success_results,
		'skipped_results': skipped_items,
		'failed_results': failed_results,
	}


def parse_args():
	parser = argparse.ArgumentParser(
		description='读取 prov_gen_smape.py 聚合后的 train/test generation SMAPE 底图，并结合 wind SMAPE 选择最终最优点。'
	)
	parser.add_argument(
		'--input-dir',
		type=Path,
		default=Path('D:\\HKU PhD\\wind data paper\\prov_gen_smape'),
		help='prov_gen_smape.py 的输出目录（会读取 aggregate/*_mean_train_gen_smape.csv）',
	)
	parser.add_argument(
		'--wind-dir',
		type=Path,
		default=Path('D:\\HKU PhD\\wind data paper\\prov_w_smape_enums'),
		help='风速 SMAPE 目录，文件名格式 {省简称}_{season}_w_smape_enums.csv',
	)
	parser.add_argument(
		'--output-dir',
		type=Path,
		default=Path('D:\\HKU PhD\\wind data paper\\optimal_kb'),
		help='结果输出根目录',
	)
	parser.add_argument(
		'--optimal-kb-csv',
		type=Path,
		default=None,
		help='最终 optimal kb CSV 路径，默认 output_dir/optimal_kb.csv',
	)
	parser.add_argument(
		'--run-mode',
		choices=RUN_MODES,
		default='compute',
		help='运行模式：compute=计算并输出 summary；plot=仅重算并作图，不重写 summary',
	)
	parser.add_argument(
		'--plot-mode',
		choices=PLOT_MODES,
		default='failed',
		help='作图模式：all=全部输出，failed=兼容保留，none=不输出图',
	)
	parser.add_argument(
		'--plot',
		dest='plot_mode',
		action='store_const',
		const='all',
		help='兼容快捷参数，等价于 --plot-mode all',
	)
	parser.add_argument(
		'--no-plot',
		dest='plot_mode',
		action='store_const',
		const='none',
		help='兼容快捷参数，等价于 --plot-mode none',
	)
	parser.add_argument('--poly-degree', type=int, default=8, help='多项式拟合阶数')
	parser.add_argument('--max-workers', type=int, default=16, help='并行子任务进程数')
	parser.add_argument('--test-abs-tol', type=float, default=DEFAULT_TEST_ABS_TOL, help='testing 验线的绝对 SMAPE 容忍阈值')
	parser.add_argument('--test-rel-tol', type=float, default=DEFAULT_TEST_REL_TOL, help='testing 验线的相对 SMAPE 容忍阈值')
	spawn_group = parser.add_mutually_exclusive_group()
	spawn_group.add_argument(
		'--force-spawn',
		dest='force_spawn',
		action='store_true',
		help='强制使用 spawn 启动子进程；Windows 默认开启。',
	)
	spawn_group.add_argument(
		'--no-force-spawn',
		dest='force_spawn',
		action='store_false',
		help='禁用强制 spawn，使用平台默认的多进程启动方式。',
	)
	parser.set_defaults(force_spawn=None)
	return parser.parse_args()


def cli_main() -> dict[str, object]:
	args = parse_args()
	return find_optimal_kb(
		input_dir=args.input_dir,
		wind_dir=args.wind_dir,
		output_dir=args.output_dir,
		optimal_kb_csv=args.optimal_kb_csv,
		run_mode=args.run_mode,
		plot_mode=args.plot_mode,
		poly_degree=args.poly_degree,
		max_workers=args.max_workers,
		force_spawn=args.force_spawn,
		test_abs_tol=args.test_abs_tol,
		test_rel_tol=args.test_rel_tol,
	)


if __name__ == '__main__':
	cli_main()