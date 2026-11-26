# -*- coding: utf-8 -*-
"""Wind-speed surface and valley utilities for the final M4 workflow.

This module contains only the generic wind-surface functions used by
``m4_selection.py``.  The old M2 generation-surface selector was moved
to ``legacy/find_optimal_kb_with_m2.py`` and is not imported here.
"""
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
