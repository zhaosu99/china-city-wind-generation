from typing import Optional, Dict, Tuple, Union
import re

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# 功率曲线实现已收敛到 power_curve.py（**唯一实现**）。
# 此处原先各有一份逐字重复的 generate_power_curve；其自适应分支存在边界索引缺陷
# （左边界 kernel[:end-start] 未按 i 偏移、切片后未重新归一化），
# 已被 tests/test_power_curve_smoothing.py 捕获。
# 保留本 import 以兼容本模块内部及外部对该名字的引用。
# ---------------------------------------------------------------------------
from power_curve import (  # noqa: E402
    generate_power_curve,
    raw_capacity_factors,
    smooth_capacity_factors,
)

def generate_daily_time_axis(year: int) -> xr.DataArray:
    """
    生成指定年份的逐日时间轴。

    Args:
        year (int): 指定年份。

    Returns:
        xr.DataArray: 该年份的逐日 time 坐标。
    """
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return xr.DataArray(
        dates,
        dims=['time'],
        coords={'time': dates},
        name='time',
    )


def cdmet_10m_w(
    ds: xr.Dataset,
    year: int,
    w10_name: str = 'win',
    time_name: str = 'time',
) -> xr.DataArray:
    """
    从 CDMet 数据集中提取 10m 风速，并重建指定年份的逐日时间轴。

    Args:
        ds (xr.Dataset): 输入 CDMet 数据集。
        year (int): 对应年份，用于生成新的逐日时间坐标。
        w10_name (str): 10m 风速变量名，默认 'win'。
        time_name (str): 时间维名称，默认 'time'。

    Returns:
        xr.DataArray: 带有新逐日时间轴的 10m 风速 DataArray。
    """
    if w10_name not in ds:
        raise KeyError(f"Dataset 中缺少变量: {w10_name}")

    da = ds[w10_name]

    if time_name not in da.dims:
        raise ValueError(f"变量 {w10_name} 必须包含时间维度: {time_name}")

    time_axis_daily = generate_daily_time_axis(year)
    if da.sizes[time_name] != time_axis_daily.sizes['time']:
        raise ValueError(
            f"{w10_name} 的时间长度与年份 {year} 的日序列长度不一致: "
            f"{da.sizes[time_name]} vs {time_axis_daily.sizes['time']}"
        )

    da = da.assign_coords({time_name: time_axis_daily.values})
    da = da.rename('w10')
    da.attrs.update({
        'long_name': 'CDMet 10m wind speed',
        'units': da.attrs.get('units', 'm s-1'),
        'source_variable_name': w10_name,
        'source_dataset': 'cdmet',
        'time_axis_type': 'daily',
        'time_axis_year': int(year),
    })
    return da


def era5_10m_w(
    ds: xr.Dataset,
    u10_name: str = 'u10',
    v10_name: str = 'v10',
) -> xr.DataArray:
    """
    根据 ERA5 的 10m 风场分量计算风速。

    Args:
        ds (xr.Dataset): 输入数据集。
        u10_name (str): 10m 纬向风分量变量名。
        v10_name (str): 10m 经向风分量变量名。

    Returns:
        xr.DataArray: 10m 风速 DataArray。
    """
    if u10_name not in ds or v10_name not in ds:
        raise KeyError(f"Dataset 中缺少变量: {u10_name} 或 {v10_name}")

    w10 = np.sqrt(ds[u10_name] ** 2 + ds[v10_name] ** 2)
    w10 = w10.rename('w10')
    w10.attrs.update({
        'long_name': 'ERA5 10m wind speed',
        'units': 'm s-1',
        'source_u': u10_name,
        'source_v': v10_name,
        'formula': 'sqrt(u10^2 + v10^2)',
    })
    return w10


def era5_100m_w(
    ds: xr.Dataset,
    u100_name: str = 'u100',
    v100_name: str = 'v100',
) -> xr.DataArray:
    """
    根据 ERA5 的 100m 风场分量计算风速。

    Args:
        ds (xr.Dataset): 输入数据集。
        u100_name (str): 100m 纬向风分量变量名。
        v100_name (str): 100m 经向风分量变量名。

    Returns:
        xr.DataArray: 100m 风速 DataArray。
    """
    if u100_name not in ds or v100_name not in ds:
        raise KeyError(f"Dataset 中缺少变量: {u100_name} 或 {v100_name}")

    w100 = np.sqrt(ds[u100_name] ** 2 + ds[v100_name] ** 2)
    w100 = w100.rename('w100')
    w100.attrs.update({
        'long_name': 'ERA5 100m wind speed',
        'units': 'm s-1',
        'source_u': u100_name,
        'source_v': v100_name,
        'formula': 'sqrt(u100^2 + v100^2)',
    })
    return w100


def era5_alpha(
    w10: xr.DataArray,
    w100: xr.DataArray,
    alpha_method: str = 'moving_average_hourly',
    alpha_window: int = 3,
) -> xr.DataArray:
    """
    根据 10m 与 100m 风速计算风切变指数 alpha。

    Args:
        w10 (xr.DataArray): 10m 风速，通常维度为 (time, lat, lon)。
        w100 (xr.DataArray): 100m 风速，通常维度为 (time, lat, lon)。
        alpha_method (str): alpha 处理方式，可选 'hourly' / 'daily' / 'moving_average_hourly'。
        alpha_window (int): 滑动平均窗口（小时），仅在 moving_average_hourly 时生效。

    Returns:
        xr.DataArray: alpha 时间序列（与输入相同维度和坐标）。
    """
    if w10.shape != w100.shape:
        raise ValueError(f"w10 与 w100 形状不一致: {w10.shape} vs {w100.shape}")

    # 避免除零和 log 非法
    w10_safe = xr.where(w10 < 1e-3, 1e-3, w10)
    w100_safe = xr.where(w100 < 1e-3, 1e-3, w100)

    hourly_alpha = np.log10(w100_safe / w10_safe)

    if alpha_method == 'hourly':
        alpha_da = hourly_alpha
        alpha_desc = 'hourly (no smoothing)'

    elif alpha_method == 'daily':
        if 'time' not in hourly_alpha.dims:
            raise ValueError("alpha_method='daily' 需要 time 维度")

        time_axis = hourly_alpha['time']
        if time_axis.ndim != 1:
            raise ValueError(f"alpha_method='daily' 需要一维 time 坐标，得到 ndim={time_axis.ndim}")
        if not np.issubdtype(time_axis.dtype, np.datetime64):
            raise ValueError(
                "alpha_method='daily' 需要 datetime64 类型的 time 坐标，"
                f"得到 {time_axis.dtype}（无法按日历日分组）"
            )
        if time_axis.size > 1:
            _steps = np.diff(time_axis.values.astype('datetime64[s]').astype('int64'))
            if np.any(_steps <= 0):
                raise ValueError(
                    "alpha_method='daily' 需要时间坐标严格递增；当前存在非递增或重复时刻，"
                    "按日历日分组会把不同顺序的小时混在一起，故拒绝静默计算"
                )

        # 按**日历日**求均值，再广播回逐小时轴。
        # ⚠️ 不使用 `.values.reshape(n_days, 24, ...)`：reshape 隐含"time 必须在第 0 维"，
        #    当 dims=(lat, lon, time) 时会直接抛 CoordinateValidationError（BUG-4）。
        #    resample('1D') 只按 time 坐标分箱 ⇒ 与维度顺序无关；
        #    再用向量化 sel 回填，结果与输入同 dims / 同坐标 / 同长度。
        day_starts = time_axis.dt.floor('D')
        daily_mean = hourly_alpha.resample(time='1D').mean()
        alpha_da = daily_mean.sel(time=day_starts)
        alpha_da = alpha_da.assign_coords(time=time_axis.values).transpose(*hourly_alpha.dims)

        alpha_desc = 'daily mean (calendar day)'

    elif alpha_method == 'moving_average_hourly':
        if not isinstance(alpha_window, int) or alpha_window <= 0:
            raise ValueError(f"alpha_window 必须为正整数，得到: {alpha_window}")
        if 'time' not in hourly_alpha.dims:
            raise ValueError("alpha_method='moving_average_hourly' 需要 time 维度")

        alpha_da = hourly_alpha.rolling(
            time=alpha_window,
            center=True,
            min_periods=1,
        ).mean()
        alpha_desc = f'rolling mean (alpha_window={alpha_window}h, center=True)'

    else:
        raise ValueError(
            f"alpha_method 必须为 'hourly', 'daily' 或 'moving_average_hourly'，得到: {alpha_method}"
        )

    alpha_da = alpha_da.rename('alpha')
    alpha_da.attrs.update({
        'long_name': 'ERA5 wind shear exponent alpha',
        'formula': 'log10(w100 / w10)',
        'source_w10_name': w10.name if w10.name is not None else 'w10',
        'source_w100_name': w100.name if w100.name is not None else 'w100',
        'alpha_method': alpha_method,
        'alpha_processing_description': alpha_desc,
        'alpha_rolling_window_hours': alpha_window if alpha_method == 'moving_average_hourly' else None,
        'min_speed_threshold': 1e-3,
    })

    return alpha_da


def era5_rou(
    ds: xr.Dataset,
    z2: Optional[float] = None,
    t2m_name: str = 't2m',
    d2m_name: str = 'd2m',
    sp_name: str = 'sp',
    z_name: str = 'z',
    z2_above_orography: Optional[float] = None,
) -> xr.DataArray:
    """
    基于 ERA5 变量计算目标高度空气密度 rho。

    参考 `farm_era5_hub_height_rou.py` 中 estimate_air_density_hourly 的核心公式，
    输入应为包含 t2m/d2m/sp 与 geopotential(z) 的数据集。

    Args:
        ds (xr.Dataset): 输入数据集。
        z2 (Optional[float]): 目标高度(相对海平面, m)。与 `z2_above_orography` 二选一。
        t2m_name (str): 2m 气温变量名。
        d2m_name (str): 2m 露点温度变量名。
        sp_name (str): 地表气压变量名。
        z_name (str): 位势变量名。
        z2_above_orography (Optional[float]): **无场址海拔时的回退模式**：目标高度 = ERA5 格点
            地形位势高度 H1 + 该值（通常传轮毂高度）。此时 `H2 − H1 ≈ 轮毂高度` 恰好是
            "从 ERA5 地面向上外推到轮毂"的正确配置。逐格点计算，因此不依赖场址 DEM。

    Returns:
        xr.DataArray: 空气密度 rho，维度与 t2m 一致。
    """
    if z2 is None and z2_above_orography is None:
        raise ValueError("era5_rou 需要 z2（场址海拔+轮毂）或 z2_above_orography（仅有轮毂高）之一")
    if z2 is not None and z2 < 0:
        raise ValueError(f"z2 不能为负数，得到: {z2}")
    if z2_above_orography is not None and z2_above_orography < 0:
        raise ValueError(f"z2_above_orography 不能为负数，得到: {z2_above_orography}")

    required_vars = [t2m_name, d2m_name, sp_name, z_name]
    missing_vars = [name for name in required_vars if name not in ds]
    if missing_vars:
        raise KeyError(f"Dataset 中缺少变量: {missing_vars}")

    t2m = ds[t2m_name]
    d2m = ds[d2m_name]
    sp = ds[sp_name]
    z = ds[z_name]

    if t2m.shape != d2m.shape or t2m.shape != sp.shape:
        raise ValueError(
            f"t2m/d2m/sp 形状不一致: t2m={t2m.shape}, d2m={d2m.shape}, sp={sp.shape}"
        )

    if 'time' not in t2m.dims:
        raise ValueError("t2m 必须包含 time 维")

    # 将 z 对齐到与 t2m 相同维度
    if 'time' not in z.dims:
        z_aligned = z.expand_dims(time=t2m.time).transpose(*t2m.dims)
    else:
        if z.shape != t2m.shape:
            raise ValueError(f"z 含 time 维时，形状需与 t2m 一致: z={z.shape}, t2m={t2m.shape}")
        z_aligned = z

    # 常数
    g0 = 9.80665
    Rd = 287.05
    epsilon = 0.622
    L = -0.0065
    e0 = 611.0
    b = 17.27
    T1_ref = 273.16
    T2_ref = 35.86
    R0 = 6.357e6

    H1 = z_aligned / g0
    e = e0 * np.exp(b * (d2m - T1_ref) / (d2m - T2_ref))
    q1 = epsilon * e / sp
    Tv1 = t2m * (1.0 + 0.61 * q1)
    if z2 is None:
        # 回退模式：以 ERA5 格点地形 H1 为地面，向**上**外推 z2_above_orography（逐格点）。
        # 这样 H2 − H1 ≈ z2_above_orography（轮毂高度），不会出现"向地下外推"。
        z2_grid = H1 + float(z2_above_orography)
        H2 = (R0 * z2_grid) / (R0 + z2_grid)
    else:
        H2 = (R0 * z2) / (R0 + z2)
    Tv2 = Tv1 + L * (H2 - H1)

    # 气压随高度的变化（静力平衡 + 理想气体 + 温度直减 T = T1 + L·Δz）：
    #     dP/P = -(g/(Rd·T)) dz,  dz = dT/L
    #  ⇒ ln(P2/P1) = -(g/(Rd·L))·ln(T2/T1)
    #  ⇒ P2/P1 = (T2/T1) ** (-g/(Rd·L))
    # 本代码 L = -0.0065 K/m（高度升高 ⇒ 温度降低），故指数应为**正**的 +5.2559。
    # ⚠️ 2026-09-30 修正：原实现写作 `g0 / (Rd * L)`（负指数），
    #    导致 P2 > sp、密度随高度【上升】（实측 z2=100→3000 m 时 rho 1.238→1.890），物理上不可能。
    #    正确写法为 `-g0 / (Rd * L)`。该缺陷与 OmniNet 的 _air_density_at_hub 同源，需一并修正。
    exponent = -g0 / (Rd * L)
    ratio = Tv2 / Tv1
    ratio = xr.where(ratio < 1e-3, 1e-3, ratio)
    ratio = xr.where(ratio > 1e3, 1e3, ratio)
    P2 = sp * (ratio ** exponent)
    rho = P2 / (Rd * Tv2)

    rho = rho.rename('rho')
    rho.attrs.update({
        'long_name': 'Air density at target elevation z2',
        'units': 'kg m**-3',
        'method': 'WAsP 12 ERA5 (Floors & Nielsen, 2019)',
        'formula': 'rho = P2 / (Rd * Tv2)',
        'reference': 'Energies 2019, 12, 2038',
        'z2_mode': 'explicit_amsl' if z2 is not None else 'era5_orography_plus_offset',
        'lapse_rate_L_K_per_m': L,
        'source_t2m_name': t2m_name,
        'source_d2m_name': d2m_name,
        'source_sp_name': sp_name,
        'source_z_name': z_name,
    })
    if z2 is not None:
        rho.attrs['z2_input_m_amsl'] = float(z2)
    else:
        rho.attrs['z2_offset_above_era5_orography_m'] = float(z2_above_orography)
    return rho



def vertical_extrap_w(
    w10: Optional[xr.DataArray],
    w100: Optional[xr.DataArray],
    alpha_da: xr.DataArray,
    hub_height: float,
) -> xr.DataArray:
    """
    根据 10m/100m 风速和风切变指数计算轮毂高度风速（垂直外推）。

    采用风切变指数法，选择最接近目标高度的参考高度进行外推。

    公式:
        若 hub_height 更接近 10m:
            W_hub = W10 * (hub_height / 10.0) ** alpha
        若 hub_height 更接近 100m:
            W_hub = W100 * (hub_height / 100.0) ** alpha

    Args:
        w10 (Optional[xr.DataArray]): 10m 风速，可为 None。
        w100 (Optional[xr.DataArray]): 100m 风速，可为 None。
        alpha_da (xr.DataArray): 风切变指数 alpha。
        hub_height (float): 目标轮毂高度（米）。

    Returns:
        xr.DataArray: 轮毂高度风速（未修正）。
    """
    if hub_height <= 0:
        raise ValueError(f"hub_height 必须为正数，得到: {hub_height}")

    if w10 is None and w100 is None:
        raise ValueError("w10 和 w100 不能同时为 None")

    if w10 is not None and w10.shape != alpha_da.shape:
        raise ValueError(f"w10 与 alpha_da 形状不一致: {w10.shape} vs {alpha_da.shape}")
    if w100 is not None and w100.shape != alpha_da.shape:
        raise ValueError(f"w100 与 alpha_da 形状不一致: {w100.shape} vs {alpha_da.shape}")

    if w10 is not None and w100 is not None and w10.shape != w100.shape:
        raise ValueError(f"w10 与 w100 形状不一致: {w10.shape} vs {w100.shape}")

    dist_to_10 = abs(hub_height - 10.0)
    dist_to_100 = abs(hub_height - 100.0)

    if w10 is None:
        reference_w = w100
        reference_height = 100.0
        selected_source = 'w100'
    elif w100 is None:
        reference_w = w10
        reference_height = 10.0
        selected_source = 'w10'
    elif dist_to_10 <= dist_to_100:
        reference_w = w10
        reference_height = 10.0
        selected_source = 'w10'
    else:
        reference_w = w100
        reference_height = 100.0
        selected_source = 'w100'

    hub_w = reference_w * ((hub_height / reference_height) ** alpha_da)

    hub_w = hub_w.rename('hub_height_w')
    hub_w.attrs.update({
        'long_name': 'Hub-height wind speed (vertical extrapolation only)',
        'units': reference_w.attrs.get('units', 'm s-1'),
        'formula': 'w_ref * (hub_height / h_ref) ** alpha',
        'reference_height_selection_rule': 'choose 10m or 100m by nearest height distance',
        'selected_reference': selected_source,
        'reference_height_m': float(reference_height),
        'target_height_m': float(hub_height),
        'source_alpha_name': alpha_da.name if alpha_da.name is not None else 'alpha',
        'rou_adjustment_applied': 0,
    })
    if w10 is not None:
        hub_w.attrs['source_w10_name'] = w10.name if w10.name is not None else 'w10'
    if w100 is not None:
        hub_w.attrs['source_w100_name'] = w100.name if w100.name is not None else 'w100'
    return hub_w


def rou_adjustment_w(
    wind_da: xr.DataArray,
    rou_da: xr.DataArray,
    reference_air_density: float = 1.225,
) -> xr.DataArray:
    """
    对风速应用空气密度修正。

    公式:
        W_adjusted = W_original * (rho / rho_ref) ** (1/3)

    Args:
        wind_da (xr.DataArray): 输入风速 DataArray。
        rou_da (xr.DataArray): 空气密度数据（kg m-3）。
        reference_air_density (float): 参考空气密度（默认 1.225 kg m-3）。

    Returns:
        xr.DataArray: 空气密度修正后的风速。
    """
    if wind_da.shape != rou_da.shape:
        raise ValueError(
            f"wind_da 与 rou_da 形状不一致: {wind_da.shape} vs {rou_da.shape}"
        )

    if reference_air_density <= 0:
        raise ValueError(f"reference_air_density 必须为正数，得到: {reference_air_density}")

    adjusted_wind = wind_da * (rou_da / reference_air_density) ** (1 / 3)

    if wind_da.name is not None:
        adjusted_wind = adjusted_wind.rename(wind_da.name)
    
    adjusted_wind.attrs.update(wind_da.attrs)
    adjusted_wind.attrs.update({
        'rou_adjustment_applied': 1,
        'rou_adjustment_formula': f'adjusted_wind = wind * (rou / {reference_air_density}) ** (1/3)',
        'source_wind_name': wind_da.name if wind_da.name is not None else 'wind',
        'source_rou_name': rou_da.name if rou_da.name is not None else 'rou',
        'reference_air_density_kg_m3': float(reference_air_density),
    })
    return adjusted_wind



def hub_w_kb_adjust(
    w_hub: xr.DataArray,
    kb_info: Optional[Union[Dict, Tuple[float, float]]],
    time_name: str = 'time',
) -> xr.DataArray:
    """
    对轮毂高度风速按时间段应用线性缩放 (w * k + b)。

    Args:
        w_hub (xr.DataArray): 轮毂高度风速，须包含 time 维度。
        kb_info (Optional[Union[Dict, Tuple[float, float]]]):
            可为两种形式：
            1) 时间段到 (k, b) 的映射，形如 {period: (k, b)}。period 可为：
            - None       : 全年
            - int (1-12) : 指定月份
            - str        : 季节 ('spring' / 'summer' / 'autumn' / 'winter')
            2) 二元组 (k, b)：对整个 w_hub 统一应用同一组线性缩放。
            若 kb_info 为 None，原数据不做任何修改直接返回。
            调用方应保证各 period 对应的时间段互不重叠。
        time_name (str): 时间维名称，默认 'time'。

    Returns:
        xr.DataArray: 调整后的风速（已 clip 至 >= 0）， attrs 与原 DataArray 一致。
    """
    if kb_info is None:
        return w_hub

    if isinstance(kb_info, tuple):
        if len(kb_info) != 2:
            raise ValueError(f"kb_info 为 tuple 时必须为 (k, b) 二元组，得到长度: {len(kb_info)}")
        kb_info = {None: (float(kb_info[0]), float(kb_info[1]))}
    elif not isinstance(kb_info, dict):
        raise TypeError(f"kb_info 类型不支持: {type(kb_info)}，应为 None、dict 或 (k, b) tuple")

    if time_name not in w_hub.dims:
        raise ValueError(f"w_hub 必须包含时间维度: {time_name}")

    _season_months = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11],
        'winter': [12, 1, 2],
    }

    def _build_mask(period, months_da: xr.DataArray) -> xr.DataArray:
        if period is None:
            return xr.ones_like(months_da, dtype=bool)
        elif isinstance(period, int):
            if not 1 <= period <= 12:
                raise ValueError(f"month period 必须在 1-12 之间，得到: {period}")
            return months_da == period
        elif isinstance(period, str):
            key = period.lower()
            if key not in _season_months:
                raise ValueError(
                    f"season 必须为 'spring'/'summer'/'autumn'/'winter'，得到: {period}"
                )
            mask = xr.zeros_like(months_da, dtype=bool)
            for m in _season_months[key]:
                mask = mask | (months_da == m)
            return mask
        else:
            raise TypeError(f"period 类型不支持: {type(period)}，应为 None、int 或 str")

    months_da = w_hub[time_name].dt.month
    result = w_hub.copy(deep=True)

    for period, (k, b) in kb_info.items():
        mask = _build_mask(period, months_da)
        adjusted = w_hub * float(k) + float(b)
        result = xr.where(mask, adjusted, result)
        result.attrs.update(w_hub.attrs)  # xr.where 会丢失 attrs，此处恢复

    result = result.clip(min=0.0)
    if w_hub.name is not None:
        result = result.rename(w_hub.name)
    result.attrs.update(w_hub.attrs)
    result.attrs['kb_adjustments_applied'] = str(kb_info)

    return result


def farm_cf(
    w_hub: xr.DataArray,
    w_grid: Optional[np.ndarray] = None,
    cf_grid: Optional[np.ndarray] = None,
    time_name: str = 'time',
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
    """
    根据轮毂高度风速和功率曲线计算风电场容量因子（CF）。

    支持两种模式：
    1. 直接提供 `w_grid` 和 `cf_grid` 作为插值曲线；
    2. 若未提供 `w_grid/cf_grid`，则使用后续功率曲线参数自动调用 generate_power_curve 生成功率曲线。

    Args:
        w_hub (xr.DataArray): 轮毂高度风速 DataArray。
        w_grid (Optional[np.ndarray]): 风速横轴数组。
        cf_grid (Optional[np.ndarray]): 与 w_grid 对应的容量因子数组。
        time_name (str): 时间维名称，默认 'time'。
        其余参数: 仅当 w_grid/cf_grid 未提供时，用于内部调用 generate_power_curve。

    Returns:
        xr.DataArray: 风电场容量因子（单位 1）。
    """
    if time_name not in w_hub.dims:
        raise ValueError(f"w_hub 必须包含时间维度: {time_name}")

    if (w_grid is None) != (cf_grid is None):
        raise ValueError('w_grid 和 cf_grid 必须同时提供，或同时为 None')

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
        curve_source = 'generated'
    else:
        w_grid_used = np.asarray(w_grid, dtype=np.float32)
        cf_grid_used = np.asarray(cf_grid, dtype=np.float32)
        if w_grid_used.ndim != 1:
            raise ValueError(f"w_grid 必须为一维数组，得到 ndim={w_grid_used.ndim}")
        if cf_grid_used.ndim != 1:
            raise ValueError(f"cf_grid 必须为一维数组，得到 ndim={cf_grid_used.ndim}")
        if len(w_grid_used) != len(cf_grid_used):
            raise ValueError(
                f"w_grid 与 cf_grid 长度必须一致，得到 {len(w_grid_used)} vs {len(cf_grid_used)}"
            )
        curve_source = 'provided'

    speed_values = np.asarray(w_hub.values, dtype=np.float32)
    # 网格上限之外的风速**没有曲线依据**：功率曲线只定义在 [w_grid[0], w_grid[-1]] 上，
    # 而网格上限（默认 40 m/s）已远高于任何切出风速（3–25 m/s）⇒ 上限之外按"停机"处理 CF=0，
    # 与左侧 left=0.0 对称。原实现取 right=cf_grid_used[-1]（曲线右端值），等于宣称
    # "高于建模上限仍按曲线右端值发电"，会给出虚假发电量（实测 40–45 m/s 处约 0.019–0.243）。
    cf_values = np.interp(
        x=speed_values,
        xp=w_grid_used,
        fp=cf_grid_used,
        left=0.0,
        right=0.0,
    ).astype(np.float32)

    cf_da = xr.DataArray(
        cf_values,
        coords=w_hub.coords,
        dims=w_hub.dims,
        name='farm_cf',
        attrs={
            'long_name': 'Wind farm capacity factor',
            'units': '1',
            'source_w_hub_name': w_hub.name if w_hub.name is not None else 'w_hub',
            'cf_curve_source': curve_source,
            'interpolation_method': 'numpy.interp',
            'interpolation_left_fill': 0.0,
            'interpolation_right_fill': 0.0,
            'interpolation_out_of_range_rule': 'out_of_curve_domain -> CF=0 (turbine stopped)',
            'power_curve_point_count': int(len(cf_grid_used)),
            'power_curve_w_min': float(w_grid_used[0]),
            'power_curve_w_max': float(w_grid_used[-1]),
        },
    )

    return cf_da


def farm_generation(
    cf_da: xr.DataArray,
    farm_capacity_mw: float,
    time_name: str = 'time',
    unit_step_hours: float = 1.0,
) -> xr.DataArray:
    """
    根据容量因子（CF）和装机容量计算风电场每时间步发电量。

    Args:
        cf_da (xr.DataArray): 容量因子 DataArray。
        farm_capacity_mw (float): 风电场总装机容量，单位 MW。
        time_name (str): 时间维名称，默认 'time'。
        unit_step_hours (float): 每个时间步对应的小时数，默认 1.0。

    Returns:
        xr.DataArray: 每时间步理论发电量（MWh）。
    """
    if farm_capacity_mw < 0:
        raise ValueError(f"farm_capacity_mw 不能为负数，得到: {farm_capacity_mw}")

    if unit_step_hours <= 0:
        raise ValueError(f"unit_step_hours 必须为正数，得到: {unit_step_hours}")

    if time_name not in cf_da.dims:
        raise ValueError(f"cf_da 必须包含时间维度: {time_name}")

    gen_values = np.asarray(cf_da.values, dtype=np.float32) * float(farm_capacity_mw) * float(unit_step_hours)

    gen_da = xr.DataArray(
        gen_values,
        coords=cf_da.coords,
        dims=cf_da.dims,
        name='farm_generation',
        attrs={
            'long_name': 'Wind farm theoretical generation per time step',
            'units': 'MWh',
            'source_cf_name': cf_da.name if cf_da.name is not None else 'farm_cf',
            'farm_capacity_mw': float(farm_capacity_mw),
            'unit_step_hours': float(unit_step_hours),
        },
    )
    # 保留 CF 中与功率曲线和插值相关的元数据。
    for key, value in cf_da.attrs.items():
        if key not in gen_da.attrs:
            gen_da.attrs[key] = value

    return gen_da




