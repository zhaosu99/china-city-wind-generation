import re
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd

from utils import safe_read_csv


def gen_cumul_capacity(
    csv_path: Union[str, Path],
    level: Literal['city', 'province'] = 'city',
    tech_life: int = 25,
    onshore_col: str = 'onshore_flag',
    install_year_col: str = '年',
    capacity_col: str = '风电场装机MW',
    cutoff_year: Optional[int] = 2024,
    cutoff_month: int = 12,
    output_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    根据风电场数据 CSV，按市或省生成每月累积在役装机量，区分陆上/海上/合计。

    寿命规则：
        安装年份为 Y 的风场，运行期为 Y年1月 至 (Y+tech_life-1)年12月（共 tech_life 年）。
        例：tech_life=20，1990年安装 → 1990年1月至2009年12月。

    时间范围：
        以数据中最早安装年份作为起始年，最晚退役年份（安装年+tech_life-1）作为终止年；
        若指定 cutoff_year，则截断至 cutoff_year年cutoff_month月。

    Args:
        csv_path: 风电场数据 CSV 路径，至少需要以下列：
            省全称, 省简称, 市全称, 市简称, 风电场装机MW（默认）, 年（默认）, onshore_flag（默认）。
        level: 聚合粒度，'city' 按市（默认）或 'province' 按省。
        tech_life: 风场技术寿命（年），默认 25。
        onshore_col: 陆/海标识列名，默认 'onshore_flag'；值为 1 表示陆上，其他表示海上。
        install_year_col: 安装年份列名，默认 '年'。
        capacity_col: 装机容量列名，默认 '风电场装机MW'。
        cutoff_year: 截止年份（含），超过此年月的列不输出；为 None 则不截断，默认 2024。
        cutoff_month: 截止月份（含），与 cutoff_year 配合使用，默认 12。
        output_csv_path: 输出 CSV 路径；为 None 时不写文件。

    Returns:
        pd.DataFrame:
            level='city'     → 列：省全称, 省简称, 市全称, 市简称, type, YYYY年M月…
            level='province' → 列：省全称, 省简称, type, YYYY年M月…
            type 取值：total / onshore / offshore
            单元格值为当月在役装机量（MW）之和。
    """
    if level not in ('city', 'province'):
        raise ValueError("level 只能为 'city' 或 'province'")

    df = safe_read_csv(csv_path)

    required_meta_city = ['省全称', '省简称', '市全称', '市简称']
    required_data = [install_year_col, capacity_col, onshore_col]
    missing = [c for c in required_meta_city + required_data if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少必要列：{missing}，实际列：{list(df.columns)}")

    # 按 level 确定元数据列和分组键
    if level == 'city':
        meta_cols = ['省全称', '省简称', '市全称', '市简称']
        group_key = '市简称'
    else:
        meta_cols = ['省全称', '省简称']
        group_key = '省简称'

    # 清理并强制类型
    df['_start_year'] = df[install_year_col].astype(int)
    df['_end_year'] = df['_start_year'] + tech_life - 1
    df['_capacity'] = df[capacity_col].astype(float).fillna(0.0)
    df['_onshore'] = df[onshore_col].fillna(0).astype(int)

    # 时间范围
    min_year = int(df['_start_year'].min())
    max_year = int(df['_end_year'].max())
    if cutoff_year is not None:
        max_year = min(max_year, cutoff_year)
    n_months = (max_year - min_year + 1) * 12
    if cutoff_year is not None:
        cutoff_total_months = (cutoff_year - min_year) * 12 + cutoff_month
        n_months = min(n_months, cutoff_total_months)

    # 生成形如 "2010年1月" 的列名
    all_month_cols = [
        f"{min_year + t // 12}年{t % 12 + 1}月"
        for t in range(n_months)
    ]

    # 按分组键去重，保留元数据
    unit_info = (
        df[meta_cols]
        .drop_duplicates(subset=[group_key])
        .reset_index(drop=True)
    )
    unit_list = unit_info[group_key].tolist()
    unit_to_idx: Dict[str, int] = {u: i for i, u in enumerate(unit_list)}
    n_units = len(unit_list)

    # 每个风场的月索引（0-based）
    start_midx = (df['_start_year'].values - min_year) * 12
    end_midx   = (df['_end_year'].values   - min_year) * 12 + 11
    t_idx = np.arange(n_months)

    # 在役布尔矩阵 (n_farms, n_months)
    active = (start_midx[:, None] <= t_idx[None, :]) & \
             (end_midx[:, None]   >= t_idx[None, :])

    capacities = df['_capacity'].values
    farm_unit_idx = np.array([unit_to_idx[u] for u in df[group_key].values])

    def _compute(farm_mask: np.ndarray) -> np.ndarray:
        """返回 (n_units, n_months) 装机量矩阵"""
        if not farm_mask.any():
            return np.zeros((n_units, n_months), dtype=float)
        subset_idx  = farm_unit_idx[farm_mask]
        subset_caps = capacities[farm_mask]
        weighted    = active[farm_mask, :] * subset_caps[:, None]  # (n_subset, n_months)
        membership  = (subset_idx[None, :] == np.arange(n_units)[:, None]).astype(float)
        return membership @ weighted  # (n_units, n_months)

    total_mask    = np.ones(len(df), dtype=bool)
    onshore_mask  = df['_onshore'].values == 1
    offshore_mask = df['_onshore'].values != 1

    total_mat    = _compute(total_mask)
    onshore_mat  = _compute(onshore_mask)
    offshore_mat = _compute(offshore_mask)

    def _to_df(mat: np.ndarray, type_label: str) -> pd.DataFrame:
        cap_df = pd.DataFrame(mat, columns=all_month_cols)
        out = pd.concat([unit_info[meta_cols].reset_index(drop=True), cap_df], axis=1)
        out.insert(len(meta_cols), 'type', type_label)
        return out

    result = pd.concat(
        [_to_df(total_mat,    'total'),
         _to_df(onshore_mat,  'onshore'),
         _to_df(offshore_mat, 'offshore')],
        ignore_index=True,
    )
    result = result.sort_values([group_key, 'type']).reset_index(drop=True)

    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False, encoding='utf-8-sig')

    return result


def gen_capacity_adjustment_factor(
    theoretical_csv_path: Union[str, Path],
    actual_csv_path: Union[str, Path],
    level: Literal['city', 'province'] = 'city',
    cutoff_year: Optional[int] = 2023,
    cutoff_month: int = 12,
    actual_capacity_col: str = '累积装机(MW)',
    actual_type_col: str = '能源类型',
    actual_year_col: str = '年',
    actual_month_col: str = '月',
    output_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    对比理论装机宽表与实际装机长表，按理论表格式生成容量调整系数。

    规则：
    1. 输出行列结构以 theoretical_csv_path 为基准（保持省/市/type + 月列格式）。
    2. 系数定义为 actual / theoretical。
    3. 若城市(或省)在某月没有实际值，系数=1。
    4. 若 theoretical<=0 或缺失，系数=1（避免除零和异常放大）。
    5. 结果仅输出 onshore / offshore 两类，不输出 total。
    6. 输出仅保留到 cutoff_year 年 cutoff_month 月（默认 2023年12月）。

    Args:
        theoretical_csv_path: 理论装机 CSV（宽表，含 YYYY年M月 列）。
        actual_csv_path: 实际装机 CSV（长表，通常含 年、月、地区列、容量列）。
        level: 对齐粒度，'city' 或 'province'，默认 'city'。
        cutoff_year: 截止年份（含），默认 2023。
        cutoff_month: 截止月份（含），默认 12。
        actual_capacity_col: 实际容量列名，默认 '累积装机(MW)'。
        actual_type_col: 实际能源类型列名，默认 '能源类型'。
        actual_year_col: 实际年份列名，默认 '年'。
        actual_month_col: 实际月份列名，默认 '月'。
        output_csv_path: 输出 CSV 路径；为 None 时不写文件。

    Returns:
        pd.DataFrame: 与理论表同结构（到截止月）的调整系数表，仅包含 onshore/offshore。
    """
    if level not in ('city', 'province'):
        raise ValueError("level 只能为 'city' 或 'province'")
    if cutoff_year is not None and not (1 <= int(cutoff_month) <= 12):
        raise ValueError('cutoff_month 必须在 1-12 之间')

    theory_df = safe_read_csv(theoretical_csv_path)
    actual_df = safe_read_csv(actual_csv_path)

    # 识别理论表月列并按截止年月筛选
    month_pattern = re.compile(r'^(\d{4})年(\d{1,2})月$')
    month_cols_all: list[tuple[str, int, int]] = []
    for col in theory_df.columns:
        m = month_pattern.match(str(col).strip())
        if m is not None:
            y, mth = int(m.group(1)), int(m.group(2))
            month_cols_all.append((col, y, mth))
    if not month_cols_all:
        raise ValueError('理论表中未找到形如 YYYY年M月 的月度列')

    month_cols_all.sort(key=lambda x: (x[1], x[2]))
    if cutoff_year is not None:
        month_cols = [c for c, y, mth in month_cols_all if (y < cutoff_year) or (y == cutoff_year and mth <= cutoff_month)]
    else:
        month_cols = [c for c, _, _ in month_cols_all]
    if not month_cols:
        raise ValueError('按 cutoff_year/cutoff_month 截断后无可用月度列')

    # 按 level 确定关键列
    if level == 'city':
        key_cols = ['省全称', '省简称', '市全称', '市简称', 'type']
        actual_group_cols = ['省全称', '省简称', '市全称', '市简称', 'type', 'year', 'month']
    else:
        key_cols = ['省全称', '省简称', 'type']
        actual_group_cols = ['省全称', '省简称', 'type', 'year', 'month']

    missing_theory = [c for c in key_cols if c not in theory_df.columns]
    if missing_theory:
        raise KeyError(f'理论表缺少必要列: {missing_theory}')

    # 固定读取实际表容量列（默认 '累积装机(MW)'）
    if actual_capacity_col not in actual_df.columns:
        raise KeyError(f"实际表缺少容量列 '{actual_capacity_col}'")

    required_actual = ['省全称', '省简称', actual_year_col, actual_month_col, actual_capacity_col]
    if level == 'city':
        required_actual.extend(['市全称', '市简称'])
    missing_actual = [c for c in required_actual if c not in actual_df.columns]
    if missing_actual:
        raise KeyError(f'实际表缺少必要列: {missing_actual}')

    # 标准化实际表 year/month/type/value
    actual = actual_df.copy()
    actual['year'] = pd.to_numeric(actual[actual_year_col], errors='coerce').astype('Int64')
    actual['month'] = pd.to_numeric(actual[actual_month_col], errors='coerce').astype('Int64')
    actual = actual.dropna(subset=['year', 'month'])
    actual['year'] = actual['year'].astype(int)
    actual['month'] = actual['month'].astype(int)
    actual = actual[(actual['month'] >= 1) & (actual['month'] <= 12)]

    if cutoff_year is not None:
        actual = actual[(actual['year'] < cutoff_year) | ((actual['year'] == cutoff_year) & (actual['month'] <= cutoff_month))]

    actual['value'] = pd.to_numeric(actual[actual_capacity_col], errors='coerce')
    actual = actual.dropna(subset=['value'])

    if actual_type_col in actual.columns:
        type_raw = actual[actual_type_col].astype(str)
        has_onshore = type_raw.str.contains('陆', na=False)
        has_offshore = type_raw.str.contains('海', na=False)
        actual['type'] = np.where(
            has_onshore & has_offshore,
            'total',
            np.where(
                has_offshore,
            'offshore',
                np.where(has_onshore, 'onshore', 'total')
            )
        )
    else:
        actual['type'] = 'total'

    actual = actual[actual['type'].isin(['onshore', 'offshore'])].copy()

    # 实际长表 -> 宽表（按 key + type + year/month 聚合）
    actual_aggr = (
        actual.groupby(actual_group_cols, as_index=False)['value']
        .sum()
    )
    actual_aggr['month_col'] = actual_aggr.apply(lambda r: f"{int(r['year'])}年{int(r['month'])}月", axis=1)
    actual_wide = actual_aggr.pivot_table(
        index=key_cols,
        columns='month_col',
        values='value',
        aggfunc='sum'
    ).reset_index()
    actual_wide.columns.name = None

    # 以理论表结构为底，左连接实际值，再生成系数（仅保留 onshore/offshore）
    base = theory_df[key_cols + month_cols].copy()
    base = base[base['type'].isin(['onshore', 'offshore'])].copy()
    merged = base.merge(actual_wide, on=key_cols, how='left', suffixes=('_theory', '_actual'))

    coeff_series_list = []
    for col in month_cols:
        theory_col = f'{col}_theory'
        actual_col = f'{col}_actual'

        if theory_col in merged.columns:
            theory_vals = pd.to_numeric(merged[theory_col], errors='coerce')
        else:
            theory_vals = pd.to_numeric(merged[col], errors='coerce')
        actual_vals = pd.to_numeric(merged[actual_col], errors='coerce') if actual_col in merged.columns else pd.Series(np.nan, index=merged.index)

        coeff = np.where(
            (theory_vals > 0) & actual_vals.notna(),
            actual_vals / theory_vals,
            1.0,
        )
        coeff_series_list.append(pd.Series(coeff, name=col, index=merged.index))

    coeff_df = pd.concat(coeff_series_list, axis=1) if coeff_series_list else pd.DataFrame(index=merged.index)
    result = pd.concat([merged[key_cols].reset_index(drop=True), coeff_df.reset_index(drop=True)], axis=1)

    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False, encoding='utf-8-sig')

    return result


if __name__ == "__main__":
    df = gen_cumul_capacity(
        csv_path=r"D:\HKU PhD\wind data paper\wind_farms_inventory_till_2023.csv",
        level='city',
        tech_life=25,
        cutoff_year=2024,
        cutoff_month=12,
        output_csv_path=r"D:\HKU PhD\wind data paper\try\try001.csv",
    )
    print("理论装机样例:")
    print(df.head())
    print(f"理论装机输出行数: {len(df)}, 列数: {len(df.columns)}")

    coef_df = gen_capacity_adjustment_factor(
        theoretical_csv_path=r"D:\HKU PhD\wind data paper\try\try001.csv",
        actual_csv_path=r"D:\HKU PhD\wind data paper\city_monthly_wind_capacity.csv",
        level='city',
        cutoff_year=2023,
        cutoff_month=12,
        actual_capacity_col='累积装机(MW)',
        output_csv_path=r"D:\HKU PhD\wind data paper\try\try001_factor.csv",
    )
    print("校正系数样例:")
    print(coef_df.head())
    print(f"校正系数输出行数: {len(coef_df)}, 列数: {len(coef_df.columns)}")
