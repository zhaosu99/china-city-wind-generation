import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union, Optional
import geopandas as gpd
from matplotlib.dates import DateFormatter
from matplotlib.dates import DateFormatter, YearLocator
import concurrent.futures
import multiprocessing as mp



plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
plt.rcParams['axes.unicode_minus'] = False    # To display negative signs correctly

def _read_csv_with_encoding(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """Helper function to read CSV with encoding fallback"""
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='gbk')

def aggregate_sites_to_annual(
    input_data: Union[str, pd.DataFrame, gpd.GeoDataFrame],
    coastal_cities_csv_path: str,
    output_csv_path: Optional[str] = None,
    lifetime_years: int = 25,
    encoding: str = 'utf-8-sig',
    start_year: int = 2000,
    end_year: int = 2023
) -> pd.DataFrame:
    """
    将风电场站点数据聚合为各城市每年12月数据 → 站点聚合值
    
    Parameters:
    -----------
    input_data : str or pd.DataFrame or gpd.GeoDataFrame
        输入的风电场数据（支持文件路径或DataFrame）
    coastal_cities_csv_path : str
        沿海城市CSV文件路径
    output_csv_path : str, optional
        输出CSV文件路径，如果为None则不保存
    lifetime_years : int
        风电机组设计寿命，默认25年
    encoding : str
        CSV编码，默认'utf-8-sig'
    start_year : int
        起始年，默认2000
    end_year : int
        结束年，默认2023
    
    Returns:
    --------
    pd.DataFrame
        包含站点聚合值的数据
    """
    # Read input data
    if isinstance(input_data, str):
        df = _read_csv_with_encoding(input_data, encoding)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif isinstance(input_data, gpd.GeoDataFrame):
        df = input_data.drop(columns=['geometry']).copy() if hasattr(input_data, 'geometry') else input_data.copy()
    else:
        raise ValueError("input_data must be str, pd.DataFrame, or gpd.GeoDataFrame")
    
    # Read coastal cities
    coastal_df = _read_csv_with_encoding(coastal_cities_csv_path, encoding)
    COASTAL_CITIES = set(coastal_df['市全称'])
    
    # Validate required columns
    required_cols = ['年', '省全称', '市全称', '风电场装机MW', '陆海1-2', '省简称', '市简称']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")
    
    # Type conversion and validation
    df['年'] = pd.to_numeric(df['年'], errors='raise').astype(int)
    df['风电场装机MW'] = pd.to_numeric(df['风电场装机MW'], errors='coerce')
    if df['风电场装机MW'].isna().any():
        raise ValueError("Column '风电场装机MW' contains non-numeric values")

    # Determine actual year range
    global_min_year = df['年'].min()
    global_max_year = df['年'].max()  
    actual_start_year = min(start_year, global_min_year)
    actual_end_year = end_year
    
    print(f"Data year range: {global_min_year} - {global_max_year}")
    print(f"Processing year range: {actual_start_year} - {actual_end_year}")

    # Separate land and sea data
    land_df = df[df['陆海1-2'] == 1].copy()
    sea_df = df[df['陆海1-2'] == 2].copy()

    all_results = []

    for name, subset in [('陆上风电', land_df), ('海上风电', sea_df)]:
        if subset.empty:
            print(f"Warning: {name} data is empty, will create empty DataFrame.")
            # Create empty DataFrame with proper structure
            empty_df = pd.DataFrame(columns=['年', '月', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)', '能源类型', '数据来源'])
            all_results.append(empty_df)
            continue
            
        # Filter sea data for coastal cities only
        if name == '海上风电':
            print(f"Filtering {name} data for coastal cities...")
            original_count = len(subset)
            subset = subset[subset['市全称'].isin(COASTAL_CITIES)]
            filtered_count = len(subset)
            print(f"Coastal cities filtering: {original_count} -> {filtered_count}")
        
        if subset.empty:
            print(f"Warning: {name} data is empty after coastal cities filtering, will create empty DataFrame.")
            empty_df = pd.DataFrame(columns=['年', '月', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)', '能源类型', '数据来源'])
            all_results.append(empty_df)
            continue
        
        # Aggregate annual additions
        print(f"Processing {name} data aggregation...")
        additions = (
            subset.groupby(['省全称', '市全称', '年'], as_index=False)[['风电场装机MW', '省简称', '市简称']]
            .agg({
                '风电场装机MW': 'sum',
                '省简称': 'first',
                '市简称': 'first'
            })
            .rename(columns={'风电场装机MW': '新增装机(MW)'})
            .sort_values(['省全称', '市全称', '年'])
            .reset_index(drop=True)
        )
        
        print(f"Aggregated {len(additions)} records for {name}")
        
        # Calculate cumulative capacity with retirement
        def compute_cumulative_with_retirement(group, prov_full, city_full, prov_abbr, city_abbr, 
                                             start_y, end_y, lifetime_inner, all_additions):
            # Create year range for the specific city
            years_range = pd.DataFrame({'年': list(range(start_y, end_y + 1))})
            
            # Merge with actual additions for this city
            group_with_all_years = pd.merge(
                years_range, 
                group[['年', '新增装机(MW)']], 
                on='年', 
                how='left'
            ).fillna({'新增装机(MW)': 0.0})

            cumulative_values = []
            for current_year in group_with_all_years['年']:
                # Find projects that are still operational in current_year
                running_mask = (
                    (all_additions['年'] >= (current_year - lifetime_inner + 1)) &
                    (all_additions['年'] <= current_year) &
                    (all_additions['省全称'] == prov_full) &
                    (all_additions['市全称'] == city_full)
                )
                running_capacity = all_additions.loc[running_mask, '新增装机(MW)'].sum()
                cumulative_values.append(running_capacity)

            group_with_all_years['累积装机(MW)'] = cumulative_values
            group_with_all_years['省全称'] = prov_full
            group_with_all_years['省简称'] = prov_abbr
            group_with_all_years['市全称'] = city_full
            group_with_all_years['市简称'] = city_abbr
            group_with_all_years['能源类型'] = name
            group_with_all_years['月'] = 12  # Annual data is for December
            
            return group_with_all_years[['年', '月', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)', '能源类型']]

        # Add month column (initialize as 12 for annual data)
        additions['月'] = 12
        
        print(f"Computing cumulative capacity with retirement for {name}...")
        result_df = (
            additions
            .groupby(['省全称', '市全称'], group_keys=False)
            .apply(
                lambda x: compute_cumulative_with_retirement(
                    x,
                    x.name[0], x.name[1],
                    x['省简称'].iloc[0], x['市简称'].iloc[0],
                    actual_start_year, actual_end_year, lifetime_years, additions
                ),
                include_groups=False
            )
            .reset_index(drop=True)
        )
        
        # Add data source
        result_df['数据来源'] = '站点聚合值'
        all_results.append(result_df)
        print(f"Processed {len(result_df)} records for {name}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df['累积装机(MW)'] = final_df['累积装机(MW)'].round(2)
        
        # Filter out records before start_year with zero capacity
        print(f"Filtering records: removing those before {start_year} with zero capacity...")
        initial_count = len(final_df)
        final_df = final_df[
            ~((final_df['年'] < start_year) & (final_df['累积装机(MW)'] == 0.0))
        ].copy()
        final_count = len(final_df)
        print(f"Filtered: {initial_count} -> {final_count}")
        
        final_df = final_df.sort_values(['年', '月', '省全称', '市全称']).reset_index(drop=True)
        print(f"Final aggregated result: {len(final_df)} records")
    else:
        final_df = pd.DataFrame(columns=['年', '月', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)', '能源类型', '数据来源'])
        print("No data processed, returning empty DataFrame")
    
    # Save to CSV if path provided
    if output_csv_path:
        final_df.to_csv(output_csv_path, index=False, encoding=encoding)
        print(f"Results saved to {output_csv_path}")
    
    return final_df


def merge_correction_and_direct_city_data(
    original_correction_data: Union[str, pd.DataFrame],
    provincial_data: Union[str, pd.DataFrame],
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    合并原始校准数据和省级数据中直辖市的数据。
    从省级数据中提取直辖市记录，并从原始校准数据中移除直辖市记录，然后合并两者。
    Parameters:
    -----------
    original_correction_data : str or pd.DataFrame
        原始的校准数据 (B data)，其城市列名为 '市'。
    provincial_data : str or pd.DataFrame
        省级装机数据，其省份列名为 '省'。
    encoding : str
        CSV编码，默认'utf-8'。
    Returns:
    --------
    pd.DataFrame
        合并后的新校准数据集。
    """
    # Read original correction data
    if isinstance(original_correction_data, str):
        df_correction = _read_csv_with_encoding(original_correction_data, encoding=encoding)
    elif isinstance(original_correction_data, pd.DataFrame):
        df_correction = original_correction_data.copy()
    else:
        raise ValueError("original_correction_data must be str or pd.DataFrame")

    # Read provincial data
    if isinstance(provincial_data, str):
        df_prov = _read_csv_with_encoding(provincial_data, encoding=encoding)
    elif isinstance(provincial_data, pd.DataFrame):
        df_prov = provincial_data.copy()
    else:
        raise ValueError("provincial_data must be str or pd.DataFrame")
    

    df_prov = df_prov.loc[
        df_prov['能源类型'].isin(['陆上风电', '海上风电'])
    ]

    # Define direct-controlled municipalities
    direct_cities = {'北京', '上海', '重庆', '天津'}

    # --- 1. Extract direct city data from provincial data ---
    # Filter rows where '省' is one of the direct cities
    df_direct_from_prov = df_prov[df_prov['省'].isin(direct_cities)].copy()

    if not df_direct_from_prov.empty:
        # Rename '省' column to '市' to match the correction data structure
        df_direct_from_prov.rename(columns={'省': '市'}, inplace=True)
        # Rename '装机容量(MW)' column to '装机量(MW)' to match the correction data structure
        if '装机容量(MW)' in df_direct_from_prov.columns:
            df_direct_from_prov.rename(columns={'装机容量(MW)': '装机量(MW)'}, inplace=True)
        df_direct_from_prov['省'] = df_direct_from_prov['市']
        df_direct_from_prov['月'] = 12  # Set month to December for annual data

    # --- 2. Remove direct city records from original correction data ---
    # Filter out rows where '市' is one of the direct cities
    df_correction_filtered = df_correction[~df_correction['市'].isin(direct_cities)].copy()

    # --- 3. Concatenate the two dataframes ---
    # Ensure both dataframes have the same columns before concatenating.
    # This assumes df_correction_filtered has the correct column names already.
    # df_direct_from_prov should now have '市' and '装机量(MW)' as needed.
    # Let's ensure all necessary columns are present and aligned.
    # For simplicity, we'll concatenate and let pandas handle potentially different column sets.
    # The `calibrate_with_collected_data` function will later process based on its required columns.
    if not df_direct_from_prov.empty:
        combined_df = pd.concat([df_correction_filtered, df_direct_from_prov], ignore_index=True)
    else:
        # If there's no direct city data from prov, just use the filtered correction data
        combined_df = df_correction_filtered

    # Sort the final dataframe for consistency (optional, but good practice)
    if not combined_df.empty and '年' in combined_df.columns and '市' in combined_df.columns and '能源类型' in combined_df.columns:
        combined_df.sort_values(['市', '能源类型', '年'], inplace=True, ignore_index=True)
    
    # 天津的陆风数据不准
    combined_df = combined_df.loc[~((combined_df['能源类型'] == '陆上风电') & (combined_df['市'].isin(['天津', '重庆'])))]
    return combined_df


def interpolate_monthly_within_years(
    input_data: Union[str, pd.DataFrame],
    output_csv_path: Optional[str] = None,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    对年度数据进行月度线性插值 → 月度内插值
    
    Parameters:
    -----------
    input_data : str or pd.DataFrame
        输入的年度数据
    output_csv_path : str, optional
        输出CSV文件路径，如果为None则不保存
    encoding : str
        CSV编码，默认'utf-8'
    
    Returns:
    --------
    pd.DataFrame
        包含月度内插值的数据
    """
    # Read input data
    if isinstance(input_data, str):
        df = _read_csv_with_encoding(input_data, encoding)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("input_data must be str or pd.DataFrame")
    
    # Ensure required columns exist
    required_cols = ['年', '月', '省全称', '省简称', '市全称', '市简称', '累积装机(MW)', '能源类型', '数据来源']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")
    
    # Create a copy to work with and sort for better performance
    result_df = df.copy()
    result_df = result_df.sort_values(['省简称', '市简称', '能源类型', '年', '月']).reset_index(drop=True)
    
    # Create a set of existing records for quick lookup
    existing_records = set(result_df[['年', '月', '省简称', '市简称', '能源类型']].apply(
        lambda x: (x['年'], x['月'], x['省简称'], x['市简称'], x['能源类型']), axis=1))
    
    # Collect all new records to add at once
    new_records = []
    
    # Group by province, city, and energy type to handle each combination separately
    grouped = result_df.groupby(['省简称', '市简称', '能源类型'])
    
    for name, group in grouped:
        prov_short, city_short, energy_type = name
        
        # Sort group by year to ensure proper order
        group = group.sort_values('年')
        
        # Get all years in this group
        years = sorted(group['年'].unique())
        
        for year in years:
            # Get December data for this year (the annual value)
            dec_data = group[(group['年'] == year) & (group['月'] == 12)]
            
            if len(dec_data) > 0:
                dec_capacity = dec_data.iloc[0]['累积装机(MW)']
                
                # Get previous year's December data for interpolation start
                prev_year_dec = group[(group['年'] == year - 1) & (group['月'] == 12)]
                
                if len(prev_year_dec) > 0:
                    prev_capacity = prev_year_dec.iloc[0]['累积装机(MW)']
                else:
                    prev_capacity = 0.0  # Start from 0 if no previous data
                
                # For months 1-11, interpolate between previous Dec and current Dec
                for month in range(1, 12):
                    # Check if this month already exists using the set
                    record_key = (year, month, prov_short, city_short, energy_type)
                    
                    if record_key not in existing_records:
                        # Calculate interpolated value
                        interpolated_capacity = prev_capacity + (dec_capacity - prev_capacity) * month / 12
                        interpolated_capacity = max(0, interpolated_capacity)
                        
                        # Create new record
                        new_record = {
                            '年': year,
                            '月': month,
                            '省全称': dec_data.iloc[0]['省全称'],
                            '省简称': prov_short,
                            '市全称': dec_data.iloc[0]['市全称'],
                            '市简称': city_short,
                            '累积装机(MW)': interpolated_capacity,
                            '能源类型': energy_type,
                            '数据来源': '月度内插值'
                        }
                        
                        # Add to new records list
                        new_records.append(new_record)
    
    # Add all new records at once
    if new_records:
        new_records_df = pd.DataFrame(new_records)
        result_df = pd.concat([result_df, new_records_df], ignore_index=True)
    
    # Round values and sort
    result_df['累积装机(MW)'] = result_df['累积装机(MW)'].round(2)
    result_df = result_df.sort_values(['年', '月', '省简称', '市简称', '能源类型']).reset_index(drop=True)
    
    # Remove duplicates, keeping the latest
    result_df = result_df.drop_duplicates(subset=['年', '月', '省简称', '市简称', '能源类型'], keep='last')
    
    # Save to CSV if path provided
    if output_csv_path:
        result_df.to_csv(output_csv_path, index=False, encoding=encoding)
    
    return result_df


def _process_combination_a_has_data_b_has_data(
    df_A_subset: pd.DataFrame,
    df_B_subset: pd.DataFrame,
    method: str = 'platform_truncation',
    start_year: int = 2000,
    end_year: int = 2023 # 用于确定校准处理的范围
) -> pd.DataFrame:
    """
    子函数：处理 A 不空, B 不空 的情况。
    执行校准逻辑。保留原始A数据中早于start_year的记录。

    Parameters:
    -----------
    df_A_subset : pd.DataFrame
        来自A数据（如插值后的月度数据）的子集。
    df_B_subset : pd.DataFrame
        来自B数据（校准数据）的子集。
    method : str
        校准方法，默认 'platform_truncation'。
    start_year : int
        校准处理的起始年份（仅用于校准逻辑，不裁剪原始数据）。
    end_year : int
        校准处理的结束年份。

    Returns:
    --------
    pd.DataFrame
        校准后的数据子集，包含原始A数据的结构，并更新了 '数据来源' 列。
        保留早于 start_year 的原始A数据记录。
    """
    # 确保日期列存在
    if 'date' not in df_A_subset.columns:
        df_A_subset = df_A_subset.copy() # Work on a copy
        df_A_subset['date'] = pd.to_datetime(df_A_subset['年'].astype(str) + '-' + df_A_subset['月'].astype(str) + '-01') + pd.offsets.MonthEnd(0)
    if 'date' not in df_B_subset.columns:
        df_B_subset = df_B_subset.copy() # Work on a copy
        df_B_subset['date'] = pd.to_datetime(df_B_subset['年'].astype(str) + '-' + df_B_subset['月'].astype(str) + '-01') + pd.offsets.MonthEnd(0)

    # 分离B数据中在A数据时间范围内的部分和超出范围的部分
    max_date_A = df_A_subset['date'].max()
    df_B_within = df_B_subset[df_B_subset['date'] <= max_date_A].copy()
    # df_B_beyond = df_B_subset[df_B_subset['date'] > max_date_A].copy() # 此函数不处理超出部分

    # 如果A数据时间范围内的B数据为空，则不校准，直接返回A数据
    if df_B_within.empty:
        df_A_result = df_A_subset.copy()
        df_A_result['数据来源'] = df_A_result.get('数据来源', '站点聚合值')
        df_A_result = df_A_result.drop(columns=['date'], errors='ignore')
        return df_A_result

    # 存储原始A数据的值用于后续比较，确保索引与 df_A_subset 完全一致
    original_a_values = df_A_subset['累积装机(MW)'].copy() # 使用 copy() 确保不修改原始 df_A_subset

    # --- 应用校准方法，但仅限于 start_year 到 end_year 的范围 ---
    if method == 'platform_truncation':
        # 筛选出需要校准的时间范围内的A数据点
        calib_mask_A = (df_A_subset['年'] >= start_year) & (df_A_subset['年'] <= end_year)
        df_A_calib_range = df_A_subset[calib_mask_A].copy().sort_values('date').reset_index(drop=True)
        original_a_values_calib_range = original_a_values[calib_mask_A].copy().reset_index(drop=True)

        if df_A_calib_range.empty:
            # 如果在指定校准范围内没有A数据，则无需校准，直接返回原始A数据（可能包含早于start_year的数据）
            df_A_result = df_A_subset.copy()
            df_A_result['数据来源'] = df_A_result.get('数据来源', '站点聚合值')
            df_A_result = df_A_result.drop(columns=['date'], errors='ignore')
            return df_A_result

        # 筛选出需要校准的时间范围内的B数据点
        calib_mask_B = (df_B_within['年'] >= start_year) & (df_B_within['年'] <= end_year)
        df_B_calib_range = df_B_within[calib_mask_B].copy().sort_values('date').reset_index(drop=True)

        if df_B_calib_range.empty:
             # 如果在指定校准范围内没有B数据，则无需校准该范围内的A数据，直接返回原始A数据
            df_A_result = df_A_subset.copy()
            df_A_result['数据来源'] = df_A_result.get('数据来源', '站点聚合值')
            df_A_result = df_A_result.drop(columns=['date'], errors='ignore')
            return df_A_result

        # 创建合并的时间网格：A日期（校准范围内） + B日期（校准范围内）
        all_dates_combined = pd.concat([
            df_A_calib_range[['date']],
            df_B_calib_range[['date']]
        ]).drop_duplicates().sort_values('date').reset_index(drop=True)

        # 将A数据（校准范围内）的值插值到合并的时间网格上
        a_interp_full = np.interp(
            all_dates_combined['date'].astype(np.int64),
            df_A_calib_range['date'].astype(np.int64),
            df_A_calib_range['累积装机(MW)'].values
        )

        # 初始化校准后的结果数组（仅校准范围内）
        calibrated_full = a_interp_full.copy()

        # 创建B日期到B值的映射，用于快速查找
        b_date_to_val_map = dict(zip(df_B_calib_range['date'], df_B_calib_range['装机量(MW)']))

        # 按照平台截断法逻辑处理合并网格
        for i, date in enumerate(all_dates_combined['date']):
            if date in b_date_to_val_map:
                b_val = b_date_to_val_map[date]
                a_val_at_date = a_interp_full[i]
                if b_val > a_val_at_date:
                    # B > A_interp: 将校准值设为B，向后（未来）强制最小值
                    calibrated_full[i] = b_val
                    j = i + 1
                    while j < len(calibrated_full) and all_dates_combined['date'].iloc[j] not in b_date_to_val_map:
                        calibrated_full[j] = max(calibrated_full[j], b_val)
                        j += 1
                else:
                    # B <= A_interp: 将校准值设为B，向前（过去）强制最大值
                    calibrated_full[i] = b_val
                    j = i - 1
                    while j >= 0 and all_dates_combined['date'].iloc[j] not in b_date_to_val_map:
                        calibrated_full[j] = min(calibrated_full[j], b_val)
                        j -= 1

        # 将校准后的值映射回 *原始A数据（校准范围内）* 的日期上
        calibrated_a_only_calib_range = np.interp(
            df_A_calib_range['date'].values.astype(np.int64),
            all_dates_combined['date'].astype(np.int64),
            calibrated_full
        )

        # 创建结果DataFrame，使用原始A数据（校准范围内）的结构，但更新累积装机值
        result_df_part_a_calib_range = df_A_calib_range.copy()
        result_df_part_a_calib_range['累积装机(MW)'] = calibrated_a_only_calib_range


        # --- 比较并标记 '连带校准值' (仅在校准范围内) ---
        # 确保 original_a_values_calib_range 与 result_df_part_a_calib_range 的索引对齐，进行比较
        original_a_values_aligned = original_a_values_calib_range
        # 计算变化掩码
        values_changed_mask = result_df_part_a_calib_range['累积装机(MW)'] != original_a_values_aligned
        # 标记变化的点
        result_df_part_a_calib_range.loc[values_changed_mask, '数据来源'] = '连带校准值'
        # 标记未变化的点，保留原始来源或设为默认值
        # 从原始 df_A_subset 中获取对应的 '数据来源'，同样需要对齐索引
        original_sources_aligned = df_A_calib_range['数据来源'].reset_index(drop=True)
        result_df_part_a_calib_range.loc[~values_changed_mask, '数据来源'] = original_sources_aligned[~values_changed_mask].fillna('站点聚合值')

        # --- 比较并标记 '收集校准值' (仅在校准范围内) ---
        # 检查校准后的结果中是否有值与B数据点完全匹配
        for _, b_row in df_B_calib_range.iterrows():
            b_year = b_row['年']
            b_month = b_row['月']
            b_value = b_row['装机量(MW)']
            # 在结果中查找匹配的年月和值
            dd_match_mask = (
                (result_df_part_a_calib_range['年'] == b_year) &
                (result_df_part_a_calib_range['月'] == b_month) &
                (result_df_part_a_calib_range['累积装机(MW)'] == b_value)
            )
            if dd_match_mask.any():
                # 如果找到匹配项，将其标记为 '收集校准值'
                result_df_part_a_calib_range.loc[dd_match_mask, '数据来源'] = '收集校准值'

        # --- 合并结果：将校准范围内的数据与范围外的数据合并 ---
        # 创建最终结果的副本
        result_df_part_a_full = df_A_subset.copy()

        # 关键修正：使用 merge 来精确匹配并更新数据，避免索引错位
        # 为校准后的数据添加一个唯一标识符（年-月），以便精确匹配
        result_df_part_a_calib_range['key'] = result_df_part_a_calib_range['年'].astype(str) + '-' + result_df_part_a_calib_range['月'].astype(str)
        # 为原始数据也创建相同的 key
        result_df_part_a_full['key'] = result_df_part_a_full['年'].astype(str) + '-' + result_df_part_a_full['月'].astype(str)

        # 使用 merge 将校准后的值和数据来源合并到原始数据中
        # 只选择校准后的 '累积装机(MW)' 和 '数据来源' 列进行合并
        calib_update = result_df_part_a_calib_range[['key', '累积装机(MW)', '数据来源']].copy()

        # 将校准后的值合并回完整数据
        result_df_part_a_full = result_df_part_a_full.merge(calib_update, on='key', how='left', suffixes=('', '_new'))

        # 对于有新值的行，更新 '累积装机(MW)' 和 '数据来源'
        mask_has_new = result_df_part_a_full['累积装机(MW)_new'].notna()
        result_df_part_a_full.loc[mask_has_new, '累积装机(MW)'] = result_df_part_a_full.loc[mask_has_new, '累积装机(MW)_new']
        result_df_part_a_full.loc[mask_has_new, '数据来源'] = result_df_part_a_full.loc[mask_has_new, '数据来源_new']

        # 删除临时列
        result_df_part_a_full = result_df_part_a_full.drop(columns=['key', '累积装机(MW)_new', '数据来源_new'], errors='ignore')

        # 删除临时的 'date' 列
        result_df_part_a_full = result_df_part_a_full.drop(columns=['date'], errors='ignore')
        return result_df_part_a_full

    else:
        raise ValueError(f"Unknown method: {method}")

def _process_combination_a_has_data_b_empty(
    df_A_subset: pd.DataFrame
) -> pd.DataFrame:
    """
    子函数：处理 A 不空, B 为空 的情况。
    直接返回 A 的原始值，不修改数据来源。

    Parameters:
    -----------
    df_A_subset : pd.DataFrame
        来自A数据（如插值后的月度数据）的子集。

    Returns:
    --------
    pd.DataFrame
        原始的A数据子集，保持其原始 '数据来源' 列不变。
    """
    # A数据存在，B数据为空。无需校准，直接返回A数据的副本。
    # 确保不修改原始数据，返回一个副本
    result_df = df_A_subset.copy()
    # 不修改 '数据来源' 列，保留其原始值。
    # 如果原始A数据中 '数据来源' 为空或未定义，后续流程可能会处理。
    return result_df

def _process_combination_a_empty_b_has_data(
    df_B_subset: pd.DataFrame,
    start_year: int,
    end_year: int, # 用于确定生成数据的时间范围
    city_mapping: dict # New parameter: mapping from (prov_abbr, city_abbr) to (prov_full, city_full)
) -> pd.DataFrame:
    """
    子函数：处理 A 空, B 不空 的情况。
    根据 B 数据生成 2023年12月及之前的数据，数据来源为 '校正内插值' 或 '收集校准值'。
    对于B数据点，来源保持为 '收集校准值'（或B数据原有的来源）。
    对于基于B数据点推导出的其他时间点，来源为 '校正内插值'。
    确保省全称和市全称正确填充。

    Parameters:
    -----------
    df_B_subset : pd.DataFrame
        来自B数据（校准数据）的子集。
    start_year : int
        生成数据的起始年份。
    end_year : int
        生成数据的结束年份（例如2023）。
    city_mapping : dict
        从 (省简称, 市简称) 映射到 (省全称, 市全称) 的字典。

    Returns:
    --------
    pd.DataFrame
        根据B数据生成的完整时间序列数据。
    """
    if df_B_subset.empty:
        # 如果B数据也为空，则应由第四个子函数处理
        # 此函数不应接收空的B数据
        return pd.DataFrame(columns=['年', '月', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)', '能源类型', '数据来源'])

    # 确保日期列存在
    if 'date' not in df_B_subset.columns:
        df_B_subset = df_B_subset.copy()
        df_B_subset['date'] = pd.to_datetime(df_B_subset['年'].astype(str) + '-' + df_B_subset['月'].astype(str) + '-01') + pd.offsets.MonthEnd(0)

    # 获取当前组合的固定信息 (从B数据获取简称)
    prov_abbr = df_B_subset['省'].iloc[0] if '省' in df_B_subset.columns else ''
    city_abbr = df_B_subset['市'].iloc[0] if '市' in df_B_subset.columns else ''
    energy_type = df_B_subset['能源类型'].iloc[0] if '能源类型' in df_B_subset.columns else ''

    # 使用映射表查找对应的全称
    prov_full, city_full = city_mapping.get((prov_abbr, city_abbr), (prov_abbr, city_abbr)) # Fallback to abbrev if not found

    # 按日期排序B数据
    df_B_sorted = df_B_subset.copy().sort_values('date').reset_index(drop=True)

    N = len(df_B_sorted)
    generated_records = []

    if N == 1:
        # 情况：只有一个B数据点
        calib_date = df_B_sorted.iloc[0]['date']
        calib_value = df_B_sorted.iloc[0]['装机量(MW)']

        # 遍历从 start_year 到 end_year 的所有月份
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                current_date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                if current_date <= calib_date:
                    # 当前日期在B点之前或就是B点
                    if current_date == calib_date:
                        # 这个点就是原始的B点
                        source = '收集校准值' # 或者保留 df_B_sorted.iloc[0]['数据来源'] 如果B数据有此列
                        value = calib_value
                    else:
                        # B点之前的点，累积装机为0
                        source = '校正内插值'
                        value = 0.0
                else:
                    # 当前日期在B点之后，累积装机为B点的值
                    source = '校正内插值'
                    value = calib_value

                record = {
                    '年': year,
                    '月': month,
                    '省全称': prov_full, # Use the full name from mapping
                    '省简称': prov_abbr,
                    '市全称': city_full, # Use the full name from mapping
                    '市简称': city_abbr,
                    '新增装机(MW)': 0.0 if month == 12 else float('nan'),
                    '累积装机(MW)': value,
                    '能源类型': energy_type,
                    '数据来源': source
                }
                generated_records.append(record)

    elif N > 1:
        # 情况：有多个B数据点
        df_B_sorted = df_B_sorted.sort_values('date').reset_index(drop=True)
        all_dates_in_range = []

        # 生成从 start_year 到 end_year 的所有月末日期
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                all_dates_in_range.append(pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0))

        # 为所有日期点初始化值和来源
        date_to_value = {}
        date_to_source = {}

        # 1. 处理B点：将B点的值和来源直接赋给对应日期
        for _, row in df_B_sorted.iterrows():
            b_date = row['date']
            b_value = row['装机量(MW)']
            if b_date in all_dates_in_range:
                date_to_value[b_date] = b_value
                date_to_source[b_date] = '收集校准值' # 或保留原始来源

        # 2. 处理非B点：根据平台截断法逻辑填充
        # 遍历所有日期点
        for i, current_date in enumerate(all_dates_in_range):
            if current_date in date_to_value:
                # 这是B点，已处理，跳过
                continue

            # 找到当前日期之前的最后一个B点
            prev_b_idx = -1
            for j in range(i - 1, -1, -1):
                if all_dates_in_range[j] in date_to_value and date_to_source[all_dates_in_range[j]] == '收集校准值':
                    prev_b_idx = j
                    break

            # 找到当前日期之后的第一个B点
            next_b_idx = -1
            for j in range(i + 1, len(all_dates_in_range)):
                if all_dates_in_range[j] in date_to_value and date_to_source[all_dates_in_range[j]] == '收集校准值':
                    next_b_idx = j
                    break

            if prev_b_idx == -1 and next_b_idx == -1:
                # 当前日期前后都没有B点
                date_to_value[current_date] = 0.0
                date_to_source[current_date] = '校正内插值'
            elif prev_b_idx == -1 and next_b_idx != -1:
                # 当前日期之前没有B点，之后有B点
                next_b_date = all_dates_in_range[next_b_idx]
                date_to_value[current_date] = 0.0
                date_to_source[current_date] = '校正内插值'
            elif prev_b_idx != -1 and next_b_idx == -1:
                # 当前日期之前有B点，之后没有B点
                prev_b_date = all_dates_in_range[prev_b_idx]
                date_to_value[current_date] = date_to_value[prev_b_date]
                date_to_source[current_date] = '校正内插值'
            else:
                # 当前日期前后都有B点
                prev_b_date = all_dates_in_range[prev_b_idx]
                next_b_date = all_dates_in_range[next_b_idx]
                # 使用线性插值（或保持前一个B点的值，取决于具体需求）
                # 这里采用平台截断法的中间处理方式，即保持前一个B点的值，直到遇到下一个B点
                date_to_value[current_date] = date_to_value[prev_b_date]
                date_to_source[current_date] = '校正内插值'

        # 3. 构建最终记录
        for date in all_dates_in_range:
            year, month = date.year, date.month
            record = {
                '年': year,
                '月': month,
                '省全称': prov_full, # Use the full name from mapping
                '省简称': prov_abbr,
                '市全称': city_full, # Use the full name from mapping
                '市简称': city_abbr,
                '新增装机(MW)': 0.0 if month == 12 else float('nan'),
                '累积装机(MW)': date_to_value[date],
                '能源类型': energy_type,
                '数据来源': date_to_source[date]
            }
            generated_records.append(record)

    if generated_records:
        generated_df = pd.DataFrame(generated_records)
        generated_df['累积装机(MW)'] = generated_df['累积装机(MW)'].round(2)
        return generated_df
    else:
        return pd.DataFrame(columns=['年', '月', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)', '能源类型', '数据来源'])


def _process_combination_a_empty_b_empty(
    prov_full: str, prov_abbr: str, city_full: str, city_abbr: str, energy_type: str,
    start_year: int = 2000,
    end_year: int = 2023 # 用于确定生成数据的时间范围
) -> pd.DataFrame:
    """
    子函数：处理 A 空, B 空 的情况。
    生成从 start_year 到 end_year 的全零数据，数据来源为 '应无装机值'。

    Parameters:
    -----------
    prov_full : str
        省全称。
    prov_abbr : str
        省简称。
    city_full : str
        市全称。
    city_abbr : str
        市简称。
    energy_type : str
        能源类型。
    start_year : int
        生成数据的起始年份。
    end_year : int
        生成数据的结束年份（例如2023）。

    Returns:
    --------
    pd.DataFrame
        包含全零数据的DataFrame。
    """
    # 创建一个空列表来存储所有生成的记录
    zero_records = []
    # 遍历从 start_year 到 end_year 的所有月份
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 对于每一个月份，创建一个记录，累积装机为0，数据来源为 '应无装机值'
            record = {
                '年': year,
                '月': month,
                '省全称': prov_full,
                '省简称': prov_abbr,
                '市全称': city_full,
                '市简称': city_abbr,
                '新增装机(MW)': 0.0 if month == 12 else float('nan'), # 12月为0，其他月为NaN
                '累积装机(MW)': 0.0,
                '能源类型': energy_type,
                '数据来源': '应无装机值'
            }
            # 将记录添加到列表中
            zero_records.append(record)

    # 如果列表不为空（正常情况），则创建DataFrame
    if zero_records:
        zero_df = pd.DataFrame(zero_records)
        # 累积装机值通常已经是0.0，但为了确保，可以再次指定
        zero_df['累积装机(MW)'] = 0.0
        return zero_df
    else:
        # 理论上不应该发生，但如果循环没有执行，返回一个空的DataFrame，但具有正确的列结构
        return pd.DataFrame(columns=['年', '月', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)', '能源类型', '数据来源'])


def calibrate_with_collected_data(
    input_data: Union[str, pd.DataFrame],
    correction_data: Union[str, pd.DataFrame],
    onshore_cities_csv_path: str,
    offshore_cities_csv_path: str,
    output_csv_path: Optional[str] = None,
    method: str = 'platform_truncation',
    encoding: str = 'utf-8',
    start_year: int = 2000,
    end_year: int = 2023
) -> pd.DataFrame:
    """
    使用B数据（收集校准值）校准A数据 → 连带校准值
    重构逻辑：基于完整的城市列表进行迭代，处理A/B数据存在性的四种情况。
    """
    # Read input data (A data)
    if isinstance(input_data, str):
        df_A = _read_csv_with_encoding(input_data, encoding)
    elif isinstance(input_data, pd.DataFrame):
        df_A = input_data.copy()
    else:
        raise ValueError("input_data must be str or pd.DataFrame")

    # Read correction data (B data)
    if isinstance(correction_data, str):
        df_B = _read_csv_with_encoding(correction_data, encoding)
    elif isinstance(correction_data, pd.DataFrame):
        df_B = correction_data.copy()
    else:
        raise ValueError("correction_data must be str or pd.DataFrame")

    # Add date column to A data once
    df_A['年'] = df_A['年'].astype(int)
    df_A['月'] = df_A['月'].astype(int)
    df_A['date'] = pd.to_datetime(df_A['年'].astype(str) + '-' + df_A['月'].astype(str) + '-01') + pd.offsets.MonthEnd(0)

    # Add date column to B data once
    df_B['年'] = df_B['年'].astype(int)
    df_B['月'] = df_B['月'].astype(int)
    df_B['date'] = pd.to_datetime(df_B['年'].astype(str) + '-' + df_B['月'].astype(str) + '-01') + pd.offsets.MonthEnd(0)

    # 处理B数据中超出A数据时间范围的部分（如2024和2025年）
    max_date_A = pd.Timestamp(year=end_year, month=12, day=31) + pd.offsets.MonthEnd(0)
    df_B_beyond = df_B[df_B['date'] > max_date_A].copy()

    # Read city lists to define the complete iteration space and create mappings
    onshore_cities = _read_csv_with_encoding(onshore_cities_csv_path, encoding)
    offshore_cities = _read_csv_with_encoding(offshore_cities_csv_path, encoding)

    # Create a mapping from (省简称, 市简称) to (省全称, 市全称) using onshore cities
    city_mapping = {}
    for _, row in onshore_cities.iterrows():
        key = (row['省简称'], row['市简称'])
        value = (row['省全称'], row['市全称'])
        city_mapping[key] = value

    # Create all possible combinations for陆上风电
    onshore_combinations = set()
    for _, row in onshore_cities.iterrows():
        onshore_combinations.add((row['省全称'], row['省简称'], row['市全称'], row['市简称'], '陆上风电'))

    # Create all possible combinations for海上风电
    offshore_combinations = set()
    for _, row in offshore_cities.iterrows():
        prov_full = row['省全称']
        city_full = row['市全称']
        # Find corresponding abbreviations using the mapping
        prov_abbr, city_abbr = None, None
        for (p_abbr, c_abbr), (p_full, c_full) in city_mapping.items():
            if p_full == prov_full and c_full == city_full:
                prov_abbr, city_abbr = p_abbr, c_abbr
                break
        if prov_abbr is None or city_abbr is None:
            raise ValueError(f"Could not find abbreviation for offshore city: {prov_full}, {city_full}")
        offshore_combinations.add((prov_full, prov_abbr, city_full, city_abbr, '海上风电'))

    all_combinations = onshore_combinations.union(offshore_combinations)

    print(f"Processing {len(all_combinations)} unique combinations based on city lists...")
    results_list = []

    for prov_full, prov_abbr, city_full, city_abbr, energy_type in all_combinations:
        # --- Step 1: Filter A and B data for this specific combination ---
        df_A_subset = df_A[
            (df_A['省全称'] == prov_full) &
            (df_A['市全称'] == city_full) &
            (df_A['能源类型'] == energy_type)
        ].copy().sort_values('date').reset_index(drop=True)

        df_B_subset = df_B[
            (df_B['市'] == city_abbr) & # Note: B data uses 市简称
            (df_B['能源类型'] == energy_type)
        ].copy().sort_values('date').reset_index(drop=True)

        # --- Step 2: Determine case and call appropriate sub-function ---
        if not df_A_subset.empty and not df_B_subset.empty:
            # Case 1: A不空, B不空
            processed_result = _process_combination_a_has_data_b_has_data(df_A_subset, df_B_subset, method, start_year, end_year)
        elif not df_A_subset.empty and df_B_subset.empty:
            # Case 2: A不空, B空
            processed_result = _process_combination_a_has_data_b_empty(df_A_subset)
        elif df_A_subset.empty and not df_B_subset.empty:
            # Case 3: A空, B不空
            # Pass the mapping to the function - THIS WAS THE MISSING PART
            processed_result = _process_combination_a_empty_b_has_data(df_B_subset, start_year, end_year, city_mapping)
        else: # df_A_subset.empty and df_B_subset.empty
            # Case 4: A空, B空
            processed_result = _process_combination_a_empty_b_empty(prov_full, prov_abbr, city_full, city_abbr, energy_type, start_year, end_year)

        # Append the processed result of this combination to the list
        results_list.append(processed_result)

    # Combine all results from the calibration process
    if results_list:
        result_df = pd.concat(results_list, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=df_A.columns)

    # 处理B数据中超出A数据时间范围的部分，添加到最终结果中
    if not df_B_beyond.empty:
        df_B_beyond_expanded = df_B_beyond.copy()
        df_B_beyond_expanded = df_B_beyond_expanded.rename(columns={'装机量(MW)': '累积装机(MW)'})
        required_cols = ['省全称', '省简称', '新增装机(MW)', '数据来源']
        for col in required_cols:
            if col not in df_B_beyond_expanded.columns:
                if col == '新增装机(MW)':
                    df_B_beyond_expanded[col] = 0.0
                elif col == '数据来源':
                    df_B_beyond_expanded[col] = '收集校准值'
                else:
                    if col.startswith('省'):
                        prov_abbr_val = df_B_beyond_expanded['省'].iloc[0] if not df_B_beyond_expanded.empty else ''
                        # Use mapping to get full name
                        prov_full_val = prov_abbr_val # Default to abbr if mapping fails
                        for (p_abbr, _), (p_full, _) in city_mapping.items():
                            if p_abbr == prov_abbr_val:
                                prov_full_val = p_full
                                break
                        df_B_beyond_expanded['省全称'] = prov_full_val
                    elif col.startswith('市'):
                        city_abbr_val = df_B_beyond_expanded['市'].iloc[0] if not df_B_beyond_expanded.empty else ''
                        # Use mapping to get full name
                        city_full_val = city_abbr_val # Default to abbr if mapping fails
                        for (p_abbr, c_abbr), (_, c_full) in city_mapping.items():
                            if c_abbr == city_abbr_val:
                                city_full_val = c_full
                                break
                        df_B_beyond_expanded['市全称'] = city_full_val
        df_B_beyond_expanded = df_B_beyond_expanded.rename(columns={'省': '省简称', '市': '市简称'})
        standard_a_columns = ['年', '月', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)', '能源类型', '数据来源']
        df_B_beyond_final = df_B_beyond_expanded[[col for col in standard_a_columns if col in df_B_beyond_expanded.columns]]
        result_df = pd.concat([result_df, df_B_beyond_final], ignore_index=True)

    # Round values and sort
    result_df['累积装机(MW)'] = result_df['累积装机(MW)'].round(2)
    result_df = result_df.sort_values(['年', '月', '省简称', '市简称', '能源类型']).reset_index(drop=True)
    result_df.loc[result_df['数据来源'] == '收集校准值', '新增装机(MW)'] = np.nan
    # Drop temporary 'date' column if it was added
    result_df = result_df.drop(columns=['date'], errors='ignore')

    # --- CRITICAL FIX: Remove B-data specific columns that might have been introduced during internal processing ---
    standard_a_columns = ['年', '月', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)', '能源类型', '数据来源']
    result_df = result_df[ [col for col in standard_a_columns if col in result_df.columns] ]

    # Save to CSV if path provided
    if output_csv_path:
        result_df.to_csv(output_csv_path, index=False, encoding=encoding)
    print("Calibration completed.")
    return result_df
def fill_missing_city_names(
    input_data: Union[str, pd.DataFrame],
    onshore_cities_csv_path: str,
    output_csv_path: Optional[str] = None,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    补充结果中缺失的市全称信息
    
    Parameters:
    -----------
    input_data : str or pd.DataFrame
        现有的结果数据
    onshore_cities_csv_path : str
        陆上风电城市列表CSV路径（包含：省全称，省简称，市全称，市简称）
    output_csv_path : str, optional
        输出CSV文件路径，如果为None则不保存
    encoding : str
        CSV编码，默认'utf-8'
    
    Returns:
    --------
    pd.DataFrame
        补充市全称后的数据
    """
    # Read input data
    if isinstance(input_data, str):
        df = _read_csv_with_encoding(input_data, encoding)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("input_data must be str or pd.DataFrame")
    
    # Read onshore cities
    onshore_cities = _read_csv_with_encoding(onshore_cities_csv_path, encoding)
    
    # Create mappings from different combinations to city full name
    # Mapping 1: (省简称, 市简称) to 市全称
    city_abbr_mapping = {}
    for _, row in onshore_cities.iterrows():
        key = (row['省简称'], row['市简称'])
        value = row['市全称']
        city_abbr_mapping[key] = value
    
    # Mapping 2: (省全称, 市全称) to 市全称 (for consistency, though this is redundant)
    # Actually we need (省全称, 市简称) to 市全称
    city_by_prov_full_mapping = {}
    for _, row in onshore_cities.iterrows():
        key = (row['省全称'], row['市简称'])
        value = row['市全称']
        city_by_prov_full_mapping[key] = value
    
    # Find records where 市全称 is missing (None, NaN, or empty string)
    missing_mask = df['市全称'].isna() | (df['市全称'] == '') | (df['市全称'] == 'nan')
    
    print(f"Found {missing_mask.sum()} records with missing city names")
    
    # Fill missing city names
    for idx in df[missing_mask].index:
        prov_abbr = df.loc[idx, '省简称']
        city_abbr = df.loc[idx, '市简称']
        prov_full = df.loc[idx, '省全称']
        
        # First try to find using (省简称, 市简称)
        if (prov_abbr, city_abbr) in city_abbr_mapping:
            df.loc[idx, '市全称'] = city_abbr_mapping[(prov_abbr, city_abbr)]
        # If not found, try using (省全称, 市简称)
        elif (prov_full, city_abbr) in city_by_prov_full_mapping:
            df.loc[idx, '市全称'] = city_by_prov_full_mapping[(prov_full, city_abbr)]
        else:
            raise ValueError(f"Could not find city name for ({prov_full}, {prov_abbr}, {city_abbr})")
    
    # Save to CSV if path provided
    if output_csv_path:
        df.to_csv(output_csv_path, index=False, encoding=encoding)
        print(f"Results with filled city names saved to {output_csv_path}")
    
    return df


def recalculate_additions(
    input_df: pd.DataFrame,
    lifetime_years: int = 25
) -> pd.DataFrame:
    """
    重新计算最终结果df中的 新增装机(MW) 列。
    非12月的月份设为 NaN。
    12月的新增装机 = 当年12月累积 - 上年12月累积 + (25年前累积 - 26年前累积) [如果相关年份数据存在]。

    Parameters:
    -----------
    input_df : pd.DataFrame
        包含最终结果的DataFrame，需要有 ['年', '月', '省简称', '市简称', '能源类型', '累积装机(MW)'] 列。
    lifetime_years : int
        风电机组设计寿命，默认25年。

    Returns:
    --------
    pd.DataFrame
        修正后的DataFrame，'新增装机(MW)' 列已更新。
    """
    # Create a copy to work with
    df = input_df.copy()

    # Ensure required columns exist
    required_cols = ['年', '月', '省简称', '市简称', '能源类型', '累积装机(MW)']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")

    # Sort the dataframe to ensure proper order for groupby operations
    df = df.sort_values(['省简称', '市简称', '能源类型', '年', '月']).reset_index(drop=True)

    # Initialize 新增装机(MW) column with NaN for all rows
    df['新增装机(MW)'] = float('nan')

    # Define a function to calculate additions for a single group (city, tech)
    def calculate_additions_for_group(group_df):
        # Ensure group is sorted by year and month
        group_df = group_df.sort_values(['年', '月']).reset_index(drop=True)

        # Get December data indices for easier access
        dec_mask = (group_df['月'] == 12)
        dec_years = group_df.loc[dec_mask, '年']
        dec_cumulative = group_df.loc[dec_mask, '累积装机(MW)']
        dec_indices = group_df.loc[dec_mask].index

        # Calculate additions for December months
        for i, idx in enumerate(dec_indices):
            current_year = group_df.loc[idx, '年']
            current_cumulative = group_df.loc[idx, '累积装机(MW)']

            # 1. Calculate base addition (current year Dec - previous year Dec)
            prev_year = current_year - 1
            # Find the cumulative value for the previous December *within this group*
            # Use group_df, not the global df
            prev_dec_mask_in_group = (group_df['月'] == 12) & (group_df['年'] == prev_year)
            if prev_dec_mask_in_group.any():
                 prev_cumulative = group_df.loc[prev_dec_mask_in_group, '累积装机(MW)'].iloc[0]
                 base_addition = current_cumulative - prev_cumulative
            else:
                # If no previous December data exists in this group, the base addition is the current cumulative value
                base_addition = current_cumulative

            # 2. Calculate retirement (cumulative from 25 years ago - cumulative from 26 years ago)
            retire_year_25 = current_year - lifetime_years
            retire_year_26 = current_year - lifetime_years - 1
            retirement = 0.0 # Default to 0

            # Find cumulative value for 25 years ago (Dec) *within this group*
            retire_25_mask_in_group = (group_df['月'] == 12) & (group_df['年'] == retire_year_25)
            # Find cumulative value for 26 years ago (Dec) *within this group*
            retire_26_mask_in_group = (group_df['月'] == 12) & (group_df['年'] == retire_year_26)

            if retire_25_mask_in_group.any() and retire_26_mask_in_group.any():
                cum_25_years_ago = group_df.loc[retire_25_mask_in_group, '累积装机(MW)'].iloc[0]
                cum_26_years_ago = group_df.loc[retire_26_mask_in_group, '累积装机(MW)'].iloc[0]
                retirement = cum_25_years_ago - cum_26_years_ago

            # 3. Final addition is base addition plus retirement (if applicable)
            final_addition = base_addition + retirement

            # 4. Assign the calculated value back to the dataframe
            group_df.loc[idx, '新增装机(MW)'] = max(0, final_addition)

        return group_df

    # Apply the calculation function to each group defined by ['省简称', '市简称', '能源类型']
    # Use groupby with apply, ensuring groups are handled correctly
    df = df.groupby(['省简称', '市简称', '能源类型'], group_keys=False).apply(calculate_additions_for_group).reset_index(drop=True)

    # Ensure the '新增装机(MW)' column is float type and round if necessary
    df['新增装机(MW)'] = df['新增装机(MW)'].round(2)

    # Sort the final dataframe again to ensure consistent order
    df = df.sort_values(['年', '月', '省简称', '市简称', '能源类型']).reset_index(drop=True)

    print(f"Re-calculated 新增装机(MW) for {len(df[df['月']==12])} December records.")
    return df

# Example usage within the workflow:
# final_df = process_wind_data_complete_workflow(...)
# final_df_with_correct_additions = recalculate_additions(final_df, lifetime_years=25)


def _plot_single_city_future(args):
    """
    concurrent.futures 多进程绘图的辅助函数，用于处理单个城市的绘图任务
    """
    city, city_final, city_site, city_calib, output_dir, color_map_et, encoding, figsize = args
    if city_final.empty and city_site.empty and city_calib.empty:
        return f"Skipped {city} - no data"

    # --- 创建日期列 ---
    # Final result data
    city_final['date'] = pd.to_datetime(city_final['年'].astype(str) + '-' + city_final['月'].astype(str) + '-01')
    # Site aggregated data (only Dec data)
    city_site['date'] = pd.to_datetime(city_site['年'].astype(str) + '-' + city_site['月'].astype(str) + '-01')
    # Merged correction data
    city_calib['date'] = pd.to_datetime(city_calib['年'].astype(str) + '-' + city_calib['月'].astype(str) + '-01')

    # --- 确定主时间范围（X轴），固定结束时间为2025年12月 ---
    all_dates = pd.concat([city_final['date'], city_site['date'], city_calib['date']]).dropna()
    if all_dates.empty:
        return f"Skipped {city} - no valid dates for any data type"
    
    main_start_date = all_dates.min()
    main_end_date = pd.Timestamp(year=2025, month=12, day=31) # 固定结束时间

    # --- 过滤用于绘制主折线图的数据 (city_final) 和聚合点 (city_site) ---
    # 只保留时间范围内的数据用于折线图和聚合点（例如，截止到2023年底）
    end_of_2023 = pd.Timestamp(year=2023, month=12, day=31)
    city_final_for_plot = city_final[(city_final['date'] >= main_start_date) & (city_final['date'] <= end_of_2023)].copy()
    city_site_for_plot = city_site[(city_site['date'] >= main_start_date) & (city_site['date'] <= end_of_2023)].copy()

    # --- 不过滤用于绘制校正点的数据 (city_calib) ---
    # 校正点可以包含超出主范围的数据，它们将显示在固定的X轴范围内
    city_calib_for_plot = city_calib.copy() # 保留所有校正点

    # --- 创建图表 ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 绘制最终结果的折线 (按能源类型) ---
    # 使用过滤后的数据 (<= 2023)
    energy_types_in_final = city_final_for_plot['能源类型'].unique()
    for energy_type in energy_types_in_final:
        et_final = city_final_for_plot[city_final_for_plot['能源类型'] == energy_type].copy()
        if et_final.empty:
            continue
        et_final = et_final.sort_values('date')
        line_color = color_map_et.get(energy_type, {}).get('line', 'black')
        ax.plot(
            et_final['date'],
            et_final['累积装机(MW)'],
            label=f'{energy_type} (最终结果)',
            color=line_color,
            linewidth=2.0, # 线条宽度
            zorder=5 # 确保折线在较上层
        )
        # --- 在最后一个有效点添加数值标签 ---
        if not et_final.empty:
            last_row = et_final.iloc[-1]
            ax.annotate(
                f"{last_row['累积装机(MW)']:.1f} MW",
                xy=(last_row['date'], last_row['累积装机(MW)']),
                xytext=(5, 5), # Offset text by 5 points in x and y
                textcoords='offset points',
                fontsize=10, # 字体大小
                color=line_color,
                weight='bold',
                zorder=6
            )

    # --- 绘制聚合点 (来自 site_aggregated_data) ---
    # 使用过滤后的数据 (<= 2023)
    for energy_type in energy_types_in_final: # 使用最终数据中存在的能源类型
        et_site = city_site_for_plot[city_site_for_plot['能源类型'] == energy_type].copy()
        if et_site.empty:
            continue
        point_color = color_map_et.get(energy_type, {}).get('point', 'gray')
        ax.scatter(
            et_site['date'],
            et_site['累积装机(MW)'],
            label=f'{energy_type} (聚合点)',
            color=point_color,
            marker='o', # 圆形标记
            s=40, # 点的大小
            edgecolors='black', # 添加黑色边框，使点更清晰
            linewidth=0.8, # 边框宽度
            zorder=4 # 确保点在折线下方
        )

    # --- 绘制校正点 (来自 merged_correction_data) ---
    # 使用 *所有* 校正点数据，它们会显示在固定的X轴范围内 (直到2025)
    for energy_type in city_calib_for_plot['能源类型'].unique(): # 使用校准数据中存在的能源类型
        et_calib = city_calib_for_plot[city_calib_for_plot['能源类型'] == energy_type].copy()
        if et_calib.empty:
            continue
        calib_color = color_map_et.get(energy_type, {}).get('calib', 'red')
        ax.scatter(
            et_calib['date'],
            et_calib['装机量(MW)'], # 注意：校准数据中的列名为 '装机量(MW)'
            label=f'{energy_type} (校正点)',
            color=calib_color,
            marker='s', # 方形标记
            s=40, # 点的大小
            edgecolors='black', # 添加黑色边框
            linewidth=0.8, # 边框宽度
            zorder=3, # 确保点在聚合点下方
            alpha=0.7 # 添加透明度，便于区分
        )

    # --- 设置图表的X轴范围为固定范围 ---
    ax.set_xlim(main_start_date, main_end_date)

    # --- 设置图表标题和标签 ---
    # 尝试获取市全称，如果不存在则使用市简称
    city_full_name = city_final_for_plot['市全称'].iloc[0] if '市全称' in city_final_for_plot.columns and not city_final_for_plot.empty else city
    ax.set_title(f'{city_full_name} 累计装机容量时间序列 (折线至2023, 点至2025)', fontsize=16, fontweight='bold')
    ax.set_xlabel('时间', fontsize=12, fontweight='bold')
    ax.set_ylabel('累计装机容量 (MW)', fontsize=12, fontweight='bold')

    # --- 设置日期格式 ---
    ax.xaxis.set_major_locator(YearLocator()) # 每年一个主刻度
    ax.xaxis.set_major_formatter(DateFormatter('%Y')) # 格式化为年份
    plt.xticks(rotation=45, ha='right') # 旋转x轴标签，使其更易读

    # --- 添加网格 ---
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

    # --- 添加图例 ---
    # 放置在合适位置，避免遮挡数据
    ax.legend(loc='upper left', fontsize=10, title='图例', title_fontsize=12)

    # --- 调整布局 ---
    plt.tight_layout()

    # --- 生成图片文件名 ---
    filename = f"{city}_wind_capacity_detailed.png"
    filepath = os.path.join(output_dir, filename)

    # --- 保存图片 ---
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    result_msg = f"Saved detailed plot for {city} to {filepath}"

    # --- 关闭当前图形以释放内存 ---
    plt.close(fig)
    return result_msg


def plot_city_wind_capacity_from_results_concurrent(
    final_result_data: Union[str, pd.DataFrame],
    site_aggregated_data: Union[str, pd.DataFrame],  # 来自 aggregate_sites_to_annual 的数据
    merged_correction_data: Union[str, pd.DataFrame], # 来自 merge_correction_and_direct_city_data 的数据
    output_dir: str,
    encoding: str = 'utf-8',
    figsize: tuple = (16, 10),
    max_workers: Optional[int] = None
):
    """
    为每个城市绘制累计装机量时间序列图，精确区分最终结果曲线、聚合点和校正点。
    使用 concurrent.futures 多进程加速绘图。

    Parameters:
    -----------
    final_result_data : str or pd.DataFrame
        包含最终处理结果的数据，必须包含 ['年', '月', '市简称', '能源类型', '累积装机(MW)'] 列。
    site_aggregated_data : str or pd.DataFrame
        来自 `aggregate_sites_to_annual` 函数的输出数据，代表“聚合点”，应包含每年12月的数据。
    merged_correction_data : str or pd.DataFrame
        来自 `merge_correction_and_direct_city_data` 函数的输出数据，代表“校正点”，应包含所有年份的校准数据。
    output_dir : str
        保存图片的目录路径。
    encoding : str
        如果输入是文件路径，CSV的编码，默认'utf-8'。
    figsize : tuple
        每张图的大小 (宽, 高)。
    max_workers : int, optional
        使用的最大工作进程数。如果为None，则使用CPU核心数。
    """
    # 读取数据
    if isinstance(final_result_data, str):
        df_final = _read_csv_with_encoding(final_result_data, encoding=encoding)
    elif isinstance(final_result_data, pd.DataFrame):
        df_final = final_result_data.copy()
    else:
        raise ValueError("final_result_data must be str or pd.DataFrame")

    if isinstance(site_aggregated_data, str):
        df_site = _read_csv_with_encoding(site_aggregated_data, encoding=encoding)
    elif isinstance(site_aggregated_data, pd.DataFrame):
        df_site = site_aggregated_data.copy()
    else:
        raise ValueError("site_aggregated_data must be str or pd.DataFrame")

    if isinstance(merged_correction_data, str):
        df_calib = _read_csv_with_encoding(merged_correction_data, encoding=encoding)
    elif isinstance(merged_correction_data, pd.DataFrame):
        df_calib = merged_correction_data.copy()
    else:
        raise ValueError("merged_correction_data must be str or pd.DataFrame")

    # 确保必要的列存在
    required_cols_final = ['年', '月', '市简称', '能源类型', '累积装机(MW)']
    required_cols_site = ['年', '月', '市简称', '能源类型', '累积装机(MW)']
    required_cols_calib = ['年', '月', '市', '能源类型', '装机量(MW)'] # 校准数据中的城市列名为 '市'

    missing_cols_final = [col for col in required_cols_final if col not in df_final.columns]
    missing_cols_site = [col for col in required_cols_site if col not in df_site.columns]
    missing_cols_calib = [col for col in required_cols_calib if col not in df_calib.columns]

    if missing_cols_final:
        raise ValueError(f"Final result data is missing required columns: {missing_cols_final}")
    if missing_cols_site:
        raise ValueError(f"Site aggregated data is missing required columns: {missing_cols_site}")
    if missing_cols_calib:
        raise ValueError(f"Merged correction data is missing required columns: {missing_cols_calib}")

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有唯一的城市
    unique_cities = df_final['市简称'].unique()
    print(f"Found {len(unique_cities)} unique cities to plot.")

    # 定义颜色映射
    color_map_et = {
        '陆上风电': {'line': '#1f77b4', 'point': '#aec7e8', 'calib': '#d62728'},  # 深蓝, 浅蓝, 红
        '海上风电': {'line': '#17becf', 'point': '#9edae5', 'calib': '#ff7f0e'}   # 深青, 浅青, 橙
    }

    # 为每个城市准备绘图数据和任务参数
    plot_tasks = []
    for city in unique_cities:
        # 筛选当前城市的数据
        city_final = df_final[df_final['市简称'] == city].copy()
        city_site = df_site[df_site['市简称'] == city].copy()
        city_calib = df_calib[df_calib['市'] == city].copy() # 注意：校准数据用的是 '市' 列

        # 将任务参数打包为元组
        task_args = (city, city_final, city_site, city_calib, output_dir, color_map_et, encoding, figsize)
        plot_tasks.append(task_args)

    # 使用 concurrent.futures.ProcessPoolExecutor 执行绘图任务
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(unique_cities)) # 限制进程数不超过城市数
    
    print(f"Starting concurrent.futures plotting with max {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_city = {executor.submit(_plot_single_city_future, task_args): task_args[0] for task_args in plot_tasks}
        
        # 收集结果
        results = []
        for future in concurrent.futures.as_completed(future_to_city):
            city = future_to_city[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'City {city} generated an exception: {exc}')
                results.append(f"Error for {city}: {exc}")

    # 打印结果
    for msg in results:
        print(msg)

    print(f"Concurrent.futures plotting completed. Plots saved to {output_dir}")

def plot_city_wind_capacity_from_results(
    final_result_data: Union[str, pd.DataFrame],
    site_aggregated_data: Union[str, pd.DataFrame],  # 来自 aggregate_sites_to_annual 的数据
    merged_correction_data: Union[str, pd.DataFrame], # 来自 merge_correction_and_direct_city_data 的数据
    output_dir: str,
    encoding: str = 'utf-8',
    figsize: tuple = (16, 10)
):
    """
    为每个城市绘制累计装机量时间序列图，精确区分聚合点、校正点和最终结果。

    Parameters:
    -----------
    final_result_data : str or pd.DataFrame
        包含最终处理结果的数据，必须包含 ['年', '月', '市简称', '能源类型', '累积装机(MW)', '数据来源'] 列。
    site_aggregated_data : str or pd.DataFrame
        来自 `aggregate_sites_to_annual` 函数的输出数据，代表“聚合点”，应包含每年12月的数据。
    merged_correction_data : str or pd.DataFrame
        来自 `merge_correction_and_direct_city_data` 函数的输出数据，代表“校正点”，应包含所有年份的校准数据。
    output_dir : str
        保存图片的目录路径。
    encoding : str
        如果输入是文件路径，CSV的编码，默认'utf-8'。
    figsize : tuple
        每张图的大小 (宽, 高)。
    """
    # 读取数据
    if isinstance(final_result_data, str):
        df_final = _read_csv_with_encoding(final_result_data, encoding=encoding)
    elif isinstance(final_result_data, pd.DataFrame):
        df_final = final_result_data.copy()
    else:
        raise ValueError("final_result_data must be str or pd.DataFrame")

    if isinstance(site_aggregated_data, str):
        df_site = _read_csv_with_encoding(site_aggregated_data, encoding=encoding)
    elif isinstance(site_aggregated_data, pd.DataFrame):
        df_site = site_aggregated_data.copy()
    else:
        raise ValueError("site_aggregated_data must be str or pd.DataFrame")

    if isinstance(merged_correction_data, str):
        df_calib = _read_csv_with_encoding(merged_correction_data, encoding=encoding)
    elif isinstance(merged_correction_data, pd.DataFrame):
        df_calib = merged_correction_data.copy()
    else:
        raise ValueError("merged_correction_data must be str or pd.DataFrame")

    # 确保必要的列存在
    required_cols_final = ['年', '月', '市简称', '能源类型', '累积装机(MW)', '数据来源']
    required_cols_site = ['年', '月', '市简称', '能源类型', '累积装机(MW)', '数据来源'] # 聚合数据也应有数据来源
    required_cols_calib = ['年', '月', '市', '能源类型', '装机量(MW)'] # 校准数据中的城市列名为 '市'

    missing_cols_final = [col for col in required_cols_final if col not in df_final.columns]
    missing_cols_site = [col for col in required_cols_site if col not in df_site.columns]
    missing_cols_calib = [col for col in required_cols_calib if col not in df_calib.columns]

    if missing_cols_final:
        raise ValueError(f"Final result data is missing required columns: {missing_cols_final}")
    if missing_cols_site:
        raise ValueError(f"Site aggregated data is missing required columns: {missing_cols_site}")
    if missing_cols_calib:
        raise ValueError(f"Merged correction data is missing required columns: {missing_cols_calib}")

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有唯一的城市
    unique_cities = df_final['市简称'].unique()

    # 定义颜色映射
    color_map_et = {
        '陆上风电': {
            'line': '#1f77b4', # 深蓝 - 最终结果折线
            'point': '#aec7e8', # 浅蓝 - 聚合点
            'calib': '#d62728', # 红 - 校正点
            'final_point': '#ff7f0e' # 橙 - 最终结果中的特定点 (如连带校准值, 收集校准值)
        },
        '海上风电': {
            'line': '#17becf', # 深青 - 最终结果折线
            'point': '#9edae5', # 浅青 - 聚合点
            'calib': '#ff7f0e', # 橙 - 校正点
            'final_point': '#d62728' # 红 - 最终结果中的特定点
        }
    }

    # 定义数据来源颜色映射
    source_color_map = {
        '站点聚合值': color_map_et['陆上风电']['point'], # 会根据能源类型调整
        '月度内插值': '#7f7f7f', # 灰色
        '连带校准值': color_map_et['陆上风电']['final_point'],
        '收集校准值': color_map_et['陆上风电']['calib'],
        '应无装机值': '#9467bd' # 紫色
    }

    for city in unique_cities:
        # 筛选当前城市的数据
        city_final = df_final[df_final['市简称'] == city].copy()
        city_site = df_site[df_site['市简称'] == city].copy()
        city_calib = df_calib[df_calib['市'] == city].copy() # 注意：校准数据用的是 '市' 列

        if city_final.empty and city_site.empty and city_calib.empty:
            continue  # 如果所有数据都为空，跳过

        # --- 创建日期列 ---
        # Final result data
        city_final['date'] = pd.to_datetime(city_final['年'].astype(str) + '-' + city_final['月'].astype(str) + '-01')
        # Site aggregated data (only Dec data)
        city_site['date'] = pd.to_datetime(city_site['年'].astype(str) + '-' + city_site['月'].astype(str) + '-01')
        # Merged correction data
        city_calib['date'] = pd.to_datetime(city_calib['年'].astype(str) + '-' + city_calib['月'].astype(str) + '-01')

        # --- 过滤时间范围 ---
        # 基于最终数据的时间范围，确保一致性
        if not city_final.empty:
            start_date = city_final['date'].min()
            end_date = city_final['date'].max()
        else:
            # 如果最终数据为空，基于其他数据确定范围
            all_dates = pd.concat([city_site['date'], city_calib['date']]).dropna()
            if not all_dates.empty:
                start_date = all_dates.min()
                end_date = all_dates.max()
            else:
                continue # 如果没有有效日期，跳过

        city_final = city_final[(city_final['date'] >= start_date) & (city_final['date'] <= end_date)]
        city_site = city_site[(city_site['date'] >= start_date) & (city_site['date'] <= end_date)]
        city_calib = city_calib[(city_calib['date'] >= start_date) & (city_calib['date'] <= end_date)]

        if city_final.empty and city_site.empty and city_calib.empty:
            print(f"Warning: No data within range for city {city}, skipping plot.")
            continue

        # --- 创建图表 ---
        fig, ax = plt.subplots(figsize=figsize)

        # --- 绘制最终结果的折线 (按能源类型) ---
        energy_types_in_final = city_final['能源类型'].unique()
        for energy_type in energy_types_in_final:
            et_final = city_final[city_final['能源类型'] == energy_type].copy()
            if et_final.empty:
                continue
            et_final = et_final.sort_values('date')
            
            line_color = color_map_et.get(energy_type, {}).get('line', 'black')
            ax.plot(
                et_final['date'],
                et_final['累积装机(MW)'],
                label=f'{energy_type} (最终结果折线)',
                color=line_color,
                linewidth=2.0, # 线条宽度
                zorder=5 # 确保折线在较上层
            )

            # --- 在最后一个有效点添加数值标签 ---
            if not et_final.empty:
                last_row = et_final.iloc[-1]
                ax.annotate(
                    f"{last_row['累积装机(MW)']:.1f} MW",
                    xy=(last_row['date'], last_row['累积装机(MW)']),
                    xytext=(5, 5), # Offset text by 5 points in x and y
                    textcoords='offset points',
                    fontsize=10, # 字体大小
                    color=line_color,
                    weight='bold',
                    zorder=6
                )

        # --- 绘制聚合点 (来自 site_aggregated_data) ---
        for energy_type in energy_types_in_final: # 使用最终数据中存在的能源类型
            et_site = city_site[city_site['能源类型'] == energy_type].copy()
            if et_site.empty:
                continue
            point_color = color_map_et.get(energy_type, {}).get('point', 'gray')
            ax.scatter(
                et_site['date'],
                et_site['累积装机(MW)'],
                label=f'{energy_type} (聚合点)',
                color=point_color,
                marker='o', # 圆形标记
                s=40, # 点的大小
                edgecolors='black', # 添加黑色边框，使点更清晰
                linewidth=0.8, # 边框宽度
                zorder=4 # 确保点在折线下方
            )

        # --- 绘制校正点 (来自 merged_correction_data) ---
        for energy_type in energy_types_in_final: # 使用最终数据中存在的能源类型
            et_calib = city_calib[city_calib['能源类型'] == energy_type].copy()
            if et_calib.empty:
                continue
            calib_color = color_map_et.get(energy_type, {}).get('calib', 'red')
            ax.scatter(
                et_calib['date'],
                et_calib['装机量(MW)'], # 注意：校准数据中的列名为 '装机量(MW)'
                label=f'{energy_type} (校正点)',
                color=calib_color,
                marker='s', # 方形标记
                s=40, # 点的大小
                edgecolors='black', # 添加黑色边框
                linewidth=0.8, # 边框宽度
                zorder=3 # 确保点在聚合点下方
            )
        
        # --- 绘制最终结果中的特定数据源点 (连带校准值, 收集校准值等) ---
        # 首先获取所有最终结果中非默认来源的点
        special_sources = ['连带校准值', '收集校准值', '应无装机值']
        for source in special_sources:
            source_mask = (city_final['数据来源'] == source)
            source_data = city_final[source_mask].copy()
            if source_data.empty:
                continue
            
            # 根据能源类型和数据源确定颜色
            for energy_type in energy_types_in_final:
                et_source_data = source_data[source_data['能源类型'] == energy_type].copy()
                if et_source_data.empty:
                    continue
                    
                # 获取对应能源类型的默认颜色，或使用通用颜色
                base_color = color_map_et.get(energy_type, {}).get('final_point', source_color_map.get(source, 'purple'))
                
                ax.scatter(
                    et_source_data['date'],
                    et_source_data['累积装机(MW)'],
                    label=f'{energy_type} ({source})',
                    color=base_color,
                    marker='^', # 三角形标记，区分其他点
                    s=50, # 点的大小
                    edgecolors='black', # 添加黑色边框
                    linewidth=0.8, # 边框宽度
                    zorder=7 # 确保特殊点在最上层
                )

        # --- 设置图表标题和标签 ---
        # 尝试获取市全称，如果不存在则使用市简称
        city_full_name = city_final['市全称'].iloc[0] if '市全称' in city_final.columns and not city_final.empty else city
        ax.set_title(f'{city_full_name} 累计装机容量时间序列', fontsize=16, fontweight='bold')
        ax.set_xlabel('时间', fontsize=12, fontweight='bold')
        ax.set_ylabel('累计装机容量 (MW)', fontsize=12, fontweight='bold')

        # --- 设置日期格式 ---
        ax.xaxis.set_major_locator(YearLocator()) # 每年一个主刻度
        ax.xaxis.set_major_formatter(DateFormatter('%Y')) # 格式化为年份
        plt.xticks(rotation=45, ha='right') # 旋转x轴标签，使其更易读

        # --- 添加网格 ---
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

        # --- 添加图例 ---
        # 放置在合适位置，避免遮挡数据
        ax.legend(loc='upper left', fontsize=10, title='图例', title_fontsize=12)

        # --- 调整布局 ---
        plt.tight_layout()

        # --- 生成图片文件名 ---
        filename = f"{city}_wind_capacity_detailed.png"
        filepath = os.path.join(output_dir, filename)

        # --- 保存图片 ---
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved detailed plot for {city} to {filepath}")

        # --- 关闭当前图形以释放内存 ---
        plt.close(fig)

# Main workflow function to run all steps in sequence
def process_wind_data_complete_workflow(
    sites_csv_path: str,
    coastal_cities_csv_path: str,
    correction_csv_path: str, # Original correction data
    prov_csv_path: str,       # Provincial data (for direct cities and provincial increase calc)
    onshore_all_cities_csv_path: str,
    offshore_coastal_cities_csv: str,
    output_csv_path: str,
    aggr_site_output_path: Optional[str] = None,
    correction_output_path: Optional[str] = None,
    plot_output_dir: Optional[str] = None,  # 新增参数：绘图输出目录
    start_year: int = 2000,
    end_year: int = 2024,
    lifetime_years: int = 25,
    method: str = 'platform_truncation',
    encoding: str = 'utf-8-sig'
) -> pd.DataFrame:
    """
    Complete workflow to process wind data through all steps, with updated logic for direct-controlled municipalities.
    Optionally generates plots for each city.
    """
    print("Step 1: Aggregating sites to annual data...")
    step1_result = aggregate_sites_to_annual(
        input_data=sites_csv_path,
        coastal_cities_csv_path=coastal_cities_csv_path,
        output_csv_path=aggr_site_output_path,
        lifetime_years=lifetime_years,
        start_year=start_year,
        end_year=end_year-1  # Process only up to 2023 initially
    )
    print("Step 2: Interpolating monthly within years...")
    step2_result = interpolate_monthly_within_years(
        input_data=step1_result,
        output_csv_path=None
    )
    step2_result.to_csv("D:/try/step2_result.csv", index=False, encoding=encoding)  # Debug output
    print("Step 3: Merging correction data with direct city provincial data...")
    # Use the new function to create the updated B data
    updated_correction_data = merge_correction_and_direct_city_data(
        original_correction_data=correction_csv_path, # e.g., "校对后市级陆海风电部分装机数据统计.csv"
        provincial_data=prov_csv_path,               # e.g., "省级装机量统计.csv"
        encoding=encoding
    )
    if correction_output_path:
        updated_correction_data.to_csv(correction_output_path, index=False, encoding=encoding)
    print("Step 4: Calibrating with the updated collected data...")
    step4_result = calibrate_with_collected_data( # Use the NEW function
        input_data=step2_result,
        correction_data=updated_correction_data, # Use the new merged data
        onshore_cities_csv_path=onshore_all_cities_csv_path,
        offshore_cities_csv_path=offshore_coastal_cities_csv,
        output_csv_path=None,
        method=method,
        start_year=start_year,
        end_year=end_year-1 # Use end_year - 1 for internal processing
    )
    step4_result.to_csv("D:/try/step4_result.csv", index=False, encoding=encoding)  # Debug output
    # Step 5: Adding missing cities is now handled by calibrate_with_collected_data
    # print("Step 5: Adding missing cities...")
    # step5_result = add_missing_cities(...) # This step is no longer needed
    print("Step 5: Filling missing city names...")
    step5_result = fill_missing_city_names(
        input_data=step4_result,
        onshore_cities_csv_path=onshore_all_cities_csv_path,
        output_csv_path=None,
        encoding=encoding
    )
    step5_result = recalculate_additions(step5_result, 25)
    # Generate plots if output directory is provided
    if plot_output_dir:
        print(f"Step 6: Generating plots to {plot_output_dir}...")
        plot_city_wind_capacity_from_results_concurrent(
            final_result_data=step5_result,
            site_aggregated_data=step1_result,
            merged_correction_data=updated_correction_data,
            output_dir="D:/try/plots/",
            encoding='utf-8',
            figsize=(16, 10),
            max_workers=16  # 例如，使用最多4个进程
        )
    # Save final result
    step5_result.to_csv(output_csv_path, index=False, encoding=encoding)
    print(f"Complete workflow finished. Final result saved to {output_csv_path}")
    return step5_result


if __name__ == "__main__":
    sites_csv_path = r"D:\HKU PhD\wind data paper\风电场信息-2023.csv"
    coastal_cities_csv_path = r"D:\Data\admin\沿海市.csv"
    correction_csv_path = r"D:\Data\自统装机数据\校对后市级陆海风电部分装机数据统计.csv"
    prov_csv_path = r"D:\Data\自统装机数据\省级装机量统计.csv" # New path for provincial data
    onshore_all_cities_csv_path = r"D:\Data\admin\省市名全简称.csv"
    offshore_coastal_cities_csv = r"D:\Data\admin\沿海市.csv"
    output_csv_path = "D:/try/final_output.csv"
    aggr_site_output_path = "D:/try/aggregated_sites.csv"
    correction_output_path = "D:/try/merged_correction_data.csv"
    plot_output_dir = "D:/try/plots/"

    final_df = process_wind_data_complete_workflow(
        sites_csv_path=sites_csv_path,
        coastal_cities_csv_path=coastal_cities_csv_path,
        correction_csv_path=correction_csv_path,
        prov_csv_path=prov_csv_path, # Pass the provincial data path
        onshore_all_cities_csv_path=onshore_all_cities_csv_path,
        offshore_coastal_cities_csv=offshore_coastal_cities_csv,
        output_csv_path=output_csv_path,
        aggr_site_output_path=aggr_site_output_path,
        correction_output_path=correction_output_path,
        plot_output_dir=plot_output_dir,
        lifetime_years=25,
        start_year=2000,
        end_year=2024,
        method='platform_truncation'
    )


    # TODO: 当没有站点数据时候，现在的逻辑是直接补0，但这应该是不对的。辽阳，天津等有校正数据