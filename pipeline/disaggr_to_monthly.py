import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
plt.rcParams['axes.unicode_minus'] = False    # To display negative signs correctly

def calibrate_city_capacity(
    original_csv: str,        # A 数据 CSV 路径
    correction_csv: str,      # B 数据 CSV 路径
    tech_name: str,           # 技术名称（如 "陆上风电"）
    city_name: str,           # 待校准城市名（如 "呼伦贝尔市" 或 "呼伦贝尔"）
    plot: bool = False,       # 是否绘图
    encoding: str = 'utf-8',  # CSV 文件编码
    method: str = 'scale_factor'  # 'scale_factor' or 'platform_truncation'
) -> pd.DataFrame:
    """
    对指定城市的装机容量数据进行校准，使校准后曲线通过权威校准点。
    如果B数据中没有对应的城市和能源类型数据，则直接返回插值后的A数据。
    
    Parameters:
    -----------
    original_csv : str
        原始年度数据 CSV 路径（包含年份、累积装机等）
    correction_csv : str
        校准点数据 CSV 路径（包含年、月、装机量等）
    tech_name : str
        能源类型（用于筛选 B 数据）
    city_name : str
        待校准城市名（用于筛选 A 和 B 数据）
    plot : bool
        是否绘制校准前后对比图
    encoding : str
        CSV 文件编码，默认 'utf-8'，若失败可尝试 'gbk' 或 'gb2312'
    method : str
        校准方法：
        - 'scale_factor': 乘性缩放因子法（默认）
        - 'platform_truncation': 平台截断法（阶梯式修正）
    
    Returns:
    --------
    pd.DataFrame
        包含 'date', 'calibrated_capacity' 两列的月度校准结果
    """
    
    if method not in ['scale_factor', 'platform_truncation']:
        raise ValueError("method must be 'scale_factor' or 'platform_truncation'")
    
    # ========================
    # 1. Load and Filter Data (with encoding support)
    # ========================
    try:
        df_A = pd.read_csv(original_csv, encoding=encoding)
    except UnicodeDecodeError:
        # Fallback to gbk for Chinese data
        df_A = pd.read_csv(original_csv, encoding='gbk')
    
    try:
        df_B = pd.read_csv(correction_csv, encoding=encoding)
    except UnicodeDecodeError:
        df_B = pd.read_csv(correction_csv, encoding='gbk')
    
    # A: Filter by city (using 市简称)
    df_A = df_A[df_A['市简称'] == city_name]
       
    if df_A.empty:
        raise ValueError(f"No data found for city '{city_name}' in A CSV")
    
    # B: Filter by city AND tech type
    df_B = df_B[df_B['市'] == city_name]
    df_B = df_B[df_B['能源类型'] == tech_name]
    
    # 如果B数据为空，直接返回插值后的A数据
    if df_B.empty:
        print(f"Warning: No data found for city '{city_name}' and tech '{tech_name}' in B CSV. Returning interpolated A data without calibration.")
        
        # ========================
        # Prepare A as Monthly DataFrame
        # ========================
        years_A = df_A['年份'].values
        caps_A = df_A['累积装机(MW)'].values
        
        # Create full monthly range for A's time span
        date_range_A = pd.date_range(start=f'{years_A[0]}-12-31', end=f'{years_A[-1]}-12-31', freq='ME')
        df_monthly = pd.DataFrame({'date': date_range_A})
        df_monthly['cumulative_capacity'] = np.nan
        
        # Fill December values
        for year, cap in zip(years_A, caps_A):
            dec_date = pd.Timestamp(f'{year}-12-31')
            mask = df_monthly['date'] == dec_date
            df_monthly.loc[mask, 'cumulative_capacity'] = cap
        
        # Interpolate to get monthly values
        df_monthly['cumulative_capacity'] = df_monthly['cumulative_capacity'].interpolate(method='linear')
        
        # Rename and return
        df_result = df_monthly[['date', 'cumulative_capacity']].copy()
        df_result.rename(columns={'cumulative_capacity': 'calibrated_capacity'}, inplace=True)
        
        # Remove any NaN values that might remain
        df_result = df_result.dropna(subset=['calibrated_capacity'])
        
        # Optional Plotting
        if plot:
            plt.figure(figsize=(14, 6))
            
            # Original A (monthly)
            plt.plot(
                df_result['date'], df_result['calibrated_capacity'], 
                'b-', linewidth=2, label='Interpolated A (no calibration needed)'
            )
            
            plt.xlabel('Year')
            plt.ylabel('Cumulative Capacity (MW)')
            plt.title(f'Capacity Data for {city_name} ({tech_name}) - No B Calibration Data Available')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return df_result
    
    # ========================
    # 2. Prepare A as Monthly DataFrame
    # ========================
    years_A = df_A['年份'].values
    caps_A = df_A['累积装机(MW)'].values
    
    # Create full monthly range for A's time span
    date_range_A = pd.date_range(start=f'{years_A[0]}-12-31', end=f'{years_A[-1]}-12-31', freq='ME')
    df_monthly = pd.DataFrame({'date': date_range_A})
    df_monthly['cumulative_capacity'] = np.nan
    
    # Fill December values
    for year, cap in zip(years_A, caps_A):
        dec_date = pd.Timestamp(f'{year}-12-31')
        mask = df_monthly['date'] == dec_date
        df_monthly.loc[mask, 'cumulative_capacity'] = cap
    
    # Interpolate to get monthly values
    df_monthly['cumulative_capacity'] = df_monthly['cumulative_capacity'].interpolate(method='linear')
    
    # ========================
    # 3. Prepare B with Supplement (first non-zero to zero)
    # ========================
    # Add supplement: first non-zero year to zero (for construction start)
    non_zero_years = df_A[df_A['累积装机(MW)'] > 0]['年份']
    if not non_zero_years.empty:
        first_non_zero_year = non_zero_years.min()
        # Check if this year is already 0 (then no need to supplement)
        if df_A[df_A['年份'] == first_non_zero_year]['累积装机(MW)'].iloc[0] > 0:
            supplement = pd.DataFrame({
                '年': [first_non_zero_year],
                '月': [12],
                '装机量(MW)': [0.0]
            })
            df_B = pd.concat([df_B, supplement], ignore_index=True)
    
    # Convert B to datetime
    df_B['date_str'] = df_B['年'].astype(str) + '-' + df_B['月'].astype(str) + '-01'
    df_B['date'] = pd.to_datetime(df_B['date_str']) + pd.offsets.MonthEnd(0)
    df_B = df_B.drop(columns=['date_str'])
    df_B = df_B.sort_values('date').reset_index(drop=True)
    
    # Remove duplicate dates in B data to prevent interpolation issues
    df_B = df_B.drop_duplicates(subset=['date'], keep='first')
    
    # ========================
    # 4. Split B into "within A range" and "beyond A range"
    # ========================
    max_date_A = df_monthly['date'].max()
    min_date_A = df_monthly['date'].min()
    
    # B points within A's time range (for calibration)
    df_B_within = df_B[df_B['date'] <= max_date_A].copy()
    # B points beyond A's time range (for extrapolation)
    df_B_beyond = df_B[df_B['date'] > max_date_A].copy()
    
    # ========================
    # 5. Apply Calibration Method
    # ========================
    if method == 'scale_factor':
        # ===================================================================
        # METHOD 1: SCALE FACTOR (乘性缩放因子法)
        # ===================================================================
        # 核心思想：对原始A曲线进行全局/分段缩放，使其通过所有B点
        # 优点：保留A的相对波动形态，平滑过渡
        # 缺点：若B点远低于A，可能导致校准后曲线在B点间出现负斜率
        # ===================================================================
        if not df_B_within.empty:
            # Extract B points for calibration
            b_dates = df_B_within['date'].values
            b_caps = df_B_within['装机量(MW)'].values
            
            # Compute A values at B dates
            a_at_b = []
            for t in b_dates:
                # Find closest A date value
                closest_idx = (df_monthly['date'] - t).abs().argmin()
                a_val = df_monthly.iloc[closest_idx]['cumulative_capacity']
                a_at_b.append(a_val)
            
            a_at_b = np.array(a_at_b)
            
            # Compute scale factors, handling potential division by zero
            s_factors = []
            for cap_a, cap_b in zip(a_at_b, b_caps):
                if cap_b == 0:
                    s = 0.0
                elif cap_a > 0:
                    s = cap_b / cap_a
                else:
                    s = 0.0  # Fallback
                s_factors.append(s)
            
            # Only create interpolation if we have more than one point
            if len(b_dates) > 1:
                # Remove duplicate dates to avoid interpolation issues
                unique_dates = []
                unique_factors = []
                seen_dates = set()
                for date, factor in zip(b_dates, s_factors):
                    date_int = date.astype(np.int64)
                    if date_int not in seen_dates:
                        unique_dates.append(date_int)
                        unique_factors.append(factor)
                        seen_dates.add(date_int)
                
                if len(unique_dates) > 1:
                    s_interp = interp1d(
                        unique_dates,
                        unique_factors,
                        kind='linear',
                        fill_value="extrapolate",
                        bounds_error=False
                    )
                else:
                    # If only one unique point, use constant interpolation
                    def s_interp(x):
                        return np.full_like(x, unique_factors[0], dtype=float)
            else:
                # If only one B point, use constant scale factor
                constant_factor = s_factors[0] if s_factors else 1.0
                def s_interp(x):
                    return np.full_like(x, constant_factor, dtype=float)
            
            # Apply to A's monthly range
            dates_full = df_monthly['date'].values.astype(np.int64)
            a_full = df_monthly['cumulative_capacity'].values
            s_full = s_interp(dates_full)
            calibrated = a_full * s_full
            calibrated = np.maximum(0, calibrated)
            
            df_result = pd.DataFrame({
                'date': df_monthly['date'].values,
                'calibrated_capacity': calibrated
            })
        else:
            # No B points within range, return A as baseline
            df_result = df_monthly[['date', 'cumulative_capacity']].copy()
            df_result.rename(columns={'cumulative_capacity': 'calibrated_capacity'}, inplace=True)
    
    elif method == 'platform_truncation':
        # ===================================================================
        # METHOD 2: PLATFORM TRUNCATION (平台截断法)
        # ===================================================================
        # 核心思想：从左到右遍历时间点，遇到B点时根据B与A的关系进行修正
        # 1. 若 B > A：B点设为B值，后续A点设为 max(A, B) → 向右拉平
        # 2. 若 B <= A：B点设为B值，向前修正A点至 min(A, B) → 向左截断
        # 优点：严格通过B点，结果非递减，保留A的基本形态
        # 缺点：可能出现阶梯状，局部修正幅度较大
        # ===================================================================
        if not df_B_within.empty:
            # Create combined time grid: A dates + B dates
            all_dates = pd.concat([
                df_monthly[['date']], 
                df_B_within[['date']]
            ]).drop_duplicates().sort_values('date').reset_index(drop=True)
            
            # Interpolate A to all dates
            a_interp = np.interp(
                all_dates['date'].astype(np.int64),
                df_monthly['date'].astype(np.int64),
                df_monthly['cumulative_capacity'].values
            )
            
            # Create result array
            calibrated = a_interp.copy()
            
            # Sort B points by date and create mapping
            b_sorted = df_B_within.sort_values('date').reset_index(drop=True)
            b_date_to_val = dict(zip(b_sorted['date'], b_sorted['装机量(MW)']))
            
            # Process from left to right
            for i, date in enumerate(all_dates['date']):
                if date in b_date_to_val:
                    # This is a B point
                    b_val = b_date_to_val[date]
                    a_val = a_interp[i]
                    
                    if b_val > a_val:
                        # B > A: set to B, then forward: max(A, B)
                        # 几何意义：B点高于A曲线 → 从B点向右画水平线，拉平后续A点
                        calibrated[i] = b_val
                        # Continue forward until next B or end
                        j = i + 1
                        while j < len(all_dates) and all_dates['date'].iloc[j] not in b_date_to_val:
                            calibrated[j] = max(calibrated[j], b_val)
                            j += 1
                    else:
                        # B <= A: set to B, then backward: min(A, B)
                        # 几何意义：B点低于A曲线 → 从B点向左画水平线，截断左侧A点
                        calibrated[i] = b_val
                        # Go backward until previous B or start
                        j = i - 1
                        while j >= 0 and all_dates['date'].iloc[j] not in b_date_to_val:
                            calibrated[j] = min(calibrated[j], b_val)
                            j -= 1
                        # Continue forward from current position
                        continue  # Already at current i, continue loop
            
            df_result = pd.DataFrame({
                'date': all_dates['date'],
                'calibrated_capacity': calibrated
            })
        else:
            # No B points within range, return A as baseline
            df_result = df_monthly[['date', 'cumulative_capacity']].copy()
            df_result.rename(columns={'cumulative_capacity': 'calibrated_capacity'}, inplace=True)
    
    # ========================
    # 6. Handle B points beyond A's range (extrapolation for both methods)
    # ========================
    if not df_B_beyond.empty:
        # Get A's last calibrated value
        last_val_A = df_result['calibrated_capacity'].iloc[-1]
        last_date_A = df_result['date'].iloc[-1]
        
        # Create extrapolation points from A's end to each B beyond
        extrapolation_data = []
        
        # Start from A's last point
        current_date = last_date_A
        current_val = last_val_A
        
        for _, row in df_B_beyond.iterrows():
            target_date = row['date']
            target_val = row['装机量(MW)']
            
            # Generate monthly dates between current and target
            if (target_date - current_date).days > 31:  # More than 1 month gap
                monthly_range = pd.date_range(start=current_date + pd.DateOffset(months=1), 
                                            end=target_date, freq='ME')
                
                # Linear interpolation between current_val and target_val
                n_months = len(monthly_range)
                if n_months > 0:
                    vals = np.linspace(current_val, target_val, n_months + 1)[1:]  # Skip current_val
                    for d, v in zip(monthly_range, vals):
                        extrapolation_data.append({'date': d, 'calibrated_capacity': v})
            
            # Update for next iteration
            current_date = target_date
            current_val = target_val
        
        # Append extrapolation data
        if extrapolation_data:
            df_extrap = pd.DataFrame(extrapolation_data)
            df_result = pd.concat([df_result, df_extrap], ignore_index=True)
        
        # Also append the B points themselves (to ensure they're exactly hit)
        df_B_beyond_reformat = df_B_beyond[['date', '装机量(MW)']].copy()
        df_B_beyond_reformat.rename(columns={'装机量(MW)': 'calibrated_capacity'}, inplace=True)
        
        # Merge to ensure B points are exactly on the curve
        df_result = pd.concat([df_result, df_B_beyond_reformat], ignore_index=True)
        df_result = df_result.sort_values('date').reset_index(drop=True)
    
    # ========================
    # 7. Ensure non-decreasing in the entire curve, starting from A's end
    # ========================
    # Find the index of A's last point
    a_last_date = df_monthly['date'].max()
    a_last_idx = df_result[df_result['date'] == a_last_date].index[0]
    
    # Process the entire curve from A's end onwards to ensure non-decreasing
    # We'll process from A's end to the end of the result
    if a_last_idx < len(df_result) - 1:
        # Get the part from A's end to the end
        processing_start_idx = a_last_idx
        processing_end_idx = len(df_result) - 1
        
        # Create a copy of the values to work with
        values_to_process = df_result['calibrated_capacity'].values.copy()
        
        # Apply non-decreasing constraint algorithm
        i = processing_start_idx
        while i < processing_end_idx:
            if values_to_process[i+1] >= values_to_process[i]:
                # Next value is >= current, continue
                i += 1
            else:
                # Found a decreasing point, look for the local minimum
                start_decrease_idx = i
                j = i + 1
                # Continue until we find where it starts to increase again or reach the end
                while j < processing_end_idx and values_to_process[j+1] < values_to_process[j]:
                    j += 1
                
                # j is now at the local minimum (or end of array)
                local_min_idx = j
                
                # Now go backwards from local minimum to start_decrease_idx (including A's end and before)
                for k in range(local_min_idx, start_decrease_idx, -1):
                    values_to_process[k-1] = min(values_to_process[k-1], values_to_process[k])
                
                # Move to the local minimum index to continue processing
                i = local_min_idx
        
        # Update the values in the main dataframe
        df_result['calibrated_capacity'] = values_to_process
    
    # ========================
    # 8. Round calibrated capacity to 2 decimal places
    # ========================
    df_result['calibrated_capacity'] = df_result['calibrated_capacity'].round(2)
    
    # ========================
    # 9. Optional Plotting
    # ========================
    if plot:
        plt.figure(figsize=(14, 6))
        
        # Original A (monthly) - only within its range
        plt.plot(
            df_monthly['date'], df_monthly['cumulative_capacity'], 
            'k--', alpha=0.6, label='Original A (monthly interpolated)'
        )
        
        # Calibrated result
        plt.plot(
            df_result['date'], df_result['calibrated_capacity'], 
            'b-', linewidth=2, label=f'Calibrated Result ({method})'
        )
        
        # B points with labels (only if B data exists)
        if not df_B.empty:
            plt.scatter(df_B['date'], df_B['装机量(MW)'], c='red', s=60, zorder=5, label='B Calibration Points')
            for _, row in df_B.iterrows():
                plt.annotate(
                    f"{row['装机量(MW)']:.1f}", 
                    (row['date'], row['装机量(MW)']), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=9
                )
        
        plt.xlabel('Year')
        plt.ylabel('Cumulative Capacity (MW)')
        plt.title(f'Capacity Calibration for {city_name} ({tech_name}) - Method: {method}\n(A through B, including auto-supplemented zero-point and extrapolation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return df_result

# ========================
# Main Function: Process all cities for both offshore and onshore wind
# ========================

def process_all_cities_for_technology(original_csv_path: str, tech_name: str, correction_csv: str, output_file: str, method: str = 'scale_factor'):
    """
    对指定技术类型的原始数据中的所有城市进行校准，并保存到一个CSV文件中。
    
    Parameters:
    -----------
    original_csv_path : str
        原始数据CSV路径
    tech_name : str
        技术名称（如 "陆上风电" 或 "海上风电"）
    correction_csv : str
        校准点数据CSV路径
    output_file : str
        输出文件路径
    method : str
        校准方法
    """
    
    # Load original data
    try:
        df_original = pd.read_csv(original_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df_original = pd.read_csv(original_csv_path, encoding='gbk')
    
    # Get unique cities
    cities = df_original['市简称'].unique()
    
    # List to store results for all cities
    all_results = []
    
    # Process each city
    for city in cities:
        print(f"Processing {city} for {tech_name}...")
        
        try:
            # Run calibration
            df_calibrated = calibrate_city_capacity(
                original_csv=original_csv_path,
                correction_csv=correction_csv,
                tech_name=tech_name,
                city_name=city,
                plot=False,  # Set to True if you want to see plots
                encoding='utf-8',
                method=method
            )
            
            # Get city information from original data
            city_info = df_original[df_original['市简称'] == city].iloc[0]
            province_full = city_info['省全称']
            province_short = city_info['省简称']
            city_full = city_info['市全称']
            
            # Add city information to calibrated results
            df_calibrated['省全称'] = province_full
            df_calibrated['省简称'] = province_short
            df_calibrated['市全称'] = city_full
            df_calibrated['市简称'] = city
            
            # Extract year and month from date
            df_calibrated['年'] = df_calibrated['date'].dt.year
            df_calibrated['月'] = df_calibrated['date'].dt.month
            
            # Select and reorder columns
            df_calibrated = df_calibrated[['年', '月', '省全称', '省简称', '市全称', '市简称', 'calibrated_capacity']]
            df_calibrated.rename(columns={'calibrated_capacity': '累积装机(MW)'}, inplace=True)
            
            # Add tech type column for identification (optional)
            df_calibrated['能源类型'] = tech_name
            
            # Add to results list
            all_results.append(df_calibrated)
            
        except Exception as e:
            print(f"Error processing {city} for {tech_name}: {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Round the cumulative capacity to 2 decimal places
        final_df['累积装机(MW)'] = final_df['累积装机(MW)'].round(2)
        
        # Save to CSV
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Successfully saved results to {output_file}")
        return final_df
    else:
        print("No results to save.")
        return None

# ========================
# Example Usage:
# ========================

# Define paths
correction_csv = r"D:\Data\自统装机数据\助研项目 - 风电装机统计_corrected.csv"

# Process陆上风电
onshore_output = r"D:\HKU PhD\wind data paper\陆上风电月度装机量.csv"
onshore_input = r"D:\HKU PhD\wind data paper\陆上风电场装机数据.csv"

print("Processing onshore wind data...")
onshore_result = process_all_cities_for_technology(
    original_csv_path=onshore_input,
    tech_name="陆上风电",
    correction_csv=correction_csv,
    output_file=onshore_output,
    method='platform_truncation'  # You can change to 'scale_factor' if needed
)

# Process海上风电
offshore_output = r"D:\HKU PhD\wind data paper\海上风电月度装机量.csv"
offshore_input = r"D:\HKU PhD\wind data paper\海上风电场装机数据.csv"

print("Processing offshore wind data...")
offshore_result = process_all_cities_for_technology(
    original_csv_path=offshore_input,
    tech_name="海上风电",
    correction_csv=correction_csv,
    output_file=offshore_output,
    method='platform_truncation'  # You can change to 'scale_factor' if needed
)

# If you want to combine both onshore and offshore results into one file
if onshore_result is not None and offshore_result is not None:
    combined_output = r"D:\HKU PhD\wind data paper\风电市级月度装机量-2023.csv"
    combined_result = pd.concat([onshore_result, offshore_result], ignore_index=True)
    
    # Round the cumulative capacity to 2 decimal places in combined result
    combined_result['累积装机(MW)'] = combined_result['累积装机(MW)'].round(2)
    
    combined_result.to_csv(combined_output, index=False, encoding='utf-8')
    print(f"Successfully saved combined results to {combined_output}")