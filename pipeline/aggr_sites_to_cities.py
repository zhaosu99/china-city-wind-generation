import pandas as pd
import os
from typing import Literal

def process_wind_farms_by_city(
    input_csv_path: str,
    coastal_cities_csv_path: str,
    output_land_csv_path: str,
    output_sea_csv_path: str,
    lifetime_years: int = 25,
    encoding: str = 'utf-8-sig',
    start_year: int = 2010, # 作为计算起始年的下限
    end_year: int = 2023    # 固定结束年
) -> None:
    """
    读取包含风电场详细信息的 CSV 文件，根据 '陆海1-2' 列将其分为陆上和海上风电，
    然后按“省全称”和“市全称”进行年度聚合，计算新增和考虑退役的累积装机容量。
    海上风电的输出仅包含沿海城市（由单独的 CSV 文件定义）。
    最终输出两个 CSV 文件，时间范围从 min(start_year, 数据最小年份) 到 end_year。
    在最终输出前，删除所有年份 < start_year 且 累积装机(MW) == 0 的记录行。

    参数:
        input_csv_path (str): 输入的风电场数据 CSV 文件路径。
        coastal_cities_csv_path (str): 包含沿海城市信息的 CSV 文件路径 (列: 省全称, 市全称)。
        output_land_csv_path (str): 输出的陆上风电数据 CSV 文件路径。
        output_sea_csv_path (str): 输出的海上风电数据 CSV 文件路径。
        lifetime_years (int, optional): 风电机组的设计寿命（年），用于计算退役时间，默认为 25 年。
        encoding (str, optional): CSV 文件的编码，默认为 'utf-8-sig'。
        start_year (int, optional): 作为过滤下限的基准年份，默认为 2010。
        end_year (int, optional): 输出结果的固定结束年份，默认为 2023。

    返回:
        None
    """
    # --- Step 0: 读取沿海城市信息 ---
    print(f"正在读取沿海城市 CSV 文件: {coastal_cities_csv_path}")
    try:
        coastal_df = pd.read_csv(coastal_cities_csv_path, encoding=encoding)
    except Exception as e:
        raise ValueError(f"读取沿海城市 CSV 失败（{coastal_cities_csv_path}）: {e}")
    
    required_coastal_cols = ['省全称', '市全称']
    missing_coastal = [c for c in required_coastal_cols if c not in coastal_df.columns]
    if missing_coastal:
        raise ValueError(f"沿海城市 CSV 文件缺失必需列: {missing_coastal}")

    # 创建沿海城市集合 (市全称)
    COASTAL_CITIES = set(coastal_df['市全称'])

    # --- Step 1: 读取与校验 ---
    if not os.path.isfile(input_csv_path):
        raise FileNotFoundError(f"输入文件不存在: {input_csv_path}")
    
    try:
        df = pd.read_csv(input_csv_path, encoding=encoding)
    except Exception as e:
        raise ValueError(f"读取风电场 CSV 失败（{input_csv_path}）: {e}")
    
    required_cols = ['年份', '省全称', '市全称', '风电场装机MW', '陆海1-2', '省简称', '市简称']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"风电场 CSV 文件缺失必需列: {missing}")

    # --- Step 1.1: 确定全局数据年份范围 & 类型转换 ---
    # 将年份转换为数值以进行比较
    df['年份'] = pd.to_numeric(df['年份'], errors='raise').astype(int)
    df['风电场装机MW'] = pd.to_numeric(df['风电场装机MW'], errors='coerce') # 在这里进行装机容量的转换
    if df['风电场装机MW'].isna().any():
        raise ValueError("列 '风电场装机MW' 含无法转换为数值的数据")

    global_min_year = df['年份'].min()
    global_max_year = df['年份'].max()
    
    # 计算实际的输出年份范围
    actual_start_year = min(start_year, global_min_year)
    actual_end_year = end_year # 固定为 2023
    
    print(f"数据中最早年份: {global_min_year}, 最晚年份: {global_max_year}")
    print(f"计算输出年份范围: {actual_start_year} - {actual_end_year}")

    # --- Step 2: 分离陆上和海上风电数据 ---
    land_df = df[df['陆海1-2'] == 1].copy()
    sea_df = df[df['陆海1-2'] == 2].copy()

    # --- Step 3: 清洗与过滤 ---
    for name, subset, output_path in [('陆上风电', land_df, output_land_csv_path), ('海上风电', sea_df, output_sea_csv_path)]:
        if subset.empty:
            print(f"警告: {name} 数据为空，将跳过处理。")
            # 如果为空，创建一个空的 DataFrame 并保存
            # 创建一个包含固定年份范围的空 DataFrame
            empty_index = pd.MultiIndex.from_product(
                [list(range(actual_start_year, actual_end_year + 1)), [''], ['']], 
                names=['年份', '省全称', '市全称']
            )
            empty_df = pd.DataFrame(index=empty_index).reset_index()
            empty_df['省简称'] = ''
            empty_df['市简称'] = ''
            empty_df['新增装机(MW)'] = 0.0
            empty_df['累积装机(MW)'] = 0.0
            # 过滤掉空的省/市行（因为 MultiIndex 会包含空值）
            empty_df = empty_df[(empty_df['省全称'] != '') & (empty_df['市全称'] != '')]
            empty_df.to_csv(output_path, index=False, encoding=encoding)
            print(f"✅ 已生成空的 {name} 输出文件: {output_path}")
            continue # 继续处理下一个

        # --- Step 3.1: 过滤海上风电数据，仅保留沿海城市 ---
        if name == '海上风电':
            print(f"  正在过滤 {name} 数据，仅保留沿海城市...")
            original_count = len(subset)
            # 使用之前读取的 COASTAL_CITIES 集合进行过滤
            subset = subset[subset['市全称'].isin(COASTAL_CITIES)]
            filtered_count = len(subset)
            print(f"  过滤前记录数: {original_count}, 过滤后记录数: {filtered_count}")

        # --- Step 4: 聚合年度新增装机 & 计算考虑退役的累积装机 ---
        def aggregate_and_compute_cumulative_with_retirement(df_input, output_path_internal, wind_type, start_y, end_y, lifetime):
            if df_input.empty:
                print(f"  {wind_type} 数据为空，跳过处理。")
                # 创建一个包含固定年份范围的空 DataFrame
                empty_index = pd.MultiIndex.from_product(
                    [list(range(start_y, end_y + 1)), [''], ['']], 
                    names=['年份', '省全称', '市全称']
                )
                empty_df = pd.DataFrame(index=empty_index).reset_index()
                empty_df['省简称'] = ''
                empty_df['市简称'] = ''
                empty_df['新增装机(MW)'] = 0.0
                empty_df['累积装机(MW)'] = 0.0
                empty_df = empty_df[(empty_df['省全称'] != '') & (empty_df['市全称'] != '')]
                empty_df.to_csv(output_path_internal, index=False, encoding=encoding)
                print(f"  ✅ 已生成空的 {wind_type} 输出文件: {output_path_internal}")
                return

            print(f"  正在处理 {wind_type} 数据...")
            
            # 按省、市、年份聚合新增装机
            # 保留省简称、市简称信息
            additions = (
                df_input.groupby(['省全称', '市全称', '年份'], as_index=False)[['风电场装机MW', '省简称', '市简称']]
                .agg({
                    '风电场装机MW': 'sum',
                    '省简称': 'first',
                    '市简称': 'first'
                })
                .rename(columns={'风电场装机MW': '新增装机(MW)'})
                .sort_values(['省全称', '市全称', '年份'])
                .reset_index(drop=True)
            )

            # 定义计算累积装机的函数 (考虑退役)
            # 该函数现在接收分组键和必要的数据作为参数
            def compute_cumulative_with_retirement(group, prov_full, city_full, prov_abbr, city_abbr, start_year_inner, end_year_inner, lifetime_inner, all_additions):
                # 创建指定年份范围的 DataFrame
                years_range = pd.DataFrame({'年份': list(range(start_year_inner, end_year_inner + 1))})
                
                # 将原始的年度新增数据合并到年份范围上，缺失值填充为 0
                group_with_all_years = pd.merge(
                    years_range, 
                    group[['年份', '新增装机(MW)']], 
                    on='年份', 
                    how='left'
                ).fillna({'新增装机(MW)': 0.0})

                # 计算每年的累积装机 (考虑退役)
                cumulative_values = []
                for current_year in group_with_all_years['年份']:
                    # 找到在当前年份仍在运行的项目
                    # 项目开始年份 >= (当前年份 - lifetime + 1) 且 项目结束年份 (开始年份 + lifetime - 1) >= 当前年份
                    # 简化为：项目开始年份 >= (current_year - lifetime_inner + 1) 且 项目开始年份 <= current_year
                    # 从 all_additions (完整的聚合数据) 中筛选当前省、市、且在运行年限内的项目
                    running_mask = (
                        (all_additions['年份'] >= (current_year - lifetime_inner + 1)) &
                        (all_additions['年份'] <= current_year) &
                        (all_additions['省全称'] == prov_full) &
                        (all_additions['市全称'] == city_full)
                    )
                    # 计算仍在运行的项目总装机
                    running_capacity = all_additions.loc[running_mask, '新增装机(MW)'].sum()
                    cumulative_values.append(running_capacity)

                group_with_all_years['累积装机(MW)'] = cumulative_values
                group_with_all_years['省全称'] = prov_full
                group_with_all_years['省简称'] = prov_abbr
                group_with_all_years['市全称'] = city_full
                group_with_all_years['市简称'] = city_abbr
                
                return group_with_all_years[['年份', '省全称', '省简称', '市全称', '市简称', '新增装机(MW)', '累积装机(MW)']]

            # 使用 lambda 函数将分组键和必要的数据传递给内部函数
            # groupby 时 include_groups=False，这样 group 中就不包含分组列
            result_df = (
                additions
                .groupby(['省全称', '市全称'], group_keys=False)
                .apply(
                    lambda x: compute_cumulative_with_retirement(
                        x, # 当前分组的 DataFrame (不包含分组列)
                        x.name[0], # 分组键 0: 省全称
                        x.name[1], # 分组键 1: 市全称
                        x['省简称'].iloc[0], # 从当前分组获取省简称
                        x['市简称'].iloc[0], # 从当前分组获取市简称
                        start_y, end_y, lifetime, additions # 传递其他必要参数
                    ),
                    include_groups=False
                )
                .reset_index(drop=True)
            )

            # --- Step 5: 最终校验、过滤与输出 ---
            if not {'新增装机(MW)', '累积装机(MW)'}.issubset(result_df.columns):
                missing_cols = {'新增装机(MW)', '累积装机(MW)'} - set(result_df.columns)
                raise RuntimeError(f"内部错误：缺失列 {missing_cols}")
            
            print(f"  正在过滤 {wind_type} 数据：移除 {start_year} 年前累积为 0 的记录行...")
            
            # 应用最终过滤条件：年份 < start_year 且 累积装机(MW) == 0
            initial_rows = len(result_df)
            result_df_filtered = result_df[
                ~((result_df['年份'] < start_year) & (result_df['累积装机(MW)'] == 0.0))
            ].copy()
            
            final_rows = len(result_df_filtered)
            print(f"    过滤前行数: {initial_rows}, 过滤后行数: {final_rows}, 删除了 {initial_rows - final_rows} 行。")

            result_df_filtered = result_df_filtered.sort_values(['年份', '省全称', '市全称']).reset_index(drop=True)
            result_df_filtered['新增装机(MW)'] = result_df_filtered['新增装机(MW)'].round(1)
            result_df_filtered['累积装机(MW)'] = result_df_filtered['累积装机(MW)'].round(1)
            
            try:
                # 修复：使用 output_path_internal 而不是未定义的 output_csv_path
                result_df_filtered.to_csv(output_path_internal, index=False, encoding=encoding)
                print(f"  ✅ {wind_type} 数据处理完成，已生成: {output_path_internal}")
                print(f"     - 行数: {len(result_df_filtered)}")
                if not result_df_filtered.empty:
                    print(f"     - 时间范围: {result_df_filtered['年份'].min()} ~ {result_df_filtered['年份'].max()}")
                    print(f"     - 城市数: {result_df_filtered[['省全称','市全称']].drop_duplicates().shape[0]}")
            except Exception as e:
                raise IOError(f"写入 CSV 失败（{output_path_internal}）: {e}")

        # 调用内部函数处理当前类型的数据
        # 使用计算得出的实际年份范围
        aggregate_and_compute_cumulative_with_retirement(subset, output_path, name, actual_start_year, actual_end_year, lifetime_years)

# --- 使用示例 ---
input_path = r"D:\HKU PhD\wind data paper\风电场信息-2023.csv" # 替换为您的输入文件路径
coastal_cities_path = r"D:\Data\admin\沿海市.csv" # 替换为您沿海城市 CSV 的路径
land_output_path = r"d:\HKU PhD\wind data paper\陆上风电场装机数据.csv"
sea_output_path = r"d:\HKU PhD\wind data paper\海上风电场装机数据.csv"
process_wind_farms_by_city(
    input_path,
    coastal_cities_path,
    land_output_path,
    sea_output_path,
    lifetime_years=25,
    start_year=2000, # 作为过滤基准年
    end_year=2023    # 固定上限
)