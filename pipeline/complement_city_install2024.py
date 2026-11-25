import pandas as pd
import numpy as np

def distribute_unknown_capacity_with_transparency(
    csv_province_data: str,
    csv_city_data: str,
    csv_onshore_weights: str,
    csv_offshore_weights: str,
    output_file: str = None,
    year: int = 2024,
    log_detailed: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    分配未知城市的装机容量（基于已处理的站点权重数据）- 增强透明度版本
    
    Parameters:
    -----------
    csv_province_data : str
        省级数据CSV文件路径（包含类型、省份、年份、装机容量）
    csv_city_data : str
        市级数据CSV文件路径（包含年、月、省全称、省简称、市全称、市简称、累积装机、能源类型）
    csv_onshore_weights : str
        2024年GEM陆上风电省市装机量统计文件
    csv_offshore_weights : str
        2024年GEM海上风电省市装机量统计文件
    output_file : str
        输出文件路径
    year : int
        年份，默认为2024
    log_detailed : bool
        是否输出详细日志

    Returns:
    --------
    tuple[pd.DataFrame, dict]
        - 更新后的市级数据
        - 详细的分配过程信息
    """
    
    print(f"Processing data for year {year}...")
    
    # 用于记录分配过程的字典
    distribution_log = {
        'provincial_summary': {},
        'city_allocation_details': {},
        'unknown_distribution': {},
        'interpolation_details': {}
    }
    
    # ========================
    # 1. Load and process provincial data (CSV乙)
    # ========================
    try:
        df_province = pd.read_csv(csv_province_data, encoding='utf-8')
    except UnicodeDecodeError:
        df_province = pd.read_csv(csv_province_data, encoding='gbk')
    
    print(f"Loaded {len(df_province)} provincial records")
    
    # 计算2024年新增装机量
    def calculate_annual_increase(df, target_year=2024):
        """计算指定年份的新增装机量"""
        current_year_data = df[df['年份'] == target_year].copy()
        previous_year_data = df[df['年份'] == target_year - 1].copy()
        
        # 合并数据以计算增量
        merged = current_year_data.merge(previous_year_data, on=['类型', '省份'], how='left', suffixes=('', '_prev'))
        merged['新增装机(MW)'] = merged['装机容量(MW)'] - merged.get('装机容量(MW)_prev', 0)
        merged['新增装机(MW)'] = merged['新增装机(MW)'].clip(lower=0)  # 确保非负
        
        return merged[['类型', '省份', '新增装机(MW)']]
    
    # 分别处理陆上和海上风电
    onshore_province = df_province[df_province['类型'] == '陆上风电'].copy()
    offshore_province = df_province[df_province['类型'] == '海上风电'].copy()
    
    onshore_increase = calculate_annual_increase(onshore_province, year)
    offshore_increase = calculate_annual_increase(offshore_province, year)
    
    print(f"Onshore increase: {len(onshore_increase)} provinces")
    print(f"Offshore increase: {len(offshore_increase)} provinces")
    
    # 记录省级数据摘要
    distribution_log['provincial_summary'] = {
        'onshore': onshore_increase.to_dict('records'),
        'offshore': offshore_increase.to_dict('records')
    }
    
    # ========================
    # 2. Load and process city data (CSV丙)
    # ========================
    try:
        df_city = pd.read_csv(csv_city_data, encoding='utf-8')
    except UnicodeDecodeError:
        df_city = pd.read_csv(csv_city_data, encoding='gbk')
    
    print(f"Loaded {len(df_city)} city records")
    
    # 获取2023年12月的数据
    df_city_dec_2023 = df_city[(df_city['年'] == year - 1) & (df_city['月'] == 12)].copy()
    
    # 获取2024年12月的数据
    df_city_dec_2024 = df_city[(df_city['年'] == year) & (df_city['月'] == 12)].copy()
    
    print(f"City data for Dec {year-1}: {len(df_city_dec_2023)} records")
    print(f"City data for Dec {year}: {len(df_city_dec_2024)} records")
    
    # 记录原始已知城市数据
    original_known_cities = {
        'onshore': df_city_dec_2024[df_city_dec_2024['能源类型'] == '陆上风电'][['省简称', '市简称', '累积装机(MW)']].to_dict('records'),
        'offshore': df_city_dec_2024[df_city_dec_2024['能源类型'] == '海上风电'][['省简称', '市简称', '累积装机(MW)']].to_dict('records')
    }
    
    # ========================
    # 3. Load the pre-processed station weights
    # ========================
    try:
        onshore_weights = pd.read_csv(csv_onshore_weights, encoding='utf-8')
    except UnicodeDecodeError:
        onshore_weights = pd.read_csv(csv_onshore_weights, encoding='gbk')
    
    try:
        offshore_weights = pd.read_csv(csv_offshore_weights, encoding='utf-8')
    except UnicodeDecodeError:
        offshore_weights = pd.read_csv(csv_offshore_weights, encoding='gbk')
    
    print(f"Loaded onshore weights: {len(onshore_weights)} records")
    print(f"Loaded offshore weights: {len(offshore_weights)} records")
    
    # ========================
    # 4. Process each province and distribute unknown capacity
    # ========================
    result_df = df_city.copy()
    
    # 为结果数据添加来源标识列
    result_df['数据来源'] = '原始已知'  # 默认为原始数据
    
    # 处理陆上风电
    for _, prov_row in onshore_increase.iterrows():
        province_name = prov_row['省份']
        total_increase = prov_row['新增装机(MW)']
        
        print(f"\n{'='*50}")
        print(f"Processing province: {province_name} (Onshore)")
        print(f"Provincial total increase: {total_increase:.2f} MW")
        
        # 获取该省在2024年12月的已知城市数据
        known_cities_2024 = df_city_dec_2024[
            (df_city_dec_2024['省简称'] == province_name) & 
            (df_city_dec_2024['能源类型'] == '陆上风电')
        ].copy()
        
        # 获取2023年12月的数据
        known_cities_2023 = df_city_dec_2023[
            (df_city_dec_2023['省简称'] == province_name) & 
            (df_city_dec_2023['能源类型'] == '陆上风电')
        ].copy()
        
        # 计算已知城市的新增量
        known_increase_sum = 0.0
        known_city_details = []
        
        for _, city_row in known_cities_2024.iterrows():
            city_name = city_row['市简称']
            city_capacity_2024 = city_row['累积装机(MW)']
            
            # 查找2023年12月的数据
            city_2023 = known_cities_2023[known_cities_2023['市简称'] == city_name]
            if len(city_2023) > 0:
                city_capacity_2023 = city_2023.iloc[0]['累积装机(MW)']
                city_increase = max(0, city_capacity_2024 - city_capacity_2023)
            else:
                # 如果没有2023年数据，假设新增量等于2024年容量
                city_increase = city_capacity_2024
            
            known_increase_sum += city_increase
            known_city_details.append({
                '市简称': city_name,
                '2023年容量': city_capacity_2023 if len(city_2023) > 0 else 0,
                '2024年容量': city_capacity_2024,
                '新增量': city_increase
            })
        
        # 计算未确定部分
        unknown_increase = max(0, total_increase - known_increase_sum)
        print(f"Known cities total increase: {known_increase_sum:.2f} MW")
        print(f"Unknown cities increase to distribute: {unknown_increase:.2f} MW")
        
        if log_detailed:
            print("\nKnown city details:")
            for detail in known_city_details:
                print(f"  {detail['市简称']}: {detail['2023年容量']:.2f} -> {detail['2024年容量']:.2f} MW (Increase: {detail['新增量']:.2f} MW)")
        
        if unknown_increase > 0:
            # 使用站点数据作为权重进行分配
            province_station_weights = onshore_weights[
                onshore_weights['省简称'] == province_name
            ].copy()
            
            if len(province_station_weights) > 0:
                # 计算总权重
                total_weight = province_station_weights[f'{year}年装机量(MW)'].sum()
                
                if total_weight > 0:
                    print(f"Total weight from GEM stations: {total_weight:.2f} MW")
                    
                    # 记录分配详情
                    allocation_details = {
                        'province': province_name,
                        'total_increase': total_increase,
                        'known_increase': known_increase_sum,
                        'unknown_to_distribute': unknown_increase,
                        'total_weight': total_weight,
                        'allocated_cities': []
                    }
                    
                    # 分配未确定部分
                    for _, weight_row in province_station_weights.iterrows():
                        city_name = weight_row['市简称']
                        weight = weight_row[f'{year}年装机量(MW)']
                        
                        if weight > 0:
                            allocated_increase = unknown_increase * (weight / total_weight)
                            allocation_ratio = weight / total_weight
                            
                            print(f"  Allocating {allocated_increase:.2f} MW to {city_name} (weight: {weight:.2f} MW, ratio: {allocation_ratio:.4f})")
                            
                            # 记录分配详情
                            allocation_details['allocated_cities'].append({
                                '市简称': city_name,
                                'weight': weight,
                                'allocation_ratio': allocation_ratio,
                                'allocated_increase': allocated_increase
                            })
                            
                            # 找到该城市在2024年12月的记录（如果存在）
                            city_record = df_city_dec_2024[
                                (df_city_dec_2024['省简称'] == province_name) & 
                                (df_city_dec_2024['市简称'] == city_name) & 
                                (df_city_dec_2024['能源类型'] == '陆上风电')
                            ]
                            
                            if len(city_record) == 0:
                                # 如果不存在，创建新记录
                                # 获取2023年12月的数据
                                city_2023 = df_city_dec_2023[
                                    (df_city_dec_2023['省简称'] == province_name) & 
                                    (df_city_dec_2023['市简称'] == city_name) & 
                                    (df_city_dec_2023['能源类型'] == '陆上风电')
                                ]
                                
                                if len(city_2023) > 0:
                                    city_capacity_2023 = city_2023.iloc[0]['累积装机(MW)']
                                    new_capacity = city_capacity_2023 + allocated_increase
                                else:
                                    new_capacity = allocated_increase
                                
                                # 获取该城市的省全称和市全称
                                if len(city_2023) > 0:
                                    prov_full = city_2023.iloc[0]['省全称']
                                    city_full = city_2023.iloc[0]['市全称']
                                else:
                                    # 如果2023年没有数据，尝试从原始数据中获取
                                    original_city = df_city[
                                        (df_city['省简称'] == province_name) & 
                                        (df_city['市简称'] == city_name)
                                    ]
                                    if len(original_city) > 0:
                                        prov_full = original_city.iloc[0]['省全称']
                                        city_full = original_city.iloc[0]['市全称']
                                    else:
                                        prov_full = province_name
                                        city_full = city_name
                                
                                # 创建新记录
                                new_record = {
                                    '年': year,
                                    '月': 12,
                                    '省全称': prov_full,
                                    '省简称': province_name,
                                    '市全称': city_full,
                                    '市简称': city_name,
                                    '累积装机(MW)': new_capacity,
                                    '能源类型': '陆上风电',
                                    '数据来源': '分配补全'
                                }
                                
                                # 添加到结果中
                                result_df = pd.concat([result_df, pd.DataFrame([new_record])], ignore_index=True)
                                print(f"    Created new record for {city_name}: {new_capacity:.2f} MW (from allocation)")
                            else:
                                # 更新现有记录
                                idx = city_record.index[0]
                                # 获取2023年12月的数据
                                city_2023 = df_city_dec_2023[
                                    (df_city_dec_2023['省简称'] == province_name) & 
                                    (df_city_dec_2023['市简称'] == city_name) & 
                                    (df_city_dec_2023['能源类型'] == '陆上风电')
                                ]
                                
                                if len(city_2023) > 0:
                                    city_capacity_2023 = city_2023.iloc[0]['累积装机(MW)']
                                    new_capacity = city_capacity_2023 + allocated_increase
                                else:
                                    new_capacity = allocated_increase
                                
                                result_df.at[idx, '累积装机(MW)'] = new_capacity
                                result_df.at[idx, '数据来源'] = '分配补全'
                                print(f"    Updated existing record for {city_name}: {new_capacity:.2f} MW (added {allocated_increase:.2f} MW from allocation)")
                    
                    # 记录到分配日志
                    distribution_log['city_allocation_details'].setdefault('onshore', {})[province_name] = allocation_details
                    
                else:
                    print("No station data for weighting, skipping allocation")
            else:
                print("No station data for this province, skipping allocation")
        else:
            print("No unknown capacity to distribute for this province")
    
    # 处理海上风电（类似逻辑）
    for _, prov_row in offshore_increase.iterrows():
        province_name = prov_row['省份']
        total_increase = prov_row['新增装机(MW)']
        
        print(f"\n{'='*50}")
        print(f"Processing province: {province_name} (Offshore)")
        print(f"Provincial total increase: {total_increase:.2f} MW")
        
        # 获取该省在2024年12月的已知城市数据
        known_cities_2024 = df_city_dec_2024[
            (df_city_dec_2024['省简称'] == province_name) & 
            (df_city_dec_2024['能源类型'] == '海上风电')
        ].copy()
        
        # 获取2023年12月的数据
        known_cities_2023 = df_city_dec_2023[
            (df_city_dec_2023['省简称'] == province_name) & 
            (df_city_dec_2023['能源类型'] == '海上风电')
        ].copy()
        
        # 计算已知城市的新增量
        known_increase_sum = 0.0
        known_city_details = []
        
        for _, city_row in known_cities_2024.iterrows():
            city_name = city_row['市简称']
            city_capacity_2024 = city_row['累积装机(MW)']
            
            # 查找2023年12月的数据
            city_2023 = known_cities_2023[known_cities_2023['市简称'] == city_name]
            if len(city_2023) > 0:
                city_capacity_2023 = city_2023.iloc[0]['累积装机(MW)']
                city_increase = max(0, city_capacity_2024 - city_capacity_2023)
            else:
                # 如果没有2023年数据，假设新增量等于2024年容量
                city_increase = city_capacity_2024
            
            known_increase_sum += city_increase
            known_city_details.append({
                '市简称': city_name,
                '2023年容量': city_capacity_2023 if len(city_2023) > 0 else 0,
                '2024年容量': city_capacity_2024,
                '新增量': city_increase
            })
        
        # 计算未确定部分
        unknown_increase = max(0, total_increase - known_increase_sum)
        print(f"Known cities total increase: {known_increase_sum:.2f} MW")
        print(f"Unknown cities increase to distribute: {unknown_increase:.2f} MW")
        
        if log_detailed:
            print("\nKnown city details:")
            for detail in known_city_details:
                print(f"  {detail['市简称']}: {detail['2023年容量']:.2f} -> {detail['2024年容量']:.2f} MW (Increase: {detail['新增量']:.2f} MW)")
        
        if unknown_increase > 0:
            # 使用站点数据作为权重进行分配
            province_station_weights = offshore_weights[
                offshore_weights['省简称'] == province_name
            ].copy()
            
            if len(province_station_weights) > 0:
                # 计算总权重
                total_weight = province_station_weights[f'{year}年装机量(MW)'].sum()
                
                if total_weight > 0:
                    print(f"Total weight from GEM stations: {total_weight:.2f} MW")
                    
                    # 记录分配详情
                    allocation_details = {
                        'province': province_name,
                        'total_increase': total_increase,
                        'known_increase': known_increase_sum,
                        'unknown_to_distribute': unknown_increase,
                        'total_weight': total_weight,
                        'allocated_cities': []
                    }
                    
                    # 分配未确定部分
                    for _, weight_row in province_station_weights.iterrows():
                        city_name = weight_row['市简称']
                        weight = weight_row[f'{year}年装机量(MW)']
                        
                        if weight > 0:
                            allocated_increase = unknown_increase * (weight / total_weight)
                            allocation_ratio = weight / total_weight
                            
                            print(f"  Allocating {allocated_increase:.2f} MW to {city_name} (weight: {weight:.2f} MW, ratio: {allocation_ratio:.4f})")
                            
                            # 记录分配详情
                            allocation_details['allocated_cities'].append({
                                '市简称': city_name,
                                'weight': weight,
                                'allocation_ratio': allocation_ratio,
                                'allocated_increase': allocated_increase
                            })
                            
                            # 找到该城市在2024年12月的记录（如果存在）
                            city_record = df_city_dec_2024[
                                (df_city_dec_2024['省简称'] == province_name) & 
                                (df_city_dec_2024['市简称'] == city_name) & 
                                (df_city_dec_2024['能源类型'] == '海上风电')
                            ]
                            
                            if len(city_record) == 0:
                                # 如果不存在，创建新记录
                                # 获取2023年12月的数据
                                city_2023 = df_city_dec_2023[
                                    (df_city_dec_2023['省简称'] == province_name) & 
                                    (df_city_dec_2023['市简称'] == city_name) & 
                                    (df_city_dec_2023['能源类型'] == '海上风电')
                                ]
                                
                                if len(city_2023) > 0:
                                    city_capacity_2023 = city_2023.iloc[0]['累积装机(MW)']
                                    new_capacity = city_capacity_2023 + allocated_increase
                                else:
                                    new_capacity = allocated_increase
                                
                                # 获取该城市的省全称和市全称
                                if len(city_2023) > 0:
                                    prov_full = city_2023.iloc[0]['省全称']
                                    city_full = city_2023.iloc[0]['市全称']
                                else:
                                    # 如果2023年没有数据，尝试从原始数据中获取
                                    original_city = df_city[
                                        (df_city['省简称'] == province_name) & 
                                        (df_city['市简称'] == city_name)
                                    ]
                                    if len(original_city) > 0:
                                        prov_full = original_city.iloc[0]['省全称']
                                        city_full = original_city.iloc[0]['市全称']
                                    else:
                                        prov_full = province_name
                                        city_full = city_name
                                
                                # 创建新记录
                                new_record = {
                                    '年': year,
                                    '月': 12,
                                    '省全称': prov_full,
                                    '省简称': province_name,
                                    '市全称': city_full,
                                    '市简称': city_name,
                                    '累积装机(MW)': new_capacity,
                                    '能源类型': '海上风电',
                                    '数据来源': '分配补全'
                                }
                                
                                # 添加到结果中
                                result_df = pd.concat([result_df, pd.DataFrame([new_record])], ignore_index=True)
                                print(f"    Created new record for {city_name}: {new_capacity:.2f} MW (from allocation)")
                            else:
                                # 更新现有记录
                                idx = city_record.index[0]
                                # 获取2023年12月的数据
                                city_2023 = df_city_dec_2023[
                                    (df_city_dec_2023['省简称'] == province_name) & 
                                    (df_city_dec_2023['市简称'] == city_name) & 
                                    (df_city_dec_2023['能源类型'] == '海上风电')
                                ]
                                
                                if len(city_2023) > 0:
                                    city_capacity_2023 = city_2023.iloc[0]['累积装机(MW)']
                                    new_capacity = city_capacity_2023 + allocated_increase
                                else:
                                    new_capacity = allocated_increase
                                
                                result_df.at[idx, '累积装机(MW)'] = new_capacity
                                result_df.at[idx, '数据来源'] = '分配补全'
                                print(f"    Updated existing record for {city_name}: {new_capacity:.2f} MW (added {allocated_increase:.2f} MW from allocation)")
                    
                    # 记录到分配日志
                    distribution_log['city_allocation_details'].setdefault('offshore', {})[province_name] = allocation_details
                    
                else:
                    print("No station data for weighting, skipping allocation")
            else:
                print("No station data for this province, skipping allocation")
        else:
            print("No unknown capacity to distribute for this province")
    
    # ========================
    # 5. Linear interpolation for months before December 2024
    # ========================
    print(f"\n{'='*50}")
    print("Filling monthly data for 2024...")
    
    # 记录插值详情
    interpolation_log = []
    
    # 对于每个省市能源类型组合
    unique_combinations = result_df[
        (result_df['年'] == year) & 
        (result_df['月'] == 12)
    ][['省简称', '市简称', '能源类型']].drop_duplicates().values
    
    for prov_short, city_short, energy_type in unique_combinations:
        # 获取该组合的所有记录
        city_records = result_df[
            (result_df['省简称'] == prov_short) & 
            (result_df['市简称'] == city_short) & 
            (result_df['能源类型'] == energy_type)
        ].copy()
        
        # 获取2024年12月的数据
        dec_2024_record = city_records[
            (city_records['年'] == year) & 
            (city_records['月'] == 12)
        ]
        
        if len(dec_2024_record) > 0:
            dec_2024_capacity = dec_2024_record.iloc[0]['累积装机(MW)']
            
            # 获取最近的非12月数据
            before_dec_records = city_records[
                (city_records['年'] == year) & 
                (city_records['月'] < 12)
            ].sort_values('月', ascending=False)
            
            if len(before_dec_records) > 0:
                last_month = before_dec_records.iloc[0]['月']
                last_capacity = before_dec_records.iloc[0]['累积装机(MW)']
                
                # 记录插值信息
                interpolation_info = {
                    'province': prov_short,
                    'city': city_short,
                    'energy_type': energy_type,
                    'start_month': last_month,
                    'end_month': 12,
                    'start_capacity': last_capacity,
                    'end_capacity': dec_2024_capacity,
                    'interpolated_months': []
                }
                
                # 线性插值填充中间月份
                for month in range(last_month + 1, 12):
                    # 线性插值
                    interpolated_capacity = last_capacity + (dec_2024_capacity - last_capacity) * (month - last_month) / (12 - last_month)
                    interpolated_capacity = max(0, interpolated_capacity)  # 确保非负
                    
                    # 创建新记录
                    new_record = {
                        '年': year,
                        '月': month,
                        '省全称': dec_2024_record.iloc[0]['省全称'],
                        '省简称': prov_short,
                        '市全称': dec_2024_record.iloc[0]['市全称'],
                        '市简称': city_short,
                        '累积装机(MW)': interpolated_capacity,
                        '能源类型': energy_type,
                        '数据来源': '插值补全'
                    }
                    
                    # 添加到结果中
                    result_df = pd.concat([result_df, pd.DataFrame([new_record])], ignore_index=True)
                    
                    interpolation_info['interpolated_months'].append({
                        'month': month,
                        'capacity': interpolated_capacity
                    })
                
                interpolation_log.append(interpolation_info)
            else:
                # 如果没有其他月份数据，从年初开始线性填充
                # 获取2023年12月的数据
                prev_year_record = result_df[
                    (result_df['省简称'] == prov_short) & 
                    (result_df['市简称'] == city_short) & 
                    (result_df['能源类型'] == energy_type) & 
                    (result_df['年'] == year - 1) & 
                    (result_df['月'] == 12)
                ]
                
                if len(prev_year_record) > 0:
                    prev_capacity = prev_year_record.iloc[0]['累积装机(MW)']
                    
                    # 记录插值信息
                    interpolation_info = {
                        'province': prov_short,
                        'city': city_short,
                        'energy_type': energy_type,
                        'start_month': 1,
                        'end_month': 12,
                        'start_capacity': prev_capacity,
                        'end_capacity': dec_2024_capacity,
                        'interpolated_months': []
                    }
                    
                    # 从2024年1月到11月线性填充
                    for month in range(1, 12):
                        interpolated_capacity = prev_capacity + (dec_2024_capacity - prev_capacity) * month / 12
                        interpolated_capacity = max(0, interpolated_capacity)  # 确保非负
                        
                        # 创建新记录
                        new_record = {
                            '年': year,
                            '月': month,
                            '省全称': dec_2024_record.iloc[0]['省全称'],
                            '省简称': prov_short,
                            '市全称': dec_2024_record.iloc[0]['市全称'],
                            '市简称': city_short,
                            '累积装机(MW)': interpolated_capacity,
                            '能源类型': energy_type,
                            '数据来源': '插值补全'
                        }
                        
                        # 添加到结果中
                        result_df = pd.concat([result_df, pd.DataFrame([new_record])], ignore_index=True)
                        
                        interpolation_info['interpolated_months'].append({
                            'month': month,
                            'capacity': interpolated_capacity
                        })
                    
                    interpolation_log.append(interpolation_info)
    
    # 记录插值详情
    distribution_log['interpolation_details'] = interpolation_log
    
    # ========================
    # 6. Round to 2 decimal places and clean up
    # ========================
    result_df['累积装机(MW)'] = result_df['累积装机(MW)'].round(2)
    
    # 去除重复记录（保留最新值）
    result_df = result_df.drop_duplicates(subset=['年', '月', '省简称', '市简称', '能源类型'], keep='last')
    
    # 排序
    result_df = result_df.sort_values(['省简称', '市简称', '能源类型', '年', '月']).reset_index(drop=True)
    
    print(f"Final result contains {len(result_df)} records")
    
    # 统计数据来源
    source_counts = result_df['数据来源'].value_counts()
    print(f"Data source distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} records")
    
    # 保存结果
    if output_file:
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Results saved to {output_file}")
    
    return result_df, distribution_log

def print_distribution_summary(distribution_log: dict):
    """
    打印分配过程的摘要信息
    """
    print(f"\n{'='*60}")
    print("DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    
    # 省级摘要
    print("\n1. Provincial Summary:")
    for energy_type, prov_data in distribution_log['provincial_summary'].items():
        print(f"\n  {energy_type.upper()}:")
        for record in prov_data:
            print(f"    {record['省份']}: {record['新增装机(MW)']:.2f} MW")
    
    # 城市分配详情
    print("\n2. City Allocation Details:")
    for energy_type, provinces in distribution_log['city_allocation_details'].items():
        print(f"\n  {energy_type.upper()}:")
        for province, details in provinces.items():
            print(f"    {province}:")
            print(f"      Provincial total increase: {details['total_increase']:.2f} MW")
            print(f"      Known cities increase: {details['known_increase']:.2f} MW")
            print(f"      Unknown to distribute: {details['unknown_to_distribute']:.2f} MW")
            print(f"      Total weight from GEM: {details['total_weight']:.2f} MW")
            print(f"      Allocated cities:")
            for city_alloc in details['allocated_cities']:
                print(f"        {city_alloc['市简称']}: {city_alloc['allocated_increase']:.2f} MW "
                      f"(weight: {city_alloc['weight']:.2f}, ratio: {city_alloc['allocation_ratio']:.4f})")
    
    # 插值详情
    print(f"\n3. Interpolation Details:")
    print(f"    Total interpolated records: {len(distribution_log['interpolation_details'])}")

# 示例使用
def main():
    # 配置文件路径
    csv_province_data = r"D:\Data\自统装机数据\省级装机量统计.csv"      # CSV乙
    csv_city_data = r"D:\HKU PhD\wind data paper\风电市级月度装机量-2023.csv"         # CSV丙
    csv_onshore_weights = r"D:\HKU PhD\wind data paper\GEM陆上风电省市装机量统计 - 2024.csv"  # GEM陆上风电权重
    csv_offshore_weights = r"D:\HKU PhD\wind data paper\GEM海上风电省市装机量统计 - 2024.csv" # GEM海上风电权重
    output_file = r"D:\HKU PhD\wind data paper\风电市级月度装机量-2024.csv"
    
    # 处理数据
    result, distribution_log = distribute_unknown_capacity_with_transparency(
        csv_province_data,
        csv_city_data,
        csv_onshore_weights,
        csv_offshore_weights,
        output_file,
        year=2024,
        log_detailed=True
    )
    
    # 打印摘要
    print_distribution_summary(distribution_log)
    
    print("Processing completed!")
    return result, distribution_log

if __name__ == "__main__":
    main()