import pandas as pd
from fuzzywuzzy import fuzz, process
import os

def merge_wind_turbine_data(csv1_path, csv2_path, csv3_path, csv4_path, output_csv_path):
    """
    合并风电场数据与风机技术参数。

    Args:
        csv1_path (str): 风电场基础数据 CSV 文件路径 (df1)
        csv2_path (str): 风机型号详细参数 CSV 文件路径 (df2)
        csv3_path (str): 风机平均参数 CSV 文件路径 (df3)
        csv4_path (str): 风机平均轮毂高度 CSV 文件路径 (df4)
        output_csv_path (str): 输出合并后数据的 CSV 文件路径。
    """

    # --- 0. 检查输出文件是否存在，决定从哪里加载 df1 ---
    if os.path.exists(output_csv_path):
        print(f"检测到输出文件 {output_csv_path} 已存在，将从该文件继续处理。")
        df1 = pd.read_csv(output_csv_path, encoding='gbk')
    else:
        print(f"输出文件 {output_csv_path} 不存在，将从输入文件 {csv1_path} 开始处理。")
        df1 = pd.read_csv(csv1_path, encoding='gbk')

    df2 = pd.read_csv(csv2_path, encoding='gbk')
    df3 = pd.read_csv(csv3_path, encoding='gbk')
    df4 = pd.read_csv(csv4_path, encoding='gbk')

    # --- 1. 确保输出列存在 ---
    # 风速相关列
    for col in ["Cut-in wind speed", "Rated wind speed", "Cut-out wind speed"]:
        if col not in df1.columns:
            df1[col] = None
    for col in ["Cut-in wind speed source", "Rated wind speed source", "Cut-out wind speed source"]:
        if col not in df1.columns:
            df1[col] = None

    # 轮毂高度相关列
    if "Hub height" not in df1.columns:
        df1["Hub height"] = None
    if "Hub height source" not in df1.columns:
        df1["Hub height source"] = None

    # --- 2. 数据预处理 ---
    # 将陆海标识统一为 1 (陆) / 0 (海)
    df1['onshore_flag'] = df1['陆海1-2'].apply(lambda x: 1 if x == 1 else 0)
    
    # 清理 Model 列的空格
    df2['Model'] = df2['Model'].astype(str).str.strip()

    # --- 3. 获取不重复的机型编码，并检查是否需要处理 ---
    unique_codes = df1['机型编码'].unique()
    print(f"共发现 {len(unique_codes)} 种不重复的机型编码。")

    codes_to_process = []
    for code in unique_codes:
        mask = df1['机型编码'] == code
        all_rows_for_code = df1.loc[mask]
        # 检查该机型编码对应的所有行的 Cut-in wind speed 是否都已填充
        if all_rows_for_code['Cut-in wind speed'].notna().all():
             print(f"  机型编码 '{code}' 的所有 {len(all_rows_for_code)} 行均已完成，跳过。")
        else:
            codes_to_process.append(code)
            print(f"  机型编码 '{code}' 存在未完成行，待处理。")
    
    print(f"需要处理的机型编码数量: {len(codes_to_process)}")

    # --- 4. 遍历每种需要处理的机型编码 ---
    for code in codes_to_process:
        print(f"\n--- 开始处理机型编码: '{code}' ---")
        
        # 获取该编码对应的行索引
        mask = df1['机型编码'] == code
        
        # 从 df2 中查找最相似的5个 Model
        choices = df2['Model'].tolist()
        matches = process.extract(code, choices, limit=5, scorer=fuzz.partial_ratio)
        
        print(f"  为 '{code}' 找到的最相似型号:")
        for i, match_info in enumerate(matches):
            model_name, score = match_info
            print(f"    [{i+1}] {model_name} (相似度: {score})")
        print("    [A] 使用平均值")
        
        # --- 5. 交互式选择 ---
        while True:
            try:
                choice_input = input("  请选择 (输入 1-5 或 'a'): ").strip().upper()
                
                if choice_input == 'A': # 统一为大写
                    selected_model = 'AVERAGE'
                    break
                elif choice_input.isdigit() and 1 <= int(choice_input) <= 5:
                    selected_model = matches[int(choice_input) - 1][0]
                    break
                else:
                    print("    输入无效，请输入 1-5 或 'a'。")
            except KeyboardInterrupt:
                print("\n  用户中断操作。")
                return
            except Exception as e:
                print(f"    输入错误: {e}")

        # --- 6. 为该机型编码的所有行填充数据 ---
        # 获取该编码对应的单风机功率和陆海标志（假设同一种机型编码下，这些值相同或取第一个）
        sample_row = df1.loc[mask].iloc[0] 
        rated_power = sample_row['单风机MW']
        onshore_val = sample_row['onshore_flag']

        for idx in df1.index[mask]:
            # 检查该行是否已完成，若已完成则跳过
            if pd.notna(df1.loc[idx, 'Cut-in wind speed']):
                 print(f"    行 {idx} 已完成，跳过。")
                 continue

            print(f"    正在处理行 {idx}...")

            # --- 6.1 填充风速参数 ---
            if selected_model == 'AVERAGE':
                # 从 df3 获取平均值
                mean_row = df3[(df3['Rated power'] == round(rated_power, 1)) & (df3['Onshore'] == onshore_val)]
                if not mean_row.empty:
                    for col in ["Cut-in wind speed", "Rated wind speed", "Cut-out wind speed"]:
                        df1.at[idx, col] = mean_row.iloc[0][col]
                        df1.at[idx, f"{col} source"] = "mean value"
                else:
                     print(f"      警告: 机型编码 '{code}', 功率 {rated_power}, 陆海 {onshore_val} 在 df3 中找不到平均值。")
                     for col in ["Cut-in wind speed", "Rated wind speed", "Cut-out wind speed"]:
                         df1.at[idx, f"{col} source"] = "mean value (missing in df3)"
            else:
                # 从 df2 获取具体型号的值
                model_row = df2[df2['Model'] == selected_model]
                if not model_row.empty:
                    for col in ["Cut-in wind speed", "Rated wind speed", "Cut-out wind speed"]:
                        val_from_df2 = model_row.iloc[0][col]
                        if pd.isna(val_from_df2):
                            # df2 中缺失，从 df3 获取平均值
                             mean_row = df3[(df3['Rated power'] == round(rated_power, 1)) & (df3['Onshore'] == onshore_val)]
                             if not mean_row.empty:
                                 df1.at[idx, col] = mean_row.iloc[0][col]
                             else:
                                print(f"      警告: 机型编码 '{code}', 型号 '{selected_model}', 功率 {rated_power}, 陆海 {onshore_val} 在 df2 和 df3 中都找不到 {col}。")
                             df1.at[idx, f"{col} source"] = "mean value"
                        else:
                            # df2 中有值，直接使用
                            df1.at[idx, col] = val_from_df2
                            df1.at[idx, f"{col} source"] = "real value"
                else:
                    # 如果 selected_model 在 df2 中找不到，理论上不应该发生，因为是从 df2 中选的
                    # 但作为兜底，仍然尝试从 df3 获取
                    print(f"      警告: 选中的型号 '{selected_model}' 在 df2 中不存在！尝试使用平均值。")
                    mean_row = df3[(df3['Rated power'] == round(rated_power, 1)) & (df3['Onshore'] == onshore_val)]
                    if not mean_row.empty:
                        for col in ["Cut-in wind speed", "Rated wind speed", "Cut-out wind speed"]:
                            df1.at[idx, col] = mean_row.iloc[0][col]
                            df1.at[idx, f"{col} source"] = "mean value (model not found in df2)"
                    else:
                        print(f"      警告: 机型编码 '{code}', 功率 {rated_power}, 陆海 {onshore_val} 在 df3 中也找不到平均值。")


            # --- 6.2 填充轮毂高度参数 ---
            if selected_model != 'AVERAGE':
                # 优先从 df2 获取
                model_row = df2[df2['Model'] == selected_model]
                if not model_row.empty:
                    val_from_df2 = model_row.iloc[0]['Hub height']
                    if pd.notna(val_from_df2):
                        df1.at[idx, 'Hub height'] = val_from_df2
                        df1.at[idx, 'Hub height source'] = "real value"
            
            # 如果 df2 中没有，或者选择了平均值，则从 df4 获取
            if pd.isna(df1.loc[idx, 'Hub height']):
                 mean_hh_row = df4[(df4['Rated power'] == round(rated_power, 1)) & (df4['Onshore'] == onshore_val)]
                 if not mean_hh_row.empty:
                     df1.at[idx, 'Hub height'] = mean_hh_row.iloc[0]['Average Hub Height']
                     df1.at[idx, 'Hub height source'] = "mean value"
                 else:
                     print(f"      警告: 机型编码 '{code}', 功率 {rated_power}, 陆海 {onshore_val} 在 df4 中找不到平均轮毂高度。")
                     df1.at[idx, 'Hub height source'] = "mean value (missing in df4)"


            # --- 7. 保存当前进度 ---
            df1.to_csv(output_csv_path, index=False, encoding='gbk') # 保持输出编码一致
            print(f"    行 {idx} 处理完成并已保存。")

    print("\n--- 所有机型编码处理完毕。最终结果已保存至:", output_csv_path, "---")


# --- 使用示例 ---
# 请将以下路径替换为您的实际文件路径
# csv1_file = "path/to/your/csv1.csv"
# csv2_file = "path/to/your/csv2.csv"
# csv3_file = "path/to/your/csv3.csv"
# csv4_file = "path/to/your/csv4.csv"
# output_file = "path/to/your/output.csv"

# merge_wind_turbine_data(csv1_file, csv2_file, csv3_file, csv4_file, output_file)

if __name__ == "__main__":
    # 示例文件路径（请根据实际情况修改）
    csv1_file = r"D:\HKU PhD\wind data paper\风电场信息-2023.csv"
    csv2_file = r"D:\Data\turbines\wind_turbines.csv"
    csv3_file = r"D:\Data\turbines\standard_wind_turbines.csv"
    csv4_file = r"D:\Data\turbines\standard_wind_turbines_with_power_curves.csv"
    output_file = r"D:\HKU PhD\wind data paper\wind_farms_inventory_till_2023.csv"

    merge_wind_turbine_data(csv1_file, csv2_file, csv3_file, csv4_file, output_file)