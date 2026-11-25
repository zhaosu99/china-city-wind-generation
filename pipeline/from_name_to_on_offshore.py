import pandas as pd

def add_land_type_column(input_csv_path, output_csv_path):
    """
    读取CSV文件，根据“项目名称”列的内容，创建“陆上”列。

    Args:
        input_csv_path (str): 输入CSV文件的路径。
        output_csv_path (str): 输出CSV文件的路径。
    """
    # 1. 读取CSV文件
    df = pd.read_csv(input_csv_path, encoding='utf-8')

    # 2. 检查是否存在“项目名称”列
    if '项目名称' not in df.columns:
        raise ValueError("输入的CSV文件中未找到名为 '项目名称' 的列。")

    # 3. 创建“陆上”列
    # 使用 pandas 的向量化操作，比循环更高效
    # isin() 与 str.contains() 结合使用，str.contains 返回布尔Series
    # ~ 代表取反，即不包含“海上”
    # np.where 是向量化的if-else
    import numpy as np
    # 检查“项目名称”是否包含“海上”，注意处理可能的空值 (NaN)
    contains_hai = df['项目名称'].astype(str).str.contains('海上', na=False)
    # 使用 np.where 进行条件赋值
    df['陆上'] = np.where(contains_hai, 0, 1)

    # 4. 保存到新的CSV文件
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"处理完成，结果已保存至: {output_csv_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    INPUT_FILE = r"D:\HKU PhD\wind data paper\风电场精准点位2023年.csv"
    OUTPUT_FILE = r"D:\HKU PhD\wind data paper\风电场精准点位2023年2.csv"

    add_land_type_column(INPUT_FILE, OUTPUT_FILE)