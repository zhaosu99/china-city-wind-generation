"""
风电功率曲线生成模块

该模块提供功能用于生成风电场的功率曲线，支持高斯平滑处理和可视化。
"""
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Union
import os
from tqdm import tqdm

from utils import safe_read_csv


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

def generate_farm_power_curves_csvs(
    csv_path: str,
    output_dir: str,
    overwrite: bool = False,
    w_grid_col_name: str = 'w_grid',
    cf_col_name: str = 'CF',
) -> None:
    """
    为风电场库中的每个风电场生成功率曲线 CSV 文件。

    遍历输入 DataFrame 的每一行，为包含完整风速参数的风电场生成一个包含风速-容量因子
    映射关系的 CSV 文件。文件名格式为 farm_<风电场ID>.csv，包含三列：
    - w_grid: 风速 (m/s)
    - CF: 容量因子
    - hub_height: 风机轮毂高度 (仅第一行包含值)

    Args:
        df1_augmented (pd.DataFrame): 包含风电场信息的 DataFrame，必须包含以下列名：
            - '风电场ID' (int): 风电场的唯一标识符
            - 'Cut-in wind speed' (float): 切入风速 (m/s)
            - 'Rated wind speed' (float): 额定风速 (m/s)
            - 'Cut-out wind speed' (float): 切出风速 (m/s)
            - 'Hub height' (float): 风机轮毂高度 (m)
        output_dir (str): 输出目录的路径，用于存放生成的 CSV 文件。
            若目录不存在，函数将自动创建。
        overwrite (bool): 输出文件已存在时是否覆盖。默认 False。
        w_grid_col_name (str): 风速网格列名。默认 'w_grid'。
        cf_col_name (str): 容量因子列名。默认 'CF'。

    Returns:
        None

    Raises:
        FileExistsError: 当输出目录创建失败时可能抛出（但通常被 os.makedirs 处理）。

    Notes:
        - 函数跳过缺少任何风速参数（Cut-in, Rated, Cut-out）的行
        - 若输出文件已存在且 overwrite=False，将跳过该风电场，不重新生成
        - 若输出文件已存在且 overwrite=True，将覆盖原文件
        - 使用 tqdm 提供进度条反馈
        - 所有异常会被捕获并记录，不会中断处理流程

    Examples:
        >>> import pandas as pd
        >>> df = pd.read_csv('wind_farms_inventory.csv', encoding='gbk')
        >>> generate_farm_power_curves_csv(df, 'output/power_curves/')
    """

    df1_augmented = safe_read_csv(csv_path)
    if not isinstance(w_grid_col_name, str) or w_grid_col_name.strip() == '':
        raise ValueError("w_grid_col_name 不能为空字符串")
    if not isinstance(cf_col_name, str) or cf_col_name.strip() == '':
        raise ValueError("cf_col_name 不能为空字符串")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 过滤：仅保留包含完整风速参数的行
    required_cols = ['Cut-in wind speed', 'Rated wind speed', 'Cut-out wind speed']
    valid_df = df1_augmented.dropna(subset=required_cols)

    total_rows = len(valid_df)
    if total_rows == 0:
        print("警告: DataFrame 中没有包含完整风速参数的行，处理中止。")
        return

    # 初始化进度条
    pbar = tqdm(total=total_rows, desc="生成功率曲线", unit="个")

    # 处理每个风电场
    for _, row in valid_df.iterrows():
        # 提取基本信息
        farm_id = int(row['风电场ID'])
        hub_height = row['Hub height']
        cut_in_speed = row['Cut-in wind speed']
        rated_speed = row['Rated wind speed']
        cut_out_speed = row['Cut-out wind speed']

        # 定义输出文件路径
        output_filename = f"farm_{farm_id}.csv"
        output_path = os.path.join(output_dir, output_filename)

        # 若文件已存在且不允许覆盖，则跳过
        if Path(output_path).exists() and not overwrite:
            pbar.update(1)
            continue

        try:
            # 调用 generate_power_curve 生成功率曲线
            w_grid_unrounded, cf_values = generate_power_curve(
                cut_in_v=cut_in_speed,
                rated_v=rated_speed,
                cut_out_v=cut_out_speed,
                # 其余参数使用默认值
            )
        except Exception as e:
            print(f"错误: 处理风电场 ID={farm_id} 时发生异常: {e}")
            pbar.update(1)
            continue

        # 风速网格四舍五入到小数点后两位
        w_grid = np.round(w_grid_unrounded, 2)

        # 创建结果 DataFrame
        result_df = pd.DataFrame({
            w_grid_col_name: w_grid,
            cf_col_name: cf_values,
        })

        # 添加 hub_height 列，仅第一行包含实际值，其余为 NaN
        hub_col = pd.Series([None] * len(result_df), dtype='object')
        hub_col.iloc[0] = hub_height
        result_df['hub_height'] = hub_col

        # 保存为 CSV 文件
        try:
            result_df.to_csv(output_path, index=False)
        except IOError as e:
            print(f"警告: 无法写入文件 {output_path}，错误: {e}")

        pbar.update(1)

    # 关闭进度条
    pbar.close()

    print(f"✓ 已完成。功率曲线 CSV 文件已保存至: {output_dir}")


