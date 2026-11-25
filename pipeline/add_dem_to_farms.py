"""
数字高程模型 (DEM) 处理模块

该模块从 NASADEM 栅格文件中提取指定位置的高程值，并支持批量为
CSV 文件添加高程信息。
"""
import numpy as np
from pathlib import Path
import pandas as pd
import warnings
from tqdm import tqdm


def get_dem(
    dem_dir: str | Path,
    lat: float,
    lon: float
) -> int:
    """
    从 NASADEM 栅格文件中获取指定坐标的高程值。

    本函数支持 NASADEM 1 弧秒 (3601×3601) 和 SRTM 3 弧秒 (1201×1201) 两种格式。
    坐标采用十进制度数制，北纬为正、东经为正。

    Args:
        dem_dir (str | Path): NASADEM 数据根目录的路径。
            目录结构应为: dem_dir/N<lat>/<file_folder>/n<lat>e<lon>.hgt
            例如: "D:/Data/DEM/Grid 30m"
        lat (float): 纬度坐标，范围 [0, 90)，北纬为正。单位为十进制度数。
        lon (float): 经度坐标，范围 [0, 180)，东经为正。单位为十进制度数。

    Returns:
        int: 指定位置的高程值，单位为米。
            - 若栅格点为有效数据，返回该点高程值
            - 若栅格点为 nodata，返回 -32768

    Raises:
        ValueError: 当坐标或文件格式无效时抛出。
            - 纬度不在 [0, 90) 范围内
            - 经度不在 [0, 180) 范围内
            - 文件尺寸不符合 NASADEM/SRTM 标准
            - 坐标超出 .hgt 文件覆盖范围（理论上不应发生）
        FileNotFoundError: 当对应的 .hgt 文件不存在时抛出。

    Notes:
        - 栅格数据采用 big-endian 格式存储，行列从北到东递增
        - 浮点坐标会自动转换为最近的栅格点
        - 函数内部采用行列索引边界检查以防止浮点误差导致的越界

    Examples:
        >>> dem = "D:/Data/DEM/Grid 30m"
        >>> elevation = get_dem_value_at_location(dem, 22.3, 114.2)
        >>> print(f"高程: {elevation} m")
    """
    # --- 参数校验 ---
    if not (0 <= lat < 90):
        raise ValueError(
            f"纬度必须在 [0, 90) 范围内，实际值: {lat}"
        )
    if not (0 <= lon < 180):
        raise ValueError(
            f"经度必须在 [0, 180) 范围内，实际值: {lon}"
        )

    # --- 确定目标 .hgt 文件 ---
    # 栅格文件以整数度数命名，计算坐标所属的网格单元
    lat_deg = int(np.floor(lat))
    lon_deg = int(np.floor(lon))

    # 构建 .hgt 文件的完整路径
    hgt_filename = f"n{lat_deg:02d}e{lon_deg:03d}.hgt"
    hgt_path = (
        Path(dem_dir)
        / f"N{lat_deg:02d}"
        / f"NASADEM_HGT_n{lat_deg:02d}e{lon_deg:03d}"
        / hgt_filename
    )

    if not hgt_path.exists():
        raise FileNotFoundError(
            f"DEM 栅格文件不存在: {hgt_path}"
        )

    # --- 读取二进制栅格数据 ---
    with open(hgt_path, 'rb') as f:
        raw_data = f.read()

    # 根据文件大小判断栅格分辨率
    byte_size = len(raw_data)
    if byte_size == 3601 * 3601 * 2:
        # NASADEM 1 弧秒分辨率
        rows, cols = 3601, 3601
        resolution = 1.0 / 3600  # 弧度 (degrees)
    elif byte_size == 1201 * 1201 * 2:
        # SRTM 3 弧秒分辨率
        rows, cols = 1201, 1201
        resolution = 3.0 / 3600  # 弧度
    else:
        raise ValueError(
            f"DEM 文件格式无效。文件大小 {byte_size} 字节，"
            f"预期为 {3601*3601*2} 或 {1201*1201*2} 字节。"
        )

    # 将二进制数据解析为 big-endian int16 数组，并重塑为二维栅格
    elevation_flat = np.frombuffer(raw_data, dtype='>i2')
    elevation = elevation_flat.reshape((rows, cols))

    # --- 计算栅格索引 ---
    # 栅格行从北向南递增（北边界对应行 0），列从西向东递增
    lat_max = lat_deg + 1
    lon_min = lon_deg

    row = int((lat_max - lat) / resolution)
    col = int((lon - lon_min) / resolution)

    # 边界检查，防止浮点误差导致的索引越界
    row = np.clip(row, 0, rows - 1)
    col = np.clip(col, 0, cols - 1)

    # 返回该坐标处的高程值
    return int(elevation[row, col])



def add_elevation_to_csv(csv_path: str | Path, dem_dir: str | Path) -> Path:
    """
    为 CSV 文件中的每个采样点批量添加高程信息。

    读取包含经纬度列的 CSV 文件，使用 `get_dem()` 函数
    为每个点查询对应的高程值，并将结果追加为新列保存为新文件。

    Args:
        csv_path (str | Path): 输入 CSV 文件的路径。该文件必须包含以下两列：
            - '经度'：十进制度数，东经为正，范围 [0, 180)
            - '纬度'：十进制度数，北纬为正，范围 [0, 90)
        dem_dir (str | Path): NASADEM 数据根目录的路径。
            详见 `get_dem()` 的 `dem_dir` 参数说明。

    Returns:
        Path: 输出 CSV 文件的路径。文件名格式为：
            <原始名>_with_elevation<扩展名>
            例如：wind_farms.csv → wind_farms_with_elevation.csv

    Raises:
        ValueError: 当输入 CSV 文件缺少必要的'经度'或'纬度'列时抛出。
        FileNotFoundError: 当输入 CSV 文件不存在时 pandas 会抛出。

    Notes:
        - 对于无法获取高程的坐标（如超出 DEM 覆盖范围），函数记录为 **NaN 并发出告警**
      （不再填 0——0 m 是合法海平面值，静默填 0 会掩盖问题并让高原场密度反向）
        - 处理过程使用 tqdm 显示进度条
        - 输出 CSV 使用相同的编码（gbk）和分隔符（逗号）

    Examples:
        >>> csv_file = "wind_farms_inventory.csv"
        >>> dem_dir = "D:/Data/DEM/Grid 30m"
        >>> output_file = add_elevation_to_csv(csv_file, dem_dir)
        >>> print(f"已保存至: {output_file}")
    """
    # --- 读取 CSV 数据 ---
    df = pd.read_csv(csv_path, encoding='gbk')

    # --- 验证必要列 ---
    required_cols = ['经度', '纬度']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"CSV 文件必须包含 {required_cols} 列。"
            f"实际列名: {list(df.columns)}"
        )

    # --- 批量查询高程值 ---
    elevations = []
    failed = []
    csv_path = Path(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="获取高程值", unit="个"):
        lon = row['经度']
        lat = row['纬度']

        try:
            # 调用 DEM 查询函数
            elevation = get_dem(dem_dir, lat, lon)
            elevations.append(int(elevation))
        except Exception as e:
            # ⚠️ 取不到高程时**不能填 0**：0 m 是合法海平面值，静默填 0 会让高原场
            #    被当成海平面场（密度修正方向反向）。这里留 NaN + 告警，让下游
            #    resolve_dem_elevation 明确报错或按显式开关回退到 ERA5 地形。
            elevations.append(np.nan)
            failed.append((float(lat), float(lon), type(e).__name__, str(e)))
            warnings.warn(
                f"无法获取坐标 ({lat:.4f}°, {lon:.4f}°) 的高程值，已写 NaN（不是 0）。"
                f"原因: {type(e).__name__}: {str(e)}",
                UserWarning,
                stacklevel=2,
            )

    # --- 添加高程列 ---
    df['Elevation'] = elevations

    if failed:
        print(
            "⚠ %d/%d 个坐标未取到高程（已写 NaN）——这些场在流水线里会报错，"
            "除非显式开启 ERA5 地形回退。" % (len(failed), len(df))
        )
    else:
        print("✓ 全部 %d 个坐标均取到高程。" % len(df))

    # --- 保存结果 ---
    output_path = csv_path.parent / f"{csv_path.stem}_with_elevation{csv_path.suffix}"
    df.to_csv(output_path, index=False, encoding='gbk')
    print(f"✓ 已完成。结果保存至: {output_path}")

    return output_path

if __name__ == "__main__":
    """
    主程序入口：演示如何使用 add_elevation_to_csv 函数。
    """
    # 输入数据路径
    csv_file = r"D:\HKU PhD\wind data paper\wind_farms_inventory_till_2023.csv"
    dem_dir = r"D:\Data\DEM\Grid 30m"

    # 为 CSV 文件添加高程列
    output_file = add_elevation_to_csv(csv_file, dem_dir)