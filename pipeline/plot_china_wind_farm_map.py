import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号

def plot_wind_farm_map(wind_farm_csv_path, province_geojson_path, city_geojson_path, min_latitude=None, output_path=None):
    """
    绘制包含中国省市级行政边界和风电场位置的地图。
    风电场按陆地(蓝色)和海上(青色)区分显示。
    可选择性地过滤掉指定纬度以下的区域。

    参数:
        wind_farm_csv_path (str): 风电场CSV文件路径。
        province_geojson_path (str): 省级行政区域GeoJSON文件路径。
        city_geojson_path (str): 市级行政区域GeoJSON文件路径。
        min_latitude (float, 可选): 最小纬度阈值，低于此纬度的区域将被过滤。如果为None，则不进行过滤。
        output_path (str, 可选): 保存图片的路径。如果为None，则只显示图像。

    返回:
        None
    """

    # --- 数据加载与预处理 ---
    print("正在加载数据...")

    # 1. 加载风电场数据
    # 假设CSV文件是制表符分隔的，且第一行是列名。
    # 根据您提供的数据，列名是中文，需要正确指定。
    try:
        wind_farm_df = pd.read_csv(wind_farm_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试GBK编码（常见于中文Windows系统）
        wind_farm_df = pd.read_csv(wind_farm_csv_path, encoding='gbk')
   
    # 创建一个GeoDataFrame，将经纬度转换为几何点
    # 注意：GeoPandas默认的CRS是WGS84 (EPSG:4326)，这与您的数据一致。
    wind_farm_gdf = gpd.GeoDataFrame(
        wind_farm_df,
        geometry=gpd.points_from_xy(wind_farm_df['经度'], wind_farm_df['纬度']),
        crs="EPSG:4326"
    )

    # 2. 加载省级和市级GeoJSON数据
    province_gdf = gpd.read_file(province_geojson_path)
    city_gdf = gpd.read_file(city_geojson_path)

    # 确保所有GeoDataFrame的CRS都是EPSG:4326
    # 如果不是，会自动转换。但最好提前确认。
    if province_gdf.crs != "EPSG:4326":
        province_gdf = province_gdf.to_crs("EPSG:4326")
        print("省级数据已转换为EPSG:4326")
    if city_gdf.crs != "EPSG:4326":
        city_gdf = city_gdf.to_crs("EPSG:4326")
        print("市级数据已转换为EPSG:4326")
    if wind_farm_gdf.crs != "EPSG:4326":
        wind_farm_gdf = wind_farm_gdf.to_crs("EPSG:4326")
        print("风电场数据已转换为EPSG:4326")

    # --- 按最小纬度过滤数据 ---
    if min_latitude is not None:
        print(f"正在过滤纬度低于 {min_latitude} 的数据...")
        
        # 过滤风电场数据
        original_count = len(wind_farm_gdf)
        wind_farm_gdf = wind_farm_gdf[wind_farm_gdf.geometry.y >= min_latitude]
        filtered_count = len(wind_farm_gdf)
        print(f"风电场数据：{original_count} -> {filtered_count}")
        
        # 过滤省级边界数据
        original_province_count = len(province_gdf)
        # 计算每个省份的中心纬度，保留中心纬度大于等于最小纬度的省份
        province_centers = province_gdf.centroid
        province_gdf = province_gdf[province_centers.y >= min_latitude]
        filtered_province_count = len(province_gdf)
        print(f"省级边界数据：{original_province_count} -> {filtered_province_count}")
        
        # 过滤市级边界数据
        original_city_count = len(city_gdf)
        city_centers = city_gdf.centroid
        city_gdf = city_gdf[city_centers.y >= min_latitude]
        filtered_city_count = len(city_gdf)
        print(f"市级边界数据：{original_city_count} -> {filtered_city_count}")

    # --- 过滤几何类型 ---
    # 移除点状几何，因为它们没有边界
    province_gdf = province_gdf[province_gdf.geometry.geom_type != 'Point']
    city_gdf = city_gdf[city_gdf.geometry.geom_type != 'Point']

    # --- 绘图 ---
    print("正在绘制地图...")

    # 创建一个更大的图形以保证美观
    fig, ax = plt.subplots(figsize=(15, 10))

    # 1. 绘制省级边界
    # 使用较粗的实线
    if not province_gdf.empty:
        province_gdf.boundary.plot(
            ax=ax,
            color='black',      # 颜色：黑色
            linewidth=0.8,      # 线宽：较粗
            linestyle='-',      # 线型：实线
            label='省级边界',
            zorder=2            # 确保边界线在点之上绘制
        )

    # 2. 绘制市级边界
    # 使用较细的虚线，颜色较浅
    if not city_gdf.empty:
        city_gdf.boundary.plot(
            ax=ax,
            color='gray',       # 颜色：灰色，较浅
            linewidth=0.3,      # 线宽：较细
            linestyle='--',     # 线型：虚线
            label='市级边界',
            zorder=2            # 确保边界线在点之上绘制
        )

    # 3. 绘制风电场点 (按陆海区分)
    if not wind_farm_gdf.empty:
        # 陆风 (陆海1-2 == 1) 用蓝色
        land_farms = wind_farm_gdf[wind_farm_gdf['陆海1-2'] == 1]
        if not land_farms.empty:
            land_farms.plot(
                ax=ax,
                marker='o',         # 标记形状：圆形
                color='blue',       # 颜色：蓝色 (陆风)
                markersize=0.1,     # 标记大小
                alpha=0.7,          # 透明度，避免遮挡太多
                label='陆上风电场',
                zorder=1            # 确保点在边界线下方绘制
            )

        # 海风 (陆海1-2 == 2) 用青色
        sea_farms = wind_farm_gdf[wind_farm_gdf['陆海1-2'] == 2]
        if not sea_farms.empty:
            sea_farms.plot(
                ax=ax,
                marker='o',         # 标记形状：圆形
                color='cyan',       # 颜色：青色 (海风)
                markersize=0.1,     # 标记大小
                alpha=0.7,          # 透明度，避免遮挡太多
                label='海上风电场',
                zorder=1            # 确保点在边界线下方绘制
            )

    # --- 图形美化 ---
    # 设置标题
    title = f'中国风电场分布图(至2023年)'
    
    ax.set_title(title, fontsize=16, fontweight='bold')

    # 设置坐标轴标签
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')

    # 添加图例
    ax.legend(loc='upper right')

    # 调整布局，防止标签被裁剪
    plt.tight_layout()

    # 如果提供了输出路径，则保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"地图已保存至: {output_path}")

    # 显示图形
    plt.show()

# --- 使用示例 ---
# 请将下面的路径替换为您实际的文件路径
if __name__ == "__main__":
    # 替换为您的实际文件路径
    wind_farm_csv = r"D:\HKU PhD\wind data paper\风电场信息-2023.csv"
    province_geojson = r"D:\Data\admin\中国_省.geojson"
    city_geojson = r"D:\Data\admin\中国_市.geojson"
    output_image = "D:/try/china_turbines.jpg"  # 可选：保存图片的路径

    # 调用函数，设置最小纬度为20度（例如：不过滤掉海南省等低纬度地区）
    # 如果不需要过滤，请将 min_latitude 设为 None 或者不传入此参数
    plot_wind_farm_map(wind_farm_csv, province_geojson, city_geojson, 
                      min_latitude=18, output_path=output_image)