import geopandas as gpd
import pandas as pd

def process_and_join_data_nearest_single_with_paths(
    csv_path, 
    admin_geojson_path, 
    output_csv_path, 
    output_geojson_path, 
    output_crs="EPSG:4326"
):
    """
    读取包含经纬度的 CSV 文件，将其转换为点，并与一个行政区域 GeoJSON 文件进行最近邻空间连接。
    输出两个文件：一个带有连接信息且不含几何列的 CSV 文件，和一个带有连接信息且含几何列的 GeoJSON 文件。
    输出文件的路径由函数调用时直接指定。

    参数:
        csv_path (str): 输入的包含经纬度的 CSV 文件路径。
        admin_geojson_path (str): 用于空间连接的行政区域 GeoJSON 文件路径 (例如，包含省简称、市简称的合并文件)。
        output_csv_path (str): 输出的 CSV 文件路径 (不含 geometry 列)。
        output_geojson_path (str): 输出的 GeoJSON 文件路径 (含 geometry 列)。
        output_crs (str, optional): 输出文件的坐标参考系统，默认为 "EPSG:4326"。

    返回:
        tuple: 
               - output_csv_path (str): 输出的 CSV 文件路径 (与输入参数相同)。
               - output_geojson_path (str): 输出的 GeoJSON 文件路径 (与输入参数相同)。
    """
    # --- 1. 读取 CSV 并创建点 GeoDataFrame ---
    print(f"正在读取 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    if '经度' not in df.columns or '纬度' not in df.columns:
        raise ValueError("CSV 文件必须包含 '经度' 和 '纬度' 列。")

    # 使用经纬度创建点几何
    geometry = gpd.points_from_xy(df['经度'], df['纬度'], crs="EPSG:4326")
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry)

    # --- 2. 读取行政区域 GeoDataFrame ---
    print(f"正在读取行政区域 GeoJSON 文件: {admin_geojson_path}")
    gdf_admin = gpd.read_file(admin_geojson_path)

    # --- 3. 确保 CRS 一致 ---
    if gdf_admin.crs != output_crs:
        print(f"行政区域 GeoJSON 的 CRS ({gdf_admin.crs}) 与目标 CRS ({output_crs}) 不一致，正在转换...")
        gdf_admin = gdf_admin.to_crs(output_crs)
    gdf_points = gdf_points.to_crs(output_crs)

    # --- 4. 执行最近邻空间连接 ---
    print("正在进行最近邻空间连接 (sjoin_nearest)...")
    # how='left': 保留左表 (gdf_points) 的所有行
    gdf_joined = gpd.sjoin_nearest(gdf_points, gdf_admin, how='left')
    gdf_joined.drop(columns=['index_right', '省代号', '市代号'], inplace=True) # 删除多余的索引列

    gdf_joined = gdf_joined.loc[:,
        ['geometry', 'ID', '年份', '省全称', '省简称', '市全称', '市简称', '经度', '纬度', 
         '直径', '机型编码', '风机MW', '装机(MW)', '陆海1-2']
        ]
    gdf_joined.rename(columns={
        'ID': '风电场ID',
        '直径': '风叶直径',
        '风机MW': '单风机MW',
        '装机(MW)': '风电场装机MW',
        }, inplace=True)

    # --- 5. 保存输出文件 ---
    # 5a. 保存为 CSV (去除 geometry 列)
    print(f"正在保存不含几何列的 CSV 文件: {output_csv_path}")
    df_output_csv = gdf_joined.drop(columns='geometry') # 移除 geometry 列
    df_output_csv.to_csv(output_csv_path, index=False, encoding='utf-8')

    # 5b. 保存为 GeoJSON (保留 geometry 列)
    print(f"正在保存含几何列的 GeoJSON 文件: {output_geojson_path}")
    gdf_joined.to_file(output_geojson_path, driver='GeoJSON')

    print("单次连接处理完成。")
    return output_csv_path, output_geojson_path

# --- 使用示例 ---
csv_input = r"D:\HKU PhD\wind data paper\风电场精准点位-2023年.csv"
admin_geojson_combined = r"D:\Data\admin\中国_市2.geojson" # 您新生成的文件路径
output_csv = r"D:\HKU PhD\wind data paper\风电场信息.csv"
output_geojson = r"D:\HKU PhD\wind data paper\风电场信息.geojson"
csv_out, geojson_out = process_and_join_data_nearest_single_with_paths(
    csv_input, admin_geojson_combined, output_csv, output_geojson
)
print(f"输出的 CSV 文件: {csv_out}")
print(f"输出的 GeoJSON 文件: {geojson_out}")