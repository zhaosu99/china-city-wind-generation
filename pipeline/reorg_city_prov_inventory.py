import geopandas as gpd
import pandas as pd

def create_admin_combined_geojson_and_csv_corrected(
    province_geojson_path,
    city_geojson_path,
    abbreviation_csv_path,
    output_geojson_path,
    output_csv_path,
    output_crs="EPSG:4326"
):
    """
    读取省和市的 GeoJSON 文件，将省名信息添加到市文件中，
    根据映射表添加省简称和市简称，过滤掉包含“境界线”的记录，
    并生成一个包含指定列的 GeoJSON 和 CSV 文件。
    此版本使用 'within' 谓词进行空间连接，以确保获取正确的行政隶属关系。

    参数:
        province_geojson_path (str): 省级 GeoJSON 文件路径。
        city_geojson_path (str): 市级 GeoJSON 文件路径。
        abbreviation_csv_path (str): 包含全称到简称映射的 CSV 文件路径。
        output_geojson_path (str): 输出的 GeoJSON 文件路径。
        output_csv_path (str): 输出的 CSV 文件路径。
        output_crs (str, optional): 输出文件的坐标参考系统，默认为 "EPSG:4326"。

    返回:
        tuple: 
               - output_geojson_path (str): 输出的 GeoJSON 文件路径 (与输入参数相同)。
               - output_csv_path (str): 输出的 CSV 文件路径 (与输入参数相同)。
    """
    print(f"正在读取省级 GeoJSON 文件: {province_geojson_path}")
    gdf_province = gpd.read_file(province_geojson_path)
    gdf_province.rename(columns={'name': '省全称', 'gb': '省代号'}, inplace=True)

    print(f"正在读取市级 GeoJSON 文件: {city_geojson_path}")
    gdf_city = gpd.read_file(city_geojson_path)
    gdf_city.rename(columns={'name': '市全称', 'gb': '市代号'}, inplace=True)

    # --- 确保 CRS 一致 ---
    if gdf_province.crs != output_crs:
        print(f"省级 GeoJSON 的 CRS ({gdf_province.crs}) 与目标 CRS ({output_crs}) 不一致，正在转换...")
        gdf_province = gdf_province.to_crs(output_crs)
    if gdf_city.crs != output_crs:
        print(f"市级 GeoJSON 的 CRS ({gdf_city.crs}) 与目标 CRS ({output_crs}) 不一致，正在转换...")
        gdf_city = gdf_city.to_crs(output_crs)

    # --- 读取全称到简称的映射表 ---
    print(f"正在读取全称到简称的映射表 CSV 文件: {abbreviation_csv_path}")
    # 尝试不同的编码来读取 CSV
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    abbrev_df = None
    for encoding in encodings_to_try:
        try:
            print(f"尝试使用编码 {encoding} 读取映射表...")
            abbrev_df = pd.read_csv(abbreviation_csv_path, encoding=encoding)
            print(f"成功使用编码 {encoding} 读取映射表。")
            break
        except UnicodeDecodeError:
            print(f"使用编码 {encoding} 失败，尝试下一个...")
            continue

    if abbrev_df is None:
        raise ValueError(f"无法使用 {encodings_to_try} 中的任何编码读取文件 {abbreviation_csv_path}。请检查文件编码。")

    # --- 使用 'within' 谓词进行空间连接 (市在省内部) ---
    print("正在进行市到省的 'within' 空间连接 (查找市属于哪个省)...")
    # how='left': 保留所有市，即使没有匹配到省 (结果中省信息将为 NaN)
    # predicate='within': 查找 gdf_city 中的几何体是否在 gdf_province 的几何体内
    gdf_city_with_prov = gpd.sjoin(gdf_city, gdf_province[['省全称', '省代号', 'geometry']], how='left', predicate='within')

    # --- 添加省简称和市简称 ---
    # 根据您补充的信息更新列名
    prov_full_col = '省全称'
    prov_abbr_col = '省简称'
    city_full_col = '市全称'
    city_abbr_col = '市简称'

    # 创建映射字典
    prov_abbrev_map = dict(zip(abbrev_df[prov_full_col], abbrev_df[prov_abbr_col]))
    city_abbrev_map = dict(zip(abbrev_df[city_full_col], abbrev_df[city_abbr_col]))

    # 应用映射
    gdf_city_with_prov['省简称'] = gdf_city_with_prov['省全称'].map(prov_abbrev_map)
    gdf_city_with_prov['市简称'] = gdf_city_with_prov['市全称'].map(city_abbrev_map)

    # --- 过滤掉名称包含“境界线”的记录 ---
    print("正在过滤包含“境界线”的记录...")
    initial_count = len(gdf_city_with_prov)
    gdf_city_with_prov = gdf_city_with_prov[
        (~gdf_city_with_prov['省全称'].str.contains('境界线', na=False)) &
        (~gdf_city_with_prov['市全称'].str.contains('境界线', na=False))
    ]
    final_count = len(gdf_city_with_prov)
    print(f"过滤前记录数: {initial_count}, 过滤后记录数: {final_count}")

    # --- 选择并排序指定的列 ---
    print("正在选择和排序指定列...")
    selected_columns = ['省简称', '市简称', '省全称', '市全称', '省代号', '市代号', 'geometry']
    # 确保所有指定的列都存在
    missing_cols = [col for col in selected_columns if col not in gdf_city_with_prov.columns]
    if missing_cols:
        print(f"警告: 以下列在数据中未找到: {missing_cols}")
        # 从 selected_columns 中移除缺失的列
        selected_columns = [col for col in selected_columns if col not in missing_cols]

    gdf_final = gdf_city_with_prov[selected_columns]

    # --- 保存输出文件 ---
    print(f"正在保存 GeoJSON 文件: {output_geojson_path}")
    gdf_final.to_file(output_geojson_path, driver='GeoJSON')

    print(f"正在保存 CSV 文件 (不含 geometry 列): {output_csv_path}")
    df_final_csv = gdf_final.drop(columns='geometry')
    df_final_csv.to_csv(output_csv_path, index=False, encoding='utf-8')

    print("处理完成。")
    return output_geojson_path, output_csv_path

# --- 使用示例 ---
province_path = r"D:\Data\admin\中国_省.geojson"
city_path = r"D:\Data\admin\中国_市.geojson"
abbrev_path = r"D:\Data\admin\省市名全简称.csv"
output_geojson = r"D:\Data\admin\中国_市2.geojson"
output_csv = r"D:\Data\admin\中国_市2.csv"
geojson_out, csv_out = create_admin_combined_geojson_and_csv_corrected(
    province_path, city_path, abbrev_path, output_geojson, output_csv
)
print(f"输出 GeoJSON: {geojson_out}")
print(f"输出 CSV: {csv_out}")