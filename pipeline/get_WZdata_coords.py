import pandas as pd
import requests
import json
import os
import threading
import queue
import time

def get_coordinates(address, api_key):
    """
    使用高德地图Web服务API查询地址的经纬度。
    """
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        'key': api_key,
        'address': address
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == '1' and data.get('info') == 'OK':
                geocodes = data.get('geocodes')
                if geocodes and len(geocodes) > 0:
                    location_str = geocodes[0].get('location')
                    if location_str:
                        longitude, latitude = location_str.split(',')
                        return float(longitude), float(latitude)
    except Exception as e:
        print(f"查询失败: {e}")
    return None, None

def input_with_timeout(prompt, timeout=5):
    """
    带有超时功能的输入函数。
    如果在 timeout 秒内没有输入，返回 None。
    """
    def target(q):
        q.put(input(prompt))

    q = queue.Queue()
    thread = threading.Thread(target=target, args=(q,))
    thread.daemon = True
    thread.start()

    try:
        user_input = q.get(timeout=timeout)
        return user_input.strip().upper()
    except queue.Empty:
        print("\n--- 超时 ---")
        return None # 超时返回 None
    
    thread.join(timeout=0)

def process_geocoding_with_confirmation(input_csv_path, output_csv_path, amap_key, timeout_seconds=5):
    """
    主函数：处理CSV文件，为每一行添加经度和纬度，并提供人工确认功能。
    增加了超时自动确认功能。
    """
    df = pd.read_csv(input_csv_path, encoding='utf-8')

    if not os.path.exists(output_csv_path):
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"已创建输出文件: {output_csv_path}")

    if '经度' not in df.columns:
        df['经度'] = ""
    if '纬度' not in df.columns:
        df['纬度'] = ""

    for index, row in df.iterrows():
        print(f"\n--- 正在处理第 {index + 1} 行 ---")

        province = str(row.get('省（自治区、直辖市）', '')).strip()
        city = str(row.get('市', '')).strip()
        county = str(row.get('县（市、）', '')).strip()
        site = str(row.get('场址', '')).strip()

        full_address = f"{province}{city}{county}{site}"
        if not full_address.strip():
            print("地址为空，跳过。")
            continue

        # --- 修改开始：处理已有数据 ---
        existing_lon = row.get('经度', '')
        existing_lat = row.get('纬度', '')

        if pd.notna(existing_lon) and existing_lon != "" and pd.notna(existing_lat) and existing_lat != "":
            print(f"警告：该行已存在经纬度数据: ({existing_lon}, {existing_lat})")
            # 立即跳过，不进入交互循环
            print("默认跳过此行。")
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            continue # 直接处理下一行
        # --- 修改结束 ---
        
        print(f"待查询地址: {full_address}")

        lon, lat = get_coordinates(full_address, amap_key)

        if lon is None or lat is None:
            print(f"未能查询到有效坐标。")
            while True:
                choice = input_with_timeout("是否手动输入坐标? [Y]es / [N]o (留空): ", timeout_seconds)
                if choice == 'Y':
                    try:
                        manual_lon = float(input("请输入经度: "))
                        manual_lat = float(input("请输入纬度: "))
                        lon, lat = manual_lon, manual_lat
                        break
                    except ValueError:
                        print("输入格式错误，请重新输入。")
                elif choice == 'N':
                    print("选择留空。")
                    lon, lat = "", ""
                    break
                elif choice is None: # 超时
                    print("超时未选择，默认留空。")
                    lon, lat = "", ""
                    break # 跳出内层循环，进入确认步骤
                else:
                    print("输入无效，请重新选择。")

        print(f"查询到的坐标: 经度={lon}, 纬度={lat}")
        while True:
            confirm = input_with_timeout("确认使用此坐标吗? [Y]es / [N]o (重新查询) / [M]anual (手动修改) / [Q]uit: ", timeout_seconds)
            if confirm == 'Y' or confirm is None: # 超时或确认
                if confirm is None:
                    print("超时，自动确认。")
                else:
                    print("用户确认。")
                break
            elif confirm == 'N':
                print("重新查询...")
                lon, lat = get_coordinates(full_address, amap_key)
                if lon is None or lat is None:
                    print("再次查询失败。")
                else:
                    print(f"新查询到的坐标: 经度={lon}, 纬度={lat}")
            elif confirm == 'M':
                try:
                    new_lon = float(input("请输入新的经度: "))
                    new_lat = float(input("请输入新的纬度: "))
                    lon, lat = new_lon, new_lat
                    print(f"已更新为: 经度={lon}, 纬度={lat}")
                except ValueError:
                    print("输入格式错误。")
            elif confirm == 'Q':
                print("用户选择退出程序。")
                df.to_csv(output_csv_path, index=False, encoding='utf-8')
                return
            else:
                print("输入无效，请重新选择。")

        df.at[index, '经度'] = lon
        df.at[index, '纬度'] = lat

        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"✅ 第 {index + 1} 行已保存至: {output_csv_path}")

    print("\n🎉 所有行处理完毕！最终结果已保存。")

if __name__ == "__main__":
    INPUT_FILE = r"D:\HKU PhD\wind data paper\风电场精准点位2023年_coords.csv"  # 原始CSV文件路径
    OUTPUT_FILE = r"D:\HKU PhD\wind data paper\风电场精准点位2023年_coords2.csv"  # 输出CSV文件路径
    AMAP_API_KEY = "1edfcefa4c07b0d64c1b416c6a938e11"  # 替换为您的高德API Key

    TIMEOUT_SECONDS = 0.1  # 设置超时时间

    process_geocoding_with_confirmation(INPUT_FILE, OUTPUT_FILE, AMAP_API_KEY, TIMEOUT_SECONDS)