import requests
import json

def get_coordinates(address, api_key):
    """
    使用高德地图Web服务API查询地址的经纬度。

    Args:
        address (str): 要查询的地址，例如 "辽宁铁岭福合村"。
        api_key (str): 您申请的高德Web服务API密钥。

    Returns:
        tuple: (longitude, latitude) 如果查询成功，否则返回 (None, None)。
    """
    # API请求的URL
    url = "https://restapi.amap.com/v3/geocode/geo"
    
    # 请求参数
    params = {
        'key': api_key,
        'address': address
        # 可选参数，可以提高查询精度
        # 'city': '铁岭' 
    }

    try:
        # 发送GET请求
        response = requests.get(url, params=params)
        
        # 检查HTTP响应状态码
        if response.status_code == 200:
            # 解析返回的JSON数据
            data = response.json()
            
            # 检查API返回的状态码
            if data.get('status') == '1' and data.get('info') == 'OK':
                geocodes = data.get('geocodes')
                if geocodes and len(geocodes) > 0:
                    location_str = geocodes[0].get('location')
                    if location_str:
                        # location格式为 "经度,纬度"
                        longitude, latitude = location_str.split(',')
                        return float(longitude), float(latitude)
                    else:
                        print(f"API返回数据中未找到location字段: {data}")
                else:
                    print(f"API返回geocodes为空: {data}")
            else:
                print(f"API返回错误: {data.get('info', 'Unknown error')}")
        else:
            print(f"HTTP请求失败，状态码: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"网络请求发生错误: {e}")
    except json.JSONDecodeError:
        print("API返回的数据不是有效的JSON格式")
    except Exception as e:
        print(f"发生未知错误: {e}")
    
    return None, None

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 请在这里填入您申请的高德Web服务API密钥
    YOUR_AMAP_KEY = "1edfcefa4c07b0d64c1b416c6a938e11" 

    # 2. 定义您要查询的地址
    province = "辽宁"
    city = "铁岭"
    village = "福合村"
    full_address = f"山西大同广灵县南村镇" # 拼接地址

    # 3. 调用函数获取坐标
    longitude, latitude = get_coordinates(full_address, YOUR_AMAP_KEY)

    # 4. 打印结果
    if longitude is not None and latitude is not None:
        print(f"地址: {full_address}")
        print(f"经度: {longitude}")
        print(f"纬度: {latitude}")
        print(f"完整坐标 (经度, 纬度): ({longitude}, {latitude})")
    else:
        print(f"未能查询到地址 '{full_address}' 的有效坐标。")
