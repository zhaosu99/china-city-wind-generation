import json
import pandas as pd
import re # 导入正则表达式模块用于处理Hub height

def parse_hub_height(height_str):
    """解析 Hub height 字符串，返回数值或 None"""
    if height_str in ["-", "site specific m", "Site-specific m", "Site and country specific m"]:
        return None
    # 使用正则表达式提取所有数字（包括小数）
    numbers = re.findall(r"(\d+\.?\d*)", height_str)
    if not numbers:
        return None
    try:
        # 将提取的数字字符串转换为浮点数
        values = [float(num) for num in numbers]
        # 计算平均值
        avg_value = sum(values) / len(values)
        return avg_value
    except ValueError:
        # 如果转换失败，返回 None
        return None

def json_to_csv_wind_data(json_path, csv_path):
    """
    从JSON文件提取风力涡轮机数据并保存为CSV文件。

    Args:
        json_path (str): 输入JSON文件的路径。
        csv_path (str): 输出CSV文件的路径。
    """
    print(f"正在读取JSON文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            # 尝试解析整个文件内容为JSON
            # 注意：您提供的文本片段不是一个完整的JSON对象，缺少最外层的 { 和 }
            # 为了使脚本正常运行，您需要确保JSON文件格式正确
            # 例如，文件内容应为 {"key1": {...}, "key2": {...}, ...}
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print("请确保您提供的文件是格式正确的JSON文件。")
            # 如果JSON格式不标准，这里会出错。您需要提供一个完整的JSON对象。
            # 基于您上传的txt内容，它似乎是由多个JSON片段拼接而成，而非一个标准对象。
            # 为了演示目的，我们假设文件内容是标准格式。
            # 如果您的文件确实缺少最外层的花括号，需要先手动添加或通过其他方式修复。
            # 这里我们假设读取成功，因为函数入口是用户指定的。
            # 如果原始文件就是不完整的片段，需要预处理将片段合并成一个对象。
            # 例如，如果文件内容是 "key1": {...}, "key2": {...} 的形式，需要在外面加上 { }
            # 这种预处理较为复杂，最好由用户提供标准JSON。
            # 为避免误解，我在此处抛出异常，要求用户提供标准格式。
            raise e # 重新抛出异常，让用户知道JSON格式有误

    print("开始处理数据...")
    processed_rows = []
    for key, value in data.items():
        if not isinstance(value, dict):
            # 如果值不是字典，跳过
            continue

        # 提取公司名和型号
        if '---' in key:
            parts = key.split('---')
            company = parts[0].strip()
            model = parts[1].strip()
        else:
            # 如果没有 '---' 分隔符，将整个key作为型号，公司名设为空
            print(f"警告: 键 '{key}' 不包含 '---' 分隔符，无法分离公司名和型号。")
            company = ""
            model = key

        # 提取其他字段
        rated_power_raw = value.get("Rated power", "-")
        cut_in_speed_raw = value.get("Cut-in wind speed", "-")
        rated_speed_raw = value.get("Rated wind speed", "-")
        cut_out_speed_raw = value.get("Cut-out wind speed", "-")
        offshore_raw = value.get("Offshore", "-")
        onshore_raw = value.get("Onshore", "-")
        hub_height_raw = value.get("Hub height", "-")

        # 处理 Rated power
        rated_power = None
        if rated_power_raw != "-":
            try:
                # 提取数值和单位
                match = re.match(r"([0-9,.]+)\s*(\w+)", rated_power_raw)
                if match:
                    num_str, unit = match.groups()
                    num_value = float(num_str.replace(',', '')) # 处理千位分隔符
                    if unit.lower() == 'kw':
                        rated_power = num_value / 1000.0  # 转换为MW
                    elif unit.lower() == 'mw':
                        rated_power = num_value
                    else:
                        print(f"未知的功率单位: {unit}，来自: {rated_power_raw}")
                        # 如果单位未知，假设其数值就是MW
                        rated_power = num_value
                else:
                    # 如果没有单位，直接转换数值
                    rated_power = float(rated_power_raw.replace(',', ''))
            except ValueError:
                print(f"无法转换功率值为数字: {rated_power_raw}")
                rated_power = None

        # 处理 Cut-in wind speed
        cut_in_speed = None
        if cut_in_speed_raw != "-":
            try:
                match = re.match(r"([0-9,.]+)\s*m/s", cut_in_speed_raw)
                if match:
                    cut_in_speed = float(match.group(1).replace(',', ''))
                else:
                    # 如果没有单位'm/s'，假设其数值就是m/s
                    cut_in_speed = float(cut_in_speed_raw.replace(',', ''))
            except ValueError:
                print(f"无法转换切入风速值为数字: {cut_in_speed_raw}")
                cut_in_speed = None
        
        cut_out_speed = None
        if cut_out_speed_raw != "-":
            try:
                match = re.match(r"([0-9,.]+)\s*m/s", cut_out_speed_raw)
                if match:
                    cut_out_speed = float(match.group(1).replace(',', ''))
                else:
                    # 如果没有单位'm/s'，假设其数值就是m/s
                    cut_out_speed = float(cut_out_speed_raw.replace(',', ''))
            except ValueError:
                print(f"无法转换切出风速值为数字: {cut_out_speed_raw}")
                cut_out_speed = None

        # 处理 Rated wind speed
        rated_speed = None
        if rated_speed_raw != "-":
            try:
                match = re.match(r"([0-9,.]+)\s*m/s", rated_speed_raw)
                if match:
                    rated_speed = float(match.group(1).replace(',', ''))
                else:
                     # 如果没有单位'm/s'，假设其数值就是m/s
                    rated_speed = float(rated_speed_raw.replace(',', ''))
            except ValueError:
                print(f"无法转换额定风速值为数字: {rated_speed_raw}")
                rated_speed = None

        # 处理 Onshore (根据 Offshore 和 Onshore 的值)
        onshore_value = None
        if onshore_raw != "-" and offshore_raw != "-":
            if onshore_raw.lower() == "yes" and offshore_raw.lower() == "no":
                onshore_value = 1
            elif onshore_raw.lower() == "no" and offshore_raw.lower() == "yes":
                onshore_value = 0
            else:
                # 如果两者都是No或Yes，或者其他组合，可以根据具体需求定义
                # 这里假设如果Onshore是Yes，就认为是1，否则是0，忽略Offshore
                onshore_value = 1 if onshore_raw.lower() == "yes" else 0
        elif onshore_raw != "-": # 如果只有Onshore有值
             onshore_value = 1 if onshore_raw.lower() == "yes" else 0
        # 如果两者都是 "-", 则 onshore_value 保持 None

        # 处理 Hub height
        hub_height = parse_hub_height(hub_height_raw)

        # 构建行数据
        row = {
            "Company": company,
            "Model": model,
            "Rated power": rated_power,
            "Cut-in wind speed": cut_in_speed,
            "Rated wind speed": rated_speed,
            "Cut-out wind speed": cut_out_speed,
            "Onshore": onshore_value,
            "Hub height": hub_height
        }
        processed_rows.append(row)

    print(f"数据处理完成，共处理 {len(processed_rows)} 条记录。")

    # 创建 DataFrame
    df = pd.DataFrame(processed_rows, columns=["Company", "Model", "Rated power", "Cut-in wind speed", "Rated wind speed", "Cut-out wind speed", "Onshore", "Hub height"])

    print(f"正在写入CSV文件: {csv_path}")

    # 四舍五入Hub height并转换为整数类型，允许空值
    df['Hub height'] = df['Hub height'].round().astype('Int64')
    # 保存为CSV
    df.to_csv(csv_path, index=False, na_rep='') # na_rep='' 将None值写入CSV时空字符串

    print("CSV文件写入完成。")

# --- 示例用法 ---


json_to_csv_wind_data(
    r"D:\Data\turbines\wind_turbines_complete.json", 
    r"D:\Data\turbines\wind_turbines.csv"
    )