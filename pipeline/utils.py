
import re
import os
import time
import shutil
import logging
from pathlib import Path
from numbers import Number
from typing import Any, Iterable, Union, Literal, Tuple, List, Optional, Dict

import chardet  
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree
from geopy.distance import distance


logger = logging.getLogger(__name__)



def sanitize_netcdf_attrs(ds: xr.Dataset, label: str) -> xr.Dataset:
    """
    清洗 Dataset 及其变量/坐标的 attrs，使其可以稳定写入 NetCDF。

    在主流程中，这个函数是所有 NetCDF 输出前的最后一道兼容性保护，
    主要服务于单风场结果落盘、city/prov 聚合结果落盘，以及中间筛选结果重写。
    它会把 NetCDF 后端不接受的 attrs 值转换成可序列化形式，避免因为布尔值、None
    或自定义对象导致整个批处理在写文件阶段失败。

    Args:
        ds (xr.Dataset): 待写出的数据集。
        label (str): 日志标识，用于指出当前清洗的是哪一类输出对象。

    Returns:
        xr.Dataset: 已完成 attrs 清洗、可继续传给 to_netcdf 的数据集。
    """
    valid_attr_types = (str, bytes, int, float, np.number, np.ndarray, list, tuple)

    def _sanitize_attrs(attrs: dict, scope: str) -> None:
        for key in list(attrs.keys()):
            value = attrs[key]
            if value is None:
                attrs.pop(key)
                continue
            if isinstance(value, (bool, np.bool_)):
                attrs[key] = int(value)
                continue
            if not isinstance(value, valid_attr_types):
                logger.warning(
                    "Converting attr %s.%s from %s to str for NetCDF compatibility",
                    scope,
                    key,
                    type(value).__name__,
                )
                attrs[key] = str(value)

    _sanitize_attrs(ds.attrs, f"{label}.attrs")
    for var_name in ds.data_vars:
        _sanitize_attrs(ds[var_name].attrs, f"{label}.{var_name}.attrs")
    for coord_name in ds.coords:
        _sanitize_attrs(ds[coord_name].attrs, f"{label}.{coord_name}.attrs")

    return ds


def detect_encoding(file_path: Union[str, Path]) -> str:
    """
    自动检测文本文件编码，并在常见中文编码间做兜底判断。

    在主流程中，这个函数是各类 CSV 读取的最底层入口，支撑 inventory、k/b、
    省市映射、官方统计表等异构来源文件的读取。它的作用不是参与业务计算，
    而是先把输入层稳定下来，避免因为 utf-8、gbk、gb2312 混用而让上层流程中断。
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000字节进行编码检测
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # 如果检测不到编码，尝试常见的几种
        if encoding is None:
            encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as test_file:
                        test_file.read(1000)  # 尝试读取一部分内容
                    return enc
                except UnicodeDecodeError:
                    continue
            return 'utf-8'  # 默认返回utf-8
        
        # 修正一些常见的检测错误
        if encoding.lower() in ['ascii', 'utf-8-sig']:
            return 'utf-8'
        elif encoding.lower() in ['gb2312']:
            return 'gbk'
        else:
            return encoding


def safe_read_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    用自动探测到的编码安全读取 CSV，并在失败时逐步回退到常见编码。

    在主流程中，这是读取外部表格数据的标准入口。无论是单风场阶段读取风场清单、
    还是聚合阶段读取容量修正表、k/b 参数表、官方统计表，都会先经过这里，
    从而把编码问题与业务逻辑解耦，减少主流程里重复写读取兜底代码。
    """
    encoding = detect_encoding(file_path)
    
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except UnicodeDecodeError:
        # 如果检测的编码失败，尝试常见编码
        fallback_encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for enc in fallback_encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                return df
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，使用错误处理
        df = pd.read_csv(file_path, encoding=encoding, errors='replace')
        return df


def normalize_text(value: Any) -> Optional[str]:
    """
    统一清洗文本值，并尽可能修复常见中文乱码。

    在主流程中，这个函数负责把来自 CSV、命令行或 attrs 的名称字段规范化为可比较的文本，
    典型场景包括风场 ID、城市简称、省简称、表头标签等。它降低了编码脏数据、前后空格、
    numpy 标量包装值对匹配逻辑的干扰，是列匹配、名称匹配和省市映射的基础预处理步骤。

    Args:
        value (Any): 原始输入值，可以是字符串、数值、numpy 标量等。

    Returns:
        Optional[str]: 清洗后的字符串；若输入为空或清洗后为空，则返回 None。
    """

    def _cjk_ratio(text: str) -> float:
        if not text:
            return 0.0
        cjk_count = 0
        for ch in text:
            code = ord(ch)
            if 0x4E00 <= code <= 0x9FFF:
                cjk_count += 1
        return cjk_count / len(text)

    def _repair_mojibake(text: str) -> str:
        candidates = [text]
        transforms = [
            ('gbk', 'utf-8'),
            ('gb18030', 'utf-8'),
            ('latin-1', 'utf-8'),
            ('cp1252', 'utf-8'),
        ]

        for src_enc, dst_enc in transforms:
            try:
                repaired = text.encode(src_enc).decode(dst_enc)
                candidates.append(repaired)
            except Exception:
                continue

        best = text
        best_ratio = _cjk_ratio(text)
        for candidate in candidates[1:]:
            ratio = _cjk_ratio(candidate)
            if ratio > best_ratio + 0.2:
                best = candidate
                best_ratio = ratio

        return best

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
    else:
        if hasattr(value, 'item'):
            value = value.item()
        text = str(value).strip()

    if not text:
        return None
    return _repair_mojibake(text)


def normalize_text_set(values: Optional[Union[str, Iterable[str]]]) -> Optional[set[str]]:
    """
    将单个或多个文本值统一规范化，并整理为去重后的集合。

    在主流程中，它主要用于把用户输入的省市筛选条件、名称候选集合等转换成稳定的匹配集合，
    便于后续在 inventory 或映射表中做包含判断。它相当于 normalize_text 的集合版封装。
    """
    if values is None:
        return None

    if isinstance(values, str):
        normalized = normalize_text(values)
        return {normalized} if normalized is not None else None

    result: set[str] = set()
    for item in values:
        normalized = normalize_text(item)
        if normalized is not None:
            result.add(normalized)
    return result or None


def resolve_existing_column(columns: Iterable[str], aliases: List[str], label: str) -> str:
    """
    从一组候选别名中解析出当前表格真实存在的列名。

    在主流程中，不同来源 CSV 的表头经常不一致，例如同一列可能叫“省简称”、
    “prov”或“province_short”。这个函数把这种输入层差异收敛成统一列名，
    让上层工作流只关心业务字段，而不必在每个模块重复写列名分支判断。
    """
    columns_list = list(columns)
    for alias in aliases:
        if alias in columns_list:
            return alias
    raise KeyError(f"无法识别 {label} 列，候选列名: {aliases}，实际列名: {columns_list}")


def sanitize_filename_part(value: str) -> str:
    """
    清洗字符串，使其适合作为文件名或路径片段。

    在主流程中，这个函数主要用于 city/prov 输出目录、CSV 名称和中间结果文件命名，
    防止省市名、参数标签中包含空格或 Windows 非法字符，导致批量落盘失败或路径混乱。
    """
    text = str(value).strip()
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[\\/:*?"<>|]+', '-', text)
    return text


def format_for_filename(value: Any) -> str:
    """
    将数值或文本转换成紧凑、稳定的文件名片段。

    在主流程中，这个函数主要用于把 k/b、period 等参数写入输出文件名。
    它的重点不是通用格式化，而是让相同参数在不同模块里生成一致的命名结果，
    便于批处理产物追踪、结果对比和后续枚举分析。
    """
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def write_netcdf_with_retry(ds: xr.Dataset, output_path: Union[str, Path], retries: int = 4, sleep_seconds: float = 0.2) -> None:
    """
    带重试地写出 NetCDF，优先处理 Windows 下常见的临时文件锁问题。

    在主流程中，这个函数承担所有关键 NetCDF 产物的最终落盘职责，尤其是 batch 聚合、
    多进程并行和频繁覆盖写文件时。它通过短暂重试和 scipy 回退，尽量避免因为系统锁文件
    或默认后端异常而让整个计算批次失败，是输出层的稳健性封装。

    Args:
        ds (xr.Dataset): 待写出的数据集。
        output_path (Union[str, Path]): 输出文件路径。
        retries (int): PermissionError 时的重试次数。
        sleep_seconds (float): 每轮重试的基础等待时间。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.load()
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            ds.to_netcdf(path, mode='w')
            return
        except PermissionError as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(sleep_seconds * (attempt + 1))

    if last_error is not None:
        logger.warning(
            "Default NetCDF write failed after retries, fallback to scipy engine: path=%s, error=%s",
            path,
            last_error,
        )
        ds.to_netcdf(path, mode='w', engine='scipy')
        return


def open_dataset_with_fallback(path: Union[str, Path]) -> xr.Dataset:
    """
    打开 NetCDF 文件，并在默认后端失败时自动回退到 scipy 引擎。

    在主流程中，这个函数用于读取聚合输出、筛选后的中间结果和后处理阶段的省级发电文件。
    它专门兜底 Windows 环境、中文路径或 netCDF4 后端不稳定场景，确保后续 city/prov 聚合、
    结果枚举与统计分析能继续执行，而不是卡在文件读取兼容性上。
    """
    resolved_path = Path(path)
    open_attempts: list[dict[str, str]] = [{}, {'engine': 'scipy'}]
    errors: list[str] = []

    for kwargs in open_attempts:
        try:
            with xr.open_dataset(str(resolved_path), **kwargs) as ds:
                return ds.load()
        except Exception as exc:  # noqa: BLE001 - preserve backend errors for diagnostics
            engine = kwargs.get('engine', 'default')
            errors.append(f"engine={engine}: {exc}")

    raise OSError(f"打开 NetCDF 失败: path={resolved_path}, errors={errors}")


SEASON_ORDER = ("spring", "summer", "autumn", "winter")


def period_key(period_item: Union[str, int], period_mode: Literal['season', 'month']) -> str:
    """
    将季节或月份 period 标准化为稳定的键名。

    在主流程中，这个函数服务于结果字典索引、输出目录命名和批量任务标识，
    保证 month 模式下 1 月与 01 月不会出现命名歧义，也让 season/month 两套周期体系
    在上层调度和持久化时都能使用统一键格式。
    """
    if period_mode == 'season':
        return str(period_item)
    return f"m{int(period_item):02d}"


def resolve_period_items(
    period: Optional[Union[str, int, List[Union[str, int]], Tuple[Union[str, int], ...], Literal['all']]],
    period_mode: Literal['season', 'month'],
    season_order: Tuple[str, ...] = SEASON_ORDER,
) -> List[Union[str, int]]:
    """
    将用户传入的 period 参数解析为经过校验、去重后的具体周期列表。

    在主流程中，这个函数位于调度层最前面，负责把 CLI 或函数调用里传入的
    None、all、单个 season、单个月份或其列表，展开成可直接迭代的任务集合。
    它决定了单风场输出和批量聚合到底按哪些 season/month 执行，是周期任务切分的入口。
    """
    if period_mode not in ('season', 'month'):
        raise ValueError("period_mode 仅支持 'season' 或 'month'")

    if period is None or period == 'all':
        if period_mode == 'season':
            return list(season_order)
        return list(range(1, 13))

    if period_mode == 'season':
        if isinstance(period, str):
            if period not in season_order:
                raise ValueError(f"period_mode='season' 时，period 仅支持 {season_order}、'all' 或 None")
            return [period]

        if isinstance(period, int):
            raise ValueError("period_mode='season' 时，period 不能是月份整数")

        periods: List[Union[str, int]] = []
        for item in period:
            if not isinstance(item, str) or item not in season_order:
                raise ValueError(f"period_mode='season' 时，period 列表元素必须属于 {season_order}，实际: {item}")
            periods.append(item)
    else:
        if isinstance(period, int):
            if 1 <= period <= 12:
                return [period]
            raise ValueError("period_mode='month' 时，period 整数必须在 1-12")

        if isinstance(period, str):
            raise ValueError("period_mode='month' 时，period 不能是季节字符串")

        periods = []
        for item in period:
            if not isinstance(item, int) or not (1 <= item <= 12):
                raise ValueError(f"period_mode='month' 时，period 列表元素必须是 1-12 的整数，实际: {item}")
            periods.append(item)

    if len(periods) == 0:
        raise ValueError('period 列表不能为空')

    unique_periods: List[Union[str, int]] = []
    for item in periods:
        if item not in unique_periods:
            unique_periods.append(item)
    return unique_periods
  



def chinese_to_pinyin(city_name: str) -> str:
    """
    将中文名称转换为适合文件系统使用的拼音字符串。

    在主流程中，它主要用于把城市、省份等中文标签转换成英文路径片段，
    便于生成跨平台、更稳定的输出目录与文件名。

    ⚠️ BUG-2：**缺少 pypinyin 时必须直接报错，不得降级。**
    原实现在 `except ImportError` 里把*每一个*汉字都替换成字面量 `chinese_char`，
    得到一个**常函数**（非单射）：「北京市」「上海市」「广东省」全部变成同一个
    slug ⇒ 城市/省级产物互相覆盖或错配，且*全程无报错*。这属于静默的数据完整性
    事故，比"跑不起来"严重得多，所以现在改为硬失败并给出安装提示。
    """
    try:
        from pypinyin import lazy_pinyin, Style
    except ImportError as exc:
        raise ImportError(
            "缺少依赖 pypinyin（用于把中文省/市名转换成唯一的拼音路径片段）。"
            "缺少它时无法生成唯一 slug，若降级会让不同省/市映射到同一个目录名、"
            "静默覆盖产物，因此这里直接报错。请安装：pip install pypinyin"
            "（已写入 requirements.txt）。"
        ) from exc

    # 转换为拼音，不带声调
    pinyin_list = lazy_pinyin(city_name, style=Style.NORMAL)
    pinyin_name = ''.join(pinyin_list)
    # 移除特殊字符，只保留字母数字下划线
    pinyin_name = re.sub(r'[^\w]', '_', pinyin_name)
    return pinyin_name


def chinese_to_pinyi(text: str) -> str:
    """
    对 chinese_to_pinyin 的兼容封装，并显式区分“山西”与“陕西”。

    在主流程中，这个函数主要用于 province/city 输出命名，尤其是生成省级 NetCDF、
    枚举统计结果和目录层级时。它解决了普通拼音转换会把山西/陕西都写成 shanxi 的冲突，
    从而避免覆盖写文件或后续读取错路径。
    """
    normalized = str(text).strip()
    if not normalized:
        return ""

    if normalized.startswith("陕西"):
        return "shaanxi" + chinese_to_pinyin(normalized[2:])
    if normalized.startswith("山西"):
        return "shanxi" + chinese_to_pinyin(normalized[2:])

    return chinese_to_pinyin(normalized)
    



def ensure_ascending_da_coords(
        da: xr.DataArray, 
        lon_name: str = 'lon', 
        lat_name: str = 'lat'
        ) -> xr.DataArray:
    """
    确保输入 DataArray 的纬度和经度坐标都按升序排列。

    在主流程中，这个函数是空间插值和多源气象数据拼接前的坐标标准化步骤。
    因为 ERA5/CDMet 等数据源可能存在降序纬度，若不先统一顺序，后续最近点搜索、
    sel 截取、重索引和插值结果都可能出现偏移或维度顺序不一致的问题。
    
    Args:
        da (xr.DataArray): 输入的数据数组
        lon_name (str, optional): 经度变量名，默认为 'lon'
        lat_name (str, optional): 纬度变量名，默认为 'lat'
    
    Returns:
        xr.DataArray: 经纬度坐标升序排列后的数据数组
    """
    # 按纬度升序排列
    da = da.sortby(lat_name)
    # 按经度升序排列
    da = da.sortby(lon_name)
    
    return da


def find_nearest_points_from_ds(
        ds: xr.Dataset, 
        target_point: Tuple[Number, Number],
        n: int,
        nearest_method: Literal['binary', 'KDTree'],
        return_distance: bool = True,
        lon_name: str = 'lon',
        lat_name: str = 'lat'
    ) -> Tuple[List[Tuple[Number, Number]], List[Number]]:
    
    """
    在规则经纬网格中寻找最接近目标点的若干网格点。

    在主流程中，这个函数为单风场气象提取提供空间定位能力。unit_farm_workflow
    会先用它锁定风场周边 ERA5/CDMet 网格，再进行子集读取和插值；horizontal_interp
    也会复用它寻找参与插值的邻近网格点。它是“从大气象场收缩到单风场局地窗口”的核心步骤。

    Args:
        ds (xr.Dataset): 包含经纬度与参数信息的数据集
        target_point (Tuple[Number, Number]): 目标点的经纬度坐标，顺序需要确保是 (latitude, longitude)
        n (int): 要考虑的最近点的数量
        nearest_method (Literal['binary', 'KDTree']): 用于寻找最近点的方法。用于控制是否使用快速模式，binary时n必须是平方根数
        return_distance (bool): 是否返回与target_point的距离
        lon_name (str): 数据集中的经度变量名
        lat_name (str): 数据集中的纬度变量名
    
    Returns:
        Tuple[List[Tuple[Number, Number]], List[Number]]: 返回最近点的坐标和与target_point的距离
    """

    logging.debug('注意target_coord要先纬度再经度')
    
    lons = np.sort(ds[lon_name].values)
    lats = np.sort(ds[lat_name].values)

    # 二分查找法, 用于快速找到最近的根号n个点
    def binary_search(
            lst: Union[List[Number], np.ndarray], 
            target: Number, 
            n: int,
            ):
        """
        使用二分法，在lst中查找距离target值最近的n个点
        """
        left, right = 0, len(lst) - 1
        while left < right:
            mid = (left + right) // 2
            if lst[mid] < target:
                left = mid + 1
            else:
                right = mid
        if left == 0:
            return lst[:n]
        if left == len(lst):
            return lst[-n:]
        candidates = lst[max(0, left-n+1):min(len(lst), left+n)]
        return sorted(candidates, key=lambda x: abs(x - target))[:n]

    if nearest_method == 'binary':
        lat, lon = target_point
        m = int(np.sqrt(n))
        assert m ** 2 == n, '使用binary时n必须是平方根数'
        nearest_lats = binary_search(lats, lat, m)
        nearest_lons = binary_search(lons, lon, m)
        nearest_coords = [(lat, lon) for lat in nearest_lats for lon in nearest_lons]

    elif nearest_method == 'KDTree': 
        # 使用KDTree来寻找最近点，n不必是平方根数，但是会更慢
        points = [(lat, lon) for lat in lats for lon in lons]
        tree = KDTree([(lat, lon) for lat in lats for lon in lons])
        _, indices = tree.query([target_point], k=n)
        # 确保 indices[0] 是可迭代的（当 k=1 时返回标量）
        indices_list = np.atleast_1d(indices[0])
        nearest_coords = [points[i] for i in indices_list]

    else:
        raise ValueError('nearest_method 参数值错误')

    if return_distance:
        nearest_distances = \
            [distance(target_point, (lat, lon)).km for lat, lon in nearest_coords]   
        return nearest_coords, nearest_distances
    else:
        return nearest_coords


def load_and_merge_subdatasets(
        nc_paths: List[str],
        target_point: Tuple[Number, Number],
        n_points: Optional[int] = 16,
        nearest_method: Literal['binary', 'KDTree'] = 'binary',
        required_varnames: Optional[List[str]] = ['u10', 'v10', 'u100', 'v100', 't2m', 'sp', 'd2m', 'z'],
        time_name_candidates: Tuple[str, ...] = ('time', 'valid_time', 'date'),
        lat_name_candidates: Tuple[str, ...] = ('lat', 'latitude', 'y'),
        lon_name_candidates: Tuple[str, ...] = ('lon', 'longitude', 'x'),
    ) -> xr.Dataset:
    """
    从多个 NetCDF 文件中提取目标点附近的小范围网格，并合并成统一坐标的数据集。

    在主流程中，这个函数是单风场计算读取 ERA5/CDMet 原始数据的入口之一。
    它负责完成“裁剪局地网格、统一 time/lat/lon 命名、坐标升序化、不同变量文件对齐合并”
    这一整套前处理，让后续风速推算、水平插值、功率曲线计算直接面对干净的输入数据。

    Args:
        nc_paths (List[str]): 多个 nc 文件路径。
        target_point (Tuple[Number, Number]): 目标点 (lat, lon)。
        n_points (Optional[int]): 最近点数量，默认16；为 None 时不做经纬度裁剪，加载完整空间范围。
        nearest_method (Literal['binary', 'KDTree']): 最近点搜索方法；n_points=None 时忽略此参数。
        required_varnames (Optional[List[str]]): 需要提取的变量名列表；
            若为 None，则读取每个文件中的全部 data_vars。
        *_name_candidates (Tuple[str, ...]): 各坐标候选名称。

    Returns:
        xr.Dataset: 合并后的子数据集，坐标统一为 time/lat/lon（若存在）。
    """
    if not nc_paths:
        raise ValueError('nc_paths 不能为空')

    requested_vars = None if required_varnames is None else list(required_varnames)

    def _resolve_coord_name(ds: xr.Dataset, candidates: Tuple[str, ...], coord_label: str) -> str:
        for name in candidates:
            if name in ds.coords or name in ds.dims:
                return name
        raise ValueError(f"无法识别 {coord_label} 坐标名，候选: {candidates}，实际: {list(ds.coords)}")

    def _is_ascending(arr: np.ndarray) -> bool:
        if len(arr) < 2:
            return True
        return bool(np.all(np.diff(arr) >= 0))

    sub_datasets: List[xr.Dataset] = []

    for nc_path in nc_paths:
        with xr.open_dataset(nc_path) as ds:
            # 自动识别坐标名
            lat_name = _resolve_coord_name(ds, lat_name_candidates, 'lat')
            lon_name = _resolve_coord_name(ds, lon_name_candidates, 'lon')
            time_name = None
            for c in time_name_candidates:
                if c in ds.coords or c in ds.dims:
                    time_name = c
                    break

            # 仅保留当前文件中存在的目标变量；若未指定则读取全部 data_vars
            if requested_vars is None:
                available_vars = list(ds.data_vars)
            else:
                available_vars = [v for v in requested_vars if v in ds.data_vars]
            if not available_vars:
                continue

            if n_points is not None:
                # 找目标点周边最近点坐标
                nearest_coords = find_nearest_points_from_ds(
                    ds=ds,
                    target_point=target_point,
                    n=n_points,
                    nearest_method=nearest_method,
                    return_distance=False,
                    lon_name=lon_name,
                    lat_name=lat_name,
                )

                nearest_lats = sorted({lat for lat, _ in nearest_coords})
                nearest_lons = sorted({lon for _, lon in nearest_coords})

                # 根据原始坐标顺序决定 sel 列表顺序
                ds_lats = ds[lat_name].values
                ds_lons = ds[lon_name].values
                if not _is_ascending(ds_lats):
                    nearest_lats = nearest_lats[::-1]
                if not _is_ascending(ds_lons):
                    nearest_lons = nearest_lons[::-1]

                sel_indexer = {lat_name: nearest_lats, lon_name: nearest_lons}
                # 先 sel 截取小子集，再 load 只读入目标格点的数据
                ds_sub = ds[available_vars].sel(sel_indexer).load()
            else:
                # n_points=None：不做经纬度裁剪，加载全部空间范围
                ds_sub = ds[available_vars].load()

        # 丢弃多余的标量/非空间坐标（如 ERA5 附带的 number、expver）
        # ds_sub 已 load 到内存，with 块关闭后继续处理
        # 丢弃多余的标量/非空间坐标（如 ERA5 附带的 number、expver）
        keep_coords = {lat_name, lon_name} | ({time_name} if time_name else set())
        extra_coords = [c for c in ds_sub.coords if c not in keep_coords]
        if extra_coords:
            ds_sub = ds_sub.drop_vars(extra_coords)

        # 坐标名统一到 time/lat/lon
        rename_map = {}
        if lat_name != 'lat':
            rename_map[lat_name] = 'lat'
        if lon_name != 'lon':
            rename_map[lon_name] = 'lon'
        if time_name is not None and time_name != 'time':
            rename_map[time_name] = 'time'
        if rename_map:
            ds_sub = ds_sub.rename(rename_map)

        # 对每个变量执行经纬度升序化
        sorted_vars = {}
        for var_name, da in ds_sub.data_vars.items():
            if 'lat' in da.coords and 'lon' in da.coords:
                sorted_vars[var_name] = ensure_ascending_da_coords(da, lon_name='lon', lat_name='lat')
            else:
                sorted_vars[var_name] = da
        ds_sub = xr.Dataset(sorted_vars)

        sub_datasets.append(ds_sub)

    if not sub_datasets:
        raise ValueError('未从输入 nc_paths 中提取到任何目标变量，请检查路径和变量名')

    # 构建统一坐标
    lat_union = sorted({float(v) for ds in sub_datasets if 'lat' in ds.coords for v in ds['lat'].values})
    lon_union = sorted({float(v) for ds in sub_datasets if 'lon' in ds.coords for v in ds['lon'].values})
    has_time = any('time' in ds.coords for ds in sub_datasets)
    if has_time:
        time_union = np.unique(np.concatenate([ds['time'].values for ds in sub_datasets if 'time' in ds.coords]))
    else:
        time_union = None

    # 统一重索引并合并
    aligned_datasets = []
    for ds in sub_datasets:
        ds_aligned = ds
        if 'lat' in ds_aligned.coords:
            ds_aligned = ds_aligned.reindex(lat=lat_union)
        if 'lon' in ds_aligned.coords:
            ds_aligned = ds_aligned.reindex(lon=lon_union)
        if has_time and 'time' in ds_aligned.coords:
            ds_aligned = ds_aligned.reindex(time=time_union)
        aligned_datasets.append(ds_aligned)

    merged = xr.Dataset()
    for ds in aligned_datasets:
        for var_name, da in ds.data_vars.items():
            if var_name in merged:
                merged[var_name] = merged[var_name].combine_first(da)
            else:
                merged[var_name] = da

    merged.attrs.update({
        'source_file_count': len(nc_paths),
        'merged_variable_count': len(merged.data_vars),
        'target_lat': float(target_point[0]),
        'target_lon': float(target_point[1]),
        'nearest_method': nearest_method,
        'n_points': int(n_points) if n_points is not None else 'all',
        'coord_standard': 'time/lat/lon',
    })
    return merged


def concat_das(
    dataarrays: List[xr.DataArray],
    time_name: str = 'time'
    ) -> xr.DataArray:
    """
    将多个按时间分段的数据对象沿时间轴拼接并排序。

    在主流程中，这个函数主要用于把不同年份、不同文件片段或不同阶段生成的序列重新接回一条
    连续时间轴，服务于单风场风速、容量因子和发电量构造。它是时间维“收口”的轻量工具函数。

    Args:
        dataarrays (List[xr.DataArray]): 待拼接的 DataArray 列表。
        time_name (str): 时间维名称，默认 'time'。

    Returns:
        xr.DataArray: 若输入为空列表，返回空 DataArray；
            否则返回按 time_name 维 concat 后并 sortby(time_name) 的结果。
    """
    da_concat = xr.DataArray()
    if dataarrays:
        first_obj = dataarrays[0]
        if isinstance(first_obj, xr.Dataset):
            # Keep current behavior explicit across xarray versions.
            da_concat = xr.concat(dataarrays, dim=time_name, data_vars='all').sortby(time_name)
        else:
            da_concat = xr.concat(dataarrays, dim=time_name).sortby(time_name)
    return da_concat


# ---------------------------------------------------------------------------
# BUG-7：可加量 vs 强度量
# ---------------------------------------------------------------------------
# `aggr_gen_ds` 是**求和**聚合器（`_sum_members` 用 fill_value=0.0 再 sum）。
# 只能喂"可加"（extensive）变量。`cf`（容量因子）是**强度量**，逐场相加物理上
# 无意义（两座 0.3 相加得 0.6），且原实现**静默**给出这种结果。
# 本集合显式声明"禁止进入求和聚合器"的强度量；`cf` 的正确聚合口径是容量加权
#     sum(cf_i * cap_i * h_i) / sum(cap_i * h_i)   （= sum(gen) / sum(cap * h)）
# 与 add_city_capacity_factors.py / add_province_capacity_factors.py 一致（并带 [0,1] 护栏）。
_INTENSIVE_AGG_VARS = frozenset({'cf'})


def aggr_gen_ds(
    ds_list: List[Union[xr.Dataset, str, Path]],
    aggr_vars: List[str] = ['gen', 'monthly_gen'],
    time_name: str = 'time',
    mode: Literal['farm2city', 'city2prov'] = 'farm2city',
    onshore_flag_attr: str = 'onshore_flag',
) -> xr.Dataset:
    """
    将多份发电量 Dataset 在统一时间轴上对齐后求和聚合。

    在主流程中，这是 farm->city 和 city->prov 两层发电量汇总的核心实现。
    aggr_farm_aux 会把筛选后的单风场或城市结果交给它，生成 total/onshore/offshore
    三类发电序列。它承担了跨文件时间对齐、缺失补零、不同层级变量命名统一等关键工作。

    典型用法：
    - farm 级 -> city 级
    - city 级 -> province 级

    规则：
    1. 每个变量必须包含 time_name 维。
    2. 除 time_name 外，其他维度长度必须为 1（允许形状如 (1,1,n)、(n,1)、(n,)）。
    3. 时间轴取并集，按时间升序排列；缺失时段补 0 后再累加。
    4. 输出仅保留 time_name 坐标。

    ⚠️ BUG-7：本函数只能聚合**可加量**（`gen`/`monthly_gen`）。**强度量**（`cf`）
    一律拒绝（见 `_INTENSIVE_AGG_VARS`）：CF 逐场相加会得到 >1 的物理无意义值，
    而原实现是**静默**这么做的。CF 要按容量加权，口径见 `_INTENSIVE_AGG_VARS` 上方注释。

    mode='farm2city':
    - 输入通常是 farm 级 ds。
    - 从 ds.attrs[onshore_flag_attr] 判断陆上/海上：1 为 onshore，0 为 offshore。
    - 对每个 aggr_var 输出 total_ 前缀；若对应类别存在，则额外输出 onshore_/offshore_ 前缀。

    mode='city2prov':
    - 输入通常是 city 级 ds。
    - 基于每个 aggr_var 直接聚合 onshore_/offshore_/total_ 三类前缀变量。
    - 若某类变量在输入中缺失，则在结果中补零。

    Args:
        ds_list: 待累加的数据列表。元素可为 xr.Dataset，
            也可为 ds 文件路径（str/Path，会自动读取为 xr.Dataset）。
        aggr_vars: 需要聚合的变量名列表。
        time_name: 时间维名称。
        mode: 聚合模式，支持 'farm2city' 和 'city2prov'。
        onshore_flag_attr: farm2city 模式下用于判断陆上/海上的 attrs 键名。

    Returns:
        xr.Dataset: 聚合后的 Dataset。
    """
    if not ds_list:
        raise ValueError('ds_list 不能为空')
    if not aggr_vars:
        raise ValueError('aggr_vars 不能为空')
    if mode not in ('farm2city', 'city2prov'):
        raise ValueError("mode 只能为 'farm2city' 或 'city2prov'")

    # ⚠️ BUG-7：强度量（cf）不得进入求和聚合器 —— 必须硬失败而不是静默给出错值
    intensive_requested = sorted(v for v in aggr_vars if v in _INTENSIVE_AGG_VARS)
    if intensive_requested:
        raise ValueError(
            "aggr_gen_ds 只能聚合**可加量**（如 gen/monthly_gen），"
            "但收到强度量 %s。CF 是容量因子（强度量），逐场相加会得到 >1 的物理无意义值，"
            "因此这里直接拒绝（原实现是静默相加）。CF 的正确聚合口径是容量加权："
            "sum(cf_i*cap_i*h_i) / sum(cap_i*h_i)，等价于 sum(gen)/sum(cap*h)，"
            "与 add_city_capacity_factors.py / add_province_capacity_factors.py 一致"
            "（并带 [0,1] 护栏）。容量信息在输入 attrs 里：'farm_capacity_mw'、'unit_step_hours'。"
            % (intensive_requested,)
        )

    normalized_ds_list: List[xr.Dataset] = []
    for i, item in enumerate(ds_list):
        if isinstance(item, xr.Dataset):
            normalized_ds_list.append(item)
            continue

        if isinstance(item, (str, Path)):
            ds_path = Path(item)
            if not ds_path.exists() or not ds_path.is_file():
                raise ValueError(f"第{i}个 ds 路径无效: {ds_path}")
            try:
                with xr.open_dataset(ds_path) as ds_loaded:
                    normalized_ds_list.append(ds_loaded.load())
            except Exception as exc:
                raise ValueError(f"第{i}个 ds 路径读取失败: {ds_path}, error={exc}")
            continue

        raise TypeError(f"ds_list 第{i}个元素类型不支持: {type(item)}")

    def _extract_1d_series(ds: xr.Dataset, var_name: str, ds_idx: int) -> xr.DataArray:
        if (time_name not in ds.coords) and (time_name not in ds.dims):
            raise ValueError(f"第{ds_idx}个 ds 缺少时间坐标/维度: {time_name}")

        da = ds[var_name]
        if time_name not in da.dims:
            raise ValueError(f"变量 {var_name} 在第{ds_idx}个 ds 中不包含时间维: {time_name}")

        non_time_dims = [d for d in da.dims if d != time_name]
        invalid_dims = [d for d in non_time_dims if da.sizes[d] != 1]
        if invalid_dims:
            raise ValueError(
                f"变量 {var_name} 在第{ds_idx}个 ds 中存在非时间且长度不为1的维度: {invalid_dims}, "
                f"shape={tuple(da.shape)}"
            )

        da_1d = da.squeeze(drop=True)
        if set(da_1d.dims) != {time_name}:
            raise ValueError(
                f"变量 {var_name} 在第{ds_idx}个 ds squeeze 后仍不是纯时间序列, dims={da_1d.dims}"
            )

        da_1d = da_1d.sortby(time_name)
        time_index = pd.Index(da_1d[time_name].values)
        if time_index.has_duplicates:
            raise ValueError(f"变量 {var_name} 在第{ds_idx}个 ds 的时间坐标存在重复值")

        return da_1d.astype(float)

    def _sum_members(da_members: List[xr.DataArray], result_name: str) -> xr.DataArray:
        union_time = np.unique(np.concatenate([da[time_name].values for da in da_members]))
        aligned = [da.reindex({time_name: union_time}, fill_value=0.0) for da in da_members]
        summed = xr.concat(aligned, dim='__member__').sum(dim='__member__', skipna=True)
        return summed.rename(result_name)

    out_vars: dict[str, xr.DataArray] = {}

    if mode == 'farm2city':
        for base_var in aggr_vars:
            total_members: List[xr.DataArray] = []
            onshore_members: List[xr.DataArray] = []
            offshore_members: List[xr.DataArray] = []

            for i, ds in enumerate(normalized_ds_list):
                if base_var not in ds.data_vars:
                    continue

                da_1d = _extract_1d_series(ds, base_var, i)
                total_members.append(da_1d)

                if onshore_flag_attr not in ds.attrs:
                    raise ValueError(f"第{i}个 ds 缺少 attrs['{onshore_flag_attr}']")

                flag = ds.attrs[onshore_flag_attr]
                try:
                    flag = int(flag)
                except Exception:
                    raise ValueError(f"第{i}个 ds 的 attrs['{onshore_flag_attr}'] 无法转为 0/1: {flag}")

                if flag == 1:
                    onshore_members.append(da_1d)
                elif flag == 0:
                    offshore_members.append(da_1d)
                else:
                    raise ValueError(f"第{i}个 ds 的 attrs['{onshore_flag_attr}'] 只能为 0 或 1，实际为 {flag}")

            if not total_members:
                continue

            out_vars[f'total_{base_var}'] = _sum_members(total_members, f'total_{base_var}')
            if onshore_members:
                out_vars[f'onshore_{base_var}'] = _sum_members(onshore_members, f'onshore_{base_var}')
            if offshore_members:
                out_vars[f'offshore_{base_var}'] = _sum_members(offshore_members, f'offshore_{base_var}')

    else:  # city2prov
        all_time_arrays: List[np.ndarray] = []
        for i, ds in enumerate(normalized_ds_list):
            if (time_name not in ds.coords) and (time_name not in ds.dims):
                raise ValueError(f"第{i}个 ds 缺少时间坐标/维度: {time_name}")
            all_time_arrays.append(np.asarray(ds[time_name].values))

        union_time = np.unique(np.concatenate(all_time_arrays))

        for base_var in aggr_vars:
            for prefix in ('onshore_', 'offshore_', 'total_'):
                full_var = f'{prefix}{base_var}'
                da_members: List[xr.DataArray] = []

                for i, ds in enumerate(normalized_ds_list):
                    if full_var in ds.data_vars:
                        da_members.append(_extract_1d_series(ds, full_var, i))
                    else:
                        zero_da = xr.DataArray(
                            np.zeros(len(union_time), dtype=float),
                            coords={time_name: union_time},
                            dims=(time_name,),
                            name=full_var,
                        )
                        da_members.append(zero_da)

                out_vars[full_var] = _sum_members(da_members, full_var)

    if not out_vars:
        raise ValueError(f'aggr_vars={aggr_vars} 在输入 ds_list 中均不存在')

    out_ds = xr.Dataset(out_vars).sortby(time_name)

    # 只保留时间坐标
    drop_coords = [c for c in out_ds.coords if c != time_name]
    if drop_coords:
        out_ds = out_ds.drop_vars(drop_coords)

    return out_ds


def adjust_da_by_time(
    gen_da: xr.DataArray,
    adjust_info: Optional[Dict[str, float]],
    time_name: str = 'time',
) -> xr.DataArray:
    """
    按指定年份、月份或年月组合，对时间序列分段乘以调整系数。

    在主流程中，这个函数是各种“后校正”逻辑的统一执行器。无论是单风场发电修正、
    月度/年度统计对齐，还是 city/prov 聚合后的再调整，最终都会转成 adjust_info
    交给这里逐段生效。它把时间条件解析与数值放缩解耦，便于上层模块只负责构造修正规则。

    支持的 key 格式：
    - "2000年"      : 某一年
    - "1月"         : 所有年份中的某个月
    - "2012年12月"  : 某年某月

    字典中的项目按迭代顺序逐项处理；若时间段重叠，会按顺序累计生效。

    Args:
        gen_da (xr.DataArray): 输入发电量（或容量因子）DataArray。
        adjust_info (Optional[Dict[str, float]]): 调整信息字典，value 为乘法因子。
        time_name (str): 时间维名称，默认 'time'。

    Returns:
        xr.DataArray: 调整后的 DataArray。
    """
    if adjust_info is None:
        return gen_da

    if time_name not in gen_da.dims:
        raise ValueError(f"gen_da 必须包含时间维度: {time_name}")

    years = gen_da[time_name].dt.year
    months = gen_da[time_name].dt.month

    result = gen_da.copy(deep=True)

    def _parse_single_key_mask(key_str: str):
        """将单个时间 key 字符串解析为布尔 mask。"""
        key_str = key_str.strip()
        match_year_month = re.fullmatch(r'(\d{4})年(\d{1,2})月', key_str)
        match_year = re.fullmatch(r'(\d{4})年', key_str)
        match_month = re.fullmatch(r'(\d{1,2})月', key_str)

        if match_year_month is not None:
            year_value = int(match_year_month.group(1))
            month_value = int(match_year_month.group(2))
            if not 1 <= month_value <= 12:
                raise ValueError(f"月份必须在 1-12 之间，得到: {month_value} (key={key_str})")
            return (years == year_value) & (months == month_value)
        elif match_year is not None:
            return years == int(match_year.group(1))
        elif match_month is not None:
            month_value = int(match_month.group(1))
            if not 1 <= month_value <= 12:
                raise ValueError(f"月份必须在 1-12 之间，得到: {month_value} (key={key_str})")
            return months == month_value
        else:
            raise ValueError(
                f"不支持的 adjust_info key 格式: {key_str}，"
                "仅支持 '2000年'、'1月'、'2012年12月'"
            )

    for key, factor in adjust_info.items():
        factor_float = float(factor)

        if isinstance(key, tuple):
            if not key:
                raise ValueError("adjust_info 的 tuple key 不能为空")
            mask = _parse_single_key_mask(str(key[0]))
            for sub_key in key[1:]:
                mask = mask | _parse_single_key_mask(str(sub_key))
        else:
            mask = _parse_single_key_mask(str(key))

        result = xr.where(mask, result * factor_float, result)
        result.attrs.update(gen_da.attrs)  # xr.where 会丢失 attrs，此处恢复

    if gen_da.name is not None:
        result = result.rename(gen_da.name)
    result.attrs.update(gen_da.attrs)
    result.attrs['generation_adjust_info'] = str(adjust_info)

    return result


def concat_w_ds(
    ds_list: List[Union[xr.Dataset, str, Path]],
    concat_vars: List[str] = ['farm_era5_w', 'farm_era5_cdmet_w'],
    time_name: str = 'time',
    output_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    将多个风速 Dataset 中的一维时间序列按输入顺序首尾拼接，并整理成表格。

    在主流程中，这个函数主要服务于 aggr_farm 的风速对比输出，
    把同一省份下多个风场的 w_compare 结果串接成一个 CSV，供后续 seasonal 风速检验、
    误差统计或人工检查使用。它不是做空间聚合，而是做批量结果的时序拼接与导出。

    规则：
    1. ds_list 元素可为 xr.Dataset 或文件路径（str/Path，会自动读取）。
    2. concat_vars 中每个变量必须在每个 ds 中存在。
    3. 每个变量必须包含 time_name 维，且除 time_name 外其他维度长度必须为 1。
       （允许形状如 (1,1,n)、(n,1)、(n,)）。
    4. 按 ds_list 原始顺序逐个拼接：
       同名变量与同名变量拼接，如 farm_era5_w -> farm_era5_w。
    5. 输出 DataFrame 的列名为 concat_vars。

    Args:
        ds_list: 数据列表（Dataset 或路径）。
        concat_vars: 待拼接变量名列表。
        time_name: 时间维名称。
        output_csv_path: 输出 CSV 路径；为 None 时不写文件。

    Returns:
        pd.DataFrame: 拼接后的表，每个变量一列。
    """
    if not ds_list:
        raise ValueError('ds_list 不能为空')
    if not concat_vars:
        raise ValueError('concat_vars 不能为空')

    normalized_ds_list: List[xr.Dataset] = []
    for i, item in enumerate(ds_list):
        if isinstance(item, xr.Dataset):
            normalized_ds_list.append(item)
            continue

        if isinstance(item, (str, Path)):
            ds_path = Path(item)
            if not ds_path.exists() or not ds_path.is_file():
                raise ValueError(f"第{i}个 ds 路径无效: {ds_path}")
            try:
                with xr.open_dataset(ds_path) as ds_loaded:
                    normalized_ds_list.append(ds_loaded.load())
            except Exception as exc:
                raise ValueError(f"第{i}个 ds 路径读取失败: {ds_path}, error={exc}")
            continue

        raise TypeError(f"ds_list 第{i}个元素类型不支持: {type(item)}")

    concatenated: dict[str, List[np.ndarray]] = {v: [] for v in concat_vars}

    for i, ds in enumerate(normalized_ds_list):
        for var_name in concat_vars:
            if var_name not in ds.data_vars:
                raise ValueError(f"第{i}个 ds 缺少变量: {var_name}")

            if (time_name not in ds.coords) and (time_name not in ds.dims):
                raise ValueError(f"第{i}个 ds 缺少时间坐标/维度: {time_name}")

            da = ds[var_name]
            if time_name not in da.dims:
                raise ValueError(f"变量 {var_name} 在第{i}个 ds 中不包含时间维: {time_name}")

            non_time_dims = [d for d in da.dims if d != time_name]
            invalid_dims = [d for d in non_time_dims if da.sizes[d] != 1]
            if invalid_dims:
                raise ValueError(
                    f"变量 {var_name} 在第{i}个 ds 中存在非时间且长度不为1的维度: {invalid_dims}, "
                    f"shape={tuple(da.shape)}"
                )

            da_1d = da.squeeze(drop=True)
            if set(da_1d.dims) != {time_name}:
                raise ValueError(
                    f"变量 {var_name} 在第{i}个 ds squeeze 后仍不是纯时间序列, dims={da_1d.dims}"
                )

            # 在单个 ds 内按时间升序，之后再按 ds 顺序首尾拼接
            da_1d = da_1d.sortby(time_name)
            concatenated[var_name].append(np.asarray(da_1d.values).reshape(-1))

    out_data: dict[str, np.ndarray] = {}
    for var_name in concat_vars:
        parts = concatenated[var_name]
        if not parts:
            raise ValueError(f"变量 {var_name} 无可拼接数据")
        out_data[var_name] = np.concatenate(parts)

    lengths = [len(out_data[v]) for v in concat_vars]
    if len(set(lengths)) != 1:
        raise ValueError(f"拼接后各变量长度不一致: {dict(zip(concat_vars, lengths))}")

    out_df = pd.DataFrame(out_data)

    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    return out_df


def filter_dataarray_by_period(
    da: xr.DataArray,
    period: Union[str, int, None] = None,
    time_name: str = 'time',
    drop: bool = True,
) -> xr.DataArray:
    """
    按 season 或 month 对单个 DataArray 进行时间筛选。

    在主流程中，这个函数常用于把全年序列切成某个季节或月份，
    为单风场模拟、对比分析和省级统计提供统一的周期过滤行为。
    它是 period 语义真正落到时间坐标上的基础工具。

    支持以下 period 取值：
    - None: 不筛选，保留全部时间；
    - str: 季节，支持 'spring'/'summer'/'autumn'/'winter'（不区分大小写）；
    - int: 月份，范围 1-12。

    Args:
        da (xr.DataArray): 待筛选的数据。
        period (Union[str, int, None]): 时间段筛选条件。
        time_name (str): 时间维度名称，默认 'time'。
        drop (bool): 是否丢弃不满足条件的时间点，默认 True。

    Returns:
        xr.DataArray: 按 period 筛选后的 DataArray。
    """
    if time_name not in da.dims:
        raise ValueError(f"DataArray 必须包含时间维度: {time_name}")

    months = da[time_name].dt.month

    if period is None:
        scaling_mask = xr.ones_like(months, dtype=bool)
    elif isinstance(period, str):
        period_lower = period.lower()
        if period_lower == 'spring':
            scaling_mask = months.isin([3, 4, 5])
        elif period_lower == 'summer':
            scaling_mask = months.isin([6, 7, 8])
        elif period_lower == 'autumn':
            scaling_mask = months.isin([9, 10, 11])
        elif period_lower == 'winter':
            scaling_mask = months.isin([12, 1, 2])
        else:
            raise ValueError(
                f"Invalid season: {period}. "
                "Use 'spring', 'summer', 'autumn', 'winter', or an integer 1-12."
            )
    elif isinstance(period, int):
        if 1 <= period <= 12:
            scaling_mask = months == period
        else:
            raise ValueError(f"Invalid month: {period}. Use an integer between 1 and 12.")
    else:
        raise ValueError(
            f"Invalid period type: {type(period)}. "
            "Use a string (season) or integer (month), or None."
        )

    filtered_da = da.where(scaling_mask, drop=drop)
    filtered_da.attrs.update(da.attrs)
    return filtered_da


def filter_xr_by_period(
    xobj: Union[xr.DataArray, xr.Dataset],
    period: Union[str, int, None] = None,
    time_name: str = 'time',
    drop: bool = True,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    按 season 或 month 对 DataArray/Dataset 进行整体时间筛选。

    在主流程中，它承担比 filter_dataarray_by_period 更通用的周期过滤职责，
    既可处理单变量，也可整体处理多变量 Dataset，用于保留某一时段的完整输入或输出对象。
    这对于按季节跑单风场、按月份重写聚合结果尤其重要。

    与 filter_dataarray_by_period 的区别在于：
    - 当输入为 Dataset 时，会基于统一的时间坐标整体筛选；
    - 默认 drop=True，会真正缩短时间轴，而不是保留全年时间后填充 NaN。

    支持以下 period 取值：
    - None: 不筛选，保留全部时间；
    - str: 季节，支持 'spring'/'summer'/'autumn'/'winter'（不区分大小写）；
    - int: 月份，范围 1-12。

    Args:
        xobj (Union[xr.DataArray, xr.Dataset]): 待筛选对象。
        period (Union[str, int, None]): 时间段筛选条件。
        time_name (str): 时间维度名称，默认 'time'。
        drop (bool):
            - True: 仅保留满足条件的时间点，时间轴会缩短；
            - False: 保留原时间轴，不满足条件的位置填充为 NaN。

    Returns:
        Union[xr.DataArray, xr.Dataset]: 按 period 筛选后的对象。
    """
    if time_name not in xobj.dims:
        raise ValueError(f"对象必须包含时间维度: {time_name}")

    time_coord = xobj[time_name]
    months = time_coord.dt.month

    if period is None:
        scaling_mask = xr.ones_like(months, dtype=bool)
    elif isinstance(period, str):
        period_lower = period.lower()
        if period_lower == 'spring':
            scaling_mask = months.isin([3, 4, 5])
        elif period_lower == 'summer':
            scaling_mask = months.isin([6, 7, 8])
        elif period_lower == 'autumn':
            scaling_mask = months.isin([9, 10, 11])
        elif period_lower == 'winter':
            scaling_mask = months.isin([12, 1, 2])
        else:
            raise ValueError(
                f"Invalid season: {period}. "
                "Use 'spring', 'summer', 'autumn', 'winter', or an integer 1-12."
            )
    elif isinstance(period, int):
        if 1 <= period <= 12:
            scaling_mask = months == period
        else:
            raise ValueError(f"Invalid month: {period}. Use an integer between 1 and 12.")
    else:
        raise ValueError(
            f"Invalid period type: {type(period)}. "
            "Use a string (season) or integer (month), or None."
        )

    if drop:
        selected_time = time_coord.where(scaling_mask, drop=True)
        filtered = xobj.sel({time_name: selected_time.values})
    else:
        filtered = xobj.where(scaling_mask, drop=False)

    filtered.attrs.update(xobj.attrs)
    return filtered


def align_da_to_reference_year_range(
    source_da: xr.DataArray,
    reference_da: xr.DataArray,
    time_name: str = 'time',
) -> xr.DataArray:
    """
    将一个时间序列按参考序列的年份范围裁剪到相同起止年。

    在主流程中，这个函数主要用于 ERA5 与 CDMet 对比前的时间范围对齐。
    因为两者可能覆盖年份不同、时间分辨率不同，但在 compare 输出里需要先保证年份窗口一致，
    它负责完成这种“按年份取交集”的预处理，避免后续对比受边界年份影响。

    该函数只依据年份对齐，不做精确时间戳交集。
    例如 source_da 为小时序列、reference_da 为日序列时，
    若 reference_da 年份范围为 2012-2018，则返回 source_da 中 2012-2018 年的全部时刻。

    Args:
        source_da (xr.DataArray): 待裁剪序列。
        reference_da (xr.DataArray): 参考序列，仅用于提取年份范围。
        time_name (str): 时间维名称，默认 'time'。

    Returns:
        xr.DataArray: 年份范围对齐后的 source_da。
    """
    if time_name not in source_da.dims:
        raise ValueError(f"source_da 必须包含时间维度: {time_name}")
    if time_name not in reference_da.dims:
        raise ValueError(f"reference_da 必须包含时间维度: {time_name}")

    if source_da.sizes[time_name] == 0:
        return source_da
    if reference_da.sizes[time_name] == 0:
        raise ValueError("reference_da 时间维为空，无法提取年份范围")

    ref_years = reference_da[time_name].dt.year
    ref_start_year = int(ref_years.min().item())
    ref_end_year = int(ref_years.max().item())

    source_years = source_da[time_name].dt.year
    mask = (source_years >= ref_start_year) & (source_years <= ref_end_year)
    aligned = source_da.where(mask, drop=True)

    aligned.attrs.update(source_da.attrs)
    aligned.attrs.update({
        'year_range_aligned': True,
        'year_range_align_source': reference_da.name if reference_da.name is not None else 'reference_da',
        'year_range_align_start_year': ref_start_year,
        'year_range_align_end_year': ref_end_year,
    })

    return aligned


def hourly_to_monthly(
    hourly_da: xr.DataArray,
    time_name: str = 'time',
    agg_method: str = 'sum',
) -> xr.DataArray:
    """
    将小时尺度序列按月聚合成月度序列。

    在主流程中，这个函数把单风场的 hourly generation 压缩成 monthly_gen，
    供后续 city/prov 聚合、官方月报对比和省级发电枚举使用。它是从高频模拟结果
    过渡到统计口径结果的重要转换步骤。

    Args:
        hourly_da (xr.DataArray): 小时序列 DataArray。
        time_name (str): 时间维名称，默认 'time'。
        agg_method (str): 聚合方式，支持
            'sum'/'mean'/'average'/'avg'/'max'/'min'/'median'。

    Returns:
        xr.DataArray: 月度聚合后的 DataArray（时间坐标为月末）。
    """
    if time_name not in hourly_da.dims:
        raise ValueError(f"hourly_da 必须包含时间维度: {time_name}")

    if hourly_da.sizes[time_name] == 0:
        return hourly_da

    agg_key = agg_method.lower()
    resampler = hourly_da.resample({time_name: 'ME'})

    if agg_key == 'sum':
        monthly_da = resampler.sum(skipna=True)
    elif agg_key in ('mean', 'average', 'avg'):
        monthly_da = resampler.mean(skipna=True)
    elif agg_key == 'max':
        monthly_da = resampler.max(skipna=True)
    elif agg_key == 'min':
        monthly_da = resampler.min(skipna=True)
    elif agg_key == 'median':
        monthly_da = resampler.median(skipna=True)
    else:
        raise ValueError(
            f"不支持的 agg_method: {agg_method}，"
            "请使用 'sum'/'mean'/'average'/'avg'/'max'/'min'/'median'"
        )

    if hourly_da.name is not None:
        monthly_da = monthly_da.rename(hourly_da.name)
    monthly_da.attrs.update(hourly_da.attrs)
    monthly_da.attrs.update({
        'monthly_aggregation_applied': 1,
        'monthly_aggregation_method': agg_key,
        'monthly_resample_rule': 'ME',
        'monthly_source_time_dim': time_name,
    })

    return monthly_da
 


def adjust_hourly_with_daily(
    hourly_da: xr.DataArray,
    daily_da: Optional[xr.DataArray] = None,
    time_name: str = 'time',
    daily_resample_rule: str = '1D',
) -> xr.DataArray:
    """
    用日尺度参考序列修正小时序列，使其逐日均值与参考值对齐。

    在主流程中，这个函数主要用于把 ERA5 推算得到的小时风速序列，
    按 CDMet 日尺度序列进行校正。它保留小时内变化形状，只调整每天的整体尺度，
    是单风场风速比较链路中连接“再分析数据”和“观测/校准数据”的关键桥梁。

    原理：
    1. 将 hourly_da 按日求均值得到 hourly_daily_mean。
    2. 与 daily_da 在共同日期对齐，计算每日修正系数：
       correction_daily = daily_da / hourly_daily_mean。
    3. 将 correction_daily 映射回每个小时并乘回 hourly_da。

    Args:
        hourly_da (xr.DataArray): 小时序列（待修正）。
        daily_da (xr.DataArray): 日序列（目标参考）。
        time_name (str): 时间维名称，默认 'time'。
        daily_resample_rule (str): 小时转日均的重采样规则，默认 '1D'。

    Returns:
        xr.DataArray: 修正后的小时序列（完整保留原 hourly_da 时间范围）。
    """

    if daily_da is None:
        return hourly_da
    
    if time_name not in hourly_da.dims:
        raise ValueError(f"hourly_da 必须包含时间维度: {time_name}")
    if time_name not in daily_da.dims:
        raise ValueError(f"daily_da 必须包含时间维度: {time_name}")

    def _to_time_series(da: xr.DataArray, label: str) -> xr.DataArray:
        """将仅含单点空间维的 DataArray 压缩为纯时间序列。"""
        non_time_dims = [d for d in da.dims if d != time_name]
        invalid_dims = [d for d in non_time_dims if da.sizes[d] != 1]
        if invalid_dims:
            raise ValueError(
                f"{label} 存在非时间且长度不为1的维度: {invalid_dims}, "
                f"shape={tuple(da.shape)}"
            )

        da_1d = da.squeeze(drop=True)
        if set(da_1d.dims) != {time_name}:
            raise ValueError(
                f"{label} squeeze 后不是纯时间序列, dims={da_1d.dims}"
            )

        return da_1d.sortby(time_name)

    hourly_series = _to_time_series(hourly_da, 'hourly_da')
    daily_series = _to_time_series(daily_da, 'daily_da')

    hourly_daily_mean = hourly_series.resample({time_name: daily_resample_rule}).mean()

    common_time = np.intersect1d(
        hourly_daily_mean[time_name].values,
        daily_series[time_name].values,
    )
    if len(common_time) > 0:
        hourly_daily_aligned = hourly_daily_mean.sel({time_name: common_time})
        daily_aligned = daily_series.sel({time_name: common_time})

        # 对分母为 0 的情况保留 NaN，避免 inf 污染。
        correction_daily = xr.where(
            np.abs(hourly_daily_aligned) > 0,
            daily_aligned / hourly_daily_aligned,
            np.nan,
        )
    else:
        # 无重叠日时，构造空 daily 系数，后续统一按 fillna(1.0) 处理为“不修正”。
        correction_daily = hourly_daily_mean.isel({time_name: slice(0, 0)})

    # 将 daily 系数按“天”映射回每个小时；无系数的小时填充为 1.0（保持原值）。
    hourly_days = hourly_series[time_name].dt.floor('D')
    correction_hourly = correction_daily.reindex({time_name: hourly_days})
    correction_hourly = correction_hourly.assign_coords({time_name: hourly_series[time_name]})
    correction_hourly = correction_hourly.fillna(1.0)

    # correction_hourly 仅按 time 维广播到 hourly_da 的原始空间维。
    corrected = hourly_da * correction_hourly

    if hourly_da.name is not None:
        corrected = corrected.rename(hourly_da.name)
    corrected.attrs.update(hourly_da.attrs)
    corrected.attrs.update({
        'daily_adjustment_applied': True,
        'daily_adjustment_source_name': daily_da.name if daily_da.name is not None else 'daily_da',
        'daily_adjustment_rule': daily_resample_rule,
        'daily_adjustment_common_days': int(len(common_time)),
        'daily_adjustment_nonoverlap_policy': 'keep_original',
        'daily_adjustment_formula': 'corrected_hourly = hourly_da * (daily_da / daily_mean(hourly_da))',
    })

    return corrected


def adjust_hourly_with_monthly(
    hourly_da: xr.DataArray,
    province_name: str,
    monthly_csv_path: Optional[Union[str, Path]] = None,
    cumul_monthly_csv_path: Optional[Union[str, Path]] = None,
    time_name: str = 'time',
) -> xr.DataArray:
    """
    用省级月度统计值对小时序列按月或按 1-2 月合计进行放缩修正。

    在主流程中，这个函数面向“让模型结果贴近官方月报”这一需求。它会从外部月度统计表中
    读取某省目标值，再把小时序列放缩到相应月份总量一致。这样上层既能保留小时级时序细节，
    又能让月度累计口径与外部统计接轨，适合省级发电校准场景。

    - 3~12 月：从当期值 CSV 按 "YYYY年M月" 列名逐月提取目标值，各月独立放缩。
    - 1~2 月：从当年累积值 CSV 的 "YYYY年2月" 列提取目标值（该列为1+2月合计），
              将1、2月视为整体统一放缩。

    两个 CSV 格式相同：
      - 第一列 "地区" 为省全称；
      - 其余列名格式为 "YYYY年M月"；
      - 单元格值为当期发电量（或其他目标量），单位与 hourly_da 一致。

    原理：
      对每个目标月（或月组），计算 hourly_da 在该时段的当前累加值，
      放缩系数 = 目标值 / 当前累加值，再调用 adjust_da_by_time 完成逐点乘法。

    Args:
        hourly_da: 小时序列 DataArray（待修正）。
        province_name: 省全称，用于在 CSV "地区" 列中定位行。
        monthly_csv_path: 当期值 CSV 路径，用于修正 3~12 月；为 None 时跳过。
        cumul_monthly_csv_path: 当年累积值 CSV 路径，用于修正 1~2 月合计；为 None 时跳过。
        time_name: 时间维名称，默认 'time'。

    Returns:
        xr.DataArray: 修正后的小时序列。
    """
    if monthly_csv_path is None and cumul_monthly_csv_path is None:
        return hourly_da

    if time_name not in hourly_da.dims:
        raise ValueError(f"hourly_da 必须包含时间维度: {time_name}")

    col_pattern = re.compile(r'^(\d{4})年(\d{1,2})月$')

    def _read_province_row(csv_path: Union[str, Path]) -> Dict[Tuple[int, int], float]:
        """读取 CSV，返回当前省份的 {(year, month): value} 字典。"""
        df = safe_read_csv(csv_path)
        if '地区' not in df.columns:
            raise ValueError(f"CSV 缺少'地区'列: {csv_path}")
        row_mask = df['地区'].astype(str).str.strip() == province_name.strip()
        if not row_mask.any():
            raise ValueError(f"CSV 中未找到省份 '{province_name}': {csv_path}")
        row = df[row_mask].iloc[0]
        result: Dict[Tuple[int, int], float] = {}
        for col in df.columns:
            m = col_pattern.match(str(col).strip())
            if m:
                year_val, month_val = int(m.group(1)), int(m.group(2))
                try:
                    val = float(row[col])
                    if not np.isnan(val):
                        result[(year_val, month_val)] = val
                except (ValueError, TypeError):
                    pass
        return result

    da_years = np.unique(hourly_da[time_name].dt.year.values)
    adjust_info: Dict = {}

    # 3~12 月：逐月放缩
    if monthly_csv_path is not None:
        monthly_targets = _read_province_row(monthly_csv_path)
        for year in da_years:
            for month in range(3, 13):
                if (year, month) not in monthly_targets:
                    continue
                target_val = monthly_targets[(year, month)]
                mask = (
                    (hourly_da[time_name].dt.year == year)
                    & (hourly_da[time_name].dt.month == month)
                )
                current_sum = float(hourly_da.where(mask, drop=True).sum(skipna=True).values)
                if current_sum == 0:
                    continue
                adjust_info[f'{year}年{month}月'] = target_val / current_sum

    # 1~2 月：合并放缩（累积值 CSV 的2月列 = 1+2月合计）
    if cumul_monthly_csv_path is not None:
        cumulative_targets = _read_province_row(cumul_monthly_csv_path)
        for year in da_years:
            if (year, 2) not in cumulative_targets:
                continue
            target_val = cumulative_targets[(year, 2)]
            mask = (
                (hourly_da[time_name].dt.year == year)
                & hourly_da[time_name].dt.month.isin([1, 2])
            )
            current_sum = float(hourly_da.where(mask, drop=True).sum(skipna=True).values)
            if current_sum == 0:
                continue
            adjust_info[(f'{year}年1月', f'{year}年2月')] = target_val / current_sum

    if not adjust_info:
        return hourly_da

    result = adjust_da_by_time(hourly_da, adjust_info, time_name=time_name)
    result.attrs.update({
        'monthly_adjustment_applied': 1,
        'monthly_adjustment_province': province_name,
        'monthly_adjustment_monthly_csv': str(monthly_csv_path) if monthly_csv_path is not None else '',
        'monthly_adjustment_cumulative_csv': str(cumul_monthly_csv_path) if cumul_monthly_csv_path is not None else '',
    })
    return result


def times_hourly_with_yearly(
    hourly_da: xr.DataArray,
    province_name: str,
    yearly_csv_path: Union[str, Path],
    prov_col: str = '省简称',
    time_name: str = 'time',
) -> xr.DataArray:
    """
    按年度系数表对小时序列逐年乘以对应修正因子。

    在主流程中，这个函数主要用于把省级年度利用率或其他年度修正系数回灌到小时序列上。
    相比 adjust_hourly_with_monthly 的月度校准，它提供更粗粒度但更轻量的年度放缩能力，
    适合快速把模型输出调到年度统计口径上。

    CSV 格式：
      - 第一列（默认列名 '省简称'）为省简称；
      - 其余列名为纯4位年份整数字符串，如 "2010"、"2023"；
      - 单元格值为该省该年的乘法系数（例如利用率、修正因子等）。

    对于 hourly_da 中出现的每个年份，若 CSV 中存在对应列且值非空，
    则将该年所有小时点乘以对应系数；否则保持原值不变。

    Args:
        hourly_da: 小时序列 DataArray（待处理）。
        province_name: 省简称，用于在 CSV 第一列中定位行。
        yearly_csv_path: 年度系数 CSV 路径。
        prov_col: CSV 中省简称列的列名，默认 '省简称'。
        time_name: 时间维名称，默认 'time'。

    Returns:
        xr.DataArray: 乘以年度系数后的 DataArray。
    """
    if time_name not in hourly_da.dims:
        raise ValueError(f"hourly_da 必须包含时间维度: {time_name}")

    df = safe_read_csv(yearly_csv_path)
    if prov_col not in df.columns:
        raise ValueError(f"CSV 缺少省简称列 '{prov_col}': {yearly_csv_path}")

    row_mask = df[prov_col].astype(str).str.strip() == province_name.strip()
    if not row_mask.any():
        raise ValueError(f"CSV 中未找到省份 '{province_name}': {yearly_csv_path}")
    row = df[row_mask].iloc[0]

    year_pattern = re.compile(r'^(\d{4})$')
    adjust_info: Dict = {}
    for col in df.columns:
        m = year_pattern.match(str(col).strip())
        if not m:
            continue
        try:
            val = float(row[col])
            if not np.isnan(val):
                adjust_info[f'{m.group(1)}年'] = val
        except (ValueError, TypeError):
            pass

    if not adjust_info:
        return hourly_da

    result = adjust_da_by_time(hourly_da, adjust_info, time_name=time_name)
    result.attrs.update({
        'yearly_factor_applied': 1,
        'yearly_factor_province': province_name,
        'yearly_factor_csv': str(yearly_csv_path),
    })
    return result


