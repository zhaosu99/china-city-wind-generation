
import logging
from numbers import Number
from time import perf_counter
from typing import Union, Literal, Tuple, List, Optional

import numpy as np
import xarray as xr
from geographiclib.geodesic import Geodesic

from utils import ensure_ascending_da_coords, find_nearest_points_from_ds


logger = logging.getLogger(__name__)


def _log_verbose(verbose: bool, message: str, *args) -> None:
    if verbose:
        logger.info(message, *args)


def _log_stage(verbose: bool, label: str, start_time: float) -> None:
    if verbose:
        logger.info("[horizontal_interp] %s | elapsed: %.2fs", label, perf_counter() - start_time)


def _coords_to_index_arrays(
        lat_values: np.ndarray,
        lon_values: np.ndarray,
        neighbor_coords: List[Tuple[Number, Number]],
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Map neighbor coordinate pairs to integer lat/lon index arrays."""
    lat_index = {float(v): i for i, v in enumerate(lat_values)}
    lon_index = {float(v): i for i, v in enumerate(lon_values)}

    lat_idx_arr = np.fromiter(
        (lat_index[float(lat)] for lat, _ in neighbor_coords),
        dtype=int,
        count=len(neighbor_coords),
    )
    lon_idx_arr = np.fromiter(
        (lon_index[float(lon)] for _, lon in neighbor_coords),
        dtype=int,
        count=len(neighbor_coords),
    )
    return lat_idx_arr, lon_idx_arr


def _extract_neighbor_values_numpy(
        da: xr.DataArray,
        neighbor_coords: List[Tuple[Number, Number]],
        lon_name: str,
        lat_name: str,
    ) -> np.ndarray:
    """Extract neighbor time series with direct numpy indexing.

    Returns an array shaped (n_neighbors, n_time), which matches the previous
    loess/idw math while avoiding repeated xarray label-based indexing.
    """
    lat_values = np.asarray(da.coords[lat_name].values)
    lon_values = np.asarray(da.coords[lon_name].values)
    lat_idx_arr, lon_idx_arr = _coords_to_index_arrays(lat_values, lon_values, neighbor_coords)

    data = np.asarray(da.values)
    return data[:, lat_idx_arr, lon_idx_arr].T


def idw_interp(
        da: xr.DataArray, 
        target_coord: Tuple[Number, Number],
        closest_points_num: int = 4,
        power: int = 2,
        nearest_method: Literal['binary', 'KDTree'] = 'binary',
        np_loc: bool = True,
        lon_name: str = 'lon',
        lat_name: str = 'lat',
        time_name: str = 'time',
        verbose: bool = False,
    ) -> xr.DataArray:
    """
    反距离加权是一种常用的空间插值方法，
    用于估算未知点的值。其基本思想是：距离越近的已知点对未知点的影响越大，
    距离越远的已知点对未知点的影响越小。

    np_loc是一种比较快速的方法

    Args:
        da (xr.DataArray): 包含经纬度与参数信息的数据数组
        target_coord (Tuple[Number, Number]): 目标点的经纬度坐标，顺序需要确保是 (latitude, longitude)
        closest_points_num (int, optional): 要考虑的最近点的数量
        power (int, optional): 用于计算权重的指数
        nearest_method (Literal['binary', 'KDTree'], optional): 用于寻找最近点的方法
        np_loc (bool, optional): 是否使用 numpy 的方法
            使用np.where来寻找最近点，如此更快
            使用da.sel来寻找最近点，如此更慢
        lon_name (str, optional): 数据集中的经度变量名
        lat_name (str, optional): 数据集中的纬度变量名
        time_name (str, optional): 数据集中的时间变量名

    Returns:
        xr.DataArray: 返回插值后的值，维度为 (time, lat, lon)，其中 lat 和 lon 各只有一个值
    """     
    total_start = perf_counter()
    logging.debug('注意target_coord要先纬度再经度')
    _log_verbose(
        verbose,
        "[horizontal_interp] idw start | target=(%.6f, %.6f) | shape=%s | dims=%s | nearest=%s",
        float(target_coord[0]),
        float(target_coord[1]),
        da.shape,
        da.dims,
        closest_points_num,
    )

    # 确保经纬度坐标按升序排列
    stage_start = perf_counter()
    da = ensure_ascending_da_coords(da, lon_name, lat_name)

    # 确保维度顺序为 (time, lat, lon)
    expected_dims = [time_name, lat_name, lon_name]
    if list(da.dims) != expected_dims:
        da = da.transpose(time_name, lat_name, lon_name)
    _log_stage(verbose, "idw preprocess", stage_start)


    # 获取时间坐标
    time_coord = da.coords[time_name]

    def get_nearest_arrs(
            da: xr.DataArray, 
            nearest_coords: List[Tuple[Number, Number]],
            np_loc: bool = np_loc
            ) -> List[np.ndarray]:
        """
        从数据数组中获取最近点的对应的子数据

        Args:
            da (xr.DataArray): 包含经纬度与参数信息的数据数组
            nearest_coords (List[Tuple[Number, Number]]): 最近点的经纬度坐标列表
            np_loc (bool, optional): 是否使用 numpy 的方法

        Returns:
            List[np.ndarray]: 返回最近点的子数据
        """
        if np_loc:
            lons = da.coords[lon_name].values
            lats = da.coords[lat_name].values
            def loc_ind(arr: np.ndarray, target: Number):
                return np.where(arr == target)[0][0]         
            return \
                [
                    da.values[:, loc_ind(lats, lat), loc_ind(lons, lon)]
                        for lat, lon in nearest_coords
                ]
        else:    
            return \
                [
                    da.sel(
                        {lat_name: lat, lon_name: lon}
                        ).values for lat, lon in nearest_coords
                ]
    
    def calculate_weights(
            distances: Union[list, np.ndarray], 
            power: int
            ) -> np.ndarray:
        """
        计算距离的反比权重

        Args:
            distances (Union[list, np.ndarray]): 距离列表
            power (int): 反比权重指数

        Returns:
            np.ndarray: 反比权重
        """
        weights = 1 / (np.array(distances) ** power)
        return weights / weights.sum()  
    
    def calculate_weighted_average(
            values: np.ndarray, 
            weights: np.ndarray,
            ) -> np.ndarray:
        """
        计算加权平均值

        Args:
            values (np.ndarray): 值数组
            weights (np.ndarray): 权重数组

        Returns:
            np.ndarray: 加权平均值
        """
        result = [v * weights[i] for i, v in enumerate(values)]
        return np.sum(result, axis=0)
    
    # 将 DataArray 转换为 Dataset 用于搜索最近点
    stage_start = perf_counter()
    ds_temp = da.to_dataset(name='__temp__')
    
    nearest_coords, nearest_distances = \
        find_nearest_points_from_ds(ds_temp, target_coord, closest_points_num, nearest_method, True,
                            lon_name, lat_name)
    _log_stage(verbose, f"idw nearest search | points={len(nearest_coords)}", stage_start)
    
    stage_start = perf_counter()
    nearest_values = get_nearest_arrs(da, nearest_coords, np_loc)
    _log_stage(verbose, "idw neighbor extraction", stage_start)

    stage_start = perf_counter()
    weights = calculate_weights(nearest_distances, power)

    result_values = calculate_weighted_average(nearest_values, weights)
    _log_stage(verbose, "idw weighting", stage_start)
    
    # 构建返回的 DataArray，维度为 (time, lat, lon)，其中 lat 和 lon 各只有一个值
    lat_target, lon_target = target_coord
    result_da = xr.DataArray(
        result_values.reshape(len(time_coord), 1, 1),
        dims=[time_name, lat_name, lon_name],
        coords={
            time_name: time_coord,
            lat_name: [lat_target],
            lon_name: [lon_target]
        }
    )

    result_da.attrs.update({
        'interp_method': 'idw',
        'interp_method_full_name': 'Inverse Distance Weighting',
        'target_coord_order': '(lat, lon)',
        'target_lat': float(lat_target),
        'target_lon': float(lon_target),
        'coord_names': f'time={time_name}, lat={lat_name}, lon={lon_name}',
        'nearest_method': nearest_method,
        'closest_points_num': int(closest_points_num),
        'idw_power': float(power),
        'idw_np_loc': bool(np_loc),
        'distance_metric': 'geopy.distance (km)',
        'source_grid_lat_size': int(da.sizes[lat_name]),
        'source_grid_lon_size': int(da.sizes[lon_name]),
        'source_time_size': int(da.sizes[time_name]),
        'output_shape': str(result_da.shape),
        'output_dims': str(result_da.dims),
    })
    _log_stage(verbose, "idw total", total_start)
    
    return result_da


def closest_interp(
        da: xr.DataArray, 
        target_coord: Tuple[Number, Number],
        lon_name: str = 'lon',
        lat_name: str = 'lat',
        time_name: str = 'time',
    verbose: bool = False,
    ) -> xr.DataArray:
    """
    在 DataArray 的现有坐标中寻找距离目标点最近的、没有空值的点，返回该点的完整时间序列数据

    Args:
        da (xr.DataArray): 输入的数据数组
        target_coord (Tuple[Number, Number]): 目标点的经纬度坐标，顺序为 (latitude, longitude)
        lon_name (str, optional): 经度变量名，默认为 'lon'
        lat_name (str, optional): 纬度变量名，默认为 'lat'
        time_name (str, optional): 时间变量名，默认为 'time'

    Returns:
        xr.DataArray: 返回最近的没有空值的点的完整时间序列数据，维度为 (time, lat, lon)，其中 lat 和 lon 各只有一个值
    """
    total_start = perf_counter()
    logging.debug('注意target_coord要先纬度再经度')
    _log_verbose(
        verbose,
        "[horizontal_interp] closest start | target=(%.6f, %.6f) | shape=%s | dims=%s",
        float(target_coord[0]),
        float(target_coord[1]),
        da.shape,
        da.dims,
    )

    def _build_result_da(series: np.ndarray, chosen_lat: Number, chosen_lon: Number) -> xr.DataArray:
        result_da = xr.DataArray(
            series.reshape(len(time_coord), 1, 1),
            dims=[time_name, lat_name, lon_name],
            coords={
                time_name: time_coord,
                lat_name: [chosen_lat],
                lon_name: [chosen_lon],
            }
        )
        print(f"closest_interp 选中最近网格点: lat={float(chosen_lat):.6f}, lon={float(chosen_lon):.6f}")
        _log_verbose(
            verbose,
            "[horizontal_interp] closest selected point | lat=%.6f | lon=%.6f",
            float(chosen_lat),
            float(chosen_lon),
        )
        return result_da

    # 确保经纬度坐标按升序排列
    stage_start = perf_counter()
    da = ensure_ascending_da_coords(da, lon_name, lat_name)

    # 确保维度顺序为 (time, lat, lon)
    expected_dims = [time_name, lat_name, lon_name]
    if list(da.dims) != expected_dims:
        da = da.transpose(time_name, lat_name, lon_name)
    _log_stage(verbose, "closest preprocess", stage_start)

    # 获取时间坐标
    time_coord = da.coords[time_name]

    # 获取数据
    lat_values = da.coords[lat_name].values
    lon_values = da.coords[lon_name].values
    data = da.values  # shape: (time, n_lat, n_lon)
    
    target_lat, target_lon = target_coord
    n_lat = len(lat_values)
    n_lon = len(lon_values)

    # 用 searchsorted 快速定位离目标点最近的网格索引
    # 使用 searchsorted 快速定位，然后从最近 4 个候选点开始，螺旋式扩展搜索
    lat_idx_right = np.searchsorted(lat_values, target_lat)
    lon_idx_right = np.searchsorted(lon_values, target_lon)
    
    lat_idx_left = lat_idx_right - 1
    lon_idx_left = lon_idx_right - 1
    
    # 检查起始 4 个候选
    candidates = [(lat_idx_left, lon_idx_left), (lat_idx_left, lon_idx_right),
                  (lat_idx_right, lon_idx_left), (lat_idx_right, lon_idx_right)]
    valid = [(i, j) for i, j in candidates if 0 <= i < n_lat and 0 <= j < n_lon]
    
    def dist_sq(i, j): return (lat_values[i] - target_lat) ** 2 + (lon_values[j] - target_lon) ** 2
    
    valid.sort(key=lambda p: dist_sq(p[0], p[1]))
    _log_verbose(verbose, "[horizontal_interp] closest initial candidates=%s", len(valid))
    
    # 首先尝试起始 4 个点
    for lat_idx, lon_idx in valid:
        series = data[:, lat_idx, lon_idx]
        if not np.isnan(series).any():
            _log_stage(verbose, "closest initial candidate hit", total_start)
            return _build_result_da(series, lat_values[lat_idx], lon_values[lon_idx])
    
    # 螺旋式扩展
    visited = set(valid)
    for radius in range(1, max(n_lat, n_lon)):
        _log_verbose(verbose, "[horizontal_interp] closest expanding radius=%s", radius)
        candidates_at_radius = []
        for i in range(max(0, lat_idx_right - radius), min(n_lat, lat_idx_right + radius + 1)):
            for j in range(max(0, lon_idx_right - radius), min(n_lon, lon_idx_right + radius + 1)):
                if (i, j) not in visited:
                    candidates_at_radius.append((i, j, dist_sq(i, j)))
                    visited.add((i, j))
        
        candidates_at_radius.sort(key=lambda x: x[2])
        for lat_idx, lon_idx, _ in candidates_at_radius:
            series = data[:, lat_idx, lon_idx]
            if not np.isnan(series).any():
                _log_stage(verbose, f"closest expanded search hit | radius={radius}", total_start)
                return _build_result_da(series, lat_values[lat_idx], lon_values[lon_idx])
    
    best_i, best_j, min_missing = 0, 0, float('inf')
    for i in range(n_lat):
        for j in range(n_lon):
            n_missing = np.isnan(data[:, i, j]).sum()
            if n_missing < min_missing:
                min_missing = n_missing
                best_i, best_j = i, j
    series = data[:, best_i, best_j]
    _log_verbose(
        verbose,
        "[horizontal_interp] closest fallback to min-missing point | missing=%s",
        min_missing,
    )
    _log_stage(verbose, "closest total", total_start)
    return _build_result_da(series, lat_values[best_i], lon_values[best_j])


def loess_interp(
        da: xr.DataArray,
        target_coord: Tuple[Number, Number],
        lon_name: str = 'lon',
        lat_name: str = 'lat',
        time_name: str = 'time',
        n_nearest_points: int = 16,
        nearest_method: str = 'binary',
    nearest_coords: Optional[List[Tuple[Number, Number]]] = None,
    verbose: bool = False,
    ) -> xr.DataArray:
    """
    使用局部二次 LOESS 对目标点进行空间插值。

    Args:
        da (xr.DataArray): 输入数据，需包含 time/lat/lon 三个维度。
        target_coord (Tuple[Number, Number]): 目标点坐标，顺序为 (lat, lon)。
        lon_name (str, optional): 经度维度名。
        lat_name (str, optional): 纬度维度名。
        time_name (str, optional): 时间维度名。
        n_nearest_points (int, optional): 用于拟合的最近点数量。
        nearest_method (str, optional): 最近点搜索方法，支持 'binary' 或 'KDTree'。

    Returns:
        xr.DataArray: 目标点插值结果，维度为 (time, lat, lon)，其中 lat/lon 长度均为 1。
    """
    total_start = perf_counter()
    logging.debug('注意target_coord要先纬度再经度')
    _log_verbose(
        verbose,
        "[horizontal_interp] loess start | target=(%.6f, %.6f) | shape=%s | dims=%s | nearest=%s",
        float(target_coord[0]),
        float(target_coord[1]),
        da.shape,
        da.dims,
        n_nearest_points,
    )

    if n_nearest_points < 6:
        raise ValueError('LOESS 二次回归至少需要 6 个点')

    # 确保经纬度坐标按升序排列
    stage_start = perf_counter()
    da = ensure_ascending_da_coords(da, lon_name, lat_name)

    # 确保维度顺序为 (time, lat, lon)
    expected_dims = [time_name, lat_name, lon_name]
    if list(da.dims) != expected_dims:
        da = da.transpose(time_name, lat_name, lon_name)
    _log_stage(verbose, "loess preprocess", stage_start)

    time_coord = da.coords[time_name]
    target_lat, target_lon = target_coord

    # 找到最近点
    stage_start = perf_counter()
    if nearest_coords is None:
        ds_temp = da.to_dataset(name='__temp__')
        nearest_coords = find_nearest_points_from_ds(
            ds=ds_temp,
            target_point=target_coord,
            n=n_nearest_points,
            nearest_method=nearest_method,
            return_distance=False,
            lon_name=lon_name,
            lat_name=lat_name,
        )
        _log_stage(verbose, f"loess nearest search | points={len(nearest_coords)}", stage_start)
    else:
        _log_verbose(verbose, "[horizontal_interp] loess reuse nearest coords | points=%s", len(nearest_coords))

    # 构建设计矩阵和权重（与 compute_loess_weights 保持一致）
    stage_start = perf_counter()
    geodesic = Geodesic.WGS84
    neighbor_lats = np.array([lat for lat, _ in nearest_coords], dtype=float)
    neighbor_lons = np.array([lon for _, lon in nearest_coords], dtype=float)

    distances_km = np.array([
        geodesic.Inverse(target_lat, target_lon, lat, lon)['s12'] / 1000.0
        for lat, lon in nearest_coords
    ], dtype=float)
    _log_stage(verbose, "loess distance computation", stage_start)

    h = float(np.max(distances_km))
    if h == 0.0:
        _log_verbose(verbose, "[horizontal_interp] loess degenerates to closest | reason=zero bandwidth")
        result_da = closest_interp(
            da=da,
            target_coord=target_coord,
            lon_name=lon_name,
            lat_name=lat_name,
            time_name=time_name,
            verbose=verbose,
        )
        result_da.attrs.update({
            'interp_method': 'loess',
            'interp_method_full_name': 'Local Quadratic Regression (LOESS)',
            'loess_fallback': 'closest_interp (all nearest points overlap target)',
            'target_coord_order': '(lat, lon)',
            'target_lat': float(target_lat),
            'target_lon': float(target_lon),
            'coord_names': f'time={time_name}, lat={lat_name}, lon={lon_name}',
            'nearest_method': nearest_method,
            'n_nearest_points': int(n_nearest_points),
            'distance_metric': 'WGS84 geodesic (GeographicLib), km for h',
            'source_grid_lat_size': int(da.sizes[lat_name]),
            'source_grid_lon_size': int(da.sizes[lon_name]),
            'source_time_size': int(da.sizes[time_name]),
            'output_shape': str(result_da.shape),
            'output_dims': str(result_da.dims),
        })
        _log_stage(verbose, "loess total", total_start)
        return result_da

    stage_start = perf_counter()
    d_norm = distances_km / h
    weights = np.where(d_norm < 1.0, (1.0 - np.abs(d_norm) ** 3) ** 3, 0.0)

    mean_lats = (target_lat + neighbor_lats) / 2.0
    cos_lats = np.cos(np.radians(mean_lats))
    dx_km = (neighbor_lons - target_lon) * cos_lats * 111.32
    dy_km = (neighbor_lats - target_lat) * 111.32

    X = np.column_stack([
        np.ones(len(nearest_coords)),
        dx_km,
        dy_km,
        dx_km ** 2,
        dx_km * dy_km,
        dy_km ** 2,
    ])
    _log_stage(verbose, "loess design matrix", stage_start)

    stage_start = perf_counter()
    sqrt_w = np.sqrt(weights)
    XTW = X.T * sqrt_w
    XTWX = XTW @ X

    try:
        M = np.linalg.solve(XTWX, XTW)
    except np.linalg.LinAlgError:
        # 点位几何关系退化时使用伪逆保证可计算性
        M = np.linalg.pinv(XTWX) @ XTW
    _log_stage(verbose, "loess linear solve", stage_start)

    # 抽取最近点时间序列，形状 (n_neighbors, n_time)
    stage_start = perf_counter()
    neighbor_values = _extract_neighbor_values_numpy(
        da,
        nearest_coords,
        lon_name=lon_name,
        lat_name=lat_name,
    )
    _log_stage(
        verbose,
        (
            "loess neighbor extraction | values_shape=%s | dtype=%s | contiguous=%s"
            % (neighbor_values.shape, neighbor_values.dtype, neighbor_values.flags.c_contiguous)
        ),
        stage_start,
    )

    # 只需要 beta[0]，直接计算目标点估计值，避免整块 6 x n_time 矩阵乘法。
    stage_start = perf_counter()
    coeff_row = np.asarray(M[0], dtype=np.float64)
    _log_verbose(
        verbose,
        "[horizontal_interp] loess regression input | coeff_shape=%s | coeff_dtype=%s",
        coeff_row.shape,
        coeff_row.dtype,
    )
    interpolated_values = coeff_row @ neighbor_values
    _log_stage(verbose, "loess regression apply target row", stage_start)

    stage_start = perf_counter()
    result_da = xr.DataArray(
        interpolated_values.reshape(len(time_coord), 1, 1),
        dims=[time_name, lat_name, lon_name],
        coords={
            time_name: time_coord,
            lat_name: [target_lat],
            lon_name: [target_lon],
        },
    )
    _log_stage(verbose, "loess result DataArray build", stage_start)

    result_da.attrs.update({
        'interp_method': 'loess',
        'interp_method_full_name': 'Local Quadratic Regression (LOESS)',
        'target_coord_order': '(lat, lon)',
        'target_lat': float(target_lat),
        'target_lon': float(target_lon),
        'coord_names': f'time={time_name}, lat={lat_name}, lon={lon_name}',
        'nearest_method': nearest_method,
        'n_nearest_points': int(n_nearest_points),
        'loess_polynomial_degree': 2,
        'loess_kernel': 'tricube',
        'loess_design_terms': '1, dx, dy, dx^2, dx*dy, dy^2',
        'distance_metric': 'WGS84 geodesic (GeographicLib), km for h',
        'loess_bandwidth_h_km': h,
        'source_grid_lat_size': int(da.sizes[lat_name]),
        'source_grid_lon_size': int(da.sizes[lon_name]),
        'source_time_size': int(da.sizes[time_name]),
        'output_shape': str(result_da.shape),
        'output_dims': str(result_da.dims),
    })

    _log_stage(verbose, "loess total", total_start)

    return result_da


def _horizontal_interp_raw(
        da: xr.DataArray,
        target_coord: Tuple[Number, Number],
        method: Literal['closest', 'idw', 'loess'] = 'closest',
        lon_name: str = 'lon',
        lat_name: str = 'lat',
        time_name: str = 'time',
    verbose: bool = False,
    ) -> xr.DataArray:
    """
    对单个时间点进行水平插值，支持 closest、IDW、LOESS 三种方法。

    兜底策略（backup method=closest）：
    - 当 method='loess' 时，若最近 4*4（16 个）候选格点存在空值，则自动回退 closest。
    - 当 method='idw' 时，若最近 2*2（4 个）候选格点存在空值，则自动回退 closest。

    Args:
        da (xr.DataArray): 输入数据，需包含 time/lat/lon 三个维度。
        target_coord (Tuple[Number, Number]): 目标点坐标，顺序为 (lat, lon)。
        method (Literal['closest', 'idw', 'loess'], optional): 插值方法，默认为 'closest'。
        lon_name (str, optional): 经度维度名，默认为 'lon'。
        lat_name (str, optional): 纬度维度名，默认为 'lat'。
        time_name (str, optional): 时间维度名，默认为 'time'。
    """

    def _nearest_grid_has_missing(
            da_in: xr.DataArray,
            nearest_points: int,
            nearest_coords: Optional[List[Tuple[Number, Number]]] = None,
        ) -> bool:
        stage_start = perf_counter()
        if nearest_coords is None:
            ds_temp = da_in.to_dataset(name='__temp__')
            nearest_coords = find_nearest_points_from_ds(
                ds=ds_temp,
                target_point=target_coord,
                n=nearest_points,
                nearest_method='binary',
                return_distance=False,
                lon_name=lon_name,
                lat_name=lat_name,
            )
        if len(nearest_coords) == 0:
            _log_stage(verbose, f"missing-check nearest search | points={nearest_points}", stage_start)
            return True

        lat_values = np.asarray(da_in.coords[lat_name].values)
        lon_values = np.asarray(da_in.coords[lon_name].values)
        lat_idx_arr, lon_idx_arr = _coords_to_index_arrays(lat_values, lon_values, nearest_coords)

        candidate_series = da_in.values[:, lat_idx_arr, lon_idx_arr]
        has_missing = bool(np.isnan(candidate_series).any())
        _log_stage(
            verbose,
            f"missing-check done | points={len(nearest_coords)} | has_missing={has_missing}",
            stage_start,
        )
        return has_missing

    # 预处理一次，避免重复排序/转置带来的额外开销
    total_start = perf_counter()
    _log_verbose(
        verbose,
        "[horizontal_interp] start | method=%s | target=(%.6f, %.6f) | shape=%s | dims=%s",
        method,
        float(target_coord[0]),
        float(target_coord[1]),
        da.shape,
        da.dims,
    )
    stage_start = perf_counter()
    da_prepared = ensure_ascending_da_coords(da, lon_name, lat_name)
    expected_dims = [time_name, lat_name, lon_name]
    if list(da_prepared.dims) != expected_dims:
        da_prepared = da_prepared.transpose(time_name, lat_name, lon_name)
    _log_stage(verbose, "top-level preprocess", stage_start)

    if method == 'closest':
        result = closest_interp(da_prepared, target_coord, lon_name, lat_name, time_name, verbose=verbose)
        _log_stage(verbose, "horizontal_interp total", total_start)
        return result
    if method == 'idw':
        if _nearest_grid_has_missing(da_prepared, nearest_points=4):
            _log_verbose(verbose, "[horizontal_interp] IDW fallback to closest | reason=nearest_2x2_has_missing")
            print("IDW 插值退化为 Closest 插值，原因：最近 2x2 网格点存在空值")
            result = closest_interp(da_prepared, target_coord, lon_name, lat_name, time_name, verbose=verbose)
            result.attrs.update({
                'requested_interp_method': 'idw',
                'fallback_to': 'closest',
                'fallback_reason': 'nearest_2x2_has_missing',
            })
            _log_stage(verbose, "horizontal_interp total", total_start)
            return result
        result = idw_interp(
            da_prepared,
            target_coord,
            lon_name=lon_name,
            lat_name=lat_name,
            time_name=time_name,
            verbose=verbose,
        )
        _log_stage(verbose, "horizontal_interp total", total_start)
        return result
    if method == 'loess':
        stage_start = perf_counter()
        ds_temp = da_prepared.to_dataset(name='__temp__')
        nearest_coords = find_nearest_points_from_ds(
            ds=ds_temp,
            target_point=target_coord,
            n=16,
            nearest_method='binary',
            return_distance=False,
            lon_name=lon_name,
            lat_name=lat_name,
        )
        _log_stage(verbose, f"top-level loess nearest search | points={len(nearest_coords)}", stage_start)
        if _nearest_grid_has_missing(da_prepared, nearest_points=16, nearest_coords=nearest_coords):
            _log_verbose(verbose, "[horizontal_interp] LOESS fallback to closest | reason=nearest_4x4_has_missing")
            print("LOESS 插值退化为 Closest 插值，原因：最近 4x4 网格点存在空值")
            result = closest_interp(da_prepared, target_coord, lon_name, lat_name, time_name, verbose=verbose)
            result.attrs.update({
                'requested_interp_method': 'loess',
                'fallback_to': 'closest',
                'fallback_reason': 'nearest_4x4_has_missing',
            })
            _log_stage(verbose, "horizontal_interp total", total_start)
            return result
        result = loess_interp(
            da_prepared,
            target_coord,
            lon_name=lon_name,
            lat_name=lat_name,
            time_name=time_name,
            nearest_coords=nearest_coords,
            verbose=verbose,
        )
        _log_stage(verbose, "horizontal_interp total", total_start)
        return result

    raise ValueError('method 参数值错误')


def _clip_nonnegative(result: xr.DataArray, method: str) -> xr.DataArray:
    """BUG-5：把水平插值输出裁剪到 ≥ 0（风速是矢量的模，物理上非负）。

    `loess`（16 点 + 二次曲面 + tricube 核）在陡梯度处会过冲出**负风速**：
    审计实测 farm 7130 `w100 min=−0.0808`；存盘侧 308 个场中 8 个为负（最负 −0.2327）。
    负风速没有物理解释（CF 曲线在 v<0 处取 0），因此在校验层统一裁剪，
    并把**被截断的个数/是否发生**写进 attrs 便于诊断。

    只改负值：非负值逐点不变（有单测锁定）。
    """
    values = np.asarray(result.values)
    n_neg = int(np.count_nonzero(values < 0))
    if n_neg:
        result = result.clip(min=0.0)
    result.attrs.update({
        'nonneg_clip_applied': bool(n_neg),
        'nonneg_clipped_count': n_neg,
        'nonneg_clip_method': str(method),
    })
    return result


def horizontal_interp(
        da: xr.DataArray,
        target_coord: Tuple[Number, Number],
        method: Literal['closest', 'idw', 'loess'] = 'closest',
        lon_name: str = 'lon',
        lat_name: str = 'lat',
        time_name: str = 'time',
    verbose: bool = False,
    ) -> xr.DataArray:
    """把 ERA5 网格水平插值到单个场址，**输出保证 ≥ 0**。

    与 `_horizontal_interp_raw` 的差别只有一处：BUG-5 修复——所有方法的输出统一
    `clip(min=0)`（并记录 `nonneg_clip_applied` / `nonneg_clipped_count` attrs）。
    """
    result = _horizontal_interp_raw(
        da, target_coord, method=method,
        lon_name=lon_name, lat_name=lat_name, time_name=time_name, verbose=verbose,
    )
    return _clip_nonnegative(result, method)
    



if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from vars_computation import cdmet_10m_w

    # ------------------------------------------------------------------ #
    # CDMet 实测测试：近海节点 closest_interp
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("CDMet 实测测试：近海节点 closest_interp")
    print("=" * 60)

    cdmet_path = r'D:\Data\CDMet\CDMet_win_2013.nc'
    ds_cdmet = xr.open_dataset(cdmet_path)
    print(f"数据集维度: {dict(ds_cdmet.sizes)}")
    print(f"lat 范围: {float(ds_cdmet.lat.min()):.2f} ~ {float(ds_cdmet.lat.max()):.2f}")
    print(f"lon 范围: {float(ds_cdmet.lon.min()):.2f} ~ {float(ds_cdmet.lon.max()):.2f}")

    da_cdmet = cdmet_10m_w(ds_cdmet, year=2013)
    print(f"w10 DA 形状: {da_cdmet.shape}, 维度: {da_cdmet.sizes}")

    # 近海测试点 (lat, lon)
    offshore_points = {
        '渤海中部':   (38.5,  120.5),
        '黄海中部':   (33.0,  123.0),
        '东海中部':   (28.0,  124.5),
        '南海北部':   (19.0,  111.5),
    }

    for name, coord in offshore_points.items():
        result = closest_interp(
            da_cdmet, coord,
            lon_name='lon', lat_name='lat', time_name='time'
        )
        selected_lat = float(result.coords['lat'].values[0])
        selected_lon = float(result.coords['lon'].values[0])
        series = result.values[:, 0, 0]
        print(f"\n[{name}]  目标=({coord[0]}, {coord[1]})")
        print(f"  → 选中网格点: lat={selected_lat:.4f}, lon={selected_lon:.4f}")
        print(f"  → 年均风速: {series.mean():.3f} m/s  |  "
              f"最大: {series.max():.3f}  最小: {series.min():.3f}")
        print(f"  → NaN 数量: {int(np.isnan(series).sum())}")
        print(f"  → 前 5 天风速: {series[:5].round(3)}")

    ds_cdmet.close()
    print("\n✓ CDMet 近海节点测试完成\n")

   
"""
目前的代码待优化

"""