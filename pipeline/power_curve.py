"""风电功率曲线：基础曲线 + 变带宽高斯平滑（**唯一实现**）

本模块是功率曲线相关计算的唯一来源。原先 `vars_computation.py` 与
`farm_power_curves.py` 各有一份**逐字重复**的 `generate_power_curve`，本次收敛到这里。

修复的缺陷（详见 tests/test_power_curve_smoothing.py）
-----------------------------------------------------
原自适应分支：

    kernel = exp(-0.5*(x/sigma_pix)**2); kernel /= kernel.sum()   # 在「全长 2r+1」上归一化
    start = max(0, i - radius); end = min(n, i + radius + 1)
    cf_grid[i] = np.sum(cf_raw[start:end] * kernel[:end - start])

kernel 下标 j 与数据下标 (start + j) 配对，故权重中心落在数据下标 `start + radius`：
    内部   (i >= r)：start = i - r  → 中心 = i      ✅
    左边界 (i < r) ：start = 0      → 中心 = r ≠ i  ❌ 权重整体偏向高风速 → 输出偏高
    右边界 (i+r>=n)：起点虽正确，但核被截断后仍沿用全长归一化 → 权重和<1 → 输出偏低

修法（从构造上消除，而不是"打补丁"）
------------------------------------
    以**格点差** `(j - i)` 直接构造权重，不再存在"核数组切片起点"这一概念；
    分母用**实际参与加权的权重之和**，因此归一化天然正确。
    ⇒ 常函数必映射为自身；两侧边界都不会错位。

为了可验证性，内部量保持与原实现一致：σ 仍由 float32 的风速网格算出、radius 同式，
因此修复后**中间区（核未触及数组边界）与旧实现逐位一致**，差异只出现在两个边界区。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ---- 默认参数（与原实现一致）----
DEFAULT_SIGMA_SLOPE = 0.2
DEFAULT_SIGMA_INTERCEPT = 0.6
DEFAULT_SIGMA_TRUNCATE = 4.0
DEFAULT_W_STEP = 0.05
DEFAULT_W_MAX = 40.0
DEFAULT_POWER_ORDER = 3


def raw_capacity_factors(w_grid, cut_in_v, rated_v, cut_out_v,
                         v_slope, v_intercept, power_order) -> np.ndarray:
    """基础（未平滑）风电功率曲线：cut_in 前 0；cut_in→rated 按 power_order 次幂上升；
    rated→cut_out 为 1；cut_out 及以后为 0。

    逐点标量计算（与原实现同构），以保证数值逐位一致。
    """
    def _cf_raw(v: float) -> float:
        v_adj = v_slope * v + v_intercept
        if v_adj < cut_in_v:
            return 0.0
        if v_adj < rated_v:
            numerator = v_adj ** power_order - cut_in_v ** power_order
            denominator = rated_v ** power_order - cut_in_v ** power_order
            return numerator / denominator if denominator != 0 else 0.0
        if v_adj < cut_out_v:
            return 1.0
        return 0.0

    return np.array([_cf_raw(w) for w in w_grid], dtype=np.float32)


def smooth_capacity_factors(cf_raw, sigma_pix, truncate: float = DEFAULT_SIGMA_TRUNCATE) -> np.ndarray:
    """★ 变带宽高斯平滑（正确实现）。σ 以**格点**为单位。

    对每个 i：
        radius = ceil(truncate * sigma_i)
        窗口   = [i-radius, i+radius] ∩ [0, n)
        权重   = exp(-0.5 * ((j - i) / sigma_i)^2)          ← 距离直接以 i 为基准
        输出   = Σ cf[j]·权重 / Σ 权重                       ← 用多少权重就归一化多少
    """
    cf_raw = np.asarray(cf_raw, dtype=np.float64)
    sigma_pix = np.asarray(sigma_pix, dtype=np.float64)
    n = cf_raw.size
    assert sigma_pix.size == n, "sigma 与基础曲线长度必须一致"
    idx = np.arange(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        sp = float(sigma_pix[i])
        if not np.isfinite(sp) or sp <= 0:
            raise ValueError(f"sigma_pix 必须为正的有限值，第 {i} 个为 {sp!r}")
        radius = int(np.ceil(truncate * sp))
        j0 = max(0, i - radius)
        j1 = min(n, i + radius + 1)
        weights = np.exp(-0.5 * ((idx[j0:j1] - i) / sp) ** 2)
        out[i] = float(np.dot(cf_raw[j0:j1], weights) / weights.sum())

    return out.astype(np.float32)


def generate_power_curve(
    cut_in_v: float = 3.0,
    rated_v: float = 12.0,
    cut_out_v: float = 25.0,
    v_slope: float = 1.0,
    v_intercept: float = 0.0,
    power_order: int = 3,
    smooth: bool = True,
    fixed_sigma=False,
    sigma_slope: float = DEFAULT_SIGMA_SLOPE,
    sigma_intercept: float = DEFAULT_SIGMA_INTERCEPT,
    sigma_truncate: float = DEFAULT_SIGMA_TRUNCATE,
    w_step: float = DEFAULT_W_STEP,
    w_max: float = DEFAULT_W_MAX,
    plot: bool = False,
    ax=None,
    plot_kwargs=None,
):
    """生成风电功率曲线（风速 -> 容量因子）。签名与返回值与原实现完全一致。

    Args:
        cut_in_v / rated_v / cut_out_v: 切入、额定、切出风速 (m/s)。
        v_slope / v_intercept: 风速的线性调整 (v_adj = v_slope*v + v_intercept)。
        power_order: 切入至额定区间的幂次。
        smooth: 是否平滑。
        fixed_sigma: False 表示**变带宽** sigma = sigma_intercept + sigma_slope*v；
                     给正数则用固定 sigma 的 gaussian_filter1d。
        sigma_slope / sigma_intercept: 变带宽参数（**物理含义：场内风速空间分布**）。
        sigma_truncate: 高斯核截断倍数。
        w_step / w_max: 风速网格步长与上限。
        plot / ax / plot_kwargs: 绘图（Stage B 将把绘图从这个数值函数中剥离）。

    Returns:
        (w_grid, cf_grid)：风速网格 (float32) 与容量因子 (float32)。
    """
    if not (cut_in_v < rated_v < cut_out_v):
        raise ValueError(
            f"风速约束条件失败: 需要 cut_in_v < rated_v < cut_out_v，"
            f"实际值为 cut_in_v={cut_in_v}, rated_v={rated_v}, cut_out_v={cut_out_v}"
        )
    if w_step <= 0 or w_max <= cut_out_v:
        raise ValueError(
            f"网格参数无效: 需要 w_step > 0 且 w_max > cut_out_v，"
            f"实际值为 w_step={w_step}, w_max={w_max}"
        )
    if power_order <= 0:
        raise ValueError(f"power_order 必须为正整数，实际值为 {power_order}")
    if plot and not isinstance(plot_kwargs, (dict, type(None))):
        raise TypeError(f"plot_kwargs 必须为字典或 None，实际类型为 {type(plot_kwargs)}")

    w_grid = np.arange(0, w_max + w_step, w_step, dtype=np.float32)
    dw = w_step

    cf_raw = raw_capacity_factors(w_grid, cut_in_v, rated_v, cut_out_v,
                                  v_slope, v_intercept, power_order)

    if not smooth:
        cf_grid = cf_raw
    else:
        if fixed_sigma is not False and isinstance(fixed_sigma, (int, float)):
            if fixed_sigma <= 0:
                raise ValueError(f"fixed_sigma 必须为正数，实际值为 {fixed_sigma}")
            sigma_pix = float(fixed_sigma) / dw
            cf_grid = gaussian_filter1d(
                cf_raw,
                sigma=sigma_pix,
                mode='nearest',
                truncate=sigma_truncate,
                output=np.float32,
            )
        elif fixed_sigma is False:
            # ★ 与原实现完全相同的 sigma 计算（float32 网格），以保证内部区逐位一致；
            #   区别只在于平滑本身改为"以 i 为基准 + 实际权重归一化"的正确形式。
            sigma_pix_grid = (sigma_intercept + sigma_slope * w_grid) / dw
            cf_grid = smooth_capacity_factors(cf_raw, sigma_pix_grid, sigma_truncate)
        else:
            raise ValueError(f"fixed_sigma 必须为 False 或正数，实际值为 {fixed_sigma}")

    if plot:
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))
            own_fig = True
        else:
            own_fig = False

        kw = {'label': 'Capacity Factor'}
        if plot_kwargs:
            kw.update(plot_kwargs)

        ax.plot(w_grid, cf_grid, **kw)
        ax.set_xlabel('风速 (m/s)')
        ax.set_ylabel('容量因子')
        ax.set_title('风电功率曲线')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, w_max)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

        if own_fig:
            plt.tight_layout()
            plt.show()

    return w_grid, cf_grid
