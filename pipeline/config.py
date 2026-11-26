"""集中式路径配置（**唯一来源**）

本地与集群的差异全部通过环境变量覆盖，不改代码：

    WINDPAPER_ROOT        项目根目录      （默认：集群 /geogfs1/groups/geog_liulb/hongyu）
    WINDPAPER_DATA        输入数据根目录  （默认 <ROOT>/data）
    WINDPAPER_RESULTS     结果根目录      （默认 <ROOT>/results_v2）
    WINDPAPER_CURVE_DIR   功率曲线目录    （默认 <DATA>/farm_power_curves）
    WINDPAPER_ERA5_DIR    ERA5 数据目录   （默认 <DATA>/ERA5_China；本地可指向 D:\\Data\\ERA5）
    WINDPAPER_CDMET_DIR   CDMet 数据目录  （默认 <DATA>/CDMet；**允许缺省**）

六个变量**互相独立**，因此可以做到"输入只读、新产物另写"。

⚠️ `results_root()` 默认值由 `<ROOT>/results` 改为 `<ROOT>/results_v2`
   理由：集群旧目录已改名为 `results_old_20260929`；若默认仍写 `<ROOT>/results`，
   忘记设环境变量时会**悄悄新建一个 results/ 目录**，与 `results_v2` 错位、且看起来"成功了"。

本地重跑布局（见 _run\\ENVIRONMENT.md 与 CalculationCode\\RUN_FULL_CHAIN.md）
--------------------------------------------------------------------------
    DATA_INPUT\\   输入（**只读**）：9 个 CSV、province_city_windfarm_mapping\\、farm_power_curves\\（旧曲线，作输入）
    Results\\      结果目录：**新产物**（farm_power_curves\\、farm_raw_w\\、kb_batch_aggr\\、final_gen\\ 等）写这里
    OldOutput\\    旧 run 的其余产物（只读对照）

⚠️ **`Results\\farm_raw_w` 不是"可复用旧产物"**：审计已逐位证明它是**修密度之前**的旧 run
   输出（可用「旧气压指数 + dem=0 + 真实 z」复现）。BUG-1 修好后必须**重算**，
   否则密度修正的 +9.4% 偏差会沿全链污染。

`curve_dir()` 单独可配是为了**读写分离**：step1 生成的逐场功率曲线属于**新产物**，
必须写到 `Results\\farm_power_curves`，不能写进只读的输入目录——否则会覆盖与集群
校验和一致的输入快照（默认值 `<DATA>/farm_power_curves` 正落在输入目录里）。

本地重跑环境变量（最终版）
-------------------------
    WINDPAPER_DATA       = "D:\\HKU PhD\\wind data paper\\DATA_INPUT"      # 只读输入
    WINDPAPER_RESULTS    = "D:\\HKU PhD\\wind data paper\\Results"         # 新结果写这里
    WINDPAPER_CURVE_DIR  = "D:\\HKU PhD\\wind data paper\\Results\\farm_power_curves"
    WINDPAPER_ERA5_DIR   = "D:\\Data\\ERA5"                                # 本地原始 ERA5（含 wind/ geopotential.nc）
    WINDPAPER_CDMET_DIR  = ""                                              # 本地无 CDMet：留空即禁用

HPC 全链重跑（用户已确认）
-------------------------
    WINDPAPER_ROOT       = /geogfs1/groups/geog_liulb/hongyu
    WINDPAPER_DATA       = /geogfs1/groups/geog_liulb/hongyu/data            （只读输入）
    WINDPAPER_RESULTS    = /geogfs1/groups/geog_liulb/hongyu/results_v2      （新输出）
    WINDPAPER_CURVE_DIR  = /geogfs1/groups/geog_liulb/hongyu/results_v2/farm_power_curves
    （旧 `${project}/results/` 已改名为 `results_old_20260929/`，不再使用；全链模式不需要符号链接）

为什么结果根要可配
------------------
`results_root()` 下的子目录名与流水线步骤输出一一对应
（`farm_raw_w` / `kb_batch_aggr` / `prov_w_compare` / `prov_w_smape_enums` / `prov_gen_enums` /
 `prov_gen_full_smape` / `optimal_kb` / `calibrated_gen` / `final_gen` / `no_kb_gen` / `kb_validation`）。
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

# Default to the repository root; cluster users should set WINDPAPER_ROOT.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return Path(value).expanduser()


def project_root() -> Path:
    """项目根目录。"""
    return _env_path("WINDPAPER_ROOT") or DEFAULT_PROJECT_ROOT


def data_root() -> Path:
    """输入数据根目录（风场清单、ERA5_China、CDMet、月度发电量等）。"""
    return _env_path("WINDPAPER_DATA") or (project_root() / "data")


def results_root() -> Path:
    """结果根目录（各步骤的输入/输出都在其子目录下）。

    默认 `<ROOT>/results`。
    """
    return _env_path("WINDPAPER_RESULTS") or (project_root() / "results")


def curve_dir() -> Path:
    """逐场功率曲线目录（step1 的输出；step2/step9/step10/step11 的输入）。

    默认 `<DATA>/farm_power_curves`（与改动前一致），可用 `WINDPAPER_CURVE_DIR` 覆盖。
    本地/HPC 重跑都应指向结果目录下的 `farm_power_curves` 实现**读写分离**：
    新曲线写进结果目录，只读的输入目录全程不被写入。
    """
    return _env_path("WINDPAPER_CURVE_DIR") or (data_root() / "farm_power_curves")


def era5_dir() -> Path:
    """ERA5 数据目录。默认 `<DATA>/ERA5_China`；本地可用 `WINDPAPER_ERA5_DIR=D:\\Data\\ERA5`。

    代码期望的布局：`{era5_dir}/wind/era5_wind_<YYYY>.nc`、`{era5_dir}/solar/...`、
    `{era5_dir}/building/...`、`{era5_dir}/geopotential.nc`。
    """
    return _env_path("WINDPAPER_ERA5_DIR") or (data_root() / "ERA5_China")


def cdmet_dir() -> Path | None:
    """CDMet 数据目录（可选）。未显式设置且 `<DATA>/CDMet` 不存在时返回 None。

    代码仅在 2000–2020 年尝试读取 CDMet，失败只记 warning ⇒ **缺省不影响全链跑通**。
    """
    explicit = _env_path("WINDPAPER_CDMET_DIR")
    if explicit is not None:
        return explicit
    candidate = data_root() / "CDMet"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# 并行预算（**防 HPC 超订惩罚**：用户硬约束并行总数 ≤ 96；节点是 384 核）
# ---------------------------------------------------------------------------
DEFAULT_MAX_PARALLEL = 96


def max_parallel_limit() -> int:
    """并行总上限：默认 96，可用 `WINDPAPER_MAX_PARALLEL` 覆盖。"""
    raw = os.environ.get("WINDPAPER_MAX_PARALLEL")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_PARALLEL
    try:
        value = int(raw)
    except ValueError:
        warnings.warn("WINDPAPER_MAX_PARALLEL=%r 不是整数，回落 %d" % (raw, DEFAULT_MAX_PARALLEL),
                      UserWarning, stacklevel=2)
        return DEFAULT_MAX_PARALLEL
    return max(1, value)


def resolve_worker_count(max_workers: Optional[int], n_tasks: int, *, context: str = "") -> int:
    """单层 worker 数。

    ⚠️ `max_workers=None` 时**绝不能**按 `cpu_count - 1` 无界回落：384 核节点会得到 383
    （远超 96，会被惩罚）。这里统一夹到 `max_parallel_limit()`。
    显式给的值只按任务数收敛（不擅自改调用方明确指定的数）。
    """
    n_tasks = max(1, int(n_tasks))
    if max_workers is None:
        cpu = os.cpu_count() or 1
        return max(1, min(n_tasks, cpu - 1, max_parallel_limit()))
    return max(1, min(int(max_workers), n_tasks))


def resolve_outer_workers(outer_max_workers: Optional[int], n_tasks: int, *,
                          context: str = "") -> int:
    """外层 worker 数：`None` 时回落成任务数，但**必须**夹到上限
    （例如 kb_batch_aggr 的 outer=None 会变成 len(kb_pairs)=1230）。"""
    n_tasks = max(1, int(n_tasks))
    if outer_max_workers is None:
        return max(1, min(n_tasks, max_parallel_limit()))
    return max(1, min(int(outer_max_workers), n_tasks))


def clamp_two_layer_workers(outer: int, inner: Optional[int], *, context: str = "") -> tuple[int, Optional[int]]:
    """保证两层步骤的 **outer × inner ≤ 上限**（两层是**相乘**，不是相加）。

    超预算时把 `inner` 夹到 `limit // outer` 并**告警**（不静默超订）。
    `inner=None` 原样返回（下游 `resolve_worker_count` 会自行夹取）。
    """
    limit = max_parallel_limit()
    outer = max(1, int(outer))
    if inner is None:
        return outer, None
    inner = max(1, int(inner))
    if outer * inner > limit:
        capped = max(1, limit // outer)
        warnings.warn(
            "两层并行 %d×%d=%d 超上限 %d%s，已把 inner 夹到 %d（防 HPC 超订惩罚）"
            % (outer, inner, outer * inner, limit, ("（%s）" % context) if context else "", capped),
            UserWarning,
            stacklevel=2,
        )
        inner = capped
    return outer, inner


def describe() -> str:
    """打印当前生效的路径（供 runner 启动时自检/留痕）。"""
    src = {
        "WINDPAPER_ROOT": os.environ.get("WINDPAPER_ROOT"),
        "WINDPAPER_DATA": os.environ.get("WINDPAPER_DATA"),
        "WINDPAPER_RESULTS": os.environ.get("WINDPAPER_RESULTS"),
        "WINDPAPER_CURVE_DIR": os.environ.get("WINDPAPER_CURVE_DIR"),
        "WINDPAPER_ERA5_DIR": os.environ.get("WINDPAPER_ERA5_DIR"),
        "WINDPAPER_CDMET_DIR": os.environ.get("WINDPAPER_CDMET_DIR"),
    }
    lines = ["路径配置（* = 由环境变量覆盖）:"]
    for label, key, path in (
        ("project_root", "WINDPAPER_ROOT", project_root()),
        ("data_root", "WINDPAPER_DATA", data_root()),
        ("results_root", "WINDPAPER_RESULTS", results_root()),
        ("curve_dir", "WINDPAPER_CURVE_DIR", curve_dir()),
        ("era5_dir", "WINDPAPER_ERA5_DIR", era5_dir()),
        ("cdmet_dir", "WINDPAPER_CDMET_DIR", cdmet_dir()),
    ):
        mark = " *" if src.get(key) else "  "
        lines.append("  %s %-16s %s" % (mark, label, path))
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
