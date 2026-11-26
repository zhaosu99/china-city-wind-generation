
import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from farm_power_curves import generate_farm_power_curves_csvs
from unit_farm_workflow import batch_farm_output_from_files
from kb_batch_aggr import kb_batch_aggr
from prov_gen_enums import prov_gen_enums
from prov_gen_full_smape import prov_gen_full_smape
from prov_w_compare import prov_w_compare
from prov_w_smape import prov_w_smape
from find_optimal_kb_full_data import find_optimal_kb_full_data
from farm_aggr_with_kb import farm_aggr_with_kb
from final_gen import final_gen
from no_kb_gen import no_kb_gen

# 路径集中到 config.py（可用 WINDPAPER_ROOT / WINDPAPER_DATA / WINDPAPER_RESULTS /
# WINDPAPER_CURVE_DIR / WINDPAPER_ERA5_DIR / WINDPAPER_CDMET_DIR 覆盖）
from config import (
    project_root,
    data_root,
    results_root,
    curve_dir,
    era5_dir as config_era5_dir,
    cdmet_dir as config_cdmet_dir,
)

project_root_dir = project_root()
data_root_dir = data_root()
results_root_dir = results_root()

# 盘点表优先用**带海拔**的增强表（BUG-1：没有海拔列会让密度修正方向反向）
_inv_elevation = data_root_dir / "wind_farms_inventory_till_2023_with_elevation.csv"
_inv_plain = data_root_dir / "wind_farms_inventory_till_2023.csv"
if _inv_elevation.exists():
    farm_inventory_csv_path = _inv_elevation
else:
    if _inv_plain.exists():
        warnings.warn(
            "未找到带海拔的盘点表 %s，回退到 %s（无海拔列 ⇒ 运行时缺海拔会报错，"
            "除非显式开启 allow_era5_orography_fallback）" % (_inv_elevation, _inv_plain),
            UserWarning,
            stacklevel=2,
        )
    farm_inventory_csv_path = _inv_plain
era5_dir = config_era5_dir()
cdmet_dir = config_cdmet_dir()
# 功率曲线目录单独可配（读写分离）：默认仍在 data_root 下；本地重跑应设
# WINDPAPER_CURVE_DIR=Results\farm_power_curves，避免把新曲线写进只读输入目录
farm_power_curve_dir = curve_dir()
city_capacity_adjust_csv_path = data_root_dir / "city_capacity_adjust_info.csv"
prov_utility_rate_adjust_csv_path = data_root_dir / "prov_wind_utility_rates.csv"
monthly_current_gen_csv_path = data_root_dir / "分省月度数据-风电-当期值(亿千瓦时).csv"
monthly_cumulative_gen_csv_path = data_root_dir / "分省月度数据-风电-当年累积值(亿千瓦时).csv"
pre2012_yearly_gen_csv_path = data_root_dir / "2012年以前风电发电量(亿千瓦时).csv"
admin_province_csv_path = data_root_dir / "省市名全简称.csv"


# batch_farm_output_from_files 关键调度参数
# 注意：旧 run 的可复用输出（farm_raw_w）就留在 results_root 下，故 step3 仍指向 results_root_dir
batch_output_dir = results_root_dir / "farm_raw_w"
batch_output_gen_subdir = "gen"
batch_output_w_compare_subdir = "w_compare"
batch_output_raw_w_subdir = "raw_w"
batch_farm_adjusted_w_varname = "raw_w"

# kb_batch_aggr 关键调度参数
kb_output_dir = results_root_dir / "kb_batch_aggr"
kb_farm_adjusted_w_dir = batch_output_dir / batch_output_raw_w_subdir

# prov_w_compare 关键调度参数
prov_w_compare_input_dir = batch_output_dir / batch_output_w_compare_subdir
prov_w_compare_output_dir = results_root_dir / "prov_w_compare"

# prov_w_smape 关键调度参数
# step5 跳过时复用 results_root 下旧 prov_w_compare 的输出（step4 亦跳过、就地复用）
prov_w_smape_input_dir = prov_w_compare_output_dir
prov_w_smape_output_dir = results_root_dir / "prov_w_smape_enums"

# prov_gen_enums 关键调度参数
prov_gen_enums_input_dir = kb_output_dir / "gen" / "prov"
prov_gen_enums_output_dir = results_root_dir / "prov_gen_enums"

# prov_gen_full_smape 关键调度参数
prov_gen_full_smape_input_dir = prov_gen_enums_output_dir
prov_gen_full_smape_output_dir = results_root_dir / "prov_gen_full_smape"

# find_optimal_kb_full_data 关键调度参数
optimal_kb_input_dir = prov_gen_full_smape_output_dir
optimal_kb_wind_dir = prov_w_smape_output_dir
optimal_kb_output_dir = results_root_dir / "optimal_kb"
optimal_kb_csv = optimal_kb_output_dir / "optimal_kb.csv"

# calibrated_gen 关键调度参数
calibrated_gen_optimal_kb_csv = optimal_kb_csv
calibrated_gen_output_dir = results_root_dir / "calibrated_gen"

# final_gen 关键调度参数
final_gen_input_dir = calibrated_gen_output_dir / "gen"
final_gen_output_dir = results_root_dir / "final_gen"

# no_kb_gen 关键调度参数
no_kb_gen_output_dir = results_root_dir / "no_kb_gen"



STEP_ARG_NAME_MAP = {
    1: "run_step_1_farm_power_curves",
    2: "run_step_2_batch_farm_output",
    3: "run_step_3_kb_batch_aggr",
    4: "run_step_4_prov_w_compare",
    5: "run_step_5_prov_w_smape",
    6: "run_step_6_prov_gen_enums",
    7: "run_step_7_prov_gen_full_smape",
    8: "run_step_8_find_optimal_kb_full_data",
    9: "run_step_9_calibrated_gen",
    10: "run_step_10_final_gen",
    11: "run_step_11_no_kb_gen",
}

STEP_LABEL_MAP = {
    1: "farm_power_curves",
    2: "batch_farm_output",
    3: "kb_batch_aggr",
    4: "prov_w_compare",
    5: "prov_w_smape",
    6: "prov_gen_enums",
    7: "prov_gen_full_smape",
    8: "find_optimal_kb_full_data",
    9: "calibrated_gen",
    10: "final_gen",
    11: "no_kb_gen",
}

STEP_DEFAULT_ENABLED = {
    "run_step_1_farm_power_curves": True,
    "run_step_2_batch_farm_output": True,
    "run_step_3_kb_batch_aggr": True,
    "run_step_4_prov_w_compare": True,
    "run_step_5_prov_w_smape": True,
    "run_step_6_prov_gen_enums": True,
    "run_step_7_prov_gen_full_smape": True,
    "run_step_8_find_optimal_kb_full_data": True,
    "run_step_9_calibrated_gen": True,
    "run_step_10_final_gen": True,
    "run_step_11_no_kb_gen": True,
}


def _report_step_status(step: int, status: str, detail: str | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[panel][{timestamp}] Step {step}/{len(STEP_LABEL_MAP)} {status}: {STEP_LABEL_MAP[step]}"
    if detail:
        message = f"{message} | {detail}"
    print(message, flush=True)


def _report_pipeline_plan(
    step_kwargs: dict[str, bool],
    max_workers: int,
    outer_max_workers: int,
    inner_max_workers: int,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    enabled_steps = [
        f"{step}:{STEP_LABEL_MAP[step]}"
        for step, arg_name in STEP_ARG_NAME_MAP.items()
        if step_kwargs[arg_name]
    ]
    skipped_steps = [
        f"{step}:{STEP_LABEL_MAP[step]}"
        for step, arg_name in STEP_ARG_NAME_MAP.items()
        if not step_kwargs[arg_name]
    ]

    print(f"[panel][{timestamp}] Pipeline started", flush=True)
    print(
        (
            f"[panel][{timestamp}] Worker config: max_workers={max_workers}, "
            f"outer_max_workers={outer_max_workers}, inner_max_workers={inner_max_workers}"
        ),
        flush=True,
    )
    print(
        f"[panel][{timestamp}] Enabled steps: {', '.join(enabled_steps) if enabled_steps else 'none'}",
        flush=True,
    )
    print(
        f"[panel][{timestamp}] Skipped steps: {', '.join(skipped_steps) if skipped_steps else 'none'}",
        flush=True,
    )


def _run_panel_step(results: dict, step: int, enabled: bool, result_key: str, func,
                    records: list | None = None, output_spec: dict | None = None,
                    **kwargs) -> None:
    if not enabled:
        _report_step_status(step, "skipped")
        if records is not None:
            records.append({"step": step, "name": STEP_LABEL_MAP[step], "failure_count": 0,
                            "total": None, "success": None, "samples": [], "skipped": True})
        return

    _report_step_status(step, "started")
    try:
        results[result_key] = func(**kwargs)
    except Exception as exc:
        _report_step_status(step, "failed", f"{type(exc).__name__}: {exc}")
        if records is not None:
            records.append({"step": step, "name": STEP_LABEL_MAP[step], "failure_count": None,
                            "total": None, "success": None,
                            "samples": ["%s: %s" % (type(exc).__name__, exc)], "raised": True})
            # 步骤抛错时也要留下部分摘要（否则前面的失败件无人知晓）
            print(format_failure_summary(records), flush=True)
        raise

    info = extract_step_failures(results[result_key])
    # 返回值里没有 failure_count 的步骤（如 final_gen / no_kb_gen）：用产物校验兜底，
    # 校验通过 ⇒ 记 WARNING 但不判失败；不通过 ⇒ 记为失败（防"5 天作业跑完却无法背书"）。
    if info.get("failure_count") is None and output_spec is not None:
        check = verify_step_outputs(output_spec)
        info["output_check"] = "ok" if check["ok"] else "failed"
        info["output_check_detail"] = check["reason"]
        if check["ok"]:
            print("[panel][%s] \u2139\ufe0f Step %d/%d %s 返回值无 failure_count，"
                  "产物校验通过：%s"
                  % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, len(STEP_LABEL_MAP),
                     STEP_LABEL_MAP[step], check["reason"]), flush=True)
        else:
            info["failure_count"] = 1
            info["samples"] = ["产物校验失败: %s" % check["reason"]]
    if records is not None:
        records.append({"step": step, "name": STEP_LABEL_MAP[step], **info})

    # 单步**醒目**告警：failure_count>0 立刻可见（不再只写 "completed"）
    if isinstance(info.get("failure_count"), int) and info["failure_count"] > 0:
        print(
            "[panel][%s] ⚠️ Step %d/%d %s 存在失败件: total=%s success=%s failure=%s "
            "（详细日志见 stderr；全链严格模式下本次 run 将判为 CHAIN INCOMPLETE）"
            % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, len(STEP_LABEL_MAP),
               STEP_LABEL_MAP[step], info.get("total"), info.get("success"), info["failure_count"]),
            flush=True,
        )
    _report_step_status(step, "completed")


def _parse_bool(value: bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise argparse.ArgumentTypeError("布尔参数不能为空")

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        f"布尔参数只能为 True/False，收到: {value}"
    )


def _parse_step_list(value: str | None) -> set[int] | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text or text == "all":
        return set(STEP_ARG_NAME_MAP)

    steps = set()
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            step = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"步骤编号必须是 1-12 的整数，收到: {item}"
            ) from exc
        if step not in STEP_ARG_NAME_MAP:
            raise argparse.ArgumentTypeError(
                f"步骤编号超出范围，收到: {step}，允许值为 1-12"
            )
        steps.add(step)

    if not steps:
        raise argparse.ArgumentTypeError("步骤列表不能为空")

    return steps



# ---------------------------------------------------------------------------
# k/b 网格（step3 kb_batch_aggr 与 step5 prov_w_smape 都要枚举）
# ---------------------------------------------------------------------------
# 正式重跑用 full（k/b 步长 0.1，≈1230 组合）；验收/冒烟建议 coarse（步长 0.5，≈63 组合），
# 只让"最优 KB 点"变粗，不损失链条完整性。
KB_GRID_PRESETS: dict[str, dict[str, float]] = {
    "full": {"k_step": 0.1, "b_step": 0.1},
    "coarse": {"k_step": 0.5, "b_step": 0.5},
}


def resolve_kb_grid(preset: str | None = None, *, k_start=None, k_end=None, k_step=None,
                    b_start=None, b_end=None, b_step=None) -> dict[str, float]:
    """把 `--kb-grid` 预设与显式参数合并成一个 dict（**显式参数优先**）。

    全都不给 ⇒ 返回空 dict ⇒ 各调用点沿用自己原来的硬编码默认（行为不变）。
    """
    if preset is not None and preset not in KB_GRID_PRESETS:
        raise ValueError("kb_grid 预设只能是 %s，得到 %r" % (sorted(KB_GRID_PRESETS), preset))
    grid: dict[str, float] = dict(KB_GRID_PRESETS.get(preset or "", {}))
    for name, value in (("k_start", k_start), ("k_end", k_end), ("k_step", k_step),
                        ("b_start", b_start), ("b_end", b_end), ("b_step", b_step)):
        if value is not None:
            grid[name] = float(value)
    return grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full panel pipeline or a selected subset of steps."
    )
    parser.add_argument("--max-workers", type=int, default=96)
    parser.add_argument("--outer-max-workers", type=int, default=12)
    parser.add_argument("--inner-max-workers", type=int, default=8)
    parser.add_argument(
        "--steps",
        type=_parse_step_list,
        default=None,
        help="仅运行这些步骤，逗号分隔，例如 3,4,5；默认运行全部步骤。",
    )
    parser.add_argument(
        "--skip-steps",
        type=_parse_step_list,
        default=None,
        help="跳过这些步骤，逗号分隔，例如 1,2。",
    )
    parser.add_argument(
        "--kb-grid",
        choices=sorted(KB_GRID_PRESETS),
        default=None,
        help="k/b 网格预设：full=步长0.1（正式重跑，≈1230 组合）；"
             "coarse=步长0.5（验收/冒烟，≈63 组合，只让最优 KB 点变粗）。"
             "也可用 --kb-k-start/--kb-k-end/--kb-k-step/--kb-b-start/--kb-b-end/--kb-b-step 单独覆盖。",
    )
    for _name, _help in (
        ("kb-k-start", "k 起点（默认按调用点，step3=0.1 / step5=0.5）"),
        ("kb-k-end", "k 终点（默认 3.0）"),
        ("kb-k-step", "k 步长（默认 0.1，full）"),
        ("kb-b-start", "b 起点（默认 -2.0）"),
        ("kb-b-end", "b 终点（默认 2.0）"),
        ("kb-b-step", "b 步长（默认 0.1，full）"),
    ):
        parser.add_argument(f"--{_name}", type=float, default=None, help=_help)

    parser.add_argument(
        "--full-chain",
        action="store_true",
        help="跑完整 step 1→12（含 step2/4/5 与 step12），不依赖任何『就地复用』的旧产物。"
             "与 --skip-steps 同时给出时，--skip-steps 仍然生效。开启后默认 strict_failures。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="任一步 failure_count>0 时**显式失败**（非零退出 + CHAIN INCOMPLETE 摘要）。"
             "--full-chain 默认已开启。",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="即使有失败件也不失败退出（仅在明确知道自己在做什么时使用）。",
    )
    parser.set_defaults(strict=False)

    for step, arg_name in STEP_ARG_NAME_MAP.items():
        parser.add_argument(
            f"--{arg_name}",
            f"--{arg_name.replace('_', '-')}",
            dest=arg_name,
            type=_parse_bool,
            default=None,
            help=f"是否运行 step {step}: {STEP_LABEL_MAP[step]}。可传 True/False。",
        )

    return parser


# ---------------------------------------------------------------------------
# 并行预算护栏（用户硬约束：HPC 并行总数必须 ≤ 96，超了会被惩罚）
# ---------------------------------------------------------------------------
# 逐 step 的并行结构（**已逐行核实**，见 RUN_FULL_CHAIN.md 的并发表）：
#   单层 max_workers      : step2, 4, 5, 6, 7, 8, 11, 12
#   两层 outer × inner   : step3 (kb_batch_aggr), step9 (farm_aggr_with_kb)
#   无并行（串行）        : step1 (generate_farm_power_curves_csvs), step10 (final_gen)
# 两层步骤的**有效并发 = outer × inner**（相乘，不是相加）⇒ 必须单独校验乘积。
DEFAULT_MAX_PARALLEL = 96

# 逐 step 的并行结构（**逐行核实**，见 RUN_FULL_CHAIN.md 的并发表；单测锁住，避免退化）：
#   single_layer -> 只用 max_workers
#   two_layer    -> outer_max_workers × inner_max_workers（**相乘**）
#   none         -> 没有进程池（串行，打不满 96）
STEP_PARALLEL_STRUCTURE: dict[int, str] = {
    1: "none",           # generate_farm_power_curves_csvs：逐场串行循环
    2: "single_layer",   # batch_farm_output_from_files(process_workers=max_workers)
    3: "two_layer",      # kb_batch_aggr(outer_max_workers, inner_max_workers)
    4: "single_layer",   # prov_w_compare(max_workers=...)
    5: "single_layer",   # prov_w_smape(max_workers=...)
    6: "single_layer",   # prov_gen_enums(max_workers=...)
    7: "single_layer",   # prov_gen_full_smape(max_workers=...)
    8: "single_layer",   # find_optimal_kb_full_data(max_workers=...)
    9: "two_layer",      # farm_aggr_with_kb(outer_max_workers, inner_max_workers)
    10: "none",          # final_gen：无进程池
    11: "single_layer",  # no_kb_gen(max_workers=...) → aggr_farm_output(max_workers)
}


def effective_concurrency(step: int, max_workers: int, outer_max_workers: int,
                          inner_max_workers: int) -> int:
    """某 step 在当前参数下的**有效并发**（供并发表与护栏使用）。"""
    structure = STEP_PARALLEL_STRUCTURE.get(step, "single_layer")
    if structure == "two_layer":
        return int(outer_max_workers) * int(inner_max_workers)
    if structure == "single_layer":
        return int(max_workers)
    return 1


# ---------------------------------------------------------------------------
# 失败门禁（failure_count > 0 必须让全链**显式失败**，否则重跑会静默缺件）
# ---------------------------------------------------------------------------
# 实证：冒烟 step3 出现 total=1230 success=1167 **failure=63**（进程池被终止），
# 但 panel 仍打印 "Step 3/12 completed" ⇒ 后续步骤建立在缺 5.1% 网格的数据上。
def extract_step_failures(result: Any) -> dict[str, Any]:
    """从步骤返回值里抽取 total / success / failure_count / 失败样例。

    返回 dict：``failure_count=None`` 表示**无法判定**（不是 0！），门禁会就此告警。
    """
    info: dict[str, Any] = {"failure_count": None, "total": None, "success": None, "samples": []}
    if not isinstance(result, dict):
        return info

    failed = None
    for key in ("failed", "failed_tasks", "failed_batches", "failures", "failed_results"):
        value = result.get(key)
        if isinstance(value, (list, tuple)):
            failed = value
            break

    fc = result.get("failure_count")
    if isinstance(fc, bool):
        fc = None
    if isinstance(fc, int):
        info["failure_count"] = int(fc)
    elif failed is not None:
        info["failure_count"] = len(failed)

    success = result.get("success")
    if isinstance(success, (list, tuple)):
        info["success"] = len(success)
    elif isinstance(success, bool):
        info["success"] = None
    elif isinstance(success, int):
        info["success"] = int(success)

    for key in ("total", "task_count", "batch_count"):
        value = result.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            info["total"] = int(value)
            break
    if info["total"] is None and info["success"] is not None and info["failure_count"] is not None:
        info["total"] = info["success"] + info["failure_count"]

    samples: list[str] = []
    errs = result.get("error_map")
    if isinstance(errs, dict):
        for key, value in list(errs.items())[:5]:
            samples.append("%s: %s" % (key, str(value)[:160]))
    if failed is not None:
        for item in list(failed):
            if len(samples) >= 5:
                break
            if isinstance(item, dict):
                # step8 等步骤用 {'prov','season','error'} 形式；优先给出可读的失败原因
                label = "/".join(str(item[k]) for k in ("prov", "season") if k in item)
                detail = item.get("error") or item.get("reason") or item.get("message") or item
                samples.append(("%s: %s" % (label, detail))[:200] if label else str(item)[:200])
            else:
                samples.append(str(item)[:200])
    info["samples"] = samples[:5]
    return info


def verify_step_outputs(spec: dict[str, Any]) -> dict[str, Any]:
    """产物存在性 + 非空 + 抽样可解析（供返回值里没有 failure_count 的步骤使用）。

    spec = {"dir": <输出目录>, "pattern": "**/*.nc", "min_files": 1}
    返回 {"ok": bool, "found": int, "reason": str}
    """
    directory = Path(str(spec.get("dir", "")))
    pattern = str(spec.get("pattern", "**/*"))
    min_files = int(spec.get("min_files", 1))
    if not directory.exists():
        return {"ok": False, "found": 0, "reason": "输出目录不存在: %s" % directory}
    files = sorted(p for p in directory.glob(pattern) if p.is_file())
    nonempty = [p for p in files if p.stat().st_size > 0]
    zero = [p for p in files if p.stat().st_size == 0]
    if len(nonempty) < min_files:
        return {"ok": False, "found": len(nonempty),
                "reason": "非空产物 %d 件 < 期望 %d 件（%s）" % (len(nonempty), min_files, directory)}
    if zero:
        return {"ok": False, "found": len(nonempty),
                "reason": "%d 件为 0 字节（例: %s）" % (len(zero), zero[0].name)}
    # 抽样可解析（只抽 2 件，代价可忽略）
    for p in nonempty[:2]:
        try:
            if p.suffix.lower() == ".csv":
                import pandas as pd
                pd.read_csv(p, nrows=0, encoding="utf-8-sig", low_memory=False)
            elif p.suffix.lower() == ".nc":
                import xarray as xr
                with xr.open_dataset(p, decode_times=False):
                    pass
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "found": len(nonempty),
                    "reason": "产物不可解析: %s（%s）" % (p.name, exc)}
    return {"ok": True, "found": len(nonempty),
            "reason": "产物 %d 件，抽样可解析" % len(nonempty)}


def format_failure_summary(records: list[dict[str, Any]]) -> str:
    """逐步 total/success/failure 摘要 + CHAIN INCOMPLETE 标记 + 日志位置。"""
    lines = ["", "=" * 92, "[panel] 逐步失败摘要（failure_count > 0 ⇒ 缺件，不可当作成功）", "=" * 92]
    lines.append("  %-4s %-26s %9s %9s %9s" % ("step", "name", "total", "success", "failure"))
    failed_steps: list[str] = []
    unknown_steps: list[str] = []
    for record in records:
        fc = record.get("failure_count")
        if fc is None:
            unknown_steps.append("step %s %s" % (record.get("step"), record.get("name")))
        elif fc > 0:
            failed_steps.append("step %s %s(failure=%d)" % (record.get("step"), record.get("name"), fc))
        lines.append("  %-4s %-26s %9s %9s %9s" % (
            record.get("step"), record.get("name"),
            record.get("total") if record.get("total") is not None else "-",
            record.get("success") if record.get("success") is not None else "-",
            fc if fc is not None else (
                "未知(产物已校验)" if record.get("output_check") == "ok" else "未知"),
        ))
        for sample in (record.get("samples") or [])[:5]:
            lines.append("        · %s" % sample)
    lines.append("-" * 92)
    if failed_steps:
        lines.append("⚠️ CHAIN INCOMPLETE：%d 个步骤有失败件 ⇒ %s" % (len(failed_steps), "; ".join(failed_steps)))
        lines.append("   后续步骤已建立在**缺件**数据上，不要把本次 run 当作成功。")
    elif unknown_steps:
        verified = [r for r in records if r.get("failure_count") is None and r.get("output_check") == "ok"]
        unverified = [r for r in records if r.get("failure_count") is None and r.get("output_check") != "ok"]
        if verified:
            lines.append("ℹ️ 以下步骤返回值无法判定 failure_count，但**产物存在性/可解析校验通过**"
                         "（不判 CHAIN INCOMPLETE）：%s"
                         % "; ".join("step %s %s(%s)" % (r.get("step"), r.get("name"),
                                                         r.get("output_check_detail", ""))
                                     for r in verified))
        if unverified:
            lines.append("⚠️ 以下步骤无法判定 failure_count（未知，不等于 0）且未做产物校验：%s"
                         % "; ".join("step %s %s" % (r.get("step"), r.get("name")) for r in unverified))
    else:
        lines.append("✅ all steps reported failure_count=0（全部步骤无失败件）")
    lines.append("各步骤详细日志位置：走 **stderr**（stdout 只有 panel 摘要）；"
                 "重定向即可留存，例：python panel.py --full-chain > run.log 2>&1")
    lines.append("=" * 92)
    return "\n".join(lines)


def enforce_failure_gate(records: list[dict[str, Any]], strict: bool) -> None:
    """任一步 failure_count>0 ⇒ strict 时 raise RuntimeError；否则告警。

    ``failure_count=None``（未知）在 strict 下也**必须告警**，不能静默当 0。
    """
    failed_steps = [r for r in records if isinstance(r.get("failure_count"), int) and r["failure_count"] > 0]
    unknown_steps = [r for r in records if r.get("failure_count") is None]

    verified_unknown = [r for r in unknown_steps if r.get("output_check") == "ok"]
    unverified_unknown = [r for r in unknown_steps if r.get("output_check") != "ok"]
    if verified_unknown:
        warnings.warn(
            "以下步骤返回值无法判定 failure_count，但**产物存在性/可解析校验通过**"
            "（不判 CHAIN INCOMPLETE）: %s"
            % ", ".join("step %s %s(%s)" % (r.get("step"), r.get("name"),
                                           r.get("output_check_detail", ""))
                        for r in verified_unknown),
            UserWarning,
            stacklevel=2,
        )
    if unverified_unknown:
        warnings.warn(
            "以下步骤无法判定 failure_count（未知，不等于 0）且未做产物校验，请核对日志: %s"
            % ", ".join("step %s %s" % (r.get("step"), r.get("name")) for r in unverified_unknown),
            UserWarning,
            stacklevel=2,
        )
    if failed_steps:
        message = "CHAIN INCOMPLETE: %d 个步骤存在失败件 —— %s" % (
            len(failed_steps),
            "; ".join("step %s %s(failure=%s, total=%s)"
                      % (r.get("step"), r.get("name"), r.get("failure_count"), r.get("total"))
                      for r in failed_steps),
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, UserWarning, stacklevel=2)


def max_parallel_limit() -> int:
    """并行总上限：默认 96，可用 `WINDPAPER_MAX_PARALLEL` 覆盖（集群/小机器各自可配）。"""
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


def validate_worker_budget(
    max_workers: int,
    outer_max_workers: int,
    inner_max_workers: int,
    limit: int | None = None,
) -> None:
    """校验并行预算，超上限直接 **raise**（不静默超订）。

    两层步骤用 `outer × inner`，单层步骤用 `max_workers`；两者都会被检查，
    因为同一组参数会同时喂给所有步骤（panel 不做逐步骤分发）。
    """
    if limit is None:
        limit = max_parallel_limit()

    problems = []
    if int(max_workers) > limit:
        problems.append("单层 max_workers=%d > 上限 %d（step2/4/5/6/7/8/11/12 用）"
                        % (max_workers, limit))
    product = int(outer_max_workers) * int(inner_max_workers)
    if product > limit:
        problems.append("两层 outer×inner=%d×%d=%d > 上限 %d（step3/step9 用，是**相乘**）"
                        % (outer_max_workers, inner_max_workers, product, limit))
    if problems:
        raise ValueError(
            "并行预算超订：" + "；".join(problems)
            + "。请降低参数，或（确有大机器时）显式设置 WINDPAPER_MAX_PARALLEL。"
        )


def _resolve_step_kwargs(
    selected_steps: set[int] | None,
    skipped_steps: set[int] | None,
    explicit_step_settings: dict[str, bool | None] | None = None,
) -> dict[str, bool]:
    step_kwargs = {
        arg_name: STEP_DEFAULT_ENABLED.get(arg_name, True) for arg_name in STEP_ARG_NAME_MAP.values()
    }

    if selected_steps is not None:
        step_kwargs = {
            arg_name: step in selected_steps
            for step, arg_name in STEP_ARG_NAME_MAP.items()
        }

    if skipped_steps is not None:
        for step in skipped_steps:
            step_kwargs[STEP_ARG_NAME_MAP[step]] = False

    if explicit_step_settings is not None:
        for arg_name, value in explicit_step_settings.items():
            if value is not None:
                step_kwargs[arg_name] = value

    return step_kwargs


def run_panel_pipeline(
    max_workers = 96,
    outer_max_workers = 12,
    inner_max_workers = 8,
    run_step_1_farm_power_curves=True,
    run_step_2_batch_farm_output=True,
    run_step_3_kb_batch_aggr=True,
    run_step_4_prov_w_compare=True,
    run_step_5_prov_w_smape=True,
    run_step_6_prov_gen_enums=True,
    run_step_7_prov_gen_full_smape=True,
    run_step_8_find_optimal_kb_full_data=True,
    run_step_9_calibrated_gen=True,
    run_step_10_final_gen=True,
    run_step_11_no_kb_gen=True,
    strict_failures: bool = False,
    kb_grid: dict | None = None,
):
    results = {}
    records: list = []
    _kb = dict(kb_grid or {})

    def _step(step: int, enabled: bool, result_key: str, func, output_spec=None, **kwargs) -> None:
        _run_panel_step(results, step, enabled, result_key, func, records=records,
                        output_spec=output_spec, **kwargs)

    _step(
        1,
        run_step_1_farm_power_curves,
        "step_1_farm_power_curves",
        generate_farm_power_curves_csvs,
        csv_path=farm_inventory_csv_path,
        output_dir=farm_power_curve_dir,
        w_grid_col_name="w_grid",
        cf_col_name="CF",
        overwrite=False,
    )

    _step(
        2,
        run_step_2_batch_farm_output,
        "step_2_batch_farm_output",
        batch_farm_output_from_files,
        farm_csv=farm_inventory_csv_path,
        era5_dir=era5_dir,
        cdmet_dir=cdmet_dir,
        farm_ids=None,
        select_periods=None,
        output_hourly_gen=True,
        output_monthly_gen=True,
        output_hourly_cf=False,
        output_w_compare=True,
        output_raw_w=True,
        process_workers=max_workers,
        output_dir=batch_output_dir,
        output_gen_subdir=batch_output_gen_subdir,
        output_w_compare_subdir=batch_output_w_compare_subdir,
        output_raw_w_subdir=batch_output_raw_w_subdir,
        farm_adjusted_w_dir=None,
        farm_adjusted_w_varname=batch_farm_adjusted_w_varname,
        cf_csv_dir=farm_power_curve_dir,
        kb_adjust_info=None,
        gen_adjust_info=None,
        return_failures_only=False,
        verbose=False,
        overwrite=False,
    )

    _step(
        3,
        run_step_3_kb_batch_aggr,
        "step_3_kb_batch_aggr",
        kb_batch_aggr,
        era5_dir=era5_dir,
        cdmet_dir=cdmet_dir,
        inventory_csv_path=farm_inventory_csv_path,
        farm_power_curve_dir=farm_power_curve_dir,
        farm_adjusted_w_dir=kb_farm_adjusted_w_dir,
        farm_adjusted_w_varname=batch_farm_adjusted_w_varname,
        output_dir=kb_output_dir,
        output_hourly_gen=False,
        output_monthly_gen=True,
        output_hourly_cf=False,
        output_w_compare=False,
        output_raw_w=False,
        select_period="all",
        period_mode="season",
        prov_short=None,
        prov_long=None,
        city_short=None,
        city_long=None,
        city_capacity_adjust_csv_path=city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=prov_utility_rate_adjust_csv_path,
        gen_adjust_info=None,
        aggr_level="both",
        horizon_interp_method="loess",
        verbose=False,
        k_start=_kb.get("k_start", 0.1),
        k_end=_kb.get("k_end", 3.0),
        k_step=_kb.get("k_step", 0.1),
        b_start=_kb.get("b_start", -2.0),
        b_end=_kb.get("b_end", 2.0),
        b_step=_kb.get("b_step", 0.1),
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        stop_on_error=False,
        overwrite=False,
    )

    _step(
        4,
        run_step_4_prov_w_compare,
        "step_4_prov_w_compare",
        prov_w_compare,
        farm_csv_path=farm_inventory_csv_path,
        input_dir=prov_w_compare_input_dir,
        output_dir=prov_w_compare_output_dir,
        era5_col_name="era5_w",
        cdmet_col_name="cdmet_w",
        filename_template="farm_{farm_id}_period_full_year_k_1_b_0_w_compare.nc",
        max_workers=max_workers,
        overwrite=False,
    )

    _step(
        5,
        run_step_5_prov_w_smape,
        "step_5_prov_w_smape",
        prov_w_smape,
        input_dir=prov_w_smape_input_dir,
        output_dir=prov_w_smape_output_dir,
        era5_col_name="era5_w",
        cdmet_col_name="cdmet_w",
        k_start=_kb.get("k_start", 0.5),
        k_end=_kb.get("k_end", 3.0),
        k_step=_kb.get("k_step", 0.1),
        b_start=_kb.get("b_start", -2.0),
        b_end=_kb.get("b_end", 2.0),
        b_step=_kb.get("b_step", 0.1),
        max_workers=max_workers,
        overwrite=False,
    )

    _step(
        6,
        run_step_6_prov_gen_enums,
        "step_6_prov_gen_enums",
        prov_gen_enums,
        input_dir=prov_gen_enums_input_dir,
        output_dir=prov_gen_enums_output_dir,
        monthly_current_csv_path=monthly_current_gen_csv_path,
        monthly_cumulative_csv_path=monthly_cumulative_gen_csv_path,
        admin_province_csv_path=admin_province_csv_path,
        max_workers=max_workers,
        overwrite=False,
    )

    _step(
        7,
        run_step_7_prov_gen_full_smape,
        "step_7_prov_gen_full_smape",
        prov_gen_full_smape,
        input_dir=prov_gen_full_smape_input_dir,
        output_dir=prov_gen_full_smape_output_dir,
        max_workers=max_workers,
        file_pattern="*_monthly_gen_enums*.csv",
        excluded_provinces=["西藏", "香港", "澳门"],
        overwrite=True,
    )

    _step(
        8,
        run_step_8_find_optimal_kb_full_data,
        "step_8_find_optimal_kb_full_data",
        find_optimal_kb_full_data,
        input_dir=optimal_kb_input_dir,
        wind_dir=optimal_kb_wind_dir,
        output_dir=optimal_kb_output_dir,
        optimal_kb_csv=optimal_kb_csv,
        compute=True,
        plot=True,
        overwrite=False,
        poly_degree=9,
        max_workers=max_workers,
    )

    _step(
        9,
        run_step_9_calibrated_gen,
        "step_9_calibrated_gen",
        farm_aggr_with_kb,
        era5_dir=era5_dir,
        optimal_kb_csv=calibrated_gen_optimal_kb_csv,
        cdmet_dir=cdmet_dir,
        inventory_csv_path=farm_inventory_csv_path,
        farm_power_curve_dir=farm_power_curve_dir,
        farm_adjusted_w_dir=kb_farm_adjusted_w_dir,
        farm_adjusted_w_varname=batch_farm_adjusted_w_varname,
        output_dir=calibrated_gen_output_dir,
        select_period=None,
        period_mode="season",
        prov_short=None,
        prov_long=None,
        city_short=None,
        city_long=None,
        city_capacity_adjust_csv_path=city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=prov_utility_rate_adjust_csv_path,
        gen_adjust_info=None,
        aggr_level="both",
        include_kb_in_gen_filenames=False,
        horizon_interp_method="loess",
        max_workers=inner_max_workers,
        verbose=False,
        kb_tmp_dir=None,
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        stop_on_error=False,
        overwrite=True,
    )

    _step(
        10,
        run_step_10_final_gen,
        "step_10_final_gen",
        final_gen,
        input_dir=final_gen_input_dir,
        output_dir=final_gen_output_dir,
        monthly_current_csv_path=monthly_current_gen_csv_path,
        monthly_cumulative_csv_path=monthly_cumulative_gen_csv_path,
        pre2012_yearly_csv_path=pre2012_yearly_gen_csv_path,
        inventory_csv_path=farm_inventory_csv_path,
        unit_to_mwh_factor=100_000.0,
        overwrite=True,
        output_spec={"dir": final_gen_output_dir, "pattern": "**/*.nc", "min_files": 1},
    )

    _step(
        11,
        run_step_11_no_kb_gen,
        "step_11_no_kb_gen",
        no_kb_gen,
        era5_dir=era5_dir,
        cdmet_dir=cdmet_dir,
        inventory_csv_path=farm_inventory_csv_path,
        farm_power_curve_dir=farm_power_curve_dir,
        farm_adjusted_w_dir=kb_farm_adjusted_w_dir,
        farm_adjusted_w_varname=batch_farm_adjusted_w_varname,
        output_dir=no_kb_gen_output_dir,
        select_period=None,
        period_mode="season",
        prov_short=None,
        prov_long=None,
        city_short=None,
        city_long=None,
        city_capacity_adjust_csv_path=city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=prov_utility_rate_adjust_csv_path,
        gen_adjust_info=None,
        overwrite=False,
        aggr_level="both",
        include_kb_in_gen_filenames=False,
        horizon_interp_method="loess",
        max_workers=max_workers,
        verbose=False,
        show_progress=False,
        output_spec={"dir": no_kb_gen_output_dir, "pattern": "**/*.nc", "min_files": 1},
    )

    # 逐步失败摘要 + 门禁：任一 step failure_count>0 ⇒ 严格模式下显式失败（防静默缺件）
    print(format_failure_summary(records), flush=True)
    enforce_failure_gate(records, strict_failures)
    return results


def resolve_cli_step_kwargs(args: argparse.Namespace) -> dict[str, bool]:
    """把命令行解析成 12 个步骤开关（唯一实现，便于单测）。

    优先级（后者覆盖前者）：
        默认表 --full-chain（全开） < --steps（子集） < --skip-steps（关闭） < --step-x（显式单参）

    注意：`--full-chain` 与显式 `--step-x` 同时给出时，**显式单参必须生效**
    （原先的实现只处理了 `--skip-steps`，单参被静默忽略）。
    """
    explicit_step_settings = {
        arg_name: getattr(args, arg_name)
        for arg_name in STEP_ARG_NAME_MAP.values()
    }

    if getattr(args, "full_chain", False):
        step_kwargs = {arg_name: True for arg_name in STEP_ARG_NAME_MAP.values()}
        if args.steps is not None:
            step_kwargs = {
                arg_name: (step in args.steps)
                for step, arg_name in STEP_ARG_NAME_MAP.items()
            }
        for step in (args.skip_steps or set()):
            step_kwargs[STEP_ARG_NAME_MAP[step]] = False
    else:
        step_kwargs = _resolve_step_kwargs(
            args.steps,
            args.skip_steps,
            explicit_step_settings,
        )
        return step_kwargs

    # 显式单参优先级最高
    for arg_name, value in explicit_step_settings.items():
        if value is not None:
            step_kwargs[arg_name] = value
    return step_kwargs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    step_kwargs = resolve_cli_step_kwargs(args)
    validate_worker_budget(args.max_workers, args.outer_max_workers, args.inner_max_workers)
    # 全链模式**默认严格**：任一步 failure_count>0 ⇒ 非零退出（防 HPC 静默缺件）
    strict_failures = bool(args.strict or args.full_chain)
    kb_grid = resolve_kb_grid(
        args.kb_grid,
        k_start=args.kb_k_start, k_end=args.kb_k_end, k_step=args.kb_k_step,
        b_start=args.kb_b_start, b_end=args.kb_b_end, b_step=args.kb_b_step,
    )
    _report_pipeline_plan(
        step_kwargs,
        args.max_workers,
        args.outer_max_workers,
        args.inner_max_workers,
    )
    print("[panel][%s] strict_failures=%s（--full-chain 默认开启；可用 --no-strict 关闭）"
          % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), strict_failures), flush=True)
    print("[panel][%s] kb_grid preset=%s 生效覆盖=%s（空=各调用点原默认；coarse 用于冒烟）"
          % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.kb_grid, kb_grid), flush=True)
    panel_results = run_panel_pipeline(
        max_workers=args.max_workers,
        outer_max_workers=args.outer_max_workers,
        inner_max_workers=args.inner_max_workers,
        strict_failures=strict_failures,
        kb_grid=kb_grid,
        **step_kwargs,
    )

    executed_steps = [
        f"{step}:{STEP_LABEL_MAP[step]}"
        for step, arg_name in STEP_ARG_NAME_MAP.items()
        if step_kwargs[arg_name]
    ]
    print("Executed steps:", ", ".join(executed_steps) if executed_steps else "none")
    print(
        "Panel summary:",
        {key: type(value).__name__ for key, value in panel_results.items()},
    )


if __name__ == "__main__":
    main()