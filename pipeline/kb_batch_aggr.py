from __future__ import annotations

import argparse
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Union, List, Tuple

from aggr_farm_aux import DEFAULT_INVENTORY_CSV_PATH, SeasonName
from aggr_farm import aggr_farm_output

from config import clamp_two_layer_workers, resolve_outer_workers  # 并行预算护栏
from parse_aux import parse_csv_list, parse_gen_adjust_info, parse_none, parse_select_period, parse_tf

logger = logging.getLogger(__name__)


def _float_range_inclusive(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")

    count = int(math.floor((end - start) / step + 1e-12)) + 1
    values: list[float] = []
    for idx in range(count):
        values.append(round(start + idx * step, 10))

    if not math.isclose(values[-1], end, rel_tol=0.0, abs_tol=1e-9):
        values.append(round(end, 10))
    return values
def _run_single_kb_task(task: dict[str, Any]) -> dict[str, Any]:
    k = float(task["k"])
    b = float(task["b"])
    params = task["params"].copy()

    params["kb_info"] = (k, b)
    params["max_workers"] = task["inner_max_workers"]

    result = aggr_farm_output(**params)
    return {
        "k": k,
        "b": b,
        "ok": True,
        "result": result,
    }


def kb_batch_aggr(
    era5_dir: str,
    cdmet_dir: Optional[str] = None,
    inventory_csv_path: Union[str, Path] = DEFAULT_INVENTORY_CSV_PATH,
    farm_power_curve_dir: Union[str, Path, None] = None,
    farm_adjusted_w_dir: Optional[Union[str, Path]] = None,
    farm_adjusted_w_varname: str = "raw_w",
    output_dir: Optional[Union[str, Path]] = "kb_batch_aggr_outputs",
    output_hourly_gen: bool = False,
    output_monthly_gen: bool = True,
    output_hourly_cf: bool = False,
    output_w_compare: Optional[bool] = False,
    output_raw_w: Optional[bool] = False,
    select_period: Optional[
        Union[
            SeasonName,
            int,
            list[Union[SeasonName, int]],
            tuple[Union[SeasonName, int], ...],
            Literal["all"], None
        ]
    ] = "all",
    period_mode: Literal["season", "month"] = "season",
    prov_short: Optional[Union[str, Iterable[str]]] = None,
    prov_long: Optional[Union[str, Iterable[str]]] = None,
    city_short: Optional[Union[str, Iterable[str]]] = None,
    city_long: Optional[Union[str, Iterable[str]]] = None,
    city_capacity_adjust_csv_path: Optional[Union[str, Path]] = r"D:\HKU PhD\wind data paper\city_capacity_adjust_info.csv",
    prov_utility_rate_adjust_csv_path: Optional[Union[str, Path]] = r"D:\HKU PhD\wind data paper\prov_wind_utility_rates.csv",
    gen_adjust_info: Optional[Union[Dict, List[Dict], Tuple[Dict, ...]]] = None,
    overwrite: bool = False,
    aggr_level: Literal["city", "prov", "both"] = "both",
    horizon_interp_method: Literal["closest", "idw", "loess"] = "loess",
    verbose: bool = True,
    k_start: float = 0.1,
    k_end: float = 3.0,
    k_step: float = 0.1,
    b_start: float = -2.0,
    b_end: float = 2.0,
    b_step: float = 0.1,
    outer_max_workers: Optional[int] = 4,
    inner_max_workers: Optional[int] = 4,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    """
    Enumerate (k, b) grid and run aggr_farm_output in outer-process parallel.

    Parallelism controls:
    - outer_max_workers: process count for enumerating kb combinations.
    - inner_max_workers: process count passed to aggr_farm_output(max_workers=...).
    """
    k_values = _float_range_inclusive(k_start, k_end, k_step)
    b_values = _float_range_inclusive(b_start, b_end, b_step)
    kb_pairs = [(k, b) for k in k_values for b in b_values]

    if len(kb_pairs) == 0:
        raise ValueError("No (k,b) tasks generated")

    base_params: dict[str, Any] = {
        "era5_dir": era5_dir,
        "cdmet_dir": cdmet_dir,
        "inventory_csv_path": str(inventory_csv_path),
        "farm_power_curve_dir": farm_power_curve_dir,
        "farm_adjusted_w_dir": farm_adjusted_w_dir,
        "farm_adjusted_w_varname": farm_adjusted_w_varname,
        "output_dir": output_dir,
        "output_hourly_gen": output_hourly_gen,
        "output_monthly_gen": output_monthly_gen,
        "output_hourly_cf": output_hourly_cf,
        "output_w_compare": output_w_compare,
        "output_raw_w": output_raw_w,
        "select_period": select_period,
        "period_mode": period_mode,
        "prov_short": prov_short,
        "prov_long": prov_long,
        "city_short": city_short,
        "city_long": city_long,
        "city_capacity_adjust_csv_path": city_capacity_adjust_csv_path,
        "prov_utility_rate_adjust_csv_path": prov_utility_rate_adjust_csv_path,
        "gen_adjust_info": gen_adjust_info,
        "overwrite": overwrite,
        "aggr_level": aggr_level,
        "horizon_interp_method": horizon_interp_method,
        "verbose": verbose,
    }

    # 并行预算：outer=None 时原先回落成 len(kb_pairs)（可达 1230！）⇒ 必须夹到上限；
    # 两层是**相乘**，因此还要按 outer × inner 再夹一次（防 HPC 超订惩罚）。
    outer_workers = resolve_outer_workers(outer_max_workers, len(kb_pairs), context="kb_batch_aggr")
    outer_workers, inner_max_workers = clamp_two_layer_workers(
        outer_workers, inner_max_workers, context="kb_batch_aggr"
    )

    logger.info(
        "Start kb_batch_aggr | kb_tasks=%s | outer_workers=%s | inner_workers=%s",
        len(kb_pairs),
        outer_workers,
        inner_max_workers,
    )

    success: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=outer_workers) as executor:
        future_map = {}
        for k, b in kb_pairs:
            task = {
                "k": k,
                "b": b,
                "params": base_params,
                "inner_max_workers": inner_max_workers,
            }
            future = executor.submit(_run_single_kb_task, task)
            future_map[future] = (k, b)

        for future in as_completed(future_map):
            k, b = future_map[future]
            try:
                result = future.result()
                success.append(result)
                logger.info("kb task done: k=%s b=%s", k, b)
            except Exception as exc:  # noqa: BLE001
                failed_item = {"k": k, "b": b, "error": str(exc)}
                failed.append(failed_item)
                logger.error("kb task failed: k=%s b=%s err=%s", k, b, exc)
                if stop_on_error:
                    raise

    summary = {
        "k_values": k_values,
        "b_values": b_values,
        "total_tasks": len(kb_pairs),
        "success_count": len(success),
        "failure_count": len(failed),
        "success": success,
        "failed": failed,
        "outer_max_workers": outer_workers,
        "inner_max_workers": inner_max_workers,
    }

    logger.info(
        "End kb_batch_aggr | total=%s success=%s failure=%s",
        summary["total_tasks"],
        summary["success_count"],
        summary["failure_count"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate kb grid and run batch aggr_farm_output")

    parser.add_argument("--era5_dir", type=str, required=True)
    parser.add_argument("--cdmet_dir", type=parse_none, default=None)
    parser.add_argument("--inventory_csv_path", type=parse_none, default=str(DEFAULT_INVENTORY_CSV_PATH))
    parser.add_argument("--farm_power_curve_dir", type=parse_none, default=None)
    parser.add_argument("--farm_adjusted_w_dir", type=parse_none, default=None)
    parser.add_argument("--farm_adjusted_w_varname", type=str, default="raw_w")

    parser.add_argument("--output_dir", type=parse_none, default="kb_batch_aggr_outputs")

    parser.add_argument("--output_hourly_gen", type=parse_tf, default=True)
    parser.add_argument("--output_monthly_gen", type=parse_tf, default=False)
    parser.add_argument("--output_hourly_cf", type=parse_tf, default=False)
    parser.add_argument("--output_w_compare", type=parse_tf, default=False)
    parser.add_argument("--output_raw_w", type=parse_tf, default=False)

    parser.add_argument("--select_period", type=parse_select_period, default="all")
    parser.add_argument("--period_mode", choices=["season", "month"], default="season")
    parser.add_argument("--prov_short", type=parse_none, default=None)
    parser.add_argument("--prov_long", type=parse_none, default=None)
    parser.add_argument("--city_short", type=parse_none, default=None)
    parser.add_argument("--city_long", type=parse_none, default=None)
    parser.add_argument(
        "--city_capacity_adjust_csv_path",
        type=parse_none,
        default=r"D:\HKU PhD\wind data paper\city_capacity_adjust_info.csv",
    )
    parser.add_argument(
        "--prov_utility_rate_adjust_csv_path",
        type=parse_none,
        default=r"D:\HKU PhD\wind data paper\prov_wind_utility_rates.csv",
    )

    parser.add_argument("--gen_adjust_info", type=parse_gen_adjust_info, default=None)
    parser.add_argument("--overwrite", type=parse_tf, default=False)
    parser.add_argument("--aggr_level", choices=["city", "prov", "both"], default="both")
    parser.add_argument("--horizon_interp_method", choices=["closest", "idw", "loess"], default="loess")
    parser.add_argument("--verbose", type=parse_tf, default=True)

    parser.add_argument("--k_start", type=float, default=0.1)
    parser.add_argument("--k_end", type=float, default=3.0)
    parser.add_argument("--k_step", type=float, default=0.1)
    parser.add_argument("--b_start", type=float, default=-2.0)
    parser.add_argument("--b_end", type=float, default=2.0)
    parser.add_argument("--b_step", type=float, default=0.1)

    parser.add_argument("--outer_max_workers", type=parse_none, default=2)
    parser.add_argument("--inner_max_workers", type=parse_none, default=2)
    parser.add_argument("--stop_on_error", type=parse_tf, default=False)

    args = parser.parse_args()

    prov_short = parse_csv_list(args.prov_short)
    prov_long = parse_csv_list(args.prov_long)
    city_short = parse_csv_list(args.city_short)
    city_long = parse_csv_list(args.city_long)
    outer_max_workers = None if args.outer_max_workers is None else int(args.outer_max_workers)
    inner_max_workers = None if args.inner_max_workers is None else int(args.inner_max_workers)

    summary = kb_batch_aggr(
        era5_dir=args.era5_dir,
        cdmet_dir=args.cdmet_dir,
        inventory_csv_path=args.inventory_csv_path,
        farm_power_curve_dir=args.farm_power_curve_dir,
        farm_adjusted_w_dir=args.farm_adjusted_w_dir,
        farm_adjusted_w_varname=args.farm_adjusted_w_varname,
        output_dir=args.output_dir,
        output_hourly_gen=args.output_hourly_gen,
        output_monthly_gen=args.output_monthly_gen,
        output_hourly_cf=args.output_hourly_cf,
        output_w_compare=args.output_w_compare,
        output_raw_w=args.output_raw_w,
        select_period=args.select_period,
        period_mode=args.period_mode,
        prov_short=prov_short,
        prov_long=prov_long,
        city_short=city_short,
        city_long=city_long,
        city_capacity_adjust_csv_path=args.city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=args.prov_utility_rate_adjust_csv_path,
        gen_adjust_info=args.gen_adjust_info,
        overwrite=args.overwrite,
        aggr_level=args.aggr_level,
        horizon_interp_method=args.horizon_interp_method,
        verbose=args.verbose,
        k_start=args.k_start,
        k_end=args.k_end,
        k_step=args.k_step,
        b_start=args.b_start,
        b_end=args.b_end,
        b_step=args.b_step,
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        stop_on_error=args.stop_on_error,
    )

    logger.info(
        "kb_batch_aggr finished | total=%s success=%s failure=%s",
        summary["total_tasks"],
        summary["success_count"],
        summary["failure_count"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    main()


