from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Union

from aggr_farm import aggr_farm_output
from aggr_farm_aux import DEFAULT_INVENTORY_CSV_PATH


DEFAULT_FARM_ADJUSTED_W_DIR = Path('/geogfs1/groups/geog_liulb/hongyu/results/farm_raw_w/raw_w')
DEFAULT_OUTPUT_DIR = Path('/geogfs1/groups/geog_liulb/hongyu/results/no_kb_gen')
NO_KB_INFO = (1.0, 0.0)


def no_kb_gen(
    era5_dir: str,
    output_dir: str | Path,
    cdmet_dir: Optional[str] = None,
    inventory_csv_path: Union[str, Path] = DEFAULT_INVENTORY_CSV_PATH,
    farm_power_curve_dir: Union[str, Path, None] = None,
    farm_adjusted_w_dir: Optional[Union[str, Path]] = DEFAULT_FARM_ADJUSTED_W_DIR,
    farm_adjusted_w_varname: str = 'raw_w',
    select_period: Any = None,
    period_mode: str = 'season',
    prov_short: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    prov_long: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    city_short: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    city_long: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    city_capacity_adjust_csv_path: Optional[Union[str, Path]] = r'D:\HKU PhD\wind data paper\city_capacity_adjust_info.csv',
    prov_utility_rate_adjust_csv_path: Optional[Union[str, Path]] = r'D:\HKU PhD\wind data paper\prov_wind_utility_rates.csv',
    gen_adjust_info: Optional[Union[dict[str, Any], list[dict[str, Any]], tuple[dict[str, Any], ...]]] = None,
    overwrite: bool = False,
    aggr_level: str = 'both',
    include_kb_in_gen_filenames: bool = False,
    horizon_interp_method: str = 'loess',
    max_workers: Optional[int] = 4,
    verbose: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    result = aggr_farm_output(
        era5_dir=era5_dir,
        cdmet_dir=cdmet_dir,
        inventory_csv_path=inventory_csv_path,
        farm_power_curve_dir=farm_power_curve_dir,
        farm_adjusted_w_dir=farm_adjusted_w_dir,
        farm_adjusted_w_varname=farm_adjusted_w_varname,
        output_dir=output_root,
        output_hourly_gen=True,
        output_monthly_gen=True,
        output_hourly_cf=False,
        output_w_compare=False,
        output_raw_w=False,
        select_period=select_period,
        period_mode=period_mode,
        prov_short=prov_short,
        prov_long=prov_long,
        city_short=city_short,
        city_long=city_long,
        kb_info=NO_KB_INFO,
        gen_adjust_info=gen_adjust_info,
        city_capacity_adjust_csv_path=city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=prov_utility_rate_adjust_csv_path,
        overwrite=overwrite,
        aggr_level=aggr_level,
        include_kb_in_gen_filenames=include_kb_in_gen_filenames,
        horizon_interp_method=horizon_interp_method,
        max_workers=max_workers,
        verbose=verbose,
        show_progress=show_progress,
        output_gen_nc_path=output_root,
    )
    result['no_kb_k'] = NO_KB_INFO[0]
    result['no_kb_b'] = NO_KB_INFO[1]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='强制使用 k=1, b=0 重新运行 aggr_farm_output，并输出 no_kb_gen 目录。'
    )
    parser.add_argument('--era5_dir', type=str, required=True, help='ERA5 风速输入目录')
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='no_kb_gen 输出根目录')
    parser.add_argument('--cdmet_dir', type=str, default=None, help='可选的 CDMet 风速输入目录')
    parser.add_argument('--inventory_csv_path', type=str, default=str(DEFAULT_INVENTORY_CSV_PATH), help='风电场清单 CSV')
    parser.add_argument('--farm_power_curve_dir', type=str, default=None, help='风机功率曲线 CSV 目录')
    parser.add_argument(
        '--farm_adjusted_w_dir',
        type=str,
        default=str(DEFAULT_FARM_ADJUSTED_W_DIR),
        help='复用的 full_year 原始风速目录，通常为 farm_raw_w/raw_w',
    )
    parser.add_argument('--farm_adjusted_w_varname', type=str, default='raw_w')
    parser.add_argument('--city_capacity_adjust_csv_path', type=str, default=r'D:\HKU PhD\wind data paper\city_capacity_adjust_info.csv')
    parser.add_argument('--prov_utility_rate_adjust_csv_path', type=str, default=r'D:\HKU PhD\wind data paper\prov_wind_utility_rates.csv')
    parser.add_argument('--overwrite', action='store_true', help='是否覆盖已存在输出')
    parser.add_argument('--aggr_level', choices=['city', 'prov', 'both'], default='both')
    parser.add_argument('--horizon_interp_method', choices=['closest', 'idw', 'loess'], default='loess')
    parser.add_argument('--max_workers', type=int, default=4)
    parser.add_argument('--verbose', action='store_true', help='是否输出详细日志')
    parser.add_argument('--show_progress', action='store_true', help='是否显示进度条')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = no_kb_gen(
        era5_dir=args.era5_dir,
        output_dir=args.output_dir,
        cdmet_dir=args.cdmet_dir,
        inventory_csv_path=args.inventory_csv_path,
        farm_power_curve_dir=args.farm_power_curve_dir,
        farm_adjusted_w_dir=args.farm_adjusted_w_dir,
        farm_adjusted_w_varname=args.farm_adjusted_w_varname,
        city_capacity_adjust_csv_path=args.city_capacity_adjust_csv_path,
        prov_utility_rate_adjust_csv_path=args.prov_utility_rate_adjust_csv_path,
        overwrite=args.overwrite,
        aggr_level=args.aggr_level,
        horizon_interp_method=args.horizon_interp_method,
        max_workers=args.max_workers,
        verbose=args.verbose,
        show_progress=args.show_progress,
    )
    print(result, flush=True)


if __name__ == '__main__':
    main()