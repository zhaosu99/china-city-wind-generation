from __future__ import annotations

import argparse
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from find_optimal_kb import (
    MIN_OPTIMAL_K,
    CANDIDATE_PATH_COLOR,
    PLOT_TEXT_SIZE,
    _draw_path,
    _style_axis,
    find_min_wind_on_gen_op,
    fit_surface_from_dataframe,
    get_optimal_path_from_dataframe,
    interpolate_smape_value,
    load_smape_csv,
    should_force_spawn,
)


FULL_GEN_PATTERN = re.compile(r"^(?P<prov>.+?)_(?P<season>[^_]+)_full_gen_smape\.csv$")


def parse_full_gen_filename(gen_csv: Path) -> dict[str, str] | None:
    match = FULL_GEN_PATTERN.match(gen_csv.name)
    if match is None:
        return None
    return {
        "prov": match.group("prov"),
        "season": match.group("season"),
    }


def _add_r2_label(ax, r2: float) -> None:
    ax.text(
        0.03,
        0.97,
        f"R2={r2:.4f}",
        transform=ax.transAxes,
        fontsize=PLOT_TEXT_SIZE,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.92, edgecolor="0.8"),
        zorder=30,
    )


def _place_figure_legend(fig, handles, labels) -> None:
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(labels),
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        prop={"size": PLOT_TEXT_SIZE},
        borderaxespad=0.3,
    )
    legend.set_zorder(30)


def _draw_full_data_selection_row(axes_row, selection: dict[str, object]):
    ax1, ax2 = axes_row

    gen_k = np.asarray(selection["gen_k"], dtype=float)
    gen_b = np.asarray(selection["gen_b"], dtype=float)
    gen_k_fine = np.asarray(selection["gen_k_fine"], dtype=float)
    gen_b_fine = np.asarray(selection["gen_b_fine"], dtype=float)
    gen_S_pred = np.asarray(selection["gen_S_pred"], dtype=float)
    wind_k_fine = np.asarray(selection["wind_k_fine"], dtype=float)
    wind_b_fine = np.asarray(selection["wind_b_fine"], dtype=float)
    wind_S_pred = np.asarray(selection["wind_S_pred"], dtype=float)
    gen_r2 = float(selection["gen_r2"])
    wind_r2 = float(selection["wind_r2"])
    min_result = selection["min_result"]

    im1 = ax1.imshow(
        np.clip(gen_S_pred, 0, 100),
        cmap="viridis_r",
        origin="lower",
        extent=[gen_k_fine.min(), gen_k_fine.max(), gen_b_fine.min(), gen_b_fine.max()],
        aspect="auto",
        vmin=0,
        vmax=100,
    )
    _style_axis(ax1, title=None)
    ax1.set_xlim(0.5, 3.0)
    ax1.set_ylim(-2.0, 2.0)
    _add_r2_label(ax1, gen_r2)
    _draw_path(ax1, gen_k, gen_b, color=CANDIDATE_PATH_COLOR, label="Optimal Path")

    ax2.imshow(
        np.clip(wind_S_pred, 0, 100),
        cmap="viridis_r",
        origin="lower",
        extent=[wind_k_fine.min(), wind_k_fine.max(), wind_b_fine.min(), wind_b_fine.max()],
        aspect="auto",
        vmin=0,
        vmax=100,
    )
    _style_axis(ax2, title=None)
    ax2.set_xlim(0.5, 3.0)
    ax2.set_ylim(-2.0, 2.0)
    _add_r2_label(ax2, wind_r2)
    _draw_path(ax2, gen_k, gen_b, color=CANDIDATE_PATH_COLOR, label="Optimal Path")

    legend_handles = [Line2D([0], [0], color=CANDIDATE_PATH_COLOR, linewidth=2.0)]
    legend_labels = ["Optimal Path"]
    if min_result is not None:
        ax2.scatter(
            min_result["k"],
            min_result["b"],
            color="gold",
            edgecolors="black",
            linewidth=1.2,
            s=96,
            zorder=10,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=9,
                linestyle="None",
                markerfacecolor="gold",
                markeredgecolor="black",
            )
        )
        legend_labels.append(f"Optimal (k, b): ({min_result['k']:.2f}, {min_result['b']:.2f})")

    return im1, legend_handles, legend_labels


def plot_full_data_selection_figure(*, selection: dict[str, object], output_plot: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im1, legend_handles, legend_labels = _draw_full_data_selection_row(axes, selection)
    _place_figure_legend(fig, legend_handles, legend_labels)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.12, wspace=0.20)
    cbar_ax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=PLOT_TEXT_SIZE)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches="tight", format="jpg")
    plt.close()


def _compute_selection(
    *,
    gen_csv: Path,
    wind_csv: Path,
    poly_degree: int,
) -> dict[str, object]:
    gen_df = load_smape_csv(gen_csv)
    wind_df = load_smape_csv(wind_csv)

    (gen_k, gen_b, gen_model, gen_poly, gen_k_fine, gen_b_fine, gen_S_pred, gen_r2,
     gen_on_boundary) = get_optimal_path_from_dataframe(
        gen_df,
        poly_degree=poly_degree,
        return_boundary_flag=True,
    )
    if gen_k is None or gen_model is None:
        raise RuntimeError(f"无法从 full generation csv 中提取有效 optimal path: {gen_csv}")

    wind_model, wind_poly, wind_k_fine, wind_b_fine, wind_S_pred, wind_r2 = fit_surface_from_dataframe(
        wind_df,
        poly_degree=poly_degree,
    )
    if wind_model is None:
        raise RuntimeError(f"无法从 wind csv 中拟合有效模型: {wind_csv}")

    min_result = find_min_wind_on_gen_op(gen_k, gen_b, wind_model, wind_poly)
    final_k = float(min_result["k"])
    final_b = float(min_result["b"])
    final_gen_smape = float(interpolate_smape_value(gen_model, gen_poly, final_k, final_b))

    return {
        "gen_k": np.asarray(gen_k, dtype=float),
        "gen_b": np.asarray(gen_b, dtype=float),
        "gen_k_fine": np.asarray(gen_k_fine, dtype=float),
        "gen_b_fine": np.asarray(gen_b_fine, dtype=float),
        "gen_S_pred": np.asarray(gen_S_pred, dtype=float),
        "gen_r2": float(gen_r2),
        "gen_on_boundary": bool(gen_on_boundary),
        "wind_k_fine": np.asarray(wind_k_fine, dtype=float),
        "wind_b_fine": np.asarray(wind_b_fine, dtype=float),
        "wind_S_pred": np.asarray(wind_S_pred, dtype=float),
        "wind_r2": float(wind_r2),
        "min_result": min_result,
        "final_row": {
            "w_smape": float(min_result["smape"]),
            "k": final_k,
            "b": final_b,
            "gen_smape": final_gen_smape,
            "candidate_point_count": int(len(gen_k)),
            # 边界解标注：路径无内点 ⇒ boundary_fallback；最终点贴 k/b 定义域边界 ⇒ on_boundary=True
            "selection_mode": "boundary_fallback" if gen_on_boundary else "full_data",
            "on_boundary": bool(
                gen_on_boundary
                or np.isclose(final_k, MIN_OPTIMAL_K, atol=1e-6)
                or np.isclose(final_k, 3.0, atol=1e-6)
                or np.isclose(abs(final_b), 2.0, atol=1e-6)
            ),
        },
    }


def process_one_full_data_combo(
    gen_csv: Path,
    wind_csv: Path,
    output_plot: Path | None,
    plot: bool,
    overwrite: bool,
    poly_degree: int = 7,
) -> dict[str, object]:
    meta = parse_full_gen_filename(gen_csv)
    if plot and output_plot is not None and output_plot.exists() and not overwrite:
        return {
            "gen_csv": str(gen_csv),
            "wind_csv": str(wind_csv),
            "prov": meta["prov"] if meta is not None else None,
            "season": meta["season"] if meta is not None else None,
            "output_plot": str(output_plot),
            "written": False,
            "skipped_existing": True,
        }

    selection = _compute_selection(
        gen_csv=gen_csv,
        wind_csv=wind_csv,
        poly_degree=poly_degree,
    )
    final_row = selection["final_row"]

    output_plot_value = None
    if plot and output_plot is not None:
        plot_full_data_selection_figure(selection=selection, output_plot=output_plot)
        output_plot_value = str(output_plot)

    return {
        "gen_csv": str(gen_csv),
        "wind_csv": str(wind_csv),
        "prov": meta["prov"] if meta is not None else None,
        "season": meta["season"] if meta is not None else None,
        "output_plot": output_plot_value,
        "written": True,
        "skipped_existing": False,
        **final_row,
    }


def _process_task(task: tuple[str, str, str, bool, bool, int]) -> dict[str, object]:
    gen_csv_str, wind_csv_str, output_plot_str, plot, overwrite, poly_degree = task
    gen_csv = Path(gen_csv_str)
    wind_csv = Path(wind_csv_str)
    output_plot = Path(output_plot_str) if output_plot_str else None
    meta = parse_full_gen_filename(gen_csv)
    try:
        result = process_one_full_data_combo(
            gen_csv=gen_csv,
            wind_csv=wind_csv,
            output_plot=output_plot,
            plot=plot,
            overwrite=overwrite,
            poly_degree=poly_degree,
        )
        result["ok"] = True
        return result
    except Exception as exc:
        return {
            "ok": False,
            "gen_csv": str(gen_csv),
            "wind_csv": str(wind_csv),
            "prov": meta["prov"] if meta is not None else None,
            "season": meta["season"] if meta is not None else None,
            "error": str(exc),
        }


def build_summary_csv(success_results: list[dict[str, object]], output_csv: Path) -> None:
    rows: list[dict[str, object]] = []
    for item in success_results:
        gen_csv = Path(str(item["gen_csv"]))
        meta = parse_full_gen_filename(gen_csv)
        if meta is None:
            continue
        rows.append(
            {
                "prov": meta["prov"],
                "season": meta["season"],
                "w_smape": float(item["w_smape"]),
                "k": float(item["k"]),
                "b": float(item["b"]),
                "gen_smape": float(item["gen_smape"]),
                "candidate_point_count": int(item["candidate_point_count"]),
                "selection_mode": str(item["selection_mode"]),
                "on_boundary": bool(item.get("on_boundary", False)),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "prov",
            "season",
            "w_smape",
            "k",
            "b",
            "gen_smape",
            "candidate_point_count",
            "selection_mode",
            "on_boundary",
        ],
    )
    if not df.empty:
        df = df.sort_values(by=["prov", "season"], kind="stable").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def find_optimal_kb_full_data(
    input_dir: Path,
    wind_dir: Path,
    output_dir: Path,
    optimal_kb_csv: Path | None = None,
    compute: bool = True,
    plot: bool = False,
    overwrite: bool = False,
    poly_degree: int = 7,
    max_workers: int | None = 8,
    force_spawn: bool | None = None,
) -> dict[str, object]:
    if not compute and not plot:
        raise ValueError("compute 和 plot 不能同时为 False")

    effective_force_spawn = should_force_spawn(force_spawn)
    if effective_force_spawn:
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    input_dir = Path(input_dir).resolve()
    wind_dir = Path(wind_dir).resolve()
    output_dir = Path(output_dir).resolve()
    optimal_kb_csv = optimal_kb_csv.resolve() if optimal_kb_csv is not None else output_dir / "optimal_kb.csv"

    if poly_degree < 1:
        raise ValueError(f"poly_degree 必须 >= 1, 当前为: {poly_degree}")
    if not input_dir.exists():
        raise FileNotFoundError(f"未找到 input_dir: {input_dir}")
    if not wind_dir.exists():
        raise FileNotFoundError(f"未找到 wind_dir: {wind_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_csv_list = sorted(input_dir.glob("*_full_gen_smape.csv"))
    if not gen_csv_list:
        raise FileNotFoundError(f"未找到任何 *_full_gen_smape.csv: {input_dir}")

    ok_count = 0
    skip_count = 0
    fail_count = 0
    success_results: list[dict[str, object]] = []
    failed_results: list[dict[str, object]] = []
    skipped_items: list[dict[str, object]] = []
    tasks: list[tuple[str, str, str, bool, bool, int]] = []

    if compute and not plot and optimal_kb_csv.exists() and not overwrite:
        print("\n处理完成:", flush=True)
        print("- 提交任务: 0", flush=True)
        print("- 成功: 0", flush=True)
        print("- 跳过: 1", flush=True)
        print("- 失败: 0", flush=True)
        return {
            "input_dir": str(input_dir),
            "wind_dir": str(wind_dir),
            "output_dir": str(output_dir),
            "optimal_kb_csv": str(optimal_kb_csv),
            "submitted_task_count": 0,
            "success_count": 0,
            "skip_count": 1,
            "failure_count": 0,
            "plot_paths": [],
            "success_results": [],
            "skipped_results": [
                {
                    "output_csv": str(optimal_kb_csv),
                    "reason": "existing_summary_csv",
                }
            ],
            "failed_results": [],
            "selection_mode": "full_data",
            "summary_written": False,
            "summary_skipped_existing": True,
        }

    for gen_csv in gen_csv_list:
        meta = parse_full_gen_filename(gen_csv)
        if meta is None:
            skip_count += 1
            skipped_items.append({"gen_csv": str(gen_csv), "reason": "unrecognized_filename"})
            continue

        prov = meta["prov"]
        season = meta["season"]
        wind_csv = wind_dir / f"{prov}_{season}_w_smape_enums.csv"
        if not wind_csv.exists():
            skip_count += 1
            skipped_items.append({"gen_csv": str(gen_csv), "wind_csv": str(wind_csv), "reason": "missing_wind_csv"})
            continue

        output_plot = output_dir / "plots" / f"{prov}_{season}_optimal_kb.jpg"
        tasks.append((str(gen_csv), str(wind_csv), str(output_plot), plot, overwrite, poly_degree))

    if tasks:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result["ok"]:
                    if result.get("skipped_existing"):
                        skip_count += 1
                        skipped_items.append(
                            {
                                "gen_csv": result.get("gen_csv"),
                                "wind_csv": result.get("wind_csv"),
                                "output_plot": result.get("output_plot"),
                                "reason": "existing_output_plot",
                            }
                        )
                    else:
                        ok_count += 1
                        success_results.append(result)
                else:
                    fail_count += 1
                    failed_results.append(result)

    plot_paths = sorted(
        str(item["output_plot"])
        for item in success_results
        if item.get("output_plot") is not None
    )

    summary_written = False
    summary_skipped_existing = False
    if compute:
        if optimal_kb_csv.exists() and not overwrite:
            summary_skipped_existing = True
            skip_count += 1
            skipped_items.append({"output_csv": str(optimal_kb_csv), "reason": "existing_summary_csv"})
        else:
            build_summary_csv(success_results, optimal_kb_csv)
            summary_written = True

    print("\n处理完成:", flush=True)
    print(f"- 提交任务: {len(tasks)}", flush=True)
    print(f"- 成功: {ok_count}", flush=True)
    print(f"- 跳过: {skip_count}", flush=True)
    print(f"- 失败: {fail_count}", flush=True)

    return {
        "input_dir": str(input_dir),
        "wind_dir": str(wind_dir),
        "output_dir": str(output_dir),
        "optimal_kb_csv": str(optimal_kb_csv),
        "submitted_task_count": len(tasks),
        "success_count": ok_count,
        "skip_count": skip_count,
        "failure_count": fail_count,
        "plot_paths": plot_paths,
        "success_results": success_results,
        "skipped_results": skipped_items,
        "failed_results": failed_results,
        "selection_mode": "full_data",
        "summary_written": summary_written,
        "summary_skipped_existing": summary_skipped_existing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 full generation SMAPE surface，并结合 wind SMAPE 选择最终最优点。",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="full generation SMAPE 目录")
    parser.add_argument("--wind-dir", type=Path, required=True, help="风速 SMAPE 目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="结果输出根目录")
    parser.add_argument(
        "--optimal-kb-csv",
        type=Path,
        default=None,
        help="最终 optimal kb CSV 路径，默认 output_dir/optimal_kb.csv",
    )
    parser.add_argument("--compute", dest="compute", action="store_true", help="计算并输出 optimal_kb.csv")
    parser.add_argument("--no-compute", dest="compute", action="store_false", help="不输出 optimal_kb.csv")
    parser.add_argument("--plot", dest="plot", action="store_true", help="输出每个省季一张 1x2 图")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="不输出图")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的图和 summary CSV")
    parser.add_argument("--poly-degree", type=int, default=8, help="多项式拟合阶数")
    parser.add_argument("--max-workers", type=int, default=16, help="并行子任务进程数")
    spawn_group = parser.add_mutually_exclusive_group()
    spawn_group.add_argument(
        "--force-spawn",
        dest="force_spawn",
        action="store_true",
        help="强制使用 spawn 启动子进程；Windows 默认开启。",
    )
    spawn_group.add_argument(
        "--no-force-spawn",
        dest="force_spawn",
        action="store_false",
        help="禁用强制 spawn，使用平台默认的多进程启动方式。",
    )
    parser.set_defaults(compute=True, plot=False)
    parser.set_defaults(force_spawn=None)
    return parser.parse_args()


def cli_main() -> dict[str, object]:
    args = parse_args()
    return find_optimal_kb_full_data(
        input_dir=args.input_dir,
        wind_dir=args.wind_dir,
        output_dir=args.output_dir,
        optimal_kb_csv=args.optimal_kb_csv,
        compute=args.compute,
        plot=args.plot,
        overwrite=args.overwrite,
        poly_degree=args.poly_degree,
        max_workers=args.max_workers,
        force_spawn=args.force_spawn,
    )


if __name__ == "__main__":
    cli_main()