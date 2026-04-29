"""Per-operator duration breakdown by source of overhead.

For each operator, decomposes the actual median duration into a stack of
contributions: theoretical compute time (peak FLOPs), instruction overhead
(MFMA flops vs theoretical), utilization, communication overlap (CDF mid),
and clock-frequency. (Paper Figure: overhead_breakdown.pdf)
"""

import re
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from chopper.common.colors import rgb
from chopper.common.cache import load_pickle
from chopper.common.annotations import (
    PaperMode,
    no_overlap_mask, assign_chunks, assign_operator_type, fix_names,
)
from chopper.common.rocm_metrics import (
    derive_duration,
    derive_tensor_flops_rocm,
    derive_tensor_util_rocm,
    derive_cycle_duration_rocm,
)
from chopper.common.trace_metrics import compute_overlap_cdf


def _agg(df, group_arr, derive_cols_before=None, derive_cols_after=None, sum_cols_map=None):
    sum_cols_map = sum_cols_map or {}
    if derive_cols_before is not None:
        for derive_col in derive_cols_before:
            df = derive_col(df)
    df_summed = df.groupby(group_arr, dropna=False).agg({**sum_cols_map}).reset_index()
    df_summed.columns = [
        "_".join(col).strip("_") if col[1] != "sum" else col[0]
        for col in df_summed.columns
    ]
    if derive_cols_after is not None:
        for derive_col in derive_cols_after:
            derive_col(df_summed)
    return df_summed


def _calc_thr_flops(bs: int, cl: int) -> dict:
    d = 4096
    hid = 14336
    kv_heads = 8
    heads = 32
    head_dim = d / heads
    return {
        "f_attn_fa": 2 * (cl ** 2) * head_dim * heads * bs,
        "f_attn_op": 2 * cl * d * d * bs,
        "f_mlp_dp": 2 * cl * d * hid * bs,
        "f_mlp_gp": 2 * cl * d * hid * bs,
        "f_mlp_up": 2 * cl * d * hid * bs,
        "f_qkv_ip": 2 * cl * d * (heads + 2 * kv_heads) * head_dim * bs,
        "b_attn_fa": 5 * (cl ** 2) * head_dim * heads * bs,
        "b_attn_op": 4 * cl * d * d * bs,
        "b_mlp_dp": 4 * cl * d * hid * bs,
        "b_mlp_gp": 4 * cl * d * hid * bs,
        "b_mlp_up": 4 * cl * d * hid * bs,
        "b_qkv_ip": 4 * cl * d * (heads + 2 * kv_heads) * head_dim * bs,
    }


def get_data(
    counter_files: list[str] = ["./counters.pkl"],
    configs: list[str] = ["default"],
    target_gpu: int = 0,
):
    """Load counters and pre-aggregate per-operator metrics for one GPU.

    For each config, filters to non-overlapping compute kernels of the target
    GPU, aggregates by (chunk, iteration, operator-name, layer), derives Tensor
    Flops / Tensor Util / Cycle Duration via rocm_metrics, and additionally
    computes the per-operator overlap CDF against communication kernels.

    Args:
        counter_files: List of paths to counters.pkl files
        configs: Config labels (e.g. "b1s4", "b2s8")
        target_gpu: Single GPU index to extract (matches paper figure)

    Returns:
        Dict mapping config -> {"data": agg DataFrame, "cdf": {op -> overlap CDF DataFrame}}
    """
    raw = {config: load_pickle(fn) for config, fn in zip(configs, counter_files)}
    breakdown_data = {}

    derive_cols = (
        derive_duration,
        derive_tensor_flops_rocm,
        derive_tensor_util_rocm,
        derive_cycle_duration_rocm,
    )

    sum_cols_map = {}
    for dc in derive_cols:
        for k, v in dc.map.items():
            sum_cols_map.setdefault(k, list(v))

    overlap_cdf_map = {
        "dur": ["sum", "last"],
        "ts": ["first", "last"],
    }
    sum_cols_map = {
        **{
            k: list(set(v + overlap_cdf_map[k])) if k in overlap_cdf_map else v
            for k, v in sum_cols_map.items()
        },
        **{k: v for k, v in overlap_cdf_map.items() if k not in sum_cols_map},
    }

    dummy_flops = _calc_thr_flops(1, 4)

    for config in configs:
        df = raw[config]
        gpu_mask = df["gpu"] == target_gpu
        overlap_mask = no_overlap_mask(df)

        df = assign_chunks(df)
        df = assign_operator_type(df)
        df = fix_names(df)

        nan_chunk_mask = df["chunk"].isna()
        op_mask = df["operator-name"].isin(dummy_flops.keys())

        agg_data = _agg(
            df[overlap_mask & gpu_mask & ~nan_chunk_mask & op_mask],
            ["gpu", "chunk", "iteration", "operator-name", "layer"],
            derive_cols_after=derive_cols,
            sum_cols_map=sum_cols_map,
        )

        cdf_input_kernels = df[~nan_chunk_mask & gpu_mask & ~overlap_mask]
        breakdown_data[config] = {
            "data": agg_data,
            "cdf": {
                op: compute_overlap_cdf(
                    agg_data[agg_data["operator-name"] == op],
                    cdf_input_kernels,
                )
                for op in dummy_flops.keys()
            },
        }

    return breakdown_data


def draw(
    fig: Figure,
    input_data,
    setup_axis_map: dict = {
        "b1s4": (0, 0),
        "b2s4": (1, 1),
        "b4s4": (2, 2),
        "b1s8": (3, 1),
        "b2s8": (4, 2),
    },
    xlims: list[list[float]] = [
        [0.11, 0.28, 0.48],
        [0.38, 1.05, 1.05],
    ],
    sanity_check: bool = False,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw stacked-source breakdown bars per operator and per config.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data() of config -> {data, cdf}
        setup_axis_map: Maps mod-param suffix (e.g. "b1s4") to (column_index, xlim_index)
        xlims: Per-row list of x-axis upper limits indexed by xlim_index
        sanity_check: If True, also draw an "Actual Duration" bar per op
        paper_mode: PaperMode settings for publication-quality figures
    """
    data = input_data
    setups = tuple(data.keys())
    ops = tuple(_calc_thr_flops(1, 4 << 10).keys())

    bars = [
        "theoretical duration",
        "instruction",
        "utilization",
        "overlap",
        "frequency",
    ]
    if sanity_check:
        bars += ["Actual Duration"]

    rgb_colors = (
        rgb(0x64, 0x8F, 0xFF),
        rgb(0x78, 0x5E, 0xF0),
        rgb(0xDC, 0x26, 0x7F),
        rgb(0xFE, 0x61, 0x00),
        rgb(0xFF, 0xB0, 0x00),
    )
    bar_color = {bar: rgb_colors[i] for i, bar in enumerate(bars[: len(rgb_colors)])}
    if sanity_check and "Actual Duration" not in bar_color:
        bar_color["Actual Duration"] = rgb(0x99, 0x99, 0x99)

    stack_data = {}
    max_dur = 0
    for s in setups:
        for op in ops:
            op_data = (
                data[s]["data"]
                .groupby(["operator-name"], dropna=False)
                .get_group((op,))
            )
            max_dur = max(max_dur, op_data["Duration"].median())

    for s in setups:
        for op in ops:
            op_slot = stack_data.setdefault(s, {}).setdefault(op, {})

            op_data = (
                data[s]["data"]
                .groupby(["operator-name"], dropna=False)
                .get_group((op,))
            )
            cdf_data = data[s]["cdf"][op]

            act_duration = op_data["Duration"].median()
            op_slot["Actual Duration"] = act_duration / max_dur

            batch_size = int(re.findall(r"b(\d+)", s)[0])
            seq_len = int(re.findall(r"s(\d+)", s)[0]) << 10

            thr_flops = _calc_thr_flops(batch_size, seq_len)
            op_thr_flops = thr_flops[op]
            thr_flops_sec = 2 * 16 * 16 * 16 * 4 * 304 / 16 * 2100e6
            op_slot["theoretical duration"] = (
                op_thr_flops / thr_flops_sec * 1e3
            ) / max_dur

            act_flops = op_data["Tensor Flops"].median()
            op_slot["inst"] = act_flops / op_thr_flops

            tensor_util = op_data["Tensor Util"].median()
            op_slot["util"] = 1 / (tensor_util / 100)

            cycle_dur = op_data["Cycle Duration"].median() * 1e-3
            op_slot["freq"] = act_duration / cycle_dur

            overlap = (
                cdf_data.groupby("cdf")["overlap_ratio"].agg(["median"]).reset_index()
            )
            elapsed = (
                cdf_data.groupby("cdf")["elapsed"].agg(["median"]).reset_index()
            )
            closest = elapsed.loc[(elapsed["cdf"] - 50).abs().idxmin(), "cdf"]
            mid = elapsed[elapsed["cdf"] == closest]["median"].to_numpy()[0]
            closest = overlap.loc[(overlap["cdf"] - 50).abs().idxmin(), "cdf"]
            ovr = overlap[overlap["cdf"] == closest]["median"].to_numpy()[0]
            first = elapsed.iloc[0]["median"]
            op_slot["overlap"] = max(mid / first, 1) if ovr != 0 else 1

    fig.clear()
    fig.patches.clear()
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    n_rows = 2
    n_cols_grid = max((setup_axis_map[m][0] for m in stack_data.keys()), default=0) + 1
    axs = tuple(
        tuple(fig.add_subplot(n_rows, n_cols_grid, r * n_cols_grid + c + 1) for c in range(n_cols_grid))
        for r in range(n_rows)
    )

    n_ops = len(ops)
    ops_per_row = n_ops // n_rows

    for config in stack_data.keys():
        i, xlim_idx = setup_axis_map[config]
        for l in range(n_rows):
            op_sidx = l * ops_per_row
            op_eidx = (l + 1) * ops_per_row + ((n_ops % n_rows) if l == n_rows - 1 else 0)
            y = np.arange(op_eidx - op_sidx)
            ax = axs[l][i]
            ax.grid(axis="x", linestyle="-", alpha=0.5)
            xlim = xlims[l][xlim_idx]
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
            ax.set_xlim(0, xlim)
            ax.set_yticks(y)
            if i == 0:
                if l == n_rows - 1:
                    ax.set_xlabel("norm duration", labelpad=0)
                    ax.xaxis.set_label_coords(-0.15, -0.250)
                ax.set_yticklabels(ops[op_sidx:op_eidx])
            else:
                ax.tick_params(axis="y", length=0)
                ax.set_yticklabels([])
            if l != 0:
                ax.text(
                    0.50, -0.4, config,
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                )
            if sanity_check:
                bar_width = 0.8 / 2
            else:
                bar_width = 0.8

            for k, (op, op_data) in enumerate(
                tuple(stack_data[config].items())[op_sidx:op_eidx]
            ):
                bar_base = 0
                acc = 0

                if sanity_check:
                    ax.barh(
                        y[k] + bar_width,
                        op_data["Actual Duration"],
                        left=bar_base,
                        label="Actual Duration",
                        color=bar_color["Actual Duration"],
                        height=bar_width * 0.9,
                        alpha=0.99,
                    )

                thr_duration = op_data["theoretical duration"]
                ax.barh(
                    y[k], thr_duration,
                    left=bar_base, label="theoretical duration",
                    color=bar_color["theoretical duration"],
                    height=bar_width * 0.9,
                    alpha=0.99,
                )
                acc = thr_duration
                bar_base = acc

                flop_ovr = op_data["inst"]
                flop_time = flop_ovr * acc
                ax.barh(
                    y[k], flop_time - acc,
                    left=bar_base,
                    color=bar_color["instruction"],
                    height=bar_width * 0.9,
                    alpha=0.99,
                )
                acc = flop_time
                bar_base = acc

                compute_ovr = op_data["util"]
                compute_time = compute_ovr * acc
                ax.barh(
                    y[k], compute_time - acc,
                    left=bar_base,
                    color=bar_color["utilization"],
                    height=bar_width * 0.9,
                    alpha=0.99,
                )
                acc = compute_time
                bar_base = acc

                overlap_ovr = op_data["overlap"]
                overlap_time = overlap_ovr * bar_base
                ax.barh(
                    y[k], overlap_time - bar_base,
                    left=bar_base,
                    color=bar_color["overlap"],
                    height=bar_width * 0.9,
                    alpha=0.99,
                )
                acc = overlap_time
                bar_base = acc

                clock_ovr = op_data["freq"] / overlap_ovr
                clock_time = clock_ovr * bar_base
                ax.barh(
                    y[k], clock_time - bar_base,
                    left=bar_base,
                    color=bar_color["frequency"],
                    height=bar_width * 0.9,
                    alpha=0.99,
                )

            ax.tick_params(axis="x", rotation=0, pad=1)
            ax.tick_params(axis="y", pad=1)

    bar_handles = [mpatches.Patch(color=bar_color[bar], label=bar) for bar in bars]
    legend_kwargs = dict(
        handles=bar_handles,
        loc="upper center",
        ncol=len(bar_handles),
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=0.5,
        frameon=False,
    )
    if paper_mode.enabled and paper_mode.legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = paper_mode.legend_bbox
    fig.legend(**legend_kwargs)

    if paper_mode.enabled:
        fig.patches.append(
            mpatches.Rectangle(
                (0, 0), 1, 1,
                transform=fig.transFigure,
                fill=False, edgecolor="black", linewidth=1, zorder=1000,
            )
        )


def main(
    counter_files: list[str] = ["./counters.pkl"],
    configs: list[str] = ["default"],
    target_gpu: int = 0,
    sanity_check: bool = False,
    paper_mode: PaperMode = PaperMode(),
    filename: str = "overhead_breakdown.png",
):
    fig = Figure()
    input_data = get_data(counter_files, configs, target_gpu)
    draw(fig, input_data, sanity_check=sanity_check, paper_mode=paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
