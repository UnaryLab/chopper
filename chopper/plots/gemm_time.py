"""GEMM operator duration distribution per configuration.

Violin plots of normalized GEMM forward and backward operator durations across
configurations and frameworks. (Paper Figure: gemm_time.pdf)
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from chopper.common.colors import rgb
from chopper.common.load import get_df
from chopper.common.annotations import PaperMode


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    fops: list[str] = [
        "f_attn_fa", "f_mlp_dp", "f_mlp_gp", "f_mlp_up",
        "f_qkv_ip", "f_attn_op", "f_lp",
    ],
    bops: list[str] = [
        "b_attn_fa", "b_mlp_dp", "b_mlp_gp", "b_mlp_up",
        "b_qkv_ip", "b_attn_op", "b_lp",
    ],
):
    """Load and process GEMM operator timing data.

    Args:
        ts_files: List of paths to ts.pkl trace files
        configs: Config labels (e.g. "b1s4", "b2s8")
        fops: Forward GEMM operator names to include
        bops: Backward GEMM operator names to include

    Returns:
        Tuple of (data, fops, bops) where data maps config -> aggregated DataFrame
        normalized by global maximum elapsed time
    """
    group_arr = ["iteration", "layer", "operator-name"]
    data = {}
    for ts_file, config in zip(ts_files, configs):
        df = get_df(
            ts_file,
            assign_chunks=True,
            remove_nan_chunks=True,
            remove_overlap=True,
            fix_names=True,
            group_arr=["gpu"] + group_arr,
            group_map={
                "ts": ["first", "last"],
                "dur": ["sum", "last"],
            },
            sort_value="ts_first",
        )
        data[config] = df

    ops = tuple(bops) + tuple(fops)
    max_dur = 0
    for setup in data.keys():
        data[setup]["elapsed_time"] = (
            data[setup]["ts_last"] + data[setup]["dur_last"] - data[setup]["ts_first"]
        )
        data[setup] = (
            data[setup]
            .groupby(["gpu", "iteration", "operator-name"], dropna=False)
            .agg(elapsed_time=("elapsed_time", "sum"))
            .reset_index()
        )
        op_mask = data[setup]["operator-name"].isin(ops)
        data[setup] = data[setup][op_mask]
        max_dur = max(data[setup]["elapsed_time"].max(), max_dur)

    for setup in data.keys():
        data[setup]["elapsed_time"] /= max_dur

    return data, list(fops), list(bops)


def draw(
    fig: Figure,
    input_data,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw GEMM operator duration violin plots per configuration.

    Two rows (forward / backward) by N columns (one per modulation parameter).

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() containing data, fops, bops
        paper_mode: PaperMode settings for publication-quality figures
    """
    data, fops, bops = input_data
    params = list(data.keys())

    fig.clear()
    fig.patches.clear()

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    n_rows, n_cols = 2, len(params)
    axs = tuple(
        tuple(fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1) for c in range(n_cols))
        for r in range(n_rows)
    )

    violin_color = rgb(0xD4, 0x11, 0x59)
    line_color = (
        max(0, violin_color[0] - 0.3),
        max(0, violin_color[1] - 0.3),
        max(0, violin_color[2] - 0.3),
    )
    violin_alpha = 0.7

    for i, setup in enumerate(params):
        for do_fops in (True, False):
            xlim = 1
            sel_ops = fops if do_fops else bops
            y_ticks = np.arange(1, len(sel_ops) + 1)
            ax = axs[0 if do_fops else 1][i]

            ax.set_yticks(y_ticks)
            ax.set_ylim(y_ticks[0] - 0.5, y_ticks[-1] + 0.5)
            if i == 0:
                ax.set_yticklabels(tuple(sel_ops))
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            if i == n_cols // 2 and not do_fops:
                ax.set_xlabel("norm duration", labelpad=0)

            op_data = [
                data[setup].groupby("operator-name").get_group(op)["elapsed_time"].values
                for op in sel_ops
            ]
            parts = ax.violinplot(
                op_data, positions=y_ticks, vert=False,
                showmedians=True, widths=0.9,
            )
            for pc in parts["bodies"]:
                pc.set_alpha(violin_alpha)
                pc.set_facecolor(violin_color)
                pc.set_edgecolor(violin_color)
            for line_type in ["cmedians", "cmins", "cmaxes", "cbars"]:
                parts[line_type].set_color(
                    tuple(line_color for _ in sel_ops)
                )
                parts[line_type].set_linewidth(0.4)

            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.grid(axis="x", linestyle="-", alpha=0.5)
            ax.set_xticks((0.5,))
            if not do_fops:
                ax.set_title(setup, fontsize=8, pad=1)
                ax.tick_params(axis="x", which="major", pad=1)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", which="major", pad=1)
            ax.set_xlim((-xlim * 5e-2, xlim + xlim * 5e-2))

    if paper_mode.enabled:
        fig.patches.append(
            mpatches.Rectangle(
                (0, 0), 1, 1,
                transform=fig.transFigure,
                fill=False, edgecolor="black", linewidth=1, zorder=1000,
            )
        )


def main(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    fops: list[str] = [
        "f_attn_fa", "f_mlp_dp", "f_mlp_gp", "f_mlp_up",
        "f_qkv_ip", "f_attn_op", "f_lp",
    ],
    bops: list[str] = [
        "b_attn_fa", "b_mlp_dp", "b_mlp_gp", "b_mlp_up",
        "b_qkv_ip", "b_attn_op", "b_lp",
    ],
    paper_mode: PaperMode = PaperMode(),
    figsize: tuple[float, float] = (7.16, 4.0),
    filename: str = "gemm_time.pdf",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(ts_files, configs, fops, bops)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
