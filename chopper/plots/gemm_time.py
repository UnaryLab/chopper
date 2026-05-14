"""GEMM operator duration distribution per configuration.

Violin plots of normalized GEMM forward and backward operator durations across
configurations. Forward and backward shown on the same tick with different colors.
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from chopper.common.colors import rgb
from chopper.common.load import get_df
from chopper.common.annotations import PaperMode, apply_paper_rcparams, paper_figsize


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    fops: list[str] = [
        "f_attn_fa", "f_mlp_dp", "f_mlp_gp", "f_mlp_up",
        "f_q_ip", "f_k_ip", "f_v_ip", "f_attn_op", "f_lp",
    ],
    bops: list[str] = [
        "b_attn_fa", "b_mlp_dp", "b_mlp_gp", "b_mlp_up",
        "b_q_ip", "b_k_ip", "b_v_ip", "b_attn_op", "b_lp",
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

    Single row with fwd/bwd on same tick using different colors.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() containing data, fops, bops
        paper_mode: PaperMode settings for publication-quality figures
    """
    data, fops, bops = input_data
    params = list(data.keys())
    assert len(fops) == len(bops), "fops and bops must have same length"

    fig.clear()
    fig.patches.clear()

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    # Strip f_/b_ prefix for labels
    labels = [op[2:] for op in fops]

    n_cols = len(params)
    axs = [fig.add_subplot(1, n_cols, c + 1) for c in range(n_cols)]

    fwd_color = rgb(0xD4, 0x11, 0x59)
    bwd_color = rgb(0x33, 0x99, 0xCC)

    y_ticks = np.arange(1, len(labels) + 1)

    for i, setup in enumerate(params):
        ax = axs[i]
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_ticks[0] - 0.5, y_ticks[-1] + 0.5)
        ax.set_xlim(-0.05, 1.05)

        if i == 0:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        ax.set_title(setup, fontsize=8, pad=2)

        # Backward violins (drawn first, behind)
        bwd_data = [
            data[setup].groupby("operator-name").get_group(op)["elapsed_time"].values
            for op in bops
        ]
        bwd_parts = ax.violinplot(
            bwd_data, positions=y_ticks, vert=False,
            showmedians=True, widths=0.8,
        )
        for pc in bwd_parts["bodies"]:
            pc.set_alpha(0.5)
            pc.set_facecolor(bwd_color)
            pc.set_edgecolor(bwd_color)
        for lt in ["cmedians", "cmins", "cmaxes", "cbars"]:
            bwd_parts[lt].set_color([bwd_color] * len(bops))
            bwd_parts[lt].set_linewidth(0.4)

        # Forward violins (drawn on top)
        fwd_data = [
            data[setup].groupby("operator-name").get_group(op)["elapsed_time"].values
            for op in fops
        ]
        fwd_parts = ax.violinplot(
            fwd_data, positions=y_ticks, vert=False,
            showmedians=True, widths=0.8,
        )
        for pc in fwd_parts["bodies"]:
            pc.set_alpha(0.7)
            pc.set_facecolor(fwd_color)
            pc.set_edgecolor(fwd_color)
        for lt in ["cmedians", "cmins", "cmaxes", "cbars"]:
            fwd_parts[lt].set_color([fwd_color] * len(fops))
            fwd_parts[lt].set_linewidth(0.4)

        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="x", linestyle="-", alpha=0.5)
        ax.set_xticks((0.5,))
        ax.tick_params(axis="y", which="major", pad=1)
        ax.tick_params(axis="x", which="major", pad=1)
        if i == n_cols // 2:
            ax.set_xlabel("norm duration", labelpad=0)

    legend_kwargs = dict(
        handles=[
            mpatches.Patch(color=fwd_color, alpha=0.7, label="fwd"),
            mpatches.Patch(color=bwd_color, alpha=0.7, label="bwd"),
        ],
        loc="upper center", ncol=2, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )
    if paper_mode.legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = paper_mode.legend_bbox
    fig.legend(**legend_kwargs)



def main(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    fops: list[str] = [
        "f_attn_fa", "f_mlp_dp", "f_mlp_gp", "f_mlp_up",
        "f_q_ip", "f_k_ip", "f_v_ip", "f_attn_op", "f_lp",
    ],
    bops: list[str] = [
        "b_attn_fa", "b_mlp_dp", "b_mlp_gp", "b_mlp_up",
        "b_q_ip", "b_k_ip", "b_v_ip", "b_attn_op", "b_lp",
    ],
    ncol: int = 1,
    figsize_ratio: float = 2.5 / 7.16,
    left: float = 0.1, right: float = 0.9,
    bottom: float = 0.1, top: float = 0.9,
    wspace: float = 0.2, hspace: float = 0.3,
    legend_x: float = 0.5, legend_y: float = 1.02,
    figsize: tuple[float, float] = (7.16, 2.5),
    filename: str = "gemm_time.pdf",
):
    paper_mode = PaperMode(
        enabled=True, ncol=ncol, figsize_ratio=figsize_ratio,
        left=left, right=right, bottom=bottom, top=top,
        wspace=wspace, hspace=hspace,
        legend_bbox=(legend_x, legend_y),
    )
    apply_paper_rcparams()
    figsize = paper_figsize(paper_mode)
    fig = Figure(figsize=figsize)
    input_data = get_data(ts_files, configs, fops, bops)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
