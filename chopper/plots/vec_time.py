"""Vector operator duration distribution per configuration.

Violin plots of normalized Vec-type operator durations split into forward,
backward, and optimizer/grad-accumulation phases. (Paper Figure: vec_time.pdf)
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
    fops: list[str] = ["f_attn_n", "f_mlp_n"],
    bops: list[str] = ["b_attn_n", "b_mlp_n"],
    oops: list[str] = ["opt_step", "b_ga"],
):
    """Load and process Vec operator timing data.

    Args:
        ts_files: List of paths to ts.pkl trace files
        configs: Config labels (e.g. "b1s4", "b2s8")
        fops: Forward Vec operator names
        bops: Backward Vec operator names
        oops: Optimizer / grad-accumulation operator names

    Returns:
        Tuple of (data, fops, bops, oops) where data maps config -> aggregated
        DataFrame normalized by global maximum elapsed time
    """
    group_arr = ["iteration", "chunk", "layer", "operator-type", "operator-name"]
    data = {}
    for ts_file, config in zip(ts_files, configs):
        df = get_df(
            ts_file,
            assign_chunks=True,
            remove_nan_chunks=True,
            remove_overlap=True,
            assign_optype=True,
            fix_names=True,
            group_arr=["gpu"] + group_arr,
            group_map={
                "ts": ["first", "last"],
                "dur": ["sum", "last"],
            },
            sort_value="ts_first",
        )
        data[config] = df

    max_dur = 0
    for setup in data.keys():
        data[setup]["elapsed_time"] = (
            data[setup]["ts_last"] + data[setup]["dur_last"] - data[setup]["ts_first"]
        )
        data[setup] = (
            data[setup]
            .groupby(
                ["gpu", "iteration", "chunk", "operator-type", "operator-name"],
                dropna=False,
            )
            .agg(elapsed_time=("elapsed_time", "sum"))
            .reset_index()
        )
        data[setup] = data[setup][data[setup]["operator-type"] == "Vec"]
        max_dur = max(data[setup]["elapsed_time"].max(), max_dur)

    for setup in data.keys():
        data[setup]["elapsed_time"] /= max_dur

    return data, list(fops), list(bops), list(oops)


def draw(
    fig: Figure,
    input_data,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw Vec operator duration violins split by training phase.

    Three phase rows (forward, backward, optimizer-style) by N columns
    (one per modulation parameter). Includes an "other" violin for
    forward/backward phases capturing residual Vec ops.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() with (data, fops, bops, oops)
        paper_mode: PaperMode settings for publication-quality figures
    """
    data, fops, bops, oops = input_data
    params = list(data.keys())

    fig.clear()
    fig.patches.clear()

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    n_rows, n_cols = 4, len(params)
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[3, 3, 0.6, 2])
    axs = tuple(
        tuple(fig.add_subplot(gs[r, c]) for c in range(n_cols))
        for r in range(n_rows)
    )
    for col in range(n_cols):
        axs[2][col].set_visible(False)

    violin_color = rgb(0xD4, 0x11, 0x59)
    line_color = (
        max(0, violin_color[0] - 0.3),
        max(0, violin_color[1] - 0.3),
        max(0, violin_color[2] - 0.3),
    )
    violin_alpha = 0.7

    for i, setup in enumerate(params):
        for phase in (0, 1, 2):
            sel_ops = fops if phase == 0 else bops if phase == 1 else oops
            if phase != 2:
                y_ticks = np.arange(1, len(sel_ops) + 2)
            else:
                y_ticks = np.arange(1, len(sel_ops) + 1)
            ax = axs[phase if phase != 2 else 3][i]

            ax.set_yticks(y_ticks)
            ax.set_ylim(y_ticks[0] - 0.5, y_ticks[-1] + 0.5)
            ax.set_xlim((-0.05, 1.05))
            if i == 0:
                if phase != 2:
                    ax.set_yticklabels(tuple(sel_ops) + ("other",))
                else:
                    ax.set_yticklabels(sel_ops)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            if i == n_cols // 2 and phase == 2:
                ax.set_xlabel("norm duration", labelpad=0)

            op_data = tuple(
                data[setup].groupby("operator-name").get_group(op)["elapsed_time"].values
                for op in sel_ops
            )
            other_data = data[setup][
                (~data[setup]["operator-name"].isin(tuple(fops) + tuple(bops) + tuple(oops)))
                & (~data[setup]["chunk"].isin(("fwd",) if phase == 0 else ("bwd",)))
            ]["elapsed_time"].values
            parts = ax.violinplot(
                op_data + (other_data,) if phase != 2 else op_data,
                positions=y_ticks, vert=False,
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
                parts[line_type].set_linewidth(0.8)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.grid(axis="x", linestyle="-", alpha=0.5)
            ax.set_xticks((0.5,))
            if phase in (0, 1):
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            else:
                ax.set_title(setup, fontsize=8, pad=1)
                ax.tick_params(axis="x", which="major", pad=1)
            ax.tick_params(axis="y", which="major", pad=1)

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
    fops: list[str] = ["f_attn_n", "f_mlp_n"],
    bops: list[str] = ["b_attn_n", "b_mlp_n"],
    oops: list[str] = ["opt_step", "b_ga"],
    paper_mode: PaperMode = PaperMode(),
    figsize: tuple[float, float] = (7.16, 5.0),
    filename: str = "vec_time.pdf",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(ts_files, configs, fops, bops, oops)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
