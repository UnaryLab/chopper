"""Vector operator duration distribution per configuration.

Violin plots of normalized Vec-type operator durations. Forward and backward
shown on the same tick with different colors, optimizer ops separate.
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
    fops: list[str] = ["f_qkv_re", "f_mlp_gu", "f_mlp_gs", "f_mlp_n", "f_attn_n", "f_cel"],
    bops: list[str] = ["b_qkv_re", "b_mlp_gu", "b_mlp_gs", "b_mlp_n", "b_attn_n", "b_cel"],
    oops: list[str] = ["opt_step", "opt_gc"],
):
    """Load and process Vec operator timing data.

    Args:
        ts_files: List of paths to ts.pkl trace files
        configs: Config labels (e.g. "b1s4", "b2s8")
        fops: Forward Vec operator names
        bops: Backward Vec operator names
        oops: Optimizer operator names

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
    """Draw Vec operator duration violins.

    Fwd/bwd on same tick with different colors, optimizer ops in separate section.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() with (data, fops, bops, oops)
        paper_mode: PaperMode settings for publication-quality figures
    """
    data, fops, bops, oops = input_data
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

    # Strip f_/b_ prefix for fwd/bwd labels
    fb_labels = [op[2:] for op in fops]

    n_cols = len(params)
    n_fb = len(fb_labels)
    n_opt = len(oops)
    n_total = n_fb + n_opt

    gs = fig.add_gridspec(2, n_cols, height_ratios=[n_fb + 1, n_opt], hspace=0.3)
    axs_fb = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    axs_opt = [fig.add_subplot(gs[1, c]) for c in range(n_cols)]

    fwd_color = rgb(0xD4, 0x11, 0x59)
    bwd_color = rgb(0x33, 0x99, 0xCC)
    opt_color = rgb(0xF0, 0xE4, 0x42)

    for i, setup in enumerate(params):
        # ── Fwd/Bwd panel (with "other") ──
        ax = axs_fb[i]
        n_fb_with_other = n_fb + 1
        y_ticks = np.arange(1, n_fb_with_other + 1)
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_ticks[0] - 0.5, y_ticks[-1] + 0.5)
        ax.set_xlim(-0.05, 1.05)

        if i == 0:
            ax.set_yticklabels(fb_labels + ["other"])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        ax.set_title(setup, fontsize=8, pad=2)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

        # Backward (drawn first, behind)
        bwd_data = [
            data[setup].groupby("operator-name").get_group(op)["elapsed_time"].values
            for op in bops
        ]
        # "other" bwd
        bwd_other = data[setup][
            (~data[setup]["operator-name"].isin(tuple(fops) + tuple(bops) + tuple(oops)))
            & (data[setup]["chunk"] == "bwd")
        ]["elapsed_time"].values
        bwd_parts = ax.violinplot(
            bwd_data + [bwd_other], positions=y_ticks, vert=False,
            showmedians=True, widths=0.8,
        )
        for pc in bwd_parts["bodies"]:
            pc.set_alpha(0.5)
            pc.set_facecolor(bwd_color)
            pc.set_edgecolor(bwd_color)
        for lt in ["cmedians", "cmins", "cmaxes", "cbars"]:
            bwd_parts[lt].set_color([bwd_color] * n_fb_with_other)
            bwd_parts[lt].set_linewidth(0.4)

        # Forward (drawn on top)
        fwd_data = [
            data[setup].groupby("operator-name").get_group(op)["elapsed_time"].values
            for op in fops
        ]
        fwd_other = data[setup][
            (~data[setup]["operator-name"].isin(tuple(fops) + tuple(bops) + tuple(oops)))
            & (data[setup]["chunk"] == "fwd")
        ]["elapsed_time"].values
        fwd_parts = ax.violinplot(
            fwd_data + [fwd_other], positions=y_ticks, vert=False,
            showmedians=True, widths=0.8,
        )
        for pc in fwd_parts["bodies"]:
            pc.set_alpha(0.7)
            pc.set_facecolor(fwd_color)
            pc.set_edgecolor(fwd_color)
        for lt in ["cmedians", "cmins", "cmaxes", "cbars"]:
            fwd_parts[lt].set_color([fwd_color] * n_fb_with_other)
            fwd_parts[lt].set_linewidth(0.4)

        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="x", linestyle="-", alpha=0.5)
        ax.set_xticks((0.5,))
        ax.tick_params(axis="y", which="major", pad=1)

        # ── Optimizer panel ──
        ax = axs_opt[i]
        y_ticks_opt = np.arange(1, n_opt + 1)
        ax.set_yticks(y_ticks_opt)
        ax.set_ylim(y_ticks_opt[0] - 0.5, y_ticks_opt[-1] + 0.5)
        ax.set_xlim(-0.05, 1.05)

        if i == 0:
            ax.set_yticklabels(oops)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        opt_data = [
            data[setup].groupby("operator-name").get_group(op)["elapsed_time"].values
            for op in oops
        ]
        opt_parts = ax.violinplot(
            opt_data, positions=y_ticks_opt, vert=False,
            showmedians=True, widths=0.7,
        )
        for pc in opt_parts["bodies"]:
            pc.set_alpha(0.7)
            pc.set_facecolor(opt_color)
            pc.set_edgecolor((0.6, 0.5, 0.0))
            pc.set_linewidth(0.8)
        for lt in ["cmedians", "cmins", "cmaxes", "cbars"]:
            opt_parts[lt].set_color([opt_color] * n_opt)
            opt_parts[lt].set_linewidth(0.4)

        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="x", linestyle="-", alpha=0.5)
        ax.set_xticks((0.5,))
        ax.tick_params(axis="y", which="major", pad=1)
        ax.tick_params(axis="x", which="major", pad=1)
        if i == n_cols // 2:
            ax.set_xlabel("norm duration", labelpad=0)

    fig.legend(
        handles=[
            mpatches.Patch(color=fwd_color, alpha=0.7, label="fwd"),
            mpatches.Patch(color=bwd_color, alpha=0.7, label="bwd"),
            mpatches.Patch(color=opt_color, alpha=0.7, label="opt"),
        ],
        loc="upper center", ncol=3, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )

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
    fops: list[str] = ["f_qkv_re", "f_mlp_gu", "f_mlp_gs", "f_mlp_n", "f_attn_n", "f_cel"],
    bops: list[str] = ["b_qkv_re", "b_mlp_gu", "b_mlp_gs", "b_mlp_n", "b_attn_n", "b_cel"],
    oops: list[str] = ["opt_step", "opt_gc"],
    paper_mode: PaperMode = PaperMode(),
    figsize: tuple[float, float] = (7.16, 2.3),
    filename: str = "vec_time.pdf",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(ts_files, configs, fops, bops, oops)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
