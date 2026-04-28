"""Vector operator duration distribution per configuration.

Violin plots of normalized Vec-type operator durations split into forward,
backward, and optimizer/grad-accumulation phases. (Paper Figure: vec_time.pdf)
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from chopper.common.colors import rgb
from chopper.common.load import get_df
from chopper.common.annotations import Framework, PaperMode


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    variants: list[str] = ["default"],
    frameworks: list[Framework] = [Framework.FSDPv2],
    fops: list[str] = ["f_attn_n", "f_mlp_n"],
    bops: list[str] = ["b_attn_n", "b_mlp_n"],
    oops: list[str] = ["opt_step", "b_ga"],
):
    """Load and process Vec operator timing data.

    Args:
        ts_files: List of paths to ts.pkl trace files
        variants: Variant labels formatted as "<framework>-<config>"
        frameworks: Framework type for each ts_file
        fops: Forward Vec operator names
        bops: Backward Vec operator names
        oops: Optimizer / grad-accumulation operator names

    Returns:
        Tuple of (data, fops, bops, oops) where data maps variant -> aggregated
        DataFrame normalized by global maximum elapsed time
    """
    group_arr = ["iteration", "chunk", "layer", "operator-type", "operator-name"]
    data = {}
    for ts_file, variant, framework in zip(ts_files, variants, frameworks):
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
            framework=framework,
        )
        data[variant] = df

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
    setups = tuple(data.keys())
    mod_params = sorted(set(setup.split("-")[1] for setup in setups))

    fig.clear()
    fig.patches.clear()

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    n_rows, n_cols = 4, len(mod_params)
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[3, 3, 0.6, 2])
    axs = tuple(
        tuple(fig.add_subplot(gs[r, c]) for c in range(n_cols))
        for r in range(n_rows)
    )
    for col in range(n_cols):
        axs[2][col].set_visible(False)

    vendors = sorted(set(s.split("-")[0] for s in setups))
    rgb_colors = (rgb(0xD4, 0x11, 0x59), rgb(0x1A, 0x85, 0xFF))
    violin_color_dict = {vc: rgb_colors[i] for i, vc in enumerate(vendors)}
    line_color_dict = {
        vc: (
            max(0, rgb_colors[i][0] - 0.3),
            max(0, rgb_colors[i][1] - 0.3),
            max(0, rgb_colors[i][2] - 0.3),
        )
        for i, vc in enumerate(vendors)
    }
    violin_alpha = 0.7

    for i, setup in enumerate(setups):
        i_ = mod_params.index(setup.split("-")[1])
        for phase in (0, 1, 2):
            sel_ops = fops if phase == 0 else bops if phase == 1 else oops
            if phase != 2:
                y_ticks = np.arange(1, len(sel_ops) + 2)
            else:
                y_ticks = np.arange(1, len(sel_ops) + 1)
            ax = axs[phase if phase != 2 else 3][i_]

            ax.set_yticks(y_ticks)
            ax.set_ylim(y_ticks[0] - 0.5, y_ticks[-1] + 0.5)
            ax.set_xlim((-0.05, 1.05))
            if i == 0:
                if phase != 2:
                    ax.set_yticklabels(tuple(sel_ops) + ("other",))
                else:
                    ax.set_yticklabels(sel_ops)
            elif i_ > 0:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            if i_ == n_cols // 2 and phase == 2:
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
            vendor, mod_param = setup.split("-")
            for pc in parts["bodies"]:
                color = violin_color_dict[vendor]
                pc.set_alpha(violin_alpha)
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
            for line_type in ["cmedians", "cmins", "cmaxes", "cbars"]:
                parts[line_type].set_color(
                    tuple(line_color_dict[vendor] for _ in sel_ops)
                )
                parts[line_type].set_linewidth(0.8)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.grid(axis="x", linestyle="-", alpha=0.5)
            ax.set_xticks((0.5,))
            if phase in (0, 1):
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            else:
                ax.set_title(mod_param, fontsize=8, pad=1)
                ax.tick_params(axis="x", which="major", pad=1)
            ax.tick_params(axis="y", which="major", pad=1)

    legend_handles = [mpatches.Patch(color=violin_color_dict[v], label=v) for v in vendors]
    legend_kwargs = dict(
        handles=legend_handles,
        loc="upper center",
        ncol=len(vendors),
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
    ts_files: list[str] = ["./ts.pkl"],
    variants: list[str] = ["default"],
    frameworks: list[Framework] = [Framework.FSDPv2],
    fops: list[str] = ["f_attn_n", "f_mlp_n"],
    bops: list[str] = ["b_attn_n", "b_mlp_n"],
    oops: list[str] = ["opt_step", "b_ga"],
    paper_mode: PaperMode = PaperMode(),
    filename: str = "vec_time.png",
):
    fig = Figure()
    input_data = get_data(ts_files, variants, frameworks, fops, bops, oops)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
