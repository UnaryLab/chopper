import numpy as np
import re
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from matplotlib.ticker import MaxNLocator

import matplotlib.patches as mpatches
from matplotlib import rcParams

from chopper.common.colors import rgb
from chopper.common.cache import load_pickle
from chopper.common.annotations import (
    no_overlap_mask,
    assign_chunks,
    fix_names,
)
from chopper.common.trace_metrics import (
    derive_launch_overhead,
    derive_prep_overhead,
    derive_call_overhead,
)


def agg(
    df,
    group_arr,
    derive_cols_before=None,
    derive_cols_after=None,
    sum_cols_map={},
):
    """Aggregate trace data with custom derivations and grouping.
    
    Groups trace data by specified columns and applies aggregation functions,
    with optional pre- and post-processing derivation steps.
    
    Args:
        df: Input DataFrame to aggregate
        group_arr: List of column names to group by
        derive_cols_before: Optional list of derivation functions to apply before aggregation
        derive_cols_after: Optional list of derivation functions to apply after aggregation
        sum_cols_map: Dict mapping column names to aggregation functions
        
    Returns:
        Aggregated DataFrame with flattened column names
    """

    if derive_cols_before is not None:
        for derive_col in derive_cols_before:
            df = derive_col(df)

    df_summed = (
        df.groupby(group_arr, dropna=False)
        .agg(
            {
                **sum_cols_map,
            }
        )
        .reset_index()
    )

    df_summed.columns = [col[0] for col in df_summed.columns]

    if derive_cols_after is not None:
        for derive_col in derive_cols_after:
            derive_col(df_summed)

    return df_summed


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
):
    """Load and process kernel launch overhead metrics.

    Processes trace files to extract launch overhead, prep overhead, and call
    overhead metrics for different training operators. Filters out communication
    kernels and aggregates metrics by operator and layer, normalizing to the
    maximum overhead value across configs.

    Args:
        ts_files: List of paths to trace pickle files
        configs: Config labels (e.g. "b1s4", "b2s8")

    Returns:
        Dict mapping config names to processed DataFrames with normalized overhead metrics
    """
    data = {
        config: load_pickle(ts_file)
        for ts_file, config in zip(ts_files, configs)
    }

    metrics = (
        "Launch Overhead",
        "Prep Overhead",
        "Call Overhead",
    )
    max_ov_sub = 0
    for setup in data.keys():
        data[setup]["layer"] = data[setup]["layer"].fillna(-1)
        weird_mask = data[setup]["iteration"].isna()
        if weird_mask.any():
            weird_df = data[setup][weird_mask]
            max_weird_ts = weird_df["ts"].max()
            min_norm_ts = data[setup][~weird_mask]["ts"].min()
            assert min_norm_ts > max_weird_ts, "Nan iteration isn't at the start"
            data[setup] = data[setup][~weird_mask]

        data[setup] = data[setup][data[setup]["name"] != "Memcpy HtoD (Host -> Device)"]

        data[setup] = assign_chunks(data[setup])
        nan_chunk_mask = data[setup]["chunk"].isna()

        data[setup] = fix_names(data[setup])

        overlap_mask = no_overlap_mask(data[setup])

        data[setup] = agg(
            data[setup][overlap_mask & ~nan_chunk_mask],
            ["gpu", "chunk", "iteration", "operator-name", "layer"],
            derive_cols_before=(
                derive_launch_overhead,
                derive_prep_overhead,
                derive_call_overhead,
            ),
            sum_cols_map={metric: ["sum"] for metric in metrics},
        )
        gb_sub = data[setup][
            ~data[setup]["operator-name"].isin(("f_ie", "b_ga", "opt_step"))
        ].groupby("operator-name")
        max_ov_sub = max(max_ov_sub, np.max(gb_sub["Launch Overhead"].mean()))

    for setup in data.keys():
        data[setup]["Call Overhead"] /= max_ov_sub
        data[setup]["Prep Overhead"] /= max_ov_sub
        data[setup]["Launch Overhead"] /= max_ov_sub

    return data


def draw(
    fig: Figure,
    input_data,
    lops: list[str] = [
        "f_ie",
        "b_ga",
        "opt_step",
    ],
    rops: list[str] = [
        "f_attn_n",
        "b_mlp_dp",
        "b_ie",
    ],
    two_axes: bool = True,
):
    """Draw launch overhead comparison across configurations.
    
    Creates a multi-panel stacked bar chart comparing prep and call overhead across
    different operators and parameter configurations. Supports split y-axes
    for operators with different overhead scales (e.g., communication vs compute).
    
    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data() containing overhead metrics
        lops: List of left-side operators to plot (typically high-overhead ops)
        rops: List of right-side operators to plot (typically low-overhead ops)
        two_axes: If True, use split y-axes for left and right operators
    """
    data = input_data
    ops = lops + rops
    rgb_colors = (
        rgb(0x66, 0xC2, 0xA5),
        rgb(0xFC, 0x8D, 0x62),
    )

    params = list(data.keys())

    metrics = (
        "Prep Overhead",
        "Call Overhead",
    )

    bar_color = {m: rgb_colors[i] for i, m in enumerate(metrics)}

    n_rows = 1
    n_cols = len(ops) + (1 if two_axes else 0)

    x = np.arange(len(params))
    fig.clear()
    width_ratios = (
        (
            tuple(1 for _ in range(len(lops)))
            + (0.1,)
            + tuple(1 for _ in range(len(rops)))
        )
        if two_axes
        else (tuple(1 for _ in range(len(ops))))
    )
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=width_ratios)
    axs = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]

    ax_idxs = (
        list(tuple(range(len(lops))) + tuple(range(len(lops) + 1, n_cols)))
        if two_axes
        else list(tuple(range(0, n_cols)))
    )
    g_ymin: float | None = None
    g_ymax: float | None = None
    if two_axes:
        axs[len(lops)].set_visible(False)

    for setup in params:
        for ax_idx in ax_idxs:
            ax = axs[ax_idx]
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
            ax.spines["top"].set_visible(False)
            if ax_idx == ax_idxs[0]:
                ax.set_ylabel("norm time", labelpad=0)
                ax.spines["right"].set_visible(False)
            elif two_axes and ax_idx == ax_idxs[len(lops)]:
                ax.spines["right"].set_visible(False)
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            elif ax_idx == ax_idxs[-1]:
                ax.spines["left"].set_visible(False)
                ax.yaxis.set_label_position("right")
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=True,
                    labelright=True,
                    labelleft=False,
                    pad=1,
                )
                ax.set_ylabel("norm time", labelpad=0)
            else:
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            if two_axes and ax_idx >= len(lops) + 1:
                ax.set_ylim((0, 1.1))
            else:
                _ymin, _ymax = ax.get_ylim()
                if g_ymin is None:
                    g_ymin = _ymin
                else:
                    g_ymin = min(g_ymin, _ymin)
                if g_ymax is None:
                    g_ymax = _ymax
                else:
                    g_ymax = max(g_ymax, _ymax)

            ax.set_xticks(x)
            ax.tick_params(axis="y", which="major", pad=1)
            ax.tick_params(axis="x", which="major", pad=1, rotation=65)
            ax.set_xticklabels(params)
            ax.set_xlim(-0.5, len(params) - 1 + 0.5)
            bar_width = 0.9
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            bottom = 0

            op = ops[ax_idxs.index(ax_idx)]
            ax.set_title(op, pad=5, fontsize=8)
            med_p = (
                data[setup].groupby("operator-name").get_group(op)["Prep Overhead"].mean()
            )
            med_c = (
                data[setup].groupby("operator-name").get_group(op)["Call Overhead"].mean()
            )

            tick = params.index(setup)

            ax.bar(
                tick,
                med_c,
                width=bar_width * 0.9,
                bottom=bottom,
                color=bar_color["Call Overhead"],
                alpha=0.99,
            )

            bottom += med_c

            ax.bar(
                tick,
                med_p,
                width=bar_width * 0.9,
                bottom=bottom,
                color=bar_color["Prep Overhead"],
                alpha=0.99,
            )

    legend_handles = [
        mpatches.Patch(color=bar_color[m], label=m) for m in reversed(metrics)
    ]

    if two_axes:
        for i in range(len(lops)):
            if g_ymin is not None and g_ymax is not None:
                axs[i].set_ylim((g_ymin, g_ymax))

    if two_axes:
        axl = axs[len(lops) - 1]
        axr = axs[len(lops) + 1]

        posl = axl.get_position()
        posr = axr.get_position()

        y_val = 1.0
        transl = axl.transData + axl.transAxes.inverted()
        transr = axr.transData + axr.transAxes.inverted()

        yl_axes = transl.transform((0, y_val))[1]
        yr_axes = transr.transform((0, y_val))[1]

        yl_fig = posl.y0 + yl_axes * posl.height
        yr_fig = posr.y0 + yr_axes * posr.height

        fig.add_artist(
            Line2D(
                [posl.x1, posr.x0],
                [yl_fig, yr_fig],
                transform=fig.transFigure,
                color=rcParams["grid.color"],
                linewidth=rcParams["grid.linewidth"],
                linestyle="--",
                alpha=0.5,
            )
        )

    fig.legend(
        handles=legend_handles,
        ncol=len(legend_handles),
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=0.5,
        frameon=False,
    )


def main(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    lops: list[str] = ["f_ie", "b_ga", "opt_step"],
    rops: list[str] = ["f_attn_n", "b_mlp_dp", "b_ie"],
    two_axes: bool = True,
    figsize: tuple[float, float] = (7.16, 2.0),
    filename: str = "launch_overhead.pdf",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(ts_files, configs)
    draw(fig, input_data, lops, rops, two_axes)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
