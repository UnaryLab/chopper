import numpy as np
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


def _load_device_kernels(path):
    """Load kernel DataFrame from a device_merged.pkl.

    Uses group 0's kernel trace (which has ts.pkl annotations).
    """
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    assert "groups" in data, f"{path} is not a device_merged.pkl"
    return data["groups"][0]["kernels"]


def _derive_gap_overhead(df):
    """Compute total inter-kernel gap when ts_cuda_runtime is unavailable.

    Returns DataFrame with 'Launch Overhead' column (ms) -- the positive
    gap between previous kernel end and current kernel start.
    """
    grp = ['gpu', 'iteration']
    df = df.sort_values(grp + ['ts']).reset_index(drop=True)

    prev_end = (df.groupby(grp)['ts'].shift(1) +
                df.groupby(grp)['dur'].shift(1))
    df['Launch Overhead'] = np.maximum(0, df['ts'] - prev_end).astype(float) * 1e-6
    df.loc[df.groupby(grp).head(1).index, 'Launch Overhead'] = 0

    return df


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
):
    """Load and process kernel launch overhead metrics.

    Computes per-kernel launch overhead (gap between consecutive compute
    kernels on the same GPU within the same iteration), then sums across
    layers per operator. Normalizes to the maximum overhead across configs.

    Supports both ts.pkl (with Prep+Call split) and device_merged.pkl
    (total inter-kernel gap only).

    Args:
        ts_files: List of paths to trace pickle files
        configs: Config labels (e.g. "b1s4", "b2s8")

    Returns:
        Dict mapping config names to processed DataFrames with normalized overhead metrics
    """
    raw = {}
    for ts_file, config in zip(ts_files, configs):
        loaded = load_pickle(ts_file)
        if isinstance(loaded, dict) and "groups" in loaded:
            raw[config] = _load_device_kernels(ts_file)
        else:
            raw[config] = loaded

    data = {}
    for setup, df in raw.items():
        has_cuda_ts = "ts_cuda_runtime" in df.columns

        if has_cuda_ts:
            metrics = ("Prep Overhead", "Call Overhead")
        else:
            metrics = ("Launch Overhead",)

        df["layer"] = df["layer"].fillna(-1)
        weird_mask = df["iteration"].isna()
        if weird_mask.any():
            if has_cuda_ts:
                assert df[~weird_mask]["ts"].min() > df[weird_mask]["ts"].max(), (
                    "NaN iteration isn't at the start"
                )
            df = df[~weird_mask]

        df = df[df["name"] != "Memcpy HtoD (Host -> Device)"]

        # First kernel ts per (gpu, iteration) from ALL kernels (including comm)
        iter_start = df.groupby(["gpu", "iteration"])["ts"].min()

        df = assign_chunks(df)
        df = df[~df["chunk"].isna()]
        df = fix_names(df)
        overlap_mask = no_overlap_mask(df)
        df = df[overlap_mask].copy()

        if has_cuda_ts:
            df = derive_prep_overhead(df)
            df = derive_call_overhead(df)
        else:
            df = _derive_gap_overhead(df)

        # For the first compute kernel per iteration, override with gap
        # from the iteration's first kernel (including comm)
        first_idx = df.groupby(["gpu", "iteration"]).head(1).index
        for idx in first_idx:
            row = df.loc[idx]
            start_ts = iter_start.loc[(row["gpu"], row["iteration"])]
            gap = max(0, row["ts"] - start_ts) * 1e-6
            if "Prep Overhead" in df.columns:
                df.loc[idx, "Prep Overhead"] = gap
                df.loc[idx, "Call Overhead"] = 0
            else:
                df.loc[idx, "Launch Overhead"] = gap

        # Sum across layers per (gpu, iteration, operator-name)
        agg_df = (
            df.groupby(["gpu", "iteration", "operator-name"])
            .agg({m: "sum" for m in metrics})
            .reset_index()
        )
        # Median across (gpu, iteration) per operator
        op_df = (
            agg_df.groupby("operator-name")
            .agg({m: "median" for m in metrics})
            .reset_index()
        )
        data[setup] = op_df

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

    # Collect all metrics across configs
    all_metrics = set()
    for df in data.values():
        for m in ("Prep Overhead", "Call Overhead", "Launch Overhead"):
            if m in df.columns:
                all_metrics.add(m)
    if "Prep Overhead" in all_metrics:
        metrics = ("Prep Overhead", "Call Overhead")
    else:
        metrics = ("Launch Overhead",)

    # Sort operators by total overhead (descending) using first config
    first_setup = next(iter(data.values())).set_index("operator-name")
    first_metrics = [m for m in ("Prep Overhead", "Call Overhead", "Launch Overhead") if m in first_setup.columns]
    ops = sorted(ops, key=lambda op: sum(first_setup.loc[op, m] for m in first_metrics), reverse=True)

    bar_color = {m: rgb_colors[i] for i, m in enumerate(metrics)}
    bar_color["Launch Overhead"] = rgb(0x8D, 0xA0, 0xCB)

    # Compute normalization factors per side
    def _max_total(op_list):
        mx = 0
        for setup in params:
            op_df = data[setup].set_index("operator-name")
            setup_metrics = [m for m in ("Prep Overhead", "Call Overhead", "Launch Overhead") if m in op_df.columns]
            for op in op_list:
                assert op in op_df.index, f"operator {op} not in {setup}"
                mx = max(mx, sum(op_df.loc[op, m] for m in setup_metrics))
        return mx

    lnorm = _max_total(lops) if two_axes else _max_total(ops)
    rnorm = _max_total(rops) if two_axes else lnorm

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
    if two_axes:
        axs[len(lops)].set_visible(False)

    for setup in params:
        op_df = data[setup].set_index("operator-name")
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

            ax.set_xticks(x)
            ax.tick_params(axis="y", which="major", pad=1)
            ax.tick_params(axis="x", which="major", pad=1, rotation=65)
            ax.set_xticklabels(params)
            ax.set_xlim(-0.5, len(params) - 1 + 0.5)
            bar_width = 0.9
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            oi = ax_idxs.index(ax_idx)
            op = ops[oi]
            is_right = two_axes and oi >= len(lops)
            norm = rnorm if is_right else lnorm

            assert op in op_df.index, f"operator {op} not in {setup}"
            row = op_df.loc[op]
            tick = params.index(setup)
            bottom = 0

            ax.set_title(op, pad=5, fontsize=8)

            if "Prep Overhead" in op_df.columns:
                # Has Prep+Call split
                call_val = row["Call Overhead"] / norm
                prep_val = row["Prep Overhead"] / norm
                ax.bar(
                    tick, call_val,
                    width=bar_width * 0.9, bottom=bottom,
                    color=bar_color["Call Overhead"], alpha=0.99,
                )
                bottom += call_val
                ax.bar(
                    tick, prep_val,
                    width=bar_width * 0.9, bottom=bottom,
                    color=bar_color["Prep Overhead"], alpha=0.99,
                )
            else:
                # Launch Overhead only (device profiling)
                val = row["Launch Overhead"] / norm
                ax.bar(
                    tick, val,
                    width=bar_width * 0.9, bottom=bottom,
                    color=bar_color.get("Launch Overhead", bar_color["Prep Overhead"]),
                    alpha=0.99,
                )

    # Sync ylims
    if two_axes:
        for side_idxs in (ax_idxs[:len(lops)], ax_idxs[len(lops):]):
            ymax = max(axs[i].get_ylim()[1] for i in side_idxs)
            for i in side_idxs:
                axs[i].set_ylim(0, ymax * 1.1)
    else:
        ymax = max(axs[i].get_ylim()[1] for i in ax_idxs)
        for i in ax_idxs:
            axs[i].set_ylim(0, ymax * 1.1)

    legend_handles = [
        mpatches.Patch(color=bar_color[m], label=m) for m in reversed(metrics)
    ]

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
