"""Overlap-ratio vs. duration correlation per operator across frameworks.

For a chosen set of backward operators, plots median overlap ratio (left) and
median normalized duration (right) over kernel samples, with min/max and
quartile bands. Reports Pearson correlation between the two.
(Paper Figure: overlap_correlation.pdf)
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from chopper.common.colors import rgb
from chopper.common.load import get_overlap_df
from chopper.common.annotations import PaperMode


def _no_digits_formatter(x, _):
    return f"{x:.0f}"


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    operators: list[str] = [
        "b_attn_n", "b_mlp_n", "b_mlp_up", "b_mlp_gp", "b_mlp_dp",
    ],
    iter_idxs: range = range(-5, -2, 1),
):
    """Load overlap data for the requested operators across configs.

    Normalizes 'elapsed' per operator by the minimum elapsed across configs.

    Args:
        ts_files: List of ts.pkl trace files
        configs: Config labels (e.g. "b1s4", "b2s8")
        operators: Operator names to include
        iter_idxs: Range of iteration indices to select

    Returns:
        Dict mapping config -> {operator -> overlap DataFrame}
    """
    data = {}
    for ts_file, config in zip(ts_files, configs):
        data[config] = get_overlap_df(
            ts_file,
            iter_idxs=list(iter_idxs),
        )

    overlap_data = {config: {} for config in configs}
    config_min_elapsed = {config: {} for config in configs}

    for config in configs:
        for op in operators:
            op_mask = data[config]["operator-name"] == op
            sub = data[config][op_mask].copy()
            sub["op_idx"] = sub.groupby("gpu").cumcount()
            overlap_data[config][op] = sub
            config_min_elapsed[config][op] = np.min(sub["elapsed"])

    for op in operators:
        global_min = min(config_min_elapsed[config][op] for config in configs)
        for config in configs:
            overlap_data[config][op]["elapsed"] /= global_min

    return overlap_data


def draw(
    fig: Figure,
    input_data,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw overlap-ratio vs. duration with Pearson correlation per operator.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data()
        paper_mode: PaperMode settings for publication-quality figures
    """
    data = input_data
    overlaps = None
    for config in data.keys():
        ops = tuple(data[config].keys())
        if overlaps is None:
            overlaps = ops
        else:
            assert overlaps == ops

    fig.clear()
    fig.patches.clear()
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    n_rows = len(overlaps)
    n_cols = 2
    axses = tuple(
        tuple(fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1) for c in range(n_cols))
        for r in range(n_rows)
    )

    params = list(data.keys())
    rgb_colors = (rgb(0xD4, 0x11, 0x59), rgb(0x1A, 0x85, 0xFF))
    color_dict = {config: rgb_colors[i % len(rgb_colors)] for i, config in enumerate(params)}
    line_color_dict = {
        config: (
            max(0, rgb_colors[i % len(rgb_colors)][0] - 0.3),
            max(0, rgb_colors[i % len(rgb_colors)][1] - 0.3),
            max(0, rgb_colors[i % len(rgb_colors)][2] - 0.3),
        )
        for i, config in enumerate(params)
    }

    for j, ov in enumerate(overlaps):
        axs = axses[j]
        for i, config in enumerate(reversed(params)):
            overlap_ratio = (
                data[config][ov]
                .groupby("op_idx")["overlap_ratio"]
                .agg(["min", lambda x: x.quantile(0.25), "median",
                      lambda x: x.quantile(0.75), "max"])
                .reset_index()
            )
            elapsed = (
                data[config][ov]
                .groupby("op_idx")["elapsed"]
                .agg(["min", lambda x: x.quantile(0.25), "median",
                      lambda x: x.quantile(0.75), "max"])
                .reset_index()
            )
            overlap_ratio.columns = ["op_idx", "min", "q1", "median", "q3", "max"]
            elapsed.columns = ["op_idx", "min", "q1", "median", "q3", "max"]

            merged = overlap_ratio.merge(
                elapsed, on="op_idx", suffixes=("_overlap", "_elapsed")
            )
            merged["norm_overlap"] = (
                merged["median_overlap"] - merged["median_overlap"].min()
            ) / (merged["median_overlap"].max() - merged["median_overlap"].min())
            merged["norm_elapsed"] = (
                merged["median_elapsed"] - merged["median_elapsed"].min()
            ) / (merged["median_elapsed"].max() - merged["median_elapsed"].min())

            corr, _ = pearsonr(
                merged["norm_overlap"].to_numpy(), merged["norm_elapsed"].to_numpy()
            )

            axs[0].text(
                1.2 + i * 0.25, 0.98, f"{corr:.2f}",
                transform=axs[0].transAxes,
                ha="left", va="bottom",
                color=line_color_dict[config],
                fontsize=8, fontweight="bold",
            )

            marker = "." if (len(overlap_ratio) == 1 and len(elapsed) == 1) else None

            axs[0].plot(
                overlap_ratio["op_idx"], overlap_ratio["median"],
                linestyle="-", marker=marker, label=config,
                color=line_color_dict[config], linewidth=1.0,
            )
            axs[0].fill_between(
                overlap_ratio["op_idx"], overlap_ratio["min"], overlap_ratio["max"],
                alpha=0.2, color=color_dict[config], edgecolor=None,
            )
            axs[0].fill_between(
                overlap_ratio["op_idx"], overlap_ratio["q1"], overlap_ratio["q3"],
                alpha=0.5, color=color_dict[config], edgecolor=None,
            )

            axs[1].plot(
                elapsed["op_idx"], elapsed["median"],
                linestyle="-", marker=marker, label=config,
                color=line_color_dict[config], linewidth=1.0,
            )
            axs[1].fill_between(
                elapsed["op_idx"], elapsed["min"], elapsed["max"],
                alpha=0.3, color=color_dict[config], edgecolor=None,
            )
            axs[1].fill_between(
                elapsed["op_idx"], elapsed["q1"], elapsed["q3"],
                alpha=0.5, color=color_dict[config], edgecolor=None,
            )

        axs[0].yaxis.set_major_formatter(FuncFormatter(_no_digits_formatter))
        axs[0].grid(axis="y", linestyle="--", alpha=0.5)
        axs[0].grid(axis="x", linestyle="-", alpha=0.5)
        axs[1].grid(axis="y", linestyle="--", alpha=0.5)
        axs[1].grid(axis="x", linestyle="-", alpha=0.5)

        axs[0].tick_params(axis="y", pad=1)
        axs[1].tick_params(
            axis="y", which="both", left=False, right=True,
            labelright=True, labelleft=False, pad=1,
        )

        if j == len(overlaps) // 2:
            axs[0].set_ylabel("overlap ratio (%)", labelpad=1)
            axs[0].tick_params(axis="x", pad=1)
            axs[1].tick_params(axis="x", pad=1)
            axs[1].yaxis.set_label_position("right")
            axs[1].set_ylabel("norm duration", labelpad=6)
        if j == len(overlaps) - 1:
            axs[0].set_xlabel("kernel sample")
            axs[0].xaxis.set_label_coords(1.00, -0.525)
            axs[0].tick_params(axis="x", pad=1)
            axs[1].tick_params(axis="x", pad=1)
        else:
            axs[0].tick_params(axis="x", length=0)
            axs[1].tick_params(axis="x", length=0)
            axs[0].set_xticklabels([])
            axs[1].set_xticklabels([])

        axs[0].set_ylim(-5, 105)
        axs[0].text(
            0.25, 1.035, ov,
            transform=axs[0].transAxes,
            ha="center", va="bottom",
            fontsize=8, fontweight="bold",
        )

    legend_handles = [mpatches.Patch(color=color_dict[config], label=config) for config in params]
    legend_kwargs = dict(
        handles=legend_handles,
        loc="upper center",
        ncol=len(params),
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
    configs: list[str] = ["default"],
    operators: list[str] = [
        "b_attn_n", "b_mlp_n", "b_mlp_up", "b_mlp_gp", "b_mlp_dp",
    ],
    iter_start: int = -5,
    iter_stop: int = -2,
    iter_step: int = 1,
    paper_mode: PaperMode = PaperMode(),
    figsize: tuple[float, float] = (7.16, 4.0),
    filename: str = "overlap_correlation.pdf",
):
    fig = Figure(figsize=figsize)
    iter_idxs = range(iter_start, iter_stop, iter_step)
    input_data = get_data(ts_files, configs, operators, iter_idxs)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
