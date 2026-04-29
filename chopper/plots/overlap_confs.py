"""Per-config overlap-ratio vs. duration correlation.

For a single operator (default b_attn_fa), one row per config:
median overlap ratio (left) and normalized duration (right) over kernel
samples, with Pearson correlation reported per config.
(Paper Figure: overlap_confs.pdf)
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
    operator: str = "b_attn_fa",
    iter_idxs: range = range(-5, -2, 1),
):
    """Load overlap data for the chosen operator across configurations.

    Normalizes 'elapsed' per config by the minimum elapsed.

    Args:
        ts_files: List of ts.pkl trace files
        configs: Config labels (e.g. "b1s4", "b2s8")
        operator: Operator to extract
        iter_idxs: Iteration indices to select

    Returns:
        Tuple (overlap_data, configs, operator) where overlap_data maps
        config -> DataFrame of operator entries with normalized 'elapsed'
        and 'op_idx'
    """
    data = {}
    for ts_file, config in zip(ts_files, configs):
        data[config] = get_overlap_df(
            ts_file, iter_idxs=list(iter_idxs)
        )

    overlap_data = {}
    config_min_elapsed = {}

    for config in configs:
        operators = set(data[config]["operator-name"])
        assert operator in operators, f"operator {operator} not found in {config}"

        op_mask = data[config]["operator-name"] == operator
        sub = data[config][op_mask].copy()
        sub["op_idx"] = sub.groupby("gpu").cumcount()
        overlap_data[config] = sub

        min_elapsed = np.min(sub["elapsed"])
        config_min_elapsed[config] = min(
            config_min_elapsed.get(config, float("inf")), min_elapsed
        )

    for config in configs:
        overlap_data[config]["elapsed"] /= config_min_elapsed[config]

    return overlap_data, list(configs), operator


def draw(
    fig: Figure,
    input_data,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw per-config overlap-ratio and duration plots with Pearson correlation.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data()
        paper_mode: PaperMode settings for publication-quality figures
    """
    data, configs, op = input_data

    fig.clear()
    fig.patches.clear()
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    n_rows = len(configs)
    n_cols = 2
    axses = tuple(
        tuple(fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1) for c in range(n_cols))
        for r in range(n_rows)
    )

    plot_color = rgb(0xD4, 0x11, 0x59)
    line_color = (
        max(0, plot_color[0] - 0.3),
        max(0, plot_color[1] - 0.3),
        max(0, plot_color[2] - 0.3),
    )

    for setup in data.keys():
        config = setup
        i = configs.index(config)
        axs = axses[i]

        overlap_ratio = (
            data[setup]
            .groupby("op_idx")["overlap_ratio"]
            .agg(["min", lambda x: x.quantile(0.25), "median",
                  lambda x: x.quantile(0.75), "max"])
            .reset_index()
        )
        elapsed = (
            data[setup]
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
            1.2, 0.98, f"{corr:.2f}",
            transform=axs[0].transAxes,
            ha="left", va="bottom",
            color=line_color,
            fontsize=8, fontweight="bold",
        )

        marker = "." if (len(overlap_ratio) == 1 and len(elapsed) == 1) else None

        axs[0].plot(
            overlap_ratio["op_idx"], overlap_ratio["median"],
            linestyle="-", marker=marker,
            color=line_color, linewidth=1.0,
        )
        axs[0].fill_between(
            overlap_ratio["op_idx"], overlap_ratio["min"], overlap_ratio["max"],
            alpha=0.2, color=plot_color, edgecolor=None,
        )
        axs[0].fill_between(
            overlap_ratio["op_idx"], overlap_ratio["q1"], overlap_ratio["q3"],
            alpha=0.5, color=plot_color, edgecolor=None,
        )

        axs[1].plot(
            elapsed["op_idx"], elapsed["median"],
            linestyle="-", marker=marker,
            color=line_color, linewidth=1.0,
        )
        axs[1].fill_between(
            elapsed["op_idx"], elapsed["min"], elapsed["max"],
            alpha=0.3, color=plot_color, edgecolor=None,
        )
        axs[1].fill_between(
            elapsed["op_idx"], elapsed["q1"], elapsed["q3"],
            alpha=0.5, color=plot_color, edgecolor=None,
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

        if i == n_rows // 2:
            axs[0].set_ylabel("overlap ratio (%)", labelpad=1)
            axs[0].tick_params(axis="x", pad=1)
            axs[1].tick_params(axis="x", pad=1)
            axs[1].yaxis.set_label_position("right")
            axs[1].set_ylabel("norm duration", labelpad=4)
        if i == n_rows - 1:
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
            0.25, 1.035, f"{op}  {config}",
            transform=axs[0].transAxes,
            ha="center", va="bottom",
            fontsize=8, fontweight="bold",
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
    operator: str = "b_attn_fa",
    iter_start: int = -5,
    iter_stop: int = -2,
    iter_step: int = 1,
    paper_mode: PaperMode = PaperMode(),
    figsize: tuple[float, float] = (7.16, 2.5),
    filename: str = "overlap_confs.pdf",
):
    fig = Figure(figsize=figsize)
    iter_idxs = range(iter_start, iter_stop, iter_step)
    input_data = get_data(ts_files, configs, operator, iter_idxs)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
