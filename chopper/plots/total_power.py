"""Per-variant socket power and graphics-clock samples (normalized).

Two-row scatter (frequency / power) by N variants, normalized per-metric to a
per-variant reference (max for power, min for frequency).
(Paper Figure: total_power_*.pdf)
"""

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from chopper.common.colors import rgb
from chopper.common.printing import info
from chopper.common.annotations import PaperMode


def get_data(
    metric_files: list[str] = ["./metric_samples.pkl"],
    variants: list[str] = ["default"],
):
    """Load per-variant metric_samples DataFrames.

    Args:
        metric_files: List of paths to metric_samples.pkl files
        variants: Variant labels (one per file)

    Returns:
        Dict mapping variant -> raw metric DataFrame
    """
    return {v: pd.read_pickle(fn) for v, fn in zip(variants, metric_files)}


def draw(
    fig: Figure,
    input_data,
    metrics: list[str] = ["current_gfxclk", "current_socket_power"],
    paper_mode: PaperMode = PaperMode(),
):
    """Draw per-variant scatter of frequency and socket power over samples.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data() of variant -> DataFrame
        metrics: Which columns to plot (rows)
        paper_mode: PaperMode settings for publication-quality figures
    """
    data = input_data
    variants = list(data.keys())

    ylabel_names = {
        "current_gfxclk": "min norm",
        "current_socket_power": "max norm",
    }
    legend_names = {
        "current_gfxclk": "Frequency",
        "current_socket_power": "Power",
    }
    rgb_colors = (rgb(0x1A, 0x85, 0xFF), rgb(0xD4, 0x11, 0x59))
    color_dict = {metric: rgb_colors[i] for i, metric in enumerate(metrics)}

    n_rows = len(metrics)
    n_cols = len(variants)

    fig.clear()
    fig.patches.clear()
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    axs = tuple(
        tuple(fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1) for c in range(n_cols))
        for r in range(n_rows)
    )

    val_range = 0.08
    ylims = {
        "current_socket_power": (1 - val_range, 1.010),
        "current_gfxclk": (0.995, 1 + val_range),
    }

    gymin = {metric: None for metric in metrics}
    gymax = {metric: None for metric in metrics}

    for variant in variants:
        info(f"Drawing: {variant}")
        metric_df = data[variant].copy()
        metric_df["gpu"] -= 2

        n_gpus = metric_df["gpu"].nunique()
        metric_df["index"] = metric_df.index // n_gpus
        start = metric_df["index"].max() * 0.05
        end = metric_df["index"].max() * 0.95
        metric_df = metric_df[
            (metric_df["index"] > start) & (metric_df["index"] < end)
        ]

        for metric in metrics:
            metric_df[metric] = metric_df[metric].astype(np.float64)

        for metric in metrics:
            tmp_m = metric_df.groupby(["index"])[metric].sum().reset_index()
            if metric == "current_socket_power":
                norm_tmp_m = tmp_m[metric].max()
            else:
                norm_tmp_m = tmp_m[metric].min()
            ax = axs[metrics.index(metric)][variants.index(variant)]
            ax.scatter(
                tmp_m["index"], tmp_m[metric] / norm_tmp_m,
                color=color_dict[metric],
                linewidth=0.5, linestyle="-", s=0.05, alpha=0.01,
            )
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            if metric in ylims:
                ax.set_ylim(ylims[metric])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=1))

            ymin, ymax = ax.get_ylim()
            gymin[metric] = ymin if gymin[metric] is None else min(gymin[metric], ymin)
            gymax[metric] = ymax if gymax[metric] is None else max(gymax[metric], ymax)

    for variant in variants:
        for metric in metrics:
            axs[metrics.index(metric)][variants.index(variant)].set_ylim(
                (gymin[metric], gymax[metric])
            )
            axs[metrics.index(metric)][variants.index(variant)].tick_params(
                axis="x", pad=1
            )
        axs[len(metrics) - 1][variants.index(variant)].set_title(
            variant, pad=1.5, fontsize=8
        )

    for metric in metrics:
        axs[metrics.index(metric)][0].set_ylabel(ylabel_names[metric], labelpad=1)
        axs[metrics.index(metric)][0].tick_params(axis="y", pad=1)
    for col in range(1, n_cols):
        for row in range(n_rows):
            axs[row][col].tick_params(axis="y", length=0)
            axs[row][col].set_yticklabels([])

    for row in range(n_rows):
        for col in range(n_cols):
            axs[row][col].tick_params(axis="x", length=0)
            axs[row][col].set_xticklabels([])

    axs[n_rows - 1][n_cols // 2].set_xlabel("sample")
    axs[n_rows - 1][n_cols // 2].xaxis.set_label_coords(0.50, -0.05)

    legend_handles = [
        mpatches.Patch(color=color_dict[metric], label=legend_names[metric])
        for metric in metrics
    ]
    legend_kwargs = dict(
        handles=legend_handles,
        loc="upper center",
        ncol=len(metrics),
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
    metric_files: list[str] = ["./metric_samples.pkl"],
    variants: list[str] = ["default"],
    metrics: list[str] = ["current_gfxclk", "current_socket_power"],
    paper_mode: PaperMode = PaperMode(),
    filename: str = "total_power.png",
):
    fig = Figure()
    input_data = get_data(metric_files, variants)
    draw(fig, input_data, metrics, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
