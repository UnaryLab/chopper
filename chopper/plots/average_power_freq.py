#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from chopper.common.colors import rgb
from chopper.common.cache import load_pickle
from chopper.common.printing import info
from chopper.common.annotations import PaperMode
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure


def get_data(
    gpu_files: list[str] = ["./gpu.pkl"],
    variants: list[str] = ["default"],
):
    """Load GPU telemetry data for power and frequency analysis.

    Loads raw GPU telemetry pickle files containing power and frequency
    metrics over time.

    Args:
        gpu_files: List of paths to GPU telemetry pickle files
        variants: List of variant names corresponding to each GPU file

    Returns:
        Dict mapping variant names to raw GPU telemetry DataFrames
    """
    return {
        variant: load_pickle(gpu_file) for gpu_file, variant in zip(gpu_files, variants)
    }


def draw(
    fig: Figure,
    input_data,
    starts: list[float] = [0.0],
    stops: list[float] = [1.0],
    ymaxs: list[float] = [float('inf')],
    ymins: list[float] = [float('-inf')],
    per_variant_norm: bool = False,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw rolling average power and frequency metrics.

    Creates a multi-panel line plot showing rolling 95th percentile power
    consumption and 5th percentile frequency over time, normalized to
    their baseline values.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data() containing GPU telemetry
        starts: Start fraction of time window (0-1) for each variant
        stops: Stop fraction of time window (0-1) for each variant
        per_variant_norm: If True, normalize each variant independently
        paper_mode: PaperMode settings for publication-quality figures
    """
    data = input_data
    variants = list(data.keys())
    metrics = (
        'current_gfxclk',
        'current_socket_power',
    )
    freq_metrics = {'current_gfxclk'}
    legend_names = {
        'current_gfxclk': 'Frequency',
        'current_socket_power': 'Power',
    }
    color_dict = {
        'current_gfxclk': rgb(0x1A, 0x85, 0xFF),  # Frequency - blue
        'current_socket_power': rgb(0xD4, 0x11, 0x59),  # Power - red
    }

    n_rows = len(metrics)
    n_cols = len(variants)

    fig.clear()
    fig.patches.clear()

    # Apply layout adjustments only in paper mode
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace
        )

    axs = tuple(tuple(fig.add_subplot(n_rows, n_cols, i*n_cols+j+1)
                for j in range(n_cols)) for i in range(n_rows))

    gymin: dict[str, float | None] = {metric: None for metric in metrics}
    gymax: dict[str, float | None] = {metric: None for metric in metrics}
    tmp_m: dict[str, dict[str, pd.DataFrame]] = {}

    gpus: list[int] = []

    # First pass: compute rolling values for all variants
    for vi, variant in enumerate(variants):
        start = starts[vi] if vi < len(starts) else starts[0]
        stop = stops[vi] if vi < len(stops) else stops[0]

        metric_trace = data[variant]
        metric_df = metric_trace.copy()
        metric_df['gpu'] -= 2

        if not gpus:
            gpus = sorted(metric_df['gpu'].unique())

        n_gpus = metric_df['gpu'].nunique()
        group_size = n_gpus

        metric_df['index'] = metric_df.index // group_size
        i_start = metric_df['index'].max() * start
        i_end = metric_df['index'].max() * stop
        metric_df = metric_df[
            (metric_df['index'] > i_start) &
            (metric_df['index'] < i_end)
        ]

        for metric in metrics:
            metric_df[metric] = metric_df[metric].astype(np.float64)

        for ci, metric in enumerate(metrics):
            metric_slot = tmp_m.setdefault(metric, {})
            tmp_m_ = metric_df.groupby(['index'])[metric].sum().reset_index()

            tmp_m_[f'{metric}_rolling'] = tmp_m_[metric].rolling(
                window=2000,
            ).quantile(.95 if metric == 'current_socket_power' else .05)
            metric_slot[variant] = tmp_m_

    # Second pass: compute normalization based on rolling values
    if per_variant_norm:
        variant_norm: dict[str, dict[str, float]] = {}
        for variant in variants:
            variant_norm[variant] = {}
            for metric in metrics:
                rolling_col = f'{metric}_rolling'
                rolling_vals = tmp_m[metric][variant][rolling_col].dropna()
                if metric in freq_metrics:
                    variant_norm[variant][metric] = rolling_vals.min()
                else:
                    variant_norm[variant][metric] = rolling_vals.max()
    else:
        global_norm: dict[str, float | None] = {metric: None for metric in metrics}
        for variant in variants:
            for metric in metrics:
                rolling_col = f'{metric}_rolling'
                rolling_vals = tmp_m[metric][variant][rolling_col].dropna()
                if metric in freq_metrics:
                    norm_val = rolling_vals.min()
                else:
                    norm_val = rolling_vals.max()

                cur = global_norm[metric]
                if cur is None:
                    global_norm[metric] = norm_val
                elif metric in freq_metrics:
                    global_norm[metric] = min(cur, norm_val)
                else:
                    global_norm[metric] = max(cur, norm_val)

    # Draw plots
    for vi, variant in enumerate(variants):
        info(f"Drawing: {variant}")
        for ci, metric in enumerate(metrics):
            ax = axs[metrics.index(metric)][vi]

            # Get norm value
            if per_variant_norm:
                norm_val = variant_norm[variant][metric]
            else:
                nv = global_norm[metric]
                assert nv is not None
                norm_val = nv

            ax.plot(
                tmp_m[metric][variant]['index'],
                tmp_m[metric][variant][f'{metric}_rolling'] / norm_val,
                color=color_dict[metric],
                linewidth=1.0,
                linestyle='-',
            )
            ax.grid(axis="y", linestyle='--', alpha=.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=1))

            ymin, ymax = ax.get_ylim()
            cur_min = gymin[metric]
            cur_max = gymax[metric]
            if cur_min is None:
                gymin[metric] = ymin
            else:
                gymin[metric] = min(cur_min, ymin)
            if cur_max is None:
                gymax[metric] = ymax
            else:
                gymax[metric] = max(cur_max, ymax)

    # Apply y-limits
    for vi, variant in enumerate(variants):
        for mi, metric in enumerate(metrics):
            gmin = gymin[metric]
            gmax = gymax[metric]
            assert gmin is not None and gmax is not None
            if len(ymaxs) == len(metrics) and ymaxs[mi] != float('inf'):
                gmax_ = ymaxs[mi]
            else:
                gmax_ = gmax
            if len(ymins) == len(metrics) and ymins[mi] != float('inf'):
                gmin_ = ymins[mi]
            else:
                gmin_ = gmin
            axs[metrics.index(metric)][vi].set_ylim((gmin_, gmax_))
            axs[metrics.index(metric)][vi].tick_params(axis='x', pad=1)

    # Set variant titles on second row (between 1st and 2nd row)
    for vi, variant in enumerate(variants):
        axs[1][vi].set_title(variant, pad=2, fontsize=8)

    for metric in metrics:
        axs[metrics.index(metric)][0].tick_params(axis='y', pad=1)
    for col in range(1, n_cols):
        for row in range(n_rows):
            axs[row][col].tick_params(axis='y', length=0)
            axs[row][col].set_yticklabels([])

    for row in range(n_rows):
        for col in range(n_cols):
            axs[row][col].tick_params(axis='x', length=0)
            axs[row][col].set_xticklabels([])

    # Get subplot bounding box to center labels relative to subplots
    fig.canvas.draw()
    bbox0 = axs[0][0].get_position()
    bbox1 = axs[1][0].get_position()
    bbox_right = axs[0][n_cols - 1].get_position()

    # Center y-label between the two rows
    y_center = (bbox0.y0 + bbox1.y1) / 2
    fig.text(0.01, y_center, "norm", ha="left", va="center", rotation=90)

    # Center x-label relative to subplot area
    x_center = (bbox0.x0 + bbox_right.x1) / 2
    fig.text(x_center, 0.01, "sample", ha="center", va="bottom")

    legend_handles = [
        mpatches.Patch(
            color=color_dict[metric], label=legend_names[metric])
        for metric in metrics
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(metrics),
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=0.5,
        frameon=False,
    )

    for ri in range(n_rows-1):
        for ci in range(n_cols):
            axs[ri][ci].tick_params(axis='x', length=0)
            axs[ri][ci].set_xticklabels([])

    for ci in range(n_cols):
        axs[n_rows - 1][ci].tick_params(axis="x", pad=1)

    # Add border around figure in paper mode (removed when saving)
    if paper_mode.enabled:
        fig.patches.append(mpatches.Rectangle(
            (0, 0), 1, 1,
            transform=fig.transFigure,
            fill=False,
            edgecolor="black",
            linewidth=1,
            zorder=1000,
        ))

def main(
    gpu_files: list[str] = ["./gpu.pkl"],
    variants: list[str] = ["default"],
    starts: list[float] = [0.0],
    stops: list[float] = [1.0],
    ymaxs: list[float] = [float('inf')],
    ymins: list[float] = [float('-inf')],
    per_variant_norm: bool = False,
    filename: str = "average_power_frequency.png"
):
    fig = Figure()
    input_data = get_data(gpu_files, variants)
    draw(fig, input_data, starts, stops, ymaxs, ymins, per_variant_norm, PaperMode())
    fig.savefig(filename, dpi=300)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
