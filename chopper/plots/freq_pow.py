import numpy as np
import matplotlib.patches as mpatches
from loguru import logger
from chopper.common.colors import rgb
from chopper.common.cache import load_pickle
from chopper.common.annotations import PaperMode
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure


def get_data(gpu_files: list[str] = ["./gpu.pkl"], variants: list[str] = ["FSDPv2"]):
    """Load and process GPU frequency and power metrics data.

    Extracts GPU clock frequency, memory clock frequency, and socket power
    metrics from GPU telemetry files. Filters to focus on steady-state
    execution periods.

    Args:
        gpu_files: List of paths to GPU telemetry pickle files
        variants: List of variant names corresponding to each GPU file

    Returns:
        Tuple containing:
            - metric_df: Dict mapping variant names to processed DataFrames
            - variants: List of variant names
    """
    metrics = (
        "current_gfxclk",
        "current_uclk",
        "current_socket_power",
    )
    metric_df = {}
    for gpu_file, variant in zip(gpu_files, variants):
        metric_trace = load_pickle(gpu_file)
        metric_df_ = metric_trace.copy()
        metric_df_["gpu"] -= 2

        n_gpus = metric_df_["gpu"].nunique()
        group_size = n_gpus

        metric_df_["index"] = metric_df_.index // group_size
        start = metric_df_["index"].max() * (0.52 if variant == "FSDPv1" else 0.31)
        end = metric_df_["index"].max() * (0.97 if variant == "FSDPv1" else 0.95)
        metric_df_ = metric_df_[
            (metric_df_["index"] > start) & (metric_df_["index"] < end)
        ]

        for metric in metrics:
            metric_df_[metric] = metric_df_[metric].astype(np.float64)
        metric_df[variant] = metric_df_

    return metric_df, variants


def draw(
    fig: Figure,
    input_data,
    show_gpus: bool = False,  # hardcode for now
    alpha: float = 1.0,
    s: float = 0.5,
    starts: list[float] = [0.0],
    stops: list[float] = [1.0],
    metrics: list[str] = [
        "current_gfxclk",
        "current_uclk",
        "current_socket_power",
    ],
    metric_y_max: list[float] = [
        float("inf"),
        float("inf"),
        float("inf"),
    ],
    metric_y_min: list[float] = [
        float("-inf"),
        float("-inf"),
        float("-inf"),
    ],
    per_variant_norm: bool = False,
    paper_mode: PaperMode = PaperMode(),
):
    logger.debug("got starts: {}", starts)
    logger.debug("got stops: {}", stops)
    """Draw normalized frequency and power metrics over time.

    Creates a multi-panel scatter plot showing GPU frequency, memory frequency,
    and socket power metrics normalized to their maximum values. Supports
    comparing multiple variants and optionally showing per-GPU breakdown.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() containing metrics and dataframes
        show_gpus: If True, show separate plots for each GPU
        alpha: Transparency of scatter points (0-1)
        s: Size of scatter points
        starts: Start fraction of time window (0-1) for each variant
        stops: Stop fraction of time window (0-1) for each variant
        metrics: List of metric names to plot
        metric_y_max: Maximum y-axis limits for each metric
        metric_y_min: Minimum y-axis limits for each metric
        per_variant_norm: If True, normalize each variant independently
        paper_mode: PaperMode settings for publication-quality figures
    """
    metric_df, variants = input_data
    # Frequency metrics normalize to min, power normalizes to max
    freq_metrics = {"current_gfxclk", "current_uclk"}
    ylabel_names = {
        "current_gfxclk": "norm",
        "current_uclk": "norm",
        "current_socket_power": "norm",
    }
    legend_names = {
        "current_gfxclk": "GPU Frequency",
        "current_uclk": "Memory Frequency",
        "current_socket_power": "Power",
    }
    color_dict = {
        "current_gfxclk": rgb(0x8D, 0xA0, 0xCB),  # GPU Frequency - blue
        "current_uclk": rgb(0x66, 0xC2, 0xA5),  # Memory Frequency - green
        "current_socket_power": rgb(0xFC, 0x8D, 0x62),  # Power - red/orange
    }

    # Layout: rows = metrics, cols = variants
    n_rows = len(metrics)
    n_cols = len(variants)

    fig.clear()
    fig.patches.clear()  # Ensure figure-level patches are also cleared

    # Apply layout adjustments only in paper mode
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace
        )

    # axs[row][col] where row=metric index, col=variant index
    axs = tuple(
        tuple(
            fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            for col in range(n_cols)
        )
        for row in range(n_rows)
    )

    gymin: dict[str, float | None] = {metric: None for metric in metrics}
    gymax: dict[str, float | None] = {metric: None for metric in metrics}

    # First pass: compute norm values from filtered regions
    # For per_variant_norm, store per-variant; otherwise store global
    if per_variant_norm:
        # Per-variant normalization: dict[variant][metric] -> value
        variant_norm_min: dict[str, dict[str, float]] = {}
        variant_norm_max: dict[str, dict[str, float]] = {}
        for vi, variant in enumerate(variants):
            start = starts[vi] if vi < len(starts) else starts[0]
            stop = stops[vi] if vi < len(stops) else stops[0]
            variant_norm_min[variant] = {}
            variant_norm_max[variant] = {}
            for mi, metric in enumerate(metrics):
                nidx = metric_df[variant]["index"].max()
                time_mask = (metric_df[variant]["index"] >= start * nidx) & (
                    metric_df[variant]["index"] <= stop * nidx
                )
                tmp_m = (
                    metric_df[variant][time_mask]
                    .groupby(["index"])[metric]
                    .sum()
                    .reset_index()
                )
                variant_norm_min[variant][metric] = tmp_m[metric].min()
                variant_norm_max[variant][metric] = tmp_m[metric].max()
    else:
        # Global normalization across all variants
        filtered_norm_min: dict[str, float | None] = {metric: None for metric in metrics}
        filtered_norm_max: dict[str, float | None] = {metric: None for metric in metrics}
        for vi, variant in enumerate(variants):
            start = starts[vi] if vi < len(starts) else starts[0]
            stop = stops[vi] if vi < len(stops) else stops[0]
            for mi, metric in enumerate(metrics):
                nidx = metric_df[variant]["index"].max()
                time_mask = (metric_df[variant]["index"] >= start * nidx) & (
                    metric_df[variant]["index"] <= stop * nidx
                )
                tmp_m = (
                    metric_df[variant][time_mask]
                    .groupby(["index"])[metric]
                    .sum()
                    .reset_index()
                )
                metric_max = tmp_m[metric].max()
                metric_min = tmp_m[metric].min()
                cur_max = filtered_norm_max[metric]
                cur_min = filtered_norm_min[metric]
                if cur_max is None:
                    filtered_norm_max[metric] = metric_max
                else:
                    filtered_norm_max[metric] = max(cur_max, metric_max)
                if cur_min is None:
                    filtered_norm_min[metric] = metric_min
                else:
                    filtered_norm_min[metric] = min(cur_min, metric_min)

    # Second pass: draw with computed norm values
    for vi, variant in enumerate(variants):
        start = starts[vi] if vi < len(starts) else starts[0]
        stop = stops[vi] if vi < len(stops) else stops[0]
        for mi, metric in enumerate(metrics):
            nidx = metric_df[variant]["index"].max()
            time_mask = (metric_df[variant]["index"] >= start * nidx) & (
                metric_df[variant]["index"] <= stop * nidx
            )

            tmp_m = (
                metric_df[variant][time_mask]
                .groupby(["index"])[metric]
                .sum()
                .reset_index()
            )
            ax = axs[mi][vi]
            # Normalize frequency metrics to min, power to max
            norm_val: float
            if per_variant_norm:
                if metric in freq_metrics:
                    norm_val = variant_norm_min[variant][metric]
                else:
                    norm_val = variant_norm_max[variant][metric]
            else:
                if metric in freq_metrics:
                    nv = filtered_norm_min[metric]
                else:
                    nv = filtered_norm_max[metric]
                assert nv is not None
                norm_val = nv
            ax.scatter(
                tmp_m["index"],
                tmp_m[metric] / norm_val,
                color=color_dict[metric],
                alpha=alpha,
                s=s,
            )
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

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

    # Apply y-limits and formatting
    for mi, metric in enumerate(metrics):
        y_min_val = metric_y_min[mi] if mi < len(metric_y_min) else float("-inf")
        y_max_val = metric_y_max[mi] if mi < len(metric_y_max) else float("inf")
        g_min = gymin[metric]
        g_max = gymax[metric]
        for vi, variant in enumerate(variants):
            ax = axs[mi][vi]
            if y_min_val != float("-inf") and y_max_val != float("inf"):
                ax.set_ylim((y_min_val, y_max_val))
            elif g_min is not None and g_max is not None:
                ax.set_ylim((g_min, g_max))
            ax.tick_params(axis="x", pad=1)

    # Set variant titles on second row (between 1st and 2nd row)
    for vi, variant in enumerate(variants):
        axs[1][vi].set_title(variant, pad=2, fontsize=8)

    # Set y-axis labels on left column only
    for mi, metric in enumerate(metrics):
        axs[mi][0].set_ylabel(ylabel_names[metric], labelpad=1)
        axs[mi][0].tick_params(axis="y", pad=1)

    # Hide y-tick labels for non-leftmost columns
    for col in range(1, n_cols):
        for row in range(n_rows):
            axs[row][col].tick_params(axis="y", length=0)
            axs[row][col].set_yticklabels([])

    # Hide x-tick labels for all
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row][col].tick_params(axis="x", length=0)
            axs[row][col].set_xticklabels([])

    # Center xlabel across all columns
    fig.text(0.5, 0.01, "sample", ha="center", va="bottom")

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

    legend_handles = [
        mpatches.Patch(color=color_dict[metric], label=legend_names[metric])
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

    for ri in range(n_rows - 1):
        for ci in range(n_cols):
            axs[ri][ci].tick_params(axis="x", length=0)
            axs[ri][ci].set_xticklabels([])

    for ci in range(n_cols):
        axs[n_rows - 1][ci].tick_params(axis="x", pad=1)

def main(
    gpu_files: list[str] = ["./gpu.pkl"],
    variants: list[str] = ["FSDPv2"],
    show_gpus: bool = False,
    alpha: float = 1.0,
    s: float = 0.5,
    starts: list[float] = [0.0],
    stops: list[float] = [1.0],
    metrics: list[str] = [
        "current_gfxclk",
        "current_uclk",
        "current_socket_power",
    ],
    metric_y_max: list[float] = [
        float("inf"),
        float("inf"),
        float("inf"),
    ],
    metric_y_min: list[float] = [
        float("-inf"),
        float("-inf"),
        float("-inf"),
    ],
    per_variant_norm: bool = False,
    filename: str = "freq_pow.png",
):
    fig = Figure()
    input_data = get_data(gpu_files, variants)
    draw(fig, input_data, show_gpus, alpha, s, starts, stops, metrics, metric_y_max, metric_y_min, per_variant_norm, PaperMode())
    fig.savefig(filename, dpi=300)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
