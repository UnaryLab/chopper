"""GPU frequency and temperature correlation over training iterations.

Visualizes how GPU clock frequency and temperature vary over time during
training, helping identify thermal throttling and its correlation with
straggling behavior. (Paper Figure 6)
"""

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from chopper.common.colors import okabe_ito
from chopper.common.cache import load_pickle
from chopper.common.printing import info
from chopper.common.annotations import PaperMode


def get_data(
    gpu_files: list[str] = ["./metric_samples.pkl"],
    variants: list[str] = ["default"],
):
    """Load and process GPU frequency and temperature metrics data.

    Extracts GPU clock frequency and temperature metrics from GPU telemetry
    files for correlation analysis.

    Args:
        gpu_files: List of paths to metric_samples.pkl files
        variants: List of variant names corresponding to each GPU file

    Returns:
        Tuple containing:
            - metric_df: Dict mapping variant names to processed DataFrames
            - variants: List of variant names
    """
    metric_df = {}
    for gpu_file, variant in zip(gpu_files, variants):
        metric_trace = load_pickle(gpu_file)
        metric_df_ = metric_trace.copy()
        # Adjust GPU indices (assuming offset of 2 for ROCm)
        if "gpu" in metric_df_.columns:
            metric_df_["gpu"] -= 2

        n_gpus = metric_df_["gpu"].nunique()
        group_size = n_gpus

        metric_df_["index"] = metric_df_.index // group_size
        metric_df[variant] = metric_df_

    return metric_df, variants


def draw(
    fig: Figure,
    input_data,
    metrics: list[str] = ["current_gfxclk", "temperature_hotspot"],
    starts: list[float] = [0.0],
    stops: list[float] = [1.0],
    global_norm: bool = True,
    target_gpus: list[int] = [0, 1, 2, 3, 4, 5, 6, 7],
    paper_mode: PaperMode = PaperMode(),
):
    """Draw GPU frequency and temperature profiles over time.

    Creates a multi-panel line plot showing GPU frequency and temperature
    metrics normalized to show correlation with straggling behavior.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() containing metrics and dataframes
        metrics: List of metric names to plot (e.g., current_gfxclk, temperature_hotspot)
        starts: Start fraction of time window (0-1) for each variant
        stops: Stop fraction of time window (0-1) for each variant
        global_norm: If True, normalize globally; otherwise normalize per-sample
        target_gpus: List of GPU indices to include in the plot
        paper_mode: PaperMode settings for publication-quality figures
    """
    metric_df, variants = input_data

    metric_names = {
        "current_gfxclk": "norm freq" if not global_norm else "Freq (MHz)",
        "current_uclk": "norm mem freq" if not global_norm else "Mem Freq (MHz)",
        "temperature_hotspot": "norm temp" if not global_norm else "Temp (C)",
        "temperature_mem": "norm mem temp" if not global_norm else "Mem Temp (C)",
        "current_socket_power": "norm power" if not global_norm else "Power (W)",
    }

    n_cols = 1
    n_rows = len(metrics)

    fig.clear()
    fig.patches.clear()

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left,
            right=paper_mode.right,
            bottom=paper_mode.bottom,
            top=paper_mode.top,
            wspace=paper_mode.wspace,
            hspace=paper_mode.hspace,
        )

    axs = tuple(
        fig.add_subplot(n_rows, n_cols, row + 1) for row in range(n_rows)
    )

    for vi, variant in enumerate(variants):
        start = starts[vi] if vi < len(starts) else starts[0]
        stop = stops[vi] if vi < len(stops) else stops[0]

        tmp_metric_df = metric_df[variant].copy()

        # Filter to target GPUs
        tmp_metric_df = tmp_metric_df[
            tmp_metric_df["gpu"].isin(target_gpus)
        ].reset_index(drop=True)

        gpus = sorted(tmp_metric_df["gpu"].unique())

        # Compute sampling statistics
        n_gpus = tmp_metric_df["gpu"].nunique()
        group_size = n_gpus
        tmp_metric_df["index"] = tmp_metric_df.index // group_size

        sampling_stats_df = tmp_metric_df.groupby(["index"]).agg(
            ts_first=("ts", "first"),
            ts_last=("ts", "last"),
        )
        sampling_stats_df["sampling_dur"] = (
            sampling_stats_df["ts_last"] - sampling_stats_df["ts_first"]
        )
        sampling_stats_df["sampling_period"] = (
            sampling_stats_df["ts_first"] - sampling_stats_df["ts_first"].shift(1)
        )

        info(
            f"Frequency sampling duration: {sampling_stats_df['sampling_dur'].median() * 1e-6:.2f}ms"
        )
        info(
            f"Frequency sampling period: {sampling_stats_df['sampling_period'].median() * 1e-6:.2f}ms"
        )

        # Apply time window filter
        max_index = tmp_metric_df["index"].max()
        start_idx = int(start * max_index)
        stop_idx = int(stop * max_index)
        mask = (tmp_metric_df["index"] >= start_idx) & (
            tmp_metric_df["index"] <= stop_idx
        )
        tmp_metric_df = tmp_metric_df[mask]

        # Compute normalization values
        if global_norm:
            agg = {
                **{f"{metric}_min": tmp_metric_df[metric].min() for metric in metrics},
                **{f"{metric}_max": tmp_metric_df[metric].max() for metric in metrics},
            }
            agg_freq = pd.DataFrame([agg])
        else:
            agg_freq = (
                tmp_metric_df.groupby(["index"])
                .agg(
                    **{
                        **{f"{metric}_min": (metric, "min") for metric in metrics},
                        **{f"{metric}_max": (metric, "max") for metric in metrics},
                    }
                )
                .reset_index()
            )

        # Convert metrics to float for normalization
        for metric in metrics:
            tmp_metric_df[metric] = tmp_metric_df[metric].astype(np.float64)

        # Apply normalization if not global
        if not global_norm:
            for gpu in gpus:
                cur_mask = tmp_metric_df["gpu"] == gpu
                for metric in metrics:
                    tmp_metric_df.loc[cur_mask, metric] = (
                        tmp_metric_df.loc[cur_mask, metric].values
                        / agg_freq[f"{metric}_min"].values
                    )

        # Create color mapping
        rgb_colors = tuple(okabe_ito.values())
        color_dict = {gpu: rgb_colors[i % len(rgb_colors)] for i, gpu in enumerate(gpus)}

        # Draw plots
        for gi, gpu in enumerate(gpus):
            for ci, metric in enumerate(metrics):
                gpu_mask = tmp_metric_df["gpu"] == gpu
                ax = axs[ci]
                ax.plot(
                    tmp_metric_df.loc[gpu_mask, "index"],
                    tmp_metric_df.loc[gpu_mask, metric],
                    color=color_dict[gpu],
                    linewidth=1.0,
                    linestyle="-",
                )
                ax.set_ylabel(metric_names.get(metric, metric), labelpad=0)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
                ax.grid(axis="y", linestyle="--", alpha=0.5)

                if ci == len(metrics) - 1:
                    ax.set_xlabel("sample", labelpad=1)

        # Create legend
        legend_handles = [
            mpatches.Patch(color=color_dict[gpu], label=str(gpu)) for gpu in gpus
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.50, 1.010),
            ncol=len(gpus),
            borderpad=0.17,
            handletextpad=0.4,
            columnspacing=0.4,
            handlelength=0.5,
            frameon=False,
        )

        # Format axes
        for ri in range(n_rows):
            axs[ri].tick_params(axis="x", length=0)
            axs[ri].set_xticklabels([])

        axs[n_rows - 1].tick_params(axis="x", pad=1)

    # Add border around figure in paper mode
    if paper_mode.enabled:
        fig.patches.append(
            mpatches.Rectangle(
                (0, 0),
                1,
                1,
                transform=fig.transFigure,
                fill=False,
                edgecolor="black",
                linewidth=1,
                zorder=1000,
            )
        )


def main(
    gpu_files: list[str] = ["./metric_samples.pkl"],
    variants: list[str] = ["default"],
    metrics: list[str] = ["current_gfxclk", "temperature_hotspot"],
    starts: list[float] = [0.0],
    stops: list[float] = [1.0],
    global_norm: bool = True,
    target_gpus: list[int] = [0, 1, 2, 3, 4, 5, 6, 7],
    filename: str = "freq_temp_corr.png",
):
    fig = Figure()
    input_data = get_data(gpu_files, variants)
    draw(fig, input_data, metrics, starts, stops, global_norm, target_gpus, PaperMode())
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
