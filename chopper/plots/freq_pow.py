import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from chopper.common.colors import rgb
from chopper.common.cache import load_pickle
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure


def get_data(gpu_files: list[str] = ("./gpu.pkl",), variants: list[str] = ("FSDPv2",)):
    metrics = (
        "current_gfxclk",
        "current_uclk",
        "current_socket_power",
    )
    norm_metric = {}
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
        for ci, metric in enumerate(metrics):
            tmp_m = metric_df_.groupby(["index"])[metric].sum().reset_index()
            norm_metric_ = tmp_m[metric].max()
            if metric not in norm_metric:
                norm_metric[metric] = norm_metric_
            else:
                norm_metric[metric] = max(norm_metric_, norm_metric[metric])

    return norm_metric, metric_df, variants


def draw(
    fig: Figure,
    input_data,
    show_gpus: bool = False,  # hardcode for now
    alpha: float = 1.0,
    s: float = 0.5,
    start: float = 0.0,
    stop: float = 1.0,
    metrics: list[str] = (
        "current_gfxclk",
        "current_uclk",
        "current_socket_power",
    ),
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
):
    norm_metric, metric_df, variants = input_data
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
    rgb_colors = (
        rgb(0x66, 0xC2, 0xA5),
        rgb(0x8D, 0xA0, 0xCB),
        rgb(0xFC, 0x8D, 0x62),
    )
    color_dict = {metric: rgb_colors[i] for i, metric in enumerate(metrics)}

    # TODO do not hardcode
    gpus = 8
    if show_gpus:
        n_cols = gpus
    else:
        n_cols = 1
    n_rows = len(variants) * len(metrics)

    fig.clear()
    axs = tuple(
        tuple(
            fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1) for j in range(n_cols)
        )
        for i in range(n_rows)
    )

    gymin = {metric: None for metric in metrics}
    gymax = gymin.copy()
    for variant in variants:
        for ci, metric in enumerate(metrics):
            for gpu in range(gpus):
                nidx = metric_df[variant]["index"].max()
                time_mask = (metric_df[variant]["index"] >= start * nidx) & (
                    metric_df[variant]["index"] <= stop * nidx
                )
                if show_gpus:
                    time_mask &= metric_df[variant]["gpu"] == gpu

                tmp_m = (
                    metric_df[variant][time_mask]
                    .groupby(["index"])[metric]
                    .sum()
                    .reset_index()
                )
                ax = axs[
                    metrics.index(metric) * len(variants) + variants.index(variant)
                ][gpu]
                if gpus > 1:
                    ax.scatter(
                        tmp_m["index"],
                        tmp_m[metric]
                        / norm_metric[metric]
                        * (gpus if show_gpus else 1),
                        color=color_dict[metric],
                        alpha=alpha,
                        s=s,
                    )
                ax.grid(axis="y", linestyle="--", alpha=0.5)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

                ymin, ymax = ax.get_ylim()
                if gymin[metric] is None:
                    gymin[metric] = ymin
                else:
                    gymin[metric] = min(gymin[metric], ymin)
                if gymax[metric] is None:
                    gymax[metric] = ymax
                else:
                    gymax[metric] = max(gymax[metric], ymax)
                if not show_gpus:
                    break

    for variant in variants:
        for y_min, y_max, metric in zip(metric_y_min, metric_y_max, metrics):
            for gpu in range(gpus):
                ax = axs[
                    metrics.index(metric) * len(variants) + variants.index(variant)
                ][gpu]

                if y_min != float("-inf") and y_max != float("inf"):
                    ax.set_ylim((y_min, y_max))
                else:
                    ax.set_ylim((gymin[metric], gymax[metric]))
                ax.tick_params(axis="x", pad=1)
                if not show_gpus:
                    break
            axs[metrics.index(metric) * len(variants) + variants.index(variant)][
                0
            ].set_title(variant, pad=1.5, fontsize=8)

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

    legend_handles = [
        mpatches.Patch(color=color_dict[metric], label=legend_names[metric])
        for metric in metrics
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=gpus,
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
