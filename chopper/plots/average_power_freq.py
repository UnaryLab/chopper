#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from chopper.common.colors import rgb
from chopper.common.cache import load_pickle
from chopper.common.printing import info
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure


def get_data(
    gpu_files: list[str] = ("./gpu.pkl",),
    variants: list[str] = ("default",),
):
    return {
        variant: load_pickle(gpu_file) for gpu_file, variant in zip(gpu_files, variants)
    }


def draw(
    fig: Figure,
    input_data,
    start: float = 0,
    end: float = 1.0,
):
    data = input_data
    metrics = (
        'current_gfxclk',
        'current_socket_power',
    )
    legend_names = {
        'current_gfxclk': 'Frequency',
        'current_socket_power': 'Power',
    }
    rgb_colors = (
        rgb(0x1A, 0x85, 0xFF),
        rgb(0xD4, 0x11, 0x59),
    )
    color_dict = {metric: rgb_colors[i]
                  for i, metric in enumerate(metrics)}

    n_rows = len(metrics)
    n_cols = len(data.keys())

    fig.clear()
    axs = tuple(tuple(fig.add_subplot(n_rows, n_cols, i*n_cols+j+1)
                for j in range(n_cols)) for i in range(n_rows))

    gymin = {metric: None for metric in metrics}
    gymax = gymin.copy()
    tmp_m = {}
    norm_tmp_m = {}
    for variant, metric_trace in data.items():
        metric_df = metric_trace.copy()
        metric_df['gpu'] -= 2

        gpus = sorted(metric_df['gpu'].unique())

        n_gpus = metric_df['gpu'].nunique()
        group_size = n_gpus

        metric_df['index'] = metric_df.index // group_size
        start = metric_df['index'].max() * start
        end = metric_df['index'].max() * end
        metric_df = metric_df[
            (metric_df['index'] > start) &
            (metric_df['index'] < end)
        ]

        for metric in metrics:
            metric_df[metric] = metric_df[metric].astype(np.float64)

        for ci, metric in enumerate(metrics):
            metric_slot = tmp_m.setdefault(metric, {})
            tmp_m_ = metric_df.groupby(
                ['index'])[metric].sum().reset_index()
            if metric == 'current_socket_power':
                norm_tmp_m_ = tmp_m_[metric].max()
                if metric in norm_tmp_m:
                    norm_tmp_m[metric] = max(norm_tmp_m_, norm_tmp_m[metric])
                else:
                    norm_tmp_m[metric] = norm_tmp_m_
            else:
                norm_tmp_m_ = tmp_m_[metric].min()
                if metric in norm_tmp_m:
                    norm_tmp_m[metric] = min(norm_tmp_m_, norm_tmp_m[metric])
                else:
                    norm_tmp_m[metric] = norm_tmp_m_

            tmp_m_[f'{metric}_rolling'] = tmp_m_[metric].rolling(
                window=2000,
            ).quantile(.95 if metric == 'current_socket_power' else .05)
            metric_slot[variant] = tmp_m_

    variants = list(data.keys())
    for variant in variants:
        info(f"Drawing: {variant}")
        for ci, metric in enumerate(metrics):
            ax = axs[metrics.index(metric)][variants.index(variant)]
            ax.plot(
                tmp_m[metric][variant]['index'],
                tmp_m[metric][variant][f'{metric}_rolling'] / norm_tmp_m[metric],
                color=color_dict[metric],
                linewidth=1.0,
                linestyle='-',
            )
            ax.grid(axis="y", linestyle='--', alpha=.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=1))

            ymin, ymax = ax.get_ylim()
            if gymin[metric] is None:
                gymin[metric] = ymin
            else:
                gymin[metric] = min(gymin[metric], ymin)
            if gymax[metric] is None:
                gymax[metric] = ymax
            else:
                gymax[metric] = max(gymax[metric], ymax)

    for variant in variants:
        for metric in metrics:
            axs[metrics.index(metric)][variants.index(variant)].set_ylim(
                (gymin[metric], gymax[metric]))
            axs[metrics.index(metric)][variants.index(
                variant)].tick_params(axis='x', pad=1)
        axs[len(metrics)-1][variants.index(variant)
                            ].set_title(variant, pad=1.5, fontsize=8)

    axs[0][0].set_ylabel("norm", labelpad=1)
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

    axs[n_rows - 1][n_cols // 2].set_xlabel("sample")

    legend_handles = [
        mpatches.Patch(
            color=color_dict[metric], label=legend_names[metric])
        for metric in metrics
    ]

    ax = axs[0][0]
    ax.figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(gpus),
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
