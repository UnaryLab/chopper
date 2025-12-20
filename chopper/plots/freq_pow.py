import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from chopper.common.colors import rgb
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from typing import Tuple


def get_data(
        dirs: Tuple[str] = (
            '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP20_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b2s4/',
            '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP21_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b2s4/',
        ),
        variants: Tuple[str] = (
            'FSDPv1',
            'FSDPv2',
        )):
    metrics = (
        'current_gfxclk',
        'current_uclk',
        'current_socket_power',
    )
    norm_metric = {}
    metric_df = {}
    for path, variant in zip(dirs, variants):
        metric_trace = pd.read_pickle(f'{path}/metric_samples.pkl')
        # print(metric_trace.columns.tolist())
        # exit(0)

        metric_df_ = metric_trace.copy()
        metric_df_['gpu'] -= 2

        n_gpus = metric_df_['gpu'].nunique()
        group_size = n_gpus

        metric_df_['index'] = metric_df_.index // group_size
        start = metric_df_['index'].max() * (.52 if variant ==
                                             'FSDPv1' else .31)
        end = metric_df_['index'].max() * (.97 if variant == 'FSDPv1' else .95)
        metric_df_ = metric_df_[
            (metric_df_['index'] > start) &
            (metric_df_['index'] < end)
        ]

        for metric in metrics:
            metric_df_[metric] = metric_df_[metric].astype(np.float64)
        metric_df[variant] = metric_df_
        for ci, metric in enumerate(metrics):
            tmp_m = metric_df_.groupby(
                ['index'])[metric].sum().reset_index()
            # if metric == 'current_socket_power':
            norm_metric_ = tmp_m[metric].max()
            if metric not in norm_metric:
                norm_metric[metric] = norm_metric_
            else:
                norm_metric[metric] = max(
                    norm_metric_, norm_metric[metric])
            # else:
            #     norm_metric_ = tmp_m[metric].min()
            #     if metric not in norm_metric:
            #         norm_metric[metric] = norm_metric_
            #     else:
            #         norm_metric[metric] = min(
            #             norm_metric_, norm_metric[metric])

    return norm_metric, metric_df, variants, metrics


def draw(
    fig: Figure,
    input_data,
    gpus: int = 8,  # hardcode for now
):
    norm_metric, metric_df, variants, metrics = input_data
    ylabel_names = {
        'current_gfxclk': 'norm',
        'current_uclk': 'norm',
        'current_socket_power': 'norm',
    }
    legend_names = {
        'current_gfxclk': 'GPU Frequency',
        'current_uclk': 'Memory Frequency',
        'current_socket_power': 'Power',
    }
    rgb_colors = (
        rgb(0x66, 0xc2, 0xa5),
        rgb(0x8d, 0xa0, 0xcb),
        rgb(0xfc, 0x8d, 0x62),
    )
    color_dict = {metric: rgb_colors[i]
                  for i, metric in enumerate(metrics)}

    n_rows = len(metrics)
    n_cols = len(variants)

    fig.clear()
    axs = tuple(tuple(fig.add_subplot(n_rows, n_cols, i*n_cols+j+1)
                for j in range(n_cols)) for i in range(n_rows))
    if n_cols == 1:
        axs = np.array([axs]).T
    if n_rows == 1:
        axs = np.array([axs])

    gymin = {metric: None for metric in metrics}
    gymax = gymin.copy()
    for variant in variants:
        for ci, metric in enumerate(metrics):
            tmp_m = metric_df[variant].groupby(
                ['index'])[metric].sum().reset_index()
            ax = axs[metrics.index(metric)][variants.index(variant)]
            ax.plot(
                tmp_m['index'],
                tmp_m[metric] / norm_metric[metric],
                color=color_dict[metric],
                linewidth=0.5,
                linestyle='-',
            )
            ax.grid(axis="y", linestyle='--', alpha=.5)
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

    for variant in variants:
        for metric in metrics:
            axs[metrics.index(metric)][variants.index(variant)].set_ylim(
                (gymin[metric], gymax[metric]))
            axs[metrics.index(metric)][variants.index(
                variant)].tick_params(axis='x', pad=1)
        axs[0][variants.index(variant)
               ].set_title(variant, pad=1.5, fontsize=8)

    for metric in metrics:
        axs[metrics.index(metric)][0].set_ylabel(
            ylabel_names[metric], labelpad=1)
        axs[metrics.index(metric)][0].tick_params(axis='y', pad=1)
    for col in range(1, n_cols):
        for row in range(n_rows):
            axs[row][col].tick_params(axis='y', length=0)
            axs[row][col].set_yticklabels([])

    for row in range(n_rows):
        for col in range(n_cols):
            axs[row][col].tick_params(axis='x', length=0)
            axs[row][col].set_xticklabels([])

    axs[n_rows-1][0].set_xlabel("sample")
    axs[n_rows-1][0].xaxis.set_label_coords(1.00, -0.05)

    legend_handles = [
        mpatches.Patch(
            color=color_dict[metric], label=legend_names[metric])
        for metric in metrics
    ]

    ax = axs[0][0]

    ax.figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(.50, 1.04),
        ncol=gpus,
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
        axs[n_rows-1][ci].tick_params(axis='x', pad=1)
