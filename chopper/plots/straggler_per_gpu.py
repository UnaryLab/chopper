#!/usr/bin/env python3

from chopper.common.load import get_straggler_df
from chopper.common.colors import okabe_ito
from matplotlib.ticker import MaxNLocator
from typing import Tuple
from matplotlib.figure import Figure
from chopper.common.annotations import Framework


def get_data(
        dirs: Tuple[str] = (
            '.',
        ),
        variants: Tuple[str] = (
            'nvidia',
        ),
):

    dfs = tuple(
        get_straggler_df(f'{d}/ts.pkl',
                         agg_meth='max',
                         framework=Framework.FSDPv2,
                         kernel_name=True)
        for d in dirs
    )
    return dfs, variants


def draw(
    fig: Figure,
    input_data,
    n_gpus: int = 8,
    idx_start: int = 0,
    idx_end: int = -1,

):

    dfs, variants = input_data

    n_rows = len(variants)
    n_cols = n_gpus

    fig.clear()
    axs = tuple(tuple(fig.add_subplot(n_rows, n_cols, i*n_cols+j+1)
                for j in range(n_cols)) for i in range(n_rows))
    color_dict = {
        'Lead': okabe_ito['Black'],
    }

    tmp_df = {}
    gymin0 = gymax0 = 0
    max_lead = 0
    for i, (variant, df) in enumerate(zip(variants, dfs)):
        for gpu in range(n_gpus):
            gpu_slot = tmp_df.setdefault(variant, {})
            ax = axs[i][gpu]
            iters = sorted(df['iteration'].unique())
            tmp_df_ = df[
                (df['gpu'] == gpu) &
                (df['iteration'].isin(iters[idx_start:idx_end]))
            ].reset_index(
                drop=True).reset_index()

            max_lead = max(max_lead, tmp_df_['s-value'].max())
            gpu_slot[gpu] = tmp_df_

    for i, variant in enumerate(variants):
        for gpu in range(n_gpus):
            ax = axs[i][gpu]
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.grid(axis="y", linestyle='--', alpha=.5)
            tmp_df_ = tmp_df[variant][gpu]
            iters = sorted(tmp_df_['iteration'].unique())
            print(iters)
            assert tmp_df_['ts_first'].is_monotonic_increasing
            iter_agg = tmp_df_.groupby('iteration')['index'].agg(
                ['min', 'median', 'max'])

            for iter in iters[::2]:
                iter_min = iter_agg.loc[iter, 'min']
                iter_max = iter_agg.loc[iter, 'max']
                ax.axvspan(
                    iter_min,
                    iter_max,
                    facecolor=okabe_ito['Pink'],
                    edgecolor=None,
                    alpha=0.25,
                    zorder=0,
                )

            ax.scatter(
                tmp_df_['index'],
                tmp_df_['s-value'] / max_lead,
                color=color_dict['Lead'],
                alpha=0.125,
                s=0.1,
            )

            ymin, ymax = ax.get_ylim()
            gymin0 = min(gymin0, ymin)
            gymax0 = max(gymax0, ymax)

            if i == 0:
                ax.text(
                    0.50, 0.9975, f"GPU{gpu}",
                    transform=ax.transAxes,
                    ha='center', va='bottom',
                    fontsize=8,
                )

        for gpu in range(n_gpus):
            axs[i][gpu].set_ylim((gymin0, gymax0))

        axs[n_rows-1][n_cols//2].set_xlabel("kernel sample")
