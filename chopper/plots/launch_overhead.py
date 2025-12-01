#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from matplotlib.ticker import MaxNLocator

import matplotlib.patches as mpatches
from matplotlib import rcParams

from chopper.common.colors import rgb
from chopper.common.annotations import (
    no_overlap_mask,
    assign_chunks,
    fix_names,
)
from chopper.common.trace_metrics import (
    derive_launch_overhead,
    derive_prep_overhead,
    derive_call_overhead,
)

from chopper.common.annotations import Framework
from typing import Tuple


def agg(df,
        group_arr,
        derive_cols_before=None,
        derive_cols_after=None,
        sum_cols_map={},
        ):

    if derive_cols_before is not None:
        for derive_col in derive_cols_before:
            df = derive_col(df)

    df_summed = df.groupby(group_arr, dropna=False).agg(
        {
            **sum_cols_map,
        }).reset_index()

    df_summed.columns = [col[0] for col in df_summed.columns]

    if derive_cols_after is not None:
        for derive_col in derive_cols_after:
            derive_col(df_summed)

    return df_summed


def get_data(
    pkl_dirs: Tuple[str] = (
        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP21_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b1s4/',
        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP21_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b2s4/',
        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP21_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b4s4/',
        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP21_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b1s8/',
        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP21_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b2s8/',

        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP20_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b1s4/',
        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP20_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b2s4/',
        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP20_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b4s4/',
        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP20_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b1s8/',
        '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP20_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC15_PROFILE_TYPE2/b2s8/',
    ),
    two_axes: bool = True,
    frameworks: Tuple[Framework] = (
        Framework.FSDPv2,
        Framework.FSDPv2,
        Framework.FSDPv2,
        Framework.FSDPv2,
        Framework.FSDPv2,

        Framework.FSDPv1,
        Framework.FSDPv1,
        Framework.FSDPv1,
        Framework.FSDPv1,
        Framework.FSDPv1,
    ),
    configs: Tuple[str] = (
        'b1s4',
        'b2s4',
        'b4s4',
        'b1s8',
        'b2s8',

        'b1s4',
        'b2s4',
        'b4s4',
        'b1s8',
        'b2s8',
    ),
):
    data = {
        f"{'FSDPv2' if fw == Framework.FSDPv2 else 'FSDPv1'}-{config}": pd.read_pickle(f"{pkl_dir}/ts.pkl")
        for fw, config, pkl_dir in zip(frameworks, configs, pkl_dirs)
    }
    if two_axes:
        ops = (
            'f_ie',
            'b_ga',
            'opt_step',
            'f_attn_n',
            'b_mlp_dp',
            'b_ie',
        )
    else:
        ops = (
            'f_ie',
            'f_attn_n',
            'b_mlp_dp',
            'b_ie',
        )

    metrics = (
        'Launch Overhead',
        'Prep Overhead',
        'Call Overhead',
    )
    max_ov_sub = 0
    for fw, setup in zip(frameworks, data.keys()):
        data[setup]['layer'] = data[setup]['layer'].fillna(-1)
        weird_mask = data[setup]['iteration'].isna()
        weird_df = data[setup][weird_mask]
        max_weird_ts = weird_df['ts'].max()
        min_norm_ts = data[setup][~weird_mask]['ts'].min()
        assert min_norm_ts > max_weird_ts, "Nan iteration isn't at the start"
        data[setup] = data[setup][~weird_mask]

        data[setup] = data[setup][data[setup]['name']
                                  != 'Memcpy HtoD (Host -> Device)']

        data[setup] = assign_chunks(data[setup])
        nan_chunk_mask = data[setup]['chunk'].isna()

        data[setup] = fix_names(data[setup])

        overlap_mask = no_overlap_mask(data[setup], framework=fw)

        data[setup] = agg(
            data[setup][overlap_mask & ~nan_chunk_mask],
            ['gpu', 'chunk', 'iteration', 'operator-name', 'layer'],
            derive_cols_before=(
                derive_launch_overhead,
                derive_prep_overhead,
                derive_call_overhead,
            ),
            sum_cols_map={metric: ['sum'] for metric in metrics}
        )
        gb_sub = data[setup][~data[setup]['operator-name'].isin(
            ('f_ie', 'b_ga', 'opt_step'))].groupby('operator-name')
        max_ov_sub = max(max_ov_sub, np.max(gb_sub['Launch Overhead'].mean()))

    for setup in data.keys():
        data[setup]['Call Overhead'] /= max_ov_sub
        data[setup]['Prep Overhead'] /= max_ov_sub
        data[setup]['Launch Overhead'] /= max_ov_sub
        # k = 7
        # for metric in metrics:
        #     print('-'*5 + setup + '-'*5)
        #     print(data[setup].groupby('operator-name')
        #           [metric].mean().nlargest(k))
        #     print('-'*(10+len(setup)))

    return data, ops, two_axes


def draw(fig: Figure, input_data):
    data, ops, two_axes = input_data
    rgb_colors = (
        rgb(0x66, 0xc2, 0xa5),
        rgb(0xfc, 0x8d, 0x62),
    )

    hatches = (
        None,
        '\\\\\\\\\\',
    )

    setups = tuple(data.keys())
    params = sorted(set(s.split('-')[1] for s in setups),
                    key=lambda x: re.findall(r'\d+', x)[::-1])
    variants = sorted(set(s.split('-')[0] for s in setups))

    metrics = (
        'Prep Overhead',
        'Call Overhead',
    )

    bar_color = {
        m: rgb_colors[i]
        for i, m in enumerate(metrics)
    }
    bar_hatch = {
        m: hatches[i]
        for i, m in enumerate(variants)
    }

    n_rows = 1
    n_cols = len(ops) + 1

    x = np.arange(len(params))
    fig.clear()
    width_ratios = (
        (tuple(1 for _ in range(3)) + (.1,) +
         tuple(1 for _ in range(3))) if two_axes else
        (
            tuple(1 for _ in range(1)) + (.1,) +
            tuple(1 for _ in range(3))
        )

    )
    gs = fig.add_gridspec(n_rows, n_cols,
                          width_ratios=width_ratios)
    axs = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]

    ax_idxs = (
        list(tuple(range(3)) + tuple(range(4, n_cols)))
        if two_axes else
        list(tuple(range(1)) + tuple(range(2, n_cols)))
    )
    g_ymin = None
    g_ymax = None
    axs[3 if two_axes else 1].set_visible(False)

    for s in setups:
        variant = s.split('-')[0]
        for ax_idx in ax_idxs:
            ax = axs[ax_idx]
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
            ax.spines['top'].set_visible(False)
            if ax_idx == ax_idxs[0]:
                ax.set_ylabel("norm time", labelpad=0)
                ax.spines['right'].set_visible(False)
            elif ax_idx == ax_idxs[3 if two_axes else 1]:
                ax.spines['right'].set_visible(False)
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)
            elif ax_idx == ax_idxs[-1]:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_label_position("right")
                ax.tick_params(axis='y', which='both', left=False,
                               right=True, labelright=True, labelleft=False, pad=1)
                ax.set_ylabel("norm time", labelpad=0)
            else:
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)

            if ax_idx > (3 if two_axes else 1):
                ax.set_ylim((0, 1.1))
            else:
                _ymin, _ymax = ax.get_ylim()
                if g_ymin is None:
                    g_ymin = _ymin
                else:
                    g_ymin = min(g_ymin, _ymin)
                if g_ymax is None:
                    g_ymax = _ymax
                else:
                    g_ymax = max(g_ymax, _ymax)

            ax.set_xticks(x)
            ax.tick_params(axis='y', which='major', pad=1)
            ax.tick_params(axis='x', which='major', pad=1, rotation=65)
            ax.set_xticklabels(params)
            ax.set_xlim(-.5, len(params)-1+.5)
            bar_width = 0.9 / len(variants)
            offset = (
                -bar_width/2 *
                (len(variants)-1) +
                bar_width*variants.index(variant))
            ax.grid(axis="y", linestyle='--', alpha=.5)

            bottom = 0

            param = s.split('-')[1]

            op = ops[ax_idxs.index(ax_idx)]
            ax.set_title(op, pad=5, fontsize=8)
            med_p = (data[s].groupby(
                'operator-name').get_group(op)['Prep Overhead'].mean())
            med_c = (data[s].groupby(
                'operator-name').get_group(op)['Call Overhead'].mean())

            tick = params.index(param)

            ax.bar(
                tick+offset,
                med_c,
                width=bar_width*.9,
                bottom=bottom,
                color=bar_color['Call Overhead'],
                alpha=.99,
                hatch=bar_hatch[variant],
            )

            bottom += med_c

            ax.bar(
                tick+offset,
                med_p,
                width=bar_width*.9,
                bottom=bottom,
                color=bar_color['Prep Overhead'],
                alpha=.99,
                hatch=bar_hatch[variant],
            )

    legend_handles = [
        mpatches.Patch(
            color=bar_color[m], label=m)
        for m in reversed(metrics)
    ]
    legend_handles.extend([
        mpatches.Patch(
            facecolor='white', edgecolor='black', label=m, hatch=bar_hatch[m],
        )
        for m in variants
    ])

    for i in range(3 if two_axes else 1):
        axs[i].set_ylim((g_ymin, g_ymax))

    axl = axs[2 if two_axes else 0]
    axr = axs[4 if two_axes else 2]

    posl = axl.get_position()
    posr = axr.get_position()

    y_val = 1.0
    transl = axl.transData + axl.transAxes.inverted()
    transr = axr.transData + axr.transAxes.inverted()

    yl_axes = transl.transform((0, y_val))[1]
    yr_axes = transr.transform((0, y_val))[1]

    yl_fig = posl.y0 + yl_axes * posl.height
    yr_fig = posr.y0 + yr_axes * posr.height

    fig.add_artist(Line2D(
        [posl.x1, posr.x0],
        [yl_fig, yr_fig],
        transform=fig.transFigure,
        color=rcParams['grid.color'],
        linewidth=rcParams['grid.linewidth'],
        linestyle='--',
        alpha=0.5
    ))

    fig.legend(
        handles=legend_handles,
        ncol=len(legend_handles),
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=0.5,
        frameon=False,
    )


if __name__ == "__main__":
    draw()
