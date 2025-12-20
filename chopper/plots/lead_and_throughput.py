from chopper.common.load import get_straggler_df
from chopper.common.colors import okabe_ito
from chopper.common.printing import info
from chopper.common.annotations import Framework
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.figure import Figure
from typing import Tuple


def get_data(
        dirs: Tuple[str] = (
            '~/data/ispass_v2/GPU-Red',
            '~/data/ispass_v2/GPU-Realloc',
            '~/data/ispass_v2/CPU-Slosh',
        ),
        variants: Tuple[str] = (
            'GPU-Red',
            'GPU-Realloc',
            'CPU-Slosh',
        )):
    dfs = []
    for dir, var in zip(dirs, variants):
        df = get_straggler_df(f'{dir}/ts.pkl',
                              agg_meth='max',
                              framework=Framework.FSDPv2,
                              kernel_name=True,
                              )

        df = df.sort_values('ts_first').reset_index()
        dfs.append(df)
    return tuple(dfs), variants


def draw(
    fig: Figure,
    input_data,
    use_elapsed: bool = False,
    adjust_steps: int = 3,
    wait_steps: int = 50,
):

    dfs, variants = input_data

    n_rows = 2
    n_cols = len(variants)

    fig.clear()
    axs = tuple(tuple(fig.add_subplot(n_rows, n_cols, i*n_cols+j+1)
                for j in range(n_cols)) for i in range(n_rows))

    gymin0 = gymax0 = gymin1 = gymax1 = None
    for df, variant in zip(dfs, variants):
        info(f"Drawing: {variant}")

        if not use_elapsed:
            iter_time = (
                df.groupby(['gpu', 'iteration'])
                .agg(**{
                     'dur': ('dur', 'sum'),
                     })
                .groupby('iteration')
                .max()
                * 1e-9
            ).reset_index()
        else:
            iter_time = (
                df.groupby(['gpu', 'iteration'])
                .agg(**{
                     'ts_first': ('ts_first', 'first'),
                     'ts_last': ('ts_last', 'last'),
                     'dur_last': ('dur_last', 'last'),
                     })
            )
            iter_time['dur'] = (
                iter_time['ts_last']
                + iter_time['dur_last']
                - iter_time['ts_first']
            )
            iter_time = (
                iter_time
                .groupby('iteration')
                ['dur']
                .max()
                * 1e-9
            ).reset_index()

        total_lead_df = (
            df.groupby(['gpu', 'iteration'])
            .agg(**{
                 'total_lead': ('s-value', 'sum'),
                 })
            .reset_index()
        )
        total_lead_df['index'] = (
            total_lead_df
            .groupby(['gpu'])
            .cumcount()
        )

        iters = sorted(df['iteration'].unique())

        color_dict = {
            'PreAdj': okabe_ito['Pink'],
            'PostAdj': okabe_ito['Cyan'],
        }
        alpha_dict = {
            'PreAdj': .25,
            'PostAdj': .25,
        }

        ax0 = axs[0][variants.index(variant)]
        ax1 = axs[1][variants.index(variant)]
        ax1.set_title(variant, pad=2, fontsize=8)
        ax1.set_zorder(1)
        ax0.set_zorder(0)
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax1.patch.set_visible(False)
        ax1.plot(
            iter_time.index,
            iter_time['dur'].max() / iter_time['dur'],
            linestyle='-',
            color=okabe_ito['Black'],
            alpha=.99,
            linewidth=1.0,
        )
        ax0.axvspan(
            0,
            wait_steps + adjust_steps - 1,
            facecolor=color_dict['PreAdj'],
            edgecolor=None,
            alpha=alpha_dict['PreAdj'],
            zorder=1,
        )
        ax0.axvspan(
            wait_steps + adjust_steps - 1,
            len(iters),
            facecolor=color_dict['PostAdj'],
            edgecolor=None,
            alpha=alpha_dict['PostAdj'],
            zorder=1,
        )
        ax1.axvspan(
            0,
            wait_steps + adjust_steps - 1,
            facecolor=color_dict['PreAdj'],
            edgecolor=None,
            alpha=alpha_dict['PreAdj'],
            zorder=1,
        )
        ax1.axvspan(
            wait_steps + adjust_steps - 1,
            len(iters),
            facecolor=color_dict['PostAdj'],
            edgecolor=None,
            alpha=alpha_dict['PostAdj'],
            zorder=1,
        )
        max_total_lead = total_lead_df['total_lead'].max()
        ax0.scatter(
            total_lead_df['index'],
            total_lead_df['total_lead'] / max_total_lead,
            color=okabe_ito['Black'],
            alpha=.4,
            s=0.1,
            zorder=2,
        )

        ymin, ymax = ax0.get_ylim()
        if gymin0 is None:
            gymin0 = ymin
        else:
            gymin0 = min(gymin0, ymin)
        if gymax0 is None:
            gymax0 = ymax
        else:
            gymax0 = max(gymax0, ymax)

        ymin, ymax = ax1.get_ylim()
        if gymin1 is None:
            gymin1 = ymin
        else:
            gymin1 = min(gymin1, ymin)
        if gymax1 is None:
            gymax1 = ymax
        else:
            gymax1 = max(gymax1, ymax)

        ax1.set_xticks([wait_steps + adjust_steps])
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
        ax0.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))

    for variant in variants:
        axs[0][variants.index(variant)].set_ylim((gymin0, gymax0))
        axs[0][variants.index(variant)].tick_params(axis='x', pad=1)
        axs[0][variants.index(variant)].grid(
            axis="y", linestyle='--', alpha=.5)
        axs[1][variants.index(variant)].set_ylim((gymin1, gymax1))
        axs[1][variants.index(variant)].tick_params(axis='x', pad=1)
        axs[1][variants.index(variant)].grid(
            axis="y", linestyle='--', alpha=.5)

    axs[1][0].set_ylabel("norm thr.",
                         labelpad=1,
                         # color=color_dict['Throughput'],
                         )
    axs[1][0].tick_params(axis='y',
                          pad=1,
                          # colors=color_dict['Throughput'],
                          )

    axs[0][0].set_ylabel("norm lead", labelpad=1)
    axs[0][0].tick_params(axis='y', pad=1)
    for col in range(1, n_cols):
        for row in range(n_rows):
            axs[row][col].tick_params(axis='y', length=0)
            axs[row][col].set_yticklabels([])

    for row in range(n_rows):
        for col in range(n_cols):
            axs[row][col].tick_params(axis='x', length=0)
            axs[row][col].set_xticklabels([])

    axs[n_rows-1][n_cols//2].set_xlabel("iteration sample", labelpad=0)

    legend_handles = [
        mpatches.Patch(
            color=color_dict[metric], label=metric, alpha=alpha_dict[metric])
        for metric in color_dict.keys()
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(color_dict.keys()),
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=0.5,
        frameon=False,
    )
