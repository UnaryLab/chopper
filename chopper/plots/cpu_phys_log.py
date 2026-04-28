"""Active logical-vs-physical CPU core distribution.

Top: count of active logical cores plus a derived "min cores" curve over time.
Bottom: heatmap of active logical cores per physical core. Annotates the
median percent of physical cores that ever go active. (Paper Figure: cpu_phys_log.pdf)
"""

import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from chopper.common.annotations import PaperMode


def _log_to_phys_cpu(fn: str) -> dict:
    df = pd.read_csv(fn, sep=r"\s+")
    return df.set_index("CPU")["CORE"].to_dict()


def get_data(
    cpu_distribution_pkl: str = "./cpu_distribution.pkl",
    phys_log_cpu_map: str = "./phys_log_cpu_map.log",
):
    """Load CPU distribution samples and the logical->physical core map.

    Args:
        cpu_distribution_pkl: Path to per-sample CPU usage DataFrame
        phys_log_cpu_map: Whitespace-separated file with CPU/CORE columns

    Returns:
        Tuple (df, lp_cpu) where df is the sample DataFrame sorted by ts and
        lp_cpu is a {logical_cpu: physical_core} dict
    """
    df = pd.read_pickle(cpu_distribution_pkl).sort_values("ts").reset_index()
    lp_cpu = _log_to_phys_cpu(phys_log_cpu_map)
    return df, lp_cpu


def draw(
    fig: Figure,
    input_data,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw active-cores time series above a per-physical-core heatmap.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple (df, lp_cpu) from get_data()
        paper_mode: PaperMode settings for publication-quality figures
    """
    df, lp_cpu = input_data
    df = df.copy()
    df["samp_idx"] = (df["ts"] != df["ts"].shift()).cumsum() - 1
    df["phys_cpu"] = df["cpu"].apply(lambda log_cpu: lp_cpu[log_cpu])

    num_phys = len(set(lp_cpu.values()))
    num_log = len(set(lp_cpu.keys()))
    assert num_log % num_phys == 0

    color_dict = {
        "$C_{active}$": "blue",
        "$C_{min}$": "red",
    }

    fig.clear()
    fig.patches.clear()
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    axs = (fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2))
    axs[0].grid(axis="y", linestyle="--", alpha=0.5)

    df["core_busy"] = df["percent"] != 0

    df_phys = (
        df.groupby(["samp_idx", "phys_cpu"]).core_busy.sum().reset_index()
    )
    df_perc = df.groupby(["samp_idx"]).percent.sum().reset_index()
    df_perc["percent"] /= 100
    df_busy = df.groupby(["samp_idx"]).core_busy.sum().reset_index()

    perc_of_phys = (
        df.groupby(["samp_idx", "phys_cpu"]).core_busy.max().reset_index()
        .groupby("samp_idx").core_busy.sum().median()
        / num_phys * 100
    )

    axs[1].set_title(
        "$C^{phys}_{active}=$" + f"{perc_of_phys:.1f}% of " + "$C^{phys}$",
        pad=3, fontsize=8,
    )

    pivoted = df_phys.pivot(index="phys_cpu", columns="samp_idx", values="core_busy")
    axs[0].plot(
        df_busy["samp_idx"], df_busy["core_busy"],
        linewidth=0.5, color=color_dict["$C_{active}$"], zorder=0,
    )
    axs[0].plot(
        df_perc["samp_idx"], df_perc["percent"],
        linewidth=0.5, color=color_dict["$C_{min}$"], zorder=1,
    )
    im = axs[1].imshow(pivoted, aspect="auto", cmap="viridis", interpolation="nearest")

    cbar = fig.colorbar(im, ax=axs[1], pad=0)
    cbar.set_label("$C^{log}_{active}$", labelpad=10, fontsize=8, rotation=0)
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()

    ylim = axs[0].get_ylim()
    new_ticks = [tick for tick in axs[0].get_yticks() if ylim[0] <= tick <= ylim[1]]
    new_ticks.append(df_busy["core_busy"].median().round(0))
    new_ticks.append(df_perc["percent"].median().round(0))
    axs[0].set_yticks(new_ticks)

    labels = axs[0].get_yticklabels()
    if len(labels) >= 2:
        x, y = labels[0].get_position()
        labels[-2].set_position((x - 0.065, y))
        labels[-2].set_color("blue")
        x, y = labels[1].get_position()
        labels[-1].set_position((x - 0.025, y))
        labels[-1].set_color("red")
        axs[0].set_yticklabels(labels)

    axs[0].tick_params(axis="x", which="major", length=0)
    axs[0].set_xticklabels([])
    x1 = df_busy["samp_idx"].max()
    assert x1 == df_perc["samp_idx"].max()
    axs[0].set_xlim((0, x1))
    axs[0].tick_params(axis="y", which="major", pad=1)
    axs[0].set_ylabel("$C^{log}$", labelpad=-1, rotation=0)
    axs[1].tick_params(axis="x", length=0)
    axs[1].set_xticklabels([])
    axs[1].tick_params(axis="y", which="major", pad=1)
    axs[1].set_ylabel("$C^{phys}$", labelpad=3, rotation=0)
    axs[1].set_xlabel("sample", labelpad=-2)

    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=color_dict[key], lw=2, label=key)
        for key in color_dict.keys()
    ]
    legend_kwargs = dict(
        handles=legend_handles,
        loc="upper right",
        ncol=1,
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.4,
        handlelength=1.8,
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
    cpu_distribution_pkl: str = "./cpu_distribution.pkl",
    phys_log_cpu_map: str = "./phys_log_cpu_map.log",
    paper_mode: PaperMode = PaperMode(),
    filename: str = "cpu_phys_log.png",
):
    fig = Figure()
    input_data = get_data(cpu_distribution_pkl, phys_log_cpu_map)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
