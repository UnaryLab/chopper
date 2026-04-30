"""CPU core activity heatmap and active core count over time.

Top panel: number of active logical cores and minimum busy cores over time.
Bottom panel: heatmap of logical threads active per physical core.
(Paper Figure: cpu_active_cores.pdf)
"""

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

from chopper.common.cache import load_pickle
from chopper.common.annotations import PaperMode


def _log_to_phys(lscpu_file):
    """Parse lscpu -e output to get logical CPU -> physical CORE mapping."""
    df = pd.read_csv(lscpu_file, sep=r'\s+')
    return df.set_index('CPU')['CORE'].to_dict()


def get_data(
    cpu_files: list[str] = ["./cpu.pkl"],
    configs: list[str] = ["default"],
    lscpu_file: str | None = None,
):
    """Load CPU telemetry and optional physical core mapping.

    Args:
        cpu_files: List of paths to cpu.pkl files
        configs: Config labels
        lscpu_file: Path to lscpu -e output for physical core mapping

    Returns:
        Dict mapping config -> {"df": DataFrame, "lp_cpu": mapping or None}
    """
    lp_cpu = _log_to_phys(lscpu_file) if lscpu_file else None
    return {
        config: {"df": load_pickle(fn), "lp_cpu": lp_cpu}
        for fn, config in zip(cpu_files, configs)
    }


def draw(
    fig: Figure,
    input_data,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw CPU active core count and physical core heatmap.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data()
        paper_mode: PaperMode settings
    """
    configs = list(input_data.keys())
    n_cols = len(configs)

    fig.clear()
    fig.patches.clear()

    color_dict = {
        '$C_{active}$': 'blue',
        '$C_{min}$': 'red',
    }

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    gs = fig.add_gridspec(2, n_cols + 1, width_ratios=[1] * n_cols + [0.03],
                          height_ratios=[1, 1])
    axs = tuple(
        tuple(fig.add_subplot(gs[r, c]) for c in range(n_cols))
        for r in range(2)
    )
    cbar_ax = fig.add_subplot(gs[1, n_cols])

    for ci, config in enumerate(configs):
        df = input_data[config]["df"].copy()
        lp_cpu = input_data[config]["lp_cpu"]

        df["samp_idx"] = (df["ts"] != df["ts"].shift()).cumsum() - 1
        df["core_busy"] = df["percent"] != 0

        if lp_cpu is not None:
            df["phys_cpu"] = df["cpu"].map(lp_cpu)
        else:
            # Assume 2 threads per core if no mapping
            df["phys_cpu"] = df["cpu"] // 2

        num_phys = df["phys_cpu"].nunique()

        # Top panel: active logical cores + min busy
        df_busy = df.groupby("samp_idx")["core_busy"].sum().reset_index()
        df_perc = df.groupby("samp_idx")["percent"].sum().reset_index()
        df_perc["percent"] /= 100

        ax0 = axs[0][ci]
        ax0.plot(
            df_busy["samp_idx"], df_busy["core_busy"],
            linewidth=0.5, color=color_dict["$C_{active}$"], zorder=0,
        )
        ax0.plot(
            df_perc["samp_idx"], df_perc["percent"],
            linewidth=0.5, color=color_dict["$C_{min}$"], zorder=1,
        )
        ax0.grid(axis="y", linestyle="--", alpha=0.5)
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.tick_params(axis="x", length=0)
        ax0.set_xticklabels([])
        ax0.set_xlim(0, df_busy["samp_idx"].max())
        if ci == 0:
            ax0.set_ylabel("$C^{log}$", labelpad=-1, rotation=0)
        else:
            ax0.tick_params(axis="y", length=0)
            ax0.set_yticklabels([])

        # Bottom panel: physical core heatmap
        df_phys = (
            df.groupby(["samp_idx", "phys_cpu"])["core_busy"]
            .sum().reset_index()
        )
        pivoted = df_phys.pivot(
            index="phys_cpu", columns="samp_idx", values="core_busy"
        )

        perc_of_phys = (
            df.groupby(["samp_idx", "phys_cpu"])["core_busy"]
            .max().reset_index()
            .groupby("samp_idx")["core_busy"].sum().median()
            / num_phys * 100
        )

        ax1 = axs[1][ci]
        im = ax1.imshow(
            pivoted, aspect="auto", cmap="viridis", interpolation="nearest",
        )
        ax1.set_title(
            "$C^{phys}_{active}=$" + f"{perc_of_phys:.1f}% of $C^{{phys}}$",
            pad=3, fontsize=8,
        )
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.tick_params(axis="x", length=0)
        ax1.set_xticklabels([])
        ax1.set_xlabel("sample", labelpad=-2)
        if ci == 0:
            ax1.set_ylabel("$C^{phys}$", labelpad=3, rotation=0)
        else:
            ax1.tick_params(axis="y", length=0)
            ax1.set_yticklabels([])

        if ci == n_cols - 1:
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(
                "$C^{log}_{active}$", labelpad=10, fontsize=8, rotation=0,
            )
            cbar.locator = MaxNLocator(integer=True)
            cbar.update_ticks()

    # Legend
    legend_handles = [
        Line2D([0], [0], color=color_dict[key], lw=2, label=key)
        for key in color_dict
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=1.5,
        frameon=False,
    )

    if paper_mode.enabled:
        fig.patches.append(
            mpatches.Rectangle(
                (0, 0), 1, 1,
                transform=fig.transFigure,
                fill=False, edgecolor="black", linewidth=1, zorder=1000,
            )
        )


def main(
    cpu_files: list[str] = ["./cpu.pkl"],
    configs: list[str] = ["default"],
    lscpu_file: str | None = None,
    paper_mode: PaperMode = PaperMode(),
    figsize: tuple[float, float] = (7.16, 4.0),
    filename: str = "cpu_active_cores.pdf",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(cpu_files, configs, lscpu_file)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
