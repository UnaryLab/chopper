from chopper.common.load import get_straggler_df
from chopper.common.colors import okabe_ito
from loguru import logger
from chopper.common.annotations import PaperMode
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.figure import Figure


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
):
    """Load and process straggler lead and throughput metrics.

    Extracts straggler lead values and iteration timing from trace files
    to analyze GPU performance imbalance and training throughput.

    Args:
        ts_files: List of paths to trace pickle files
        configs: Config labels (e.g. "b1s4", "b2s8")

    Returns:
        Tuple containing:
            - dfs: List of processed DataFrames with straggler metrics
            - configs: List of config names
    """
    dfs = []
    for ts_file in ts_files:
        df = get_straggler_df(
            ts_file,
            agg_meth="max",
            kernel_name=True,
        )
        df = df.sort_values("ts_first").reset_index()
        dfs.append(df)
    return dfs, configs


def draw(
    fig: Figure,
    input_data,
    use_elapsed: bool = False,
    adjust_steps: int = 3,
    wait_steps: int = 50,
    y_max: float = float("inf"),
    y_min: float = float("-inf"),
    scatter_alpha: float = 1.0,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw normalized lead and throughput over iterations.

    Creates a dual-axis plot showing straggler lead (performance imbalance)
    and normalized throughput across training iterations. Highlights
    pre-adjustment and post-adjustment phases.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() containing straggler DataFrames
        use_elapsed: If True, use elapsed time instead of sum duration
        adjust_steps: Number of adjustment steps
        wait_steps: Number of warmup steps before adjustment
        y_max: Maximum y-axis limit
        y_min: Minimum y-axis limit
        paper_mode: PaperMode settings for publication-quality figures
    """

    dfs, configs = input_data

    n_rows = 2
    n_cols = len(configs)

    fig.clear()
    fig.patches.clear()  # Ensure figure-level patches are also cleared

    # Apply layout adjustments only in paper mode
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace
        )

    axs = tuple(
        tuple(
            fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1) for j in range(n_cols)
        )
        for i in range(n_rows)
    )

    gymin0: float | None = None
    gymax0: float | None = None
    gymin1: float | None = None
    gymax1: float | None = None

    # Compute global max lead across all configs
    global_max_lead = 0.0
    for df in dfs:
        total_lead_df = (
            df.groupby(["gpu", "iteration"])
            .agg(
                **{
                    "total_lead": ("s-value", "sum"),
                }
            )
            .reset_index()
        )
        global_max_lead = max(global_max_lead, total_lead_df["total_lead"].max())

    for df, config in zip(dfs, configs):
        logger.info(f"Drawing: {config}")

        if not use_elapsed:
            iter_time = (
                df.groupby(["gpu", "iteration"])
                .agg(
                    **{
                        "dur": ("dur", "sum"),
                    }
                )
                .groupby("iteration")
                .max()
                * 1e-9
            ).reset_index()
        else:
            iter_time = df.groupby(["gpu", "iteration"]).agg(
                **{
                    "ts_first": ("ts_first", "first"),
                    "ts_last": ("ts_last", "last"),
                    "dur_last": ("dur_last", "last"),
                }
            )
            iter_time["dur"] = (
                iter_time["ts_last"] + iter_time["dur_last"] - iter_time["ts_first"]
            )
            iter_time = (
                iter_time.groupby("iteration")["dur"].max() * 1e-9
            ).reset_index()

        total_lead_df = (
            df.groupby(["gpu", "iteration"])
            .agg(
                **{
                    "total_lead": ("s-value", "sum"),
                }
            )
            .reset_index()
        )
        total_lead_df["index"] = total_lead_df.groupby(["gpu"]).cumcount()

        iters = sorted(df["iteration"].unique())

        color_dict = {
            "PreAdj": okabe_ito["Pink"],
            "PostAdj": okabe_ito["Cyan"],
        }
        alpha_dict = {
            "PreAdj": 0.25,
            "PostAdj": 0.25,
        }

        ax0 = axs[0][configs.index(config)]
        ax1 = axs[1][configs.index(config)]
        ax1.set_title(config, pad=2, fontsize=8)
        ax1.set_zorder(1)
        ax0.set_zorder(0)
        ax0.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax1.patch.set_visible(False)
        ax1.plot(
            iter_time.index,
            iter_time["dur"].max() / iter_time["dur"],
            linestyle="-",
            color=okabe_ito["Black"],
            alpha=0.99,
            linewidth=1.0,
        )
        ax0.axvspan(
            0,
            wait_steps + adjust_steps - 1,
            facecolor=color_dict["PreAdj"],
            edgecolor=None,
            alpha=alpha_dict["PreAdj"],
            zorder=1,
        )
        ax0.axvspan(
            wait_steps + adjust_steps - 1,
            len(iters),
            facecolor=color_dict["PostAdj"],
            edgecolor=None,
            alpha=alpha_dict["PostAdj"],
            zorder=1,
        )
        ax1.axvspan(
            0,
            wait_steps + adjust_steps - 1,
            facecolor=color_dict["PreAdj"],
            edgecolor=None,
            alpha=alpha_dict["PreAdj"],
            zorder=1,
        )
        ax1.axvspan(
            wait_steps + adjust_steps - 1,
            len(iters),
            facecolor=color_dict["PostAdj"],
            edgecolor=None,
            alpha=alpha_dict["PostAdj"],
            zorder=1,
        )
        ax0.scatter(
            total_lead_df["index"],
            total_lead_df["total_lead"] / global_max_lead,
            color=okabe_ito["Black"],
            alpha=scatter_alpha,
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

    for config in configs:
        if y_min != float("-inf") and y_max != float("inf"):
            axs[0][configs.index(config)].set_ylim((y_min, y_max))
        else:
            axs[0][configs.index(config)].set_ylim((gymin0, gymax0))
        axs[0][configs.index(config)].tick_params(axis="x", pad=1)
        axs[0][configs.index(config)].grid(axis="y", linestyle="--", alpha=0.5)
        if y_min != float("-inf") and y_max != float("inf"):
            axs[1][configs.index(config)].set_ylim((y_min, y_max))
        else:
            axs[1][configs.index(config)].set_ylim((gymin1, gymax1))
        axs[1][configs.index(config)].tick_params(axis="x", pad=1)
        axs[1][configs.index(config)].grid(axis="y", linestyle="--", alpha=0.5)

    axs[1][0].set_ylabel(
        "norm thr.",
        labelpad=1,
        # color=color_dict['Throughput'],
    )
    axs[1][0].tick_params(
        axis="y",
        pad=1,
        # colors=color_dict['Throughput'],
    )

    axs[0][0].set_ylabel("norm lead", labelpad=1)
    axs[0][0].tick_params(axis="y", pad=1)
    for col in range(1, n_cols):
        for row in range(n_rows):
            axs[row][col].tick_params(axis="y", length=0)
            axs[row][col].set_yticklabels([])

    for row in range(n_rows):
        for col in range(n_cols):
            axs[row][col].tick_params(axis="x", length=0)
            axs[row][col].set_xticklabels([])

    # Center xlabel across all columns
    fig.text(0.5, 0.01, "iteration sample", ha="center", va="bottom")

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
        mpatches.Patch(color=color_dict[metric], label=metric, alpha=alpha_dict[metric])
        for metric in color_dict.keys()
    ]

    legend_kwargs = dict(
        handles=legend_handles,
        loc="upper center",
        ncol=len(color_dict.keys()),
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=0.5,
        frameon=False,
    )
    if paper_mode.enabled and paper_mode.legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = paper_mode.legend_bbox
    fig.legend(**legend_kwargs)

def main(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    use_elapsed: bool = False,
    adjust_steps: int = 3,
    wait_steps: int = 50,
    y_max: float = float("inf"),
    y_min: float = float("-inf"),
    paper_mode: PaperMode = PaperMode(),
    figsize: tuple[float, float] = (7.16, 3.5),
    filename: str = "lead_and_throughput.pdf",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(ts_files, configs)
    draw(fig, input_data, use_elapsed, adjust_steps, wait_steps, y_max, y_min, paper_mode=paper_mode)

    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
