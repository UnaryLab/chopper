#!/usr/bin/env python3

import pandas as pd
from chopper.common.load import get_straggler_df
from chopper.common.colors import okabe_ito
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from chopper.common.annotations import PaperMode
import matplotlib.patches as mpatches


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
):
    """Load and process per-GPU straggler metrics.

    Extracts straggler lead values for each individual GPU to analyze
    per-GPU performance imbalance across training iterations.

    Args:
        ts_files: List of paths to trace pickle files
        configs: Config labels (e.g. "b1s4", "b2s8")

    Returns:
        Tuple containing:
            - dfs: List of processed DataFrames with per-GPU straggler metrics
            - configs: List of config names
    """
    dfs = [
        get_straggler_df(ts_file, agg_meth="max", kernel_name=True)
        for ts_file in ts_files
    ]
    return dfs, configs


def draw(
    fig: Figure,
    input_data,
    idx_start: int = 0,
    idx_end: int = -1,
    y_maxs: list[float] = [float("inf")],
    y_mins: list[float] = [float("-inf")],
    alpha: float = 0.025,
    s: float = 0.1,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw per-GPU straggler lead over kernel samples.

    Creates a multi-panel scatter plot showing straggler lead for each GPU
    individually. Alternating iterations are shaded for visual clarity.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() containing straggler DataFrames
        idx_start: Starting iteration index
        idx_end: Ending iteration index (-1 for last)
        y_maxs: List of maximum y-axis limits, one per row (config)
        y_mins: List of minimum y-axis limits, one per row (config)
        alpha: Transparency of scatter points (0-1)
        paper_mode: PaperMode settings for publication-quality figures
    """

    dfs, configs = input_data

    n_gpus: int | None = None
    for df in dfs:
        if n_gpus is None:
            n_gpus = df["gpu"].nunique()
        else:
            assert n_gpus == df["gpu"].nunique(), "number of GPUs don't match"

    n_rows = len(configs)
    n_cols = n_gpus

    assert n_gpus is not None, "n_gpus should not be None"
    assert n_cols is not None, "n_cols should not be None"

    fig.clear()
    fig.patches.clear()

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
    color_dict = {
        "Lead": okabe_ito["Black"],
    }

    tmp_df: dict[str, dict[int, pd.DataFrame]] = {}
    gymin0: float = 0.0
    gymax0: float = 0.0
    max_lead = 0
    for i, (config, df) in enumerate(zip(configs, dfs)):
        for gpu in range(n_gpus):
            gpu_slot = tmp_df.setdefault(config, {})
            ax = axs[i][gpu]
            iters = sorted(df["iteration"].unique())
            tmp_df_ = (
                df[
                    (df["gpu"] == gpu)
                    & (df["iteration"].isin(iters[idx_start:idx_end]))
                ]
                .reset_index(drop=True)
                .reset_index()
            )

            max_lead = max(max_lead, tmp_df_["s-value"].max())
            gpu_slot[gpu] = tmp_df_

    for i, config in enumerate(configs):
        for gpu in range(n_gpus):
            ax = axs[i][gpu]
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            tmp_df_ = tmp_df[config][gpu]
            iters = sorted(tmp_df_["iteration"].unique())
            assert tmp_df_["ts_first"].is_monotonic_increasing
            iter_agg = tmp_df_.groupby("iteration")["index"].agg(
                ["min", "median", "max"]
            )

            for iter in iters[::2]:
                iter_min = iter_agg.loc[iter, "min"]
                iter_max = iter_agg.loc[iter, "max"]
                ax.axvspan(
                    iter_min,
                    iter_max,
                    facecolor=okabe_ito["Pink"],
                    edgecolor=None,
                    alpha=0.25,
                    zorder=0,
                )

            ax.scatter(
                tmp_df_["index"],
                tmp_df_['s-value'] / max_lead,
                # tmp_df_["s-value"],
                color=color_dict["Lead"],
                alpha=alpha,
                s=s,
            )

            ymin, ymax = ax.get_ylim()
            gymin0 = min(gymin0, ymin)
            gymax0 = max(gymax0, ymax)

            if i == 0:
                ax.text(
                    0.50,
                    0.9975,
                    f"GPU{gpu}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Apply y-axis limits per row
    for i in range(n_rows):
        # Get y_min/y_max for this row, defaulting to last value if list is shorter
        row_y_min = y_mins[i] if i < len(y_mins) else y_mins[-1]
        row_y_max = y_maxs[i] if i < len(y_maxs) else y_maxs[-1]

        for gpu in range(n_gpus):
            ax = axs[i][gpu]
            if row_y_min != float("-inf") and row_y_max != float("inf"):
                ax.set_ylim((row_y_min, row_y_max))
            else:
                ax.set_ylim((gymin0, gymax0))

            # Remove x ticks and labels
            ax.tick_params(axis="x", length=0)
            ax.set_xticklabels([])

            # Only show y axis values on first column
            if gpu > 0:
                ax.tick_params(axis="y", length=0)
                ax.set_yticklabels([])

    # Center ylabel across all rows
    fig.text(0.01, 0.5, "norm lead value", ha="left", va="center", rotation="vertical")

    # Center xlabel across all columns
    fig.text(0.5, 0.01, "kernel sample", ha="center", va="bottom")

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

def main(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    idx_start: int = 0,
    idx_end: int = -1,
    y_maxs: list[float] = [float("inf")],
    y_mins: list[float] = [float("-inf")],
    alpha: float = 0.025,
    s: float = 0.1,
    paper_mode: PaperMode = PaperMode(),
    filename: str = "straggler_per_gpu.png",
):
    fig = Figure()
    input_data = get_data(ts_files, configs)
    draw(fig, input_data, idx_start, idx_end, y_maxs, y_mins, alpha, s, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
