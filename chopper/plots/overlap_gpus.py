"""Per-GPU CDFs of overlap ratio and normalized duration for f_attn_op.

For a single operator (default f_attn_op), shows per-GPU CDFs of overlap
ratio (dashed) and elapsed duration (solid) for each config.
(Paper Figure: overlap_gpus.pdf)
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from chopper.common.colors import rgb
from chopper.common.load import get_overlap_df
from chopper.common.annotations import PaperMode


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    operator: str = "f_attn_op",
    iter_idxs: range = range(-5, -2, 1),
):
    """Load overlap data for a single operator across configs and GPUs.

    Normalizes per-GPU elapsed by the per-GPU minimum elapsed across configs.

    Args:
        ts_files: List of ts.pkl trace files
        configs: Config labels (e.g. "b1s4", "b2s8")
        operator: Operator name to analyze
        iter_idxs: Iteration indices to select

    Returns:
        Tuple (overlap_data, gpus, operator) where overlap_data maps config ->
        DataFrame of operator entries with normalized 'elapsed' and 'op_idx'
    """
    data = {}
    for ts_file, config in zip(ts_files, configs):
        data[config] = get_overlap_df(
            ts_file, iter_idxs=list(iter_idxs)
        )

    overlap_data = {}
    config_min_elapsed = {}
    gpus = None

    for config in configs:
        operators = data[config]["operator-name"].unique()
        assert operator in operators, f"operator {operator} not found in {config}"
        cur_gpus = sorted(data[config]["gpu"].unique())
        if gpus is None:
            gpus = cur_gpus
        else:
            assert gpus == cur_gpus, "GPUs do not match across configs"

        op_mask = data[config]["operator-name"] == operator
        sub = data[config][op_mask].copy()
        sub["op_idx"] = sub.groupby("gpu").cumcount()
        overlap_data[config] = sub

        for gpu in gpus:
            min_elapsed = np.min(sub["elapsed"])
            config_min_elapsed[gpu] = min(
                config_min_elapsed.get(gpu, float("inf")), min_elapsed
            )

    for config in configs:
        overlap_data[config]["elapsed"] = overlap_data[config]["elapsed"].astype(float)
        for gpu in gpus:
            gpu_mask = overlap_data[config]["gpu"] == gpu
            overlap_data[config].loc[gpu_mask, "elapsed"] = (
                overlap_data[config].loc[gpu_mask, "elapsed"]
                / config_min_elapsed[gpu]
            )

    return overlap_data, gpus, operator


def draw(
    fig: Figure,
    input_data,
    n_cols: int = 4,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw per-GPU CDFs of overlap ratio and normalized duration.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data()
        n_cols: Number of columns in the GPU grid
        paper_mode: PaperMode settings for publication-quality figures
    """
    data, gpus, _op = input_data
    n_rows = max(1, len(gpus) // n_cols)

    fig.clear()
    fig.patches.clear()
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    axs = tuple(
        tuple(fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1) for c in range(n_cols))
        for r in range(n_rows)
    )

    params = list(data.keys())
    rgb_colors = (rgb(0xD4, 0x11, 0x59), rgb(0x1A, 0x85, 0xFF))
    color_dict = {config: rgb_colors[i % len(rgb_colors)] for i, config in enumerate(params)}
    line_color_dict = {
        config: (
            max(0, rgb_colors[i % len(rgb_colors)][0] - 0.3),
            max(0, rgb_colors[i % len(rgb_colors)][1] - 0.3),
            max(0, rgb_colors[i % len(rgb_colors)][2] - 0.3),
        )
        for i, config in enumerate(params)
    }

    for i, gpu in enumerate(gpus):
        ax = axs[i // n_cols][i % n_cols]
        for setup in data.keys():
            tmp_df = data[setup][data[setup]["gpu"] == gpu].copy()
            tmp_df["overlap_ratio"] /= 100

            overlap_ratio = (
                tmp_df.groupby("op_idx")["overlap_ratio"]
                .agg(["min", lambda x: x.quantile(0.25), "median",
                      lambda x: x.quantile(0.75), "max"])
                .reset_index()
            )
            elapsed = (
                tmp_df.groupby("op_idx")["elapsed"]
                .agg(["min", lambda x: x.quantile(0.25), "median",
                      lambda x: x.quantile(0.75), "max"])
                .reset_index()
            )
            overlap_ratio.columns = ["op_idx", "min", "q1", "median", "q3", "max"]
            elapsed.columns = ["op_idx", "min", "q1", "median", "q3", "max"]

            marker = "." if (len(overlap_ratio) == 1 and len(elapsed) == 1) else None

            if overlap_ratio.index.max() and overlap_ratio.index.max() > 0:
                prob = (overlap_ratio.index + 1) / overlap_ratio.index.max()
                ax.plot(
                    sorted(overlap_ratio["median"]), prob,
                    linestyle="--", marker=marker,
                    color=line_color_dict[setup], linewidth=1.0,
                )
            if elapsed.index.max() and elapsed.index.max() > 0:
                prob = (elapsed.index + 1) / elapsed.index.max()
                ax.plot(
                    sorted(elapsed["median"]), prob,
                    linestyle="-", marker=marker,
                    color=line_color_dict[setup], linewidth=1.0,
                )

        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="x", linestyle="-", alpha=0.5)
        ax.tick_params(axis="y", pad=1)
        ax.tick_params(axis="x", pad=1, rotation=25)
        ax.set_xticks((0, 0.5, 1, 1.5, 2))
        ax.set_yticks((0, 0.25, 0.5, 0.75, 1))

        if i == 0:
            ax.set_ylabel("probability")
            ax.yaxis.set_label_coords(-0.45, -0.125)
        if i == 5:
            ax.tick_params(axis="x", pad=1)
            ax.set_xlabel("norm")
            ax.xaxis.set_label_coords(1.00, -0.3)
        if i // n_cols < n_rows - 1:
            ax.tick_params(axis="x", length=0)
            ax.set_xticklabels([])
        if i % n_cols:
            ax.tick_params(axis="y", length=0)
            ax.set_yticklabels([])

        ax.set_xlim((-0.05, 2.15))
        ax.set_title(f"GPU{gpu}", fontsize=8, pad=2)

    legend_handles = [mpatches.Patch(color=color_dict[config], label=config) for config in params]
    line_handles = [
        Line2D([0], [0], color="black", linewidth=1, linestyle="--", label="overlap ratio"),
        Line2D([0], [0], color="black", linewidth=1, linestyle="-", label="duration"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=len(params),
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=0.5,
        frameon=False,
    )
    fig.legend(
        handles=line_handles,
        loc="upper right",
        ncol=2,
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=1.0,
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
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    operator: str = "f_attn_op",
    iter_start: int = -5,
    iter_stop: int = -2,
    iter_step: int = 1,
    n_cols: int = 4,
    paper_mode: PaperMode = PaperMode(),
    figsize: tuple[float, float] = (7.16, 2.5),
    filename: str = "overlap_gpus.pdf",
):
    fig = Figure(figsize=figsize)
    iter_idxs = range(iter_start, iter_stop, iter_step)
    input_data = get_data(ts_files, configs, operator, iter_idxs)
    draw(fig, input_data, n_cols, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
