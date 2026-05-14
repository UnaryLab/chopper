"""Per-config communication operator duration distribution.

Violin plots of normalized durations for all-gather, reduce-scatter, and
"other" (vector / all-reduce) operations across configurations and
frameworks. (Paper Figure: comm_violin.pdf)
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from chopper.common.colors import rgb
from chopper.common.cache import load_pickle
from chopper.common.annotations import (
    PaperMode, apply_paper_rcparams, paper_figsize, no_overlap_mask, assign_chunks,
)
def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
):
    """Load and aggregate communication kernel durations.

    For each trace file, classifies kernels into all-gather, reduce-scatter,
    and "other" (non-overlap or all-reduce) and aggregates duration per
    (gpu, iteration). Normalizes by global maximum duration.

    Args:
        ts_files: List of paths to ts.pkl trace files
        configs: Config labels (e.g. "b1s4", "b2s8")

    Returns:
        Dict mapping config -> {"ag", "rs", "other"} -> aggregated DataFrame
    """
    data = {config: load_pickle(fn) for config, fn in zip(configs, ts_files)}
    overlap_data = {config: {} for config in configs}

    overlaps_keys = ("ag", "rs", "other")
    max_dur = 0
    for setup in configs:
        df = data[setup]
        df = df[df["iteration"].notna() & df["layer"].notna()].copy()
        last_iter = df["iteration"].max()
        df = df[df["iteration"] != last_iter]

        df = assign_chunks(df)
        overlap_mask = no_overlap_mask(df)
        nccl_mask = df["name"].str.startswith("ncclDevKernel", na=False)

        allgather_mask = nccl_mask & df["operator-name"].str.contains("all_gather", na=False)
        reduce_scatter_mask = nccl_mask & (
            df["operator-name"].str.contains("post_backward_reduce", na=False)
            | df["operator-name"].str.startswith("b_FSDP::pre_forward", na=False)
        )
        other_mask = ~overlap_mask & ~allgather_mask & ~reduce_scatter_mask

        overlaps = {
            "ag": allgather_mask,
            "rs": reduce_scatter_mask,
            "other": other_mask,
        }
        for ov, ov_mask in overlaps.items():
            ov_df = df[ov_mask].groupby(
                ["gpu", "iteration", "layer", "operator-name"], dropna=False
            ).agg(dur=("dur", "sum")).reset_index()
            ov_df["Duration"] = ov_df["dur"].astype(float) * 1e-6
            overlap_data[setup][ov] = ov_df
            max_dur = max(max_dur, ov_df["Duration"].max())

    for setup in configs:
        for ov in overlaps_keys:
            overlap_data[setup][ov]["Duration"] /= max_dur

    return overlap_data


def draw(
    fig: Figure,
    input_data,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw per-config communication-kernel duration violins.

    One row, one column per modulation parameter; per-vendor color identifies
    framework. Y-axis lists comm-kernel categories (ag, rs, other).

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data()
        paper_mode: PaperMode settings for publication-quality figures
    """
    data = input_data
    params = list(data.keys())
    n_cols = len(params)
    n_rows = 1

    fig.clear()
    fig.patches.clear()
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )
    else:
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.9)

    axs = tuple(
        tuple(fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1) for c in range(n_cols))
        for r in range(n_rows)
    )

    violin_color = rgb(0xD4, 0x11, 0x59)
    line_color = (
        max(0, violin_color[0] - 0.3),
        max(0, violin_color[1] - 0.3),
        max(0, violin_color[2] - 0.3),
    )
    violin_alpha = 0.7

    comm_kerns = None
    for s in params:
        keys = tuple(data[s].keys())
        if comm_kerns is None:
            comm_kerns = keys
        else:
            assert comm_kerns == keys

    y_ticks = np.arange(1, len(comm_kerns) + 1)
    for i, s in enumerate(params):
        ax = axs[0][i]
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.set_yticks(y_ticks)
        ax.set_xticks((0.5,))
        ax.set_xlim((-0.05, 1.05))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="x", linestyle="-", alpha=0.5)
        ax.set_title(s, pad=2, fontsize=8)

        ck_data = [data[s][c]["Duration"].values for c in comm_kerns]
        parts = ax.violinplot(
            ck_data, positions=y_ticks, vert=False,
            showmedians=True, widths=0.9,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(violin_alpha)
            pc.set_facecolor(violin_color)
            pc.set_edgecolor(violin_color)
        for line_type in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[line_type].set_color(
                tuple(line_color for _ in comm_kerns)
            )
            parts[line_type].set_linewidth(0.8)

    for col in range(n_cols):
        ax = axs[0][col]
        ax.tick_params(axis="x", pad=1)
        if col == 0:
            ax.set_yticklabels(comm_kerns)
            ax.tick_params(axis="y", pad=1)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

    axs[0][n_cols // 2].set_xlabel("norm dur", labelpad=0)



def main(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    ncol: int = 1,
    figsize_ratio: float = 1.8 / 7.16,
    left: float = 0.06, right: float = 0.98,
    bottom: float = 0.2, top: float = 0.9,
    wspace: float = 0.2, hspace: float = 0.3,
    legend_x: float = 0.5, legend_y: float = 1.0,
    figsize: tuple[float, float] = (7.16, 1.8),
    filename: str = "comm_violin.pdf",
):
    paper_mode = PaperMode(
        enabled=True, ncol=ncol, figsize_ratio=figsize_ratio,
        left=left, right=right, bottom=bottom, top=top,
        wspace=wspace, hspace=hspace,
        legend_bbox=(legend_x, legend_y),
    )
    apply_paper_rcparams()
    figsize = paper_figsize(paper_mode)
    fig = Figure(figsize=figsize)
    input_data = get_data(ts_files, configs)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
