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
    Framework, PaperMode, no_overlap_mask, assign_chunks,
)
from chopper.common.rocm_metrics import derive_duration


def _agg(df, group_arr, derive_cols=None):
    df_summed = df.groupby(group_arr, dropna=False).agg(
        {
            "dur": ["sum", "count"],
            "ts": ["first", "last"],
        }
    ).reset_index()
    df_summed.columns = [
        "_".join(col).strip("_") if col[1] != "sum" else col[0]
        for col in df_summed.columns
    ]
    if derive_cols is not None:
        for dc in derive_cols:
            dc(df_summed)
    return df_summed


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    variants: list[str] = ["default"],
    frameworks: list[Framework] = [Framework.FSDPv2],
):
    """Load and aggregate communication kernel durations.

    For each trace file, classifies kernels into all-gather, reduce-scatter,
    and "other" (non-overlap or all-reduce) and aggregates duration per
    (gpu, iteration). Normalizes by global maximum duration.

    Args:
        ts_files: List of paths to ts.pkl trace files
        variants: Variant labels (e.g. "FSDPv1-b1s4")
        frameworks: Framework type for each ts_file

    Returns:
        Dict mapping variant -> {"ag", "rs", "other"} -> aggregated DataFrame
    """
    data = {v: load_pickle(fn) for v, fn in zip(variants, ts_files)}
    overlap_data = {v: {} for v in variants}

    overlaps_keys = ("ag", "rs", "other")
    max_dur = 0
    for framework, setup in zip(frameworks, variants):
        df = data[setup]
        df["layer"] = df["layer"].fillna(-1)
        weird_mask = df["iteration"].isna()
        if weird_mask.any():
            assert df[~weird_mask]["ts"].min() > df[weird_mask]["ts"].max(), (
                "Nan iteration isn't at the start"
            )
            df = df[~weird_mask]
        df = df[~df["name"].isin((
            "Memcpy HtoD (Host -> Device)",
            "MEMORY_COPY_HOST_TO_DEVICE",
        ))]
        df = assign_chunks(df)
        overlap_mask = no_overlap_mask(df, framework=framework)

        allgather_mask = df["name_cpu_op"].str.endswith("allgather_base")
        allreduce_mask = df["name_cpu_op"].str.endswith("allreduce")
        reduce_scatter_mask = df["name_cpu_op"].str.endswith("reduce_scatter_base")
        vector_mask = (
            ~overlap_mask & ~df["name"].str.startswith("ncclDevKernel")
        ) | allreduce_mask

        overlaps = {
            "ag": allgather_mask,
            "rs": reduce_scatter_mask,
            "other": vector_mask,
        }
        for ov, ov_mask in overlaps.items():
            overlap_data[setup][ov] = _agg(
                df[ov_mask],
                ["gpu", "iteration"],
                derive_cols=(derive_duration,),
            )
            max_dur = max(max_dur, np.max(overlap_data[setup][ov]["Duration"]))

    for setup in variants:
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
    setups = tuple(data.keys())
    n_cols = max(1, len(setups) // 2)
    n_rows = 1

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

    rgb_colors = (rgb(0xD4, 0x11, 0x59), rgb(0x1A, 0x85, 0xFF))
    fsdp_versions = sorted(set(s.split("-")[0] for s in setups))
    violin_colors = {vc: rgb_colors[i] for i, vc in enumerate(fsdp_versions)}
    line_color_dict = {
        vc: (
            max(0, rgb_colors[i][0] - 0.3),
            max(0, rgb_colors[i][1] - 0.3),
            max(0, rgb_colors[i][2] - 0.3),
        )
        for i, vc in enumerate(fsdp_versions)
    }
    violin_alpha = 0.7

    comm_kerns = None
    for s in setups:
        keys = tuple(data[s].keys())
        if comm_kerns is None:
            comm_kerns = keys
        else:
            assert comm_kerns == keys

    y_ticks = np.arange(1, len(comm_kerns) + 1)
    for i, s in enumerate(setups):
        mod_param = "-".join(s.split("-")[1:])
        vendor = s.split("-")[0]
        ax = axs[0][i % n_cols]
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.set_yticks(y_ticks)
        ax.set_xticks((0.5,))
        ax.set_xlim((-0.05, 1.05))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="x", linestyle="-", alpha=0.5)
        ax.set_title(mod_param, pad=2, fontsize=8)

        ck_data = [data[s][c]["Duration"].values for c in comm_kerns]
        parts = ax.violinplot(
            ck_data, positions=y_ticks, vert=False,
            showmedians=True, widths=0.9,
        )
        for pc in parts["bodies"]:
            color = violin_colors[vendor]
            pc.set_alpha(violin_alpha)
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
        for line_type in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[line_type].set_color(
                tuple(line_color_dict[vendor] for _ in comm_kerns)
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

    legend_handles = [
        mpatches.Patch(color=violin_colors[v], label=v) for v in violin_colors.keys()
    ]
    legend_kwargs = dict(
        handles=legend_handles,
        loc="upper center",
        ncol=len(fsdp_versions),
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=0.5,
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
    ts_files: list[str] = ["./ts.pkl"],
    variants: list[str] = ["default"],
    frameworks: list[Framework] = [Framework.FSDPv2],
    paper_mode: PaperMode = PaperMode(),
    filename: str = "comm_violin.png",
):
    fig = Figure()
    input_data = get_data(ts_files, variants, frameworks)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
