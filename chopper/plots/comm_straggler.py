"""Per-GPU communication kernel duration across a single iteration.

Shows all-gather and reduce-scatter NCCL kernel durations indexed by
occurrence order. Highlights the gap between isolated (straggler) GPUs
and the rest (leaders), visualizing how stragglers inflate collective
wait times. (Paper Figure: comm_straggler.pdf)
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from chopper.common.colors import rgb
from chopper.common.cache import load_pickle
from chopper.common.annotations import (
    PaperMode, assign_chunks, fix_names,
)


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    iteration: int = -1,
):
    """Load NCCL comm kernel durations for a single iteration.

    Classifies NCCL kernels as all-gather or reduce-scatter, assigns a
    unified index sorted by timestamp, and returns per-GPU duration data.

    Args:
        ts_files: List of paths to ts.pkl trace files
        configs: Config labels
        iteration: Iteration index to select (negative indexes from end)

    Returns:
        Dict mapping config -> DataFrame with columns:
        gpu, kernel_idx, dur_ms, comm_type
    """
    data = {}
    for ts_file, config in zip(ts_files, configs):
        df = load_pickle(ts_file)
        df = df[~df["iteration"].isna()]
        df = assign_chunks(df)
        df = fix_names(df)

        iters = sorted(df["iteration"].unique())
        it_val = iters[iteration]
        it = df[df["iteration"] == it_val].copy()

        nccl = it[it["name"].str.startswith("ncclDevKernel", na=False)].copy()

        ag_mask = (
            nccl["operator-name"].str.contains("all_gather", na=False)
            & ~nccl["operator-name"].str.startswith("b_", na=False)
        )
        rs_mask = (
            nccl["operator-name"].str.contains("post_backward_reduce", na=False)
            | nccl["operator-name"].str.startswith("b_FSDP::pre_forward", na=False)
        )

        nccl.loc[ag_mask, "comm_type"] = "AG"
        nccl.loc[rs_mask, "comm_type"] = "RS"
        nccl = nccl[nccl["comm_type"].isin(["AG", "RS"])]

        nccl = nccl.sort_values(["gpu", "ts"])
        nccl["kernel_idx"] = nccl.groupby("gpu").cumcount()
        nccl["dur_ms"] = nccl["dur"] * 1e-6

        data[config] = nccl[["gpu", "kernel_idx", "dur_ms", "comm_type"]].copy()

    return data


def draw(
    fig: Figure,
    input_data,
    isolated_gpus: list[int] | None = None,
    isolated_name: str = "straggler",
    group_name: str = "leader",
    paper_mode: PaperMode = PaperMode(),
):
    """Draw comm kernel duration per index with leader/straggler bands.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data()
        isolated_gpus: GPU indices to isolate. If None, auto-selects the
            GPU with lowest mean comm duration.
        isolated_name: Legend label for isolated GPU(s)
        group_name: Legend label for non-isolated GPU group
        paper_mode: PaperMode settings for publication-quality figures
    """
    fig.clear()
    fig.patches.clear()

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    configs = list(input_data.keys())
    n_cols = len(configs)
    axs = [fig.add_subplot(1, n_cols, i + 1) for i in range(n_cols)]

    ag_color = rgb(0x4C, 0x72, 0xB0)
    rs_color = rgb(0xDD, 0x84, 0x52)

    for ci, config in enumerate(configs):
        ax = axs[ci]
        nccl = input_data[config]
        gpus = sorted(nccl["gpu"].unique())

        # Auto-detect straggler if not specified
        if isolated_gpus is None:
            mean_dur = nccl.groupby("gpu")["dur_ms"].mean()
            iso_gpus = [int(mean_dur.idxmin())]
        else:
            iso_gpus = list(isolated_gpus)

        group_gpus = [g for g in gpus if g not in iso_gpus]

        pivot = nccl.pivot_table(index="kernel_idx", columns="gpu", values="dur_ms")
        ref = nccl[nccl["gpu"] == gpus[0]].sort_values("kernel_idx")
        ct_per_idx = ref.set_index("kernel_idx")["comm_type"]

        for ct, color in [("AG", ag_color), ("RS", rs_color)]:
            mask = ct_per_idx == ct
            idxs = mask[mask].index

            # Group band
            g_min = pivot[group_gpus].min(axis=1)
            g_max = pivot[group_gpus].max(axis=1)
            g_med = pivot[group_gpus].median(axis=1)

            ax.fill_between(
                idxs, g_min[idxs], g_max[idxs],
                alpha=0.25, color=color, linewidth=0,
            )
            ax.plot(
                idxs, g_med[idxs],
                color=color, alpha=0.7, linewidth=1.0,
            )

            # Isolated band or line
            if len(iso_gpus) == 1:
                ax.plot(
                    idxs, pivot[iso_gpus[0]][idxs],
                    color=color, alpha=0.9, linewidth=1.0, linestyle="--",
                )
            else:
                i_min = pivot[iso_gpus].min(axis=1)
                i_max = pivot[iso_gpus].max(axis=1)
                i_med = pivot[iso_gpus].median(axis=1)

                ax.fill_between(
                    idxs, i_min[idxs], i_max[idxs],
                    alpha=0.2, color=color, linewidth=0,
                    hatch="//",
                )
                ax.plot(
                    idxs, i_med[idxs],
                    color=color, alpha=0.9, linewidth=1.0, linestyle="--",
                )

        ax.set_xlabel("comm kernel index", fontsize=8, labelpad=2)
        ax.set_xlim(0, len(ct_per_idx) - 1)
        ax.grid(axis="both", linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7, pad=2)

        if ci == 0:
            ax.set_ylabel("duration (ms)", fontsize=8, labelpad=2)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        if n_cols > 1:
            ax.set_title(config, pad=2, fontsize=8)

    # Legend above figure
    iso_label = isolated_name
    if isolated_gpus is None or len(iso_gpus) == 1:
        gpu_str = f" (GPU {iso_gpus[0]})"
    else:
        gpu_str = f" (GPUs {','.join(str(g) for g in iso_gpus)})"

    handles = [
        Line2D([0], [0], color=ag_color, linewidth=1.5, label="all-gather"),
        Line2D([0], [0], color=rs_color, linewidth=1.5, label="reduce-scatter"),
        Line2D([0], [0], color="gray", linewidth=1.0, label=f"{group_name} median"),
        mpatches.Patch(facecolor="gray", alpha=0.25, label=f"{group_name} min-max"),
        Line2D([0], [0], color="gray", linewidth=1.0, linestyle="--",
               label=f"{iso_label}{gpu_str}"),
    ]

    if len(iso_gpus) > 1:
        handles.append(
            mpatches.Patch(facecolor="gray", alpha=0.2, hatch="//",
                           label=f"{iso_label} min-max"),
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=7,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
        handlelength=1.5,
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
    iteration: int = -1,
    isolated_gpus: list[int] | None = None,
    isolated_name: str = "straggler",
    group_name: str = "leader",
    paper_mode: PaperMode = PaperMode(),
    figsize: tuple[float, float] = (7.16, 2.5),
    filename: str = "comm_straggler.pdf",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(ts_files, configs, iteration)
    draw(fig, input_data, isolated_gpus, isolated_name, group_name, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
