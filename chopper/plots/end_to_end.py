"""End-to-end throughput and time-per-token breakdown.

Per-config throughput (normalized) alongside time-per-token decomposed by
operator type (GEMM/FA/Vec) with fwd/bwd split, plus launch overhead.
"""

import re
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from chopper.common.colors import okabe_ito
from chopper.common.load import get_df
from chopper.common.annotations import PaperMode, apply_paper_rcparams, paper_figsize


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["b1s4"],
):
    """Load and pre-aggregate trace data for end-to-end visualization.

    Computes per-kernel launch overhead before grouping so that overhead
    reflects actual GPU idle gaps, not gaps between operator groups.

    Args:
        ts_files: List of paths to ts.pkl trace files
        configs: Config labels (e.g. "b1s4", "b2s8")

    Returns:
        Dict mapping config name to aggregated DataFrame
    """
    dfs = {}
    for ts_file, config in zip(ts_files, configs):
        # Load all kernels first to get iteration start timestamps
        all_df = get_df(ts_file, iter_idxs=None)
        iter_start = all_df.groupby(["gpu", "iteration"])["ts"].min()

        df = get_df(
            ts_file,
            iter_idxs=None,
            assign_chunks=True,
            assign_optype=True,
            remove_nan_chunks=True,
            remove_overlap=True,
            fix_names=True,
        )
        # Compute per-kernel launch overhead (intra-iteration only)
        df = df.sort_values(["gpu", "iteration", "ts"]).reset_index(drop=True)
        prev_end = (
            df.groupby(["gpu", "iteration"])["ts"].shift(1)
            + df.groupby(["gpu", "iteration"])["dur"].shift(1)
        )
        df["launch_overhead"] = np.maximum(0, df["ts"] - prev_end)
        # First compute kernel: gap from first kernel (including comm)
        first_idx = df.groupby(["gpu", "iteration"]).head(1).index
        for idx in first_idx:
            row = df.loc[idx]
            start_ts = iter_start.loc[(row["gpu"], row["iteration"])]
            df.loc[idx, "launch_overhead"] = max(0, row["ts"] - start_ts)

        # Now group, summing launch_overhead alongside dur/ts
        df = df.groupby(
            ["gpu", "chunk", "iteration", "layer",
             "operator-type", "operator-name", "name"],
            dropna=False,
        ).agg(
            dur=("dur", "sum"),
            dur_last=("dur", "last"),
            ts_first=("ts", "first"),
            ts_last=("ts", "last"),
            launch_overhead=("launch_overhead", "sum"),
        ).sort_values("ts_first").reset_index()

        dfs[config] = df
    return dfs


def draw(
    fig: Figure,
    input_data,
    norm_setup: str | None = None,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw 1x2: throughput (left), time-per-token breakdown (right).

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data() mapping variant -> DataFrame
        norm_setup: Variant name used to normalize throughput and durations
        paper_mode: PaperMode settings for publication-quality figures
    """
    dfs = input_data
    if norm_setup is None:
        norm_setup = next(iter(dfs.keys()))
    assert norm_setup in dfs.keys(), f"norm_setup '{norm_setup}' not in data"

    fig.clear()
    fig.patches.clear()

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    ax_tput, ax_tpt = fig.subplots(1, 2, gridspec_kw={"wspace": 0.05})
    params = list(dfs.keys())

    # ── Compute throughput ──
    tok_p_sec = {}
    for setup, df in dfs.items():
        batch_size = int(re.findall(r"b(\d+)", setup)[0])
        seq_len = int(re.findall(r"s(\d+)", setup)[0]) << 10
        tokens = batch_size * seq_len
        iter_time = (
            (df.groupby(["gpu", "iteration"])["dur"].sum()
             + df.groupby(["gpu", "iteration"])["launch_overhead"].sum())
            .groupby("iteration").max().median()
        )
        tok_p_sec[setup] = tokens / iter_time

    # ── Compute time-per-token components ──
    tpt = {}  # {config: {component: time_per_token}}
    for setup, df in dfs.items():
        batch_size = int(re.findall(r"b(\d+)", setup)[0])
        seq_len = int(re.findall(r"s(\d+)", setup)[0]) << 10
        tokens = batch_size * seq_len
        components = {}
        for chunk in ["fwd", "bwd"]:
            for optype in ["FA", "Vec", "GEMM"]:
                mask = (df["operator-type"] == optype) & (df["chunk"] == chunk)
                dur = df[mask].groupby(["gpu", "iteration"])["dur"].sum().median()
                components[f"{chunk}_{optype}"] = dur / tokens
        overhead = df.groupby(["gpu", "iteration"])["launch_overhead"].sum().median()
        components["overhead"] = overhead / tokens
        opt_dur = df[df["chunk"] == "opt"].groupby(["gpu", "iteration"])["dur"].sum().median()
        components["opt"] = opt_dur / tokens
        tpt[setup] = components

    # Find best config for normalization (lowest total time per token)
    tpt_norm_setup = min(params, key=lambda c: sum(tpt[c].values()))
    norm_total = sum(tpt[tpt_norm_setup].values())

    # ── Left panel: throughput ──
    x = np.arange(len(params))
    bar_width = 0.85
    colors = list(okabe_ito.values())

    ax_tput.set_title("throughput", pad=2, fontsize=8)
    for i, setup in enumerate(params):
        value = tok_p_sec[setup] / tok_p_sec[norm_setup] * 100
        ax_tput.bar(i, value, width=bar_width, color=colors[1],
                    linewidth=0, alpha=0.999)
        ax_tput.text(i, value / 2, f"{value:.0f}",
                     ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
    ax_tput.set_ylabel(f"norm to {norm_setup} (%)", labelpad=1)
    ax_tput.set_xticks(x)
    ax_tput.set_xticklabels(params)
    ax_tput.tick_params(axis="x", which="major", pad=1)
    ax_tput.tick_params(axis="y", which="major", pad=1)
    ax_tput.grid(axis="y", linestyle="--", alpha=0.5)
    ax_tput.spines["top"].set_visible(False)
    ax_tput.spines["right"].set_visible(False)
    ax_tput.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # ── Right panel: time per token breakdown ──
    split_categories = [
        ("GEMM", "fwd_GEMM", "bwd_GEMM", "#cc79a7"),
        ("FA", "fwd_FA", "bwd_FA", "#e69f00"),
        ("Vec", "fwd_Vec", "bwd_Vec", "#d55e00"),
    ]
    single_categories = [
        ("Optimizer", lambda r: r["opt"], "#f0e442"),
        ("Launch overhead", lambda r: r["overhead"], "#000000"),
    ]

    ax_tpt.set_title("time per token", pad=2, fontsize=8)
    total_width = 0.85
    bottom = np.zeros(len(params))

    for label, fwd_key, bwd_key, color in split_categories:
        fwd_vals = np.array([tpt[c][fwd_key] / norm_total * 100 for c in params])
        bwd_vals = np.array([tpt[c][bwd_key] / norm_total * 100 for c in params])
        total_vals = fwd_vals + bwd_vals

        for i in range(len(params)):
            fwd_frac = fwd_vals[i] / total_vals[i] if total_vals[i] > 0 else 0.5
            fwd_w = total_width * fwd_frac
            bwd_w = total_width * (1 - fwd_frac)

            ax_tpt.bar(x[i] - total_width / 2 + fwd_w / 2, total_vals[i],
                       fwd_w, bottom=bottom[i], color=color,
                       linewidth=0, alpha=0.999)
            ax_tpt.bar(x[i] + total_width / 2 - bwd_w / 2, total_vals[i],
                       bwd_w, bottom=bottom[i], color=color,
                       linewidth=0, alpha=0.7)

            if total_vals[i] > 6:
                mid_y = bottom[i] + total_vals[i] / 2
                ax_tpt.text(x[i] - total_width / 2 + fwd_w / 2, mid_y,
                            f"{fwd_vals[i]:.0f}", ha="center", va="center",
                            fontsize=6, color="white", fontweight="bold")
                ax_tpt.text(x[i] + total_width / 2 - bwd_w / 2, mid_y,
                            f"{bwd_vals[i]:.0f}", ha="center", va="center",
                            fontsize=6, color="white", fontweight="bold")

        bottom += total_vals

    for label, fn, color in single_categories:
        vals = np.array([fn(tpt[c]) / norm_total * 100 for c in params])
        ax_tpt.bar(x, vals, total_width, bottom=bottom, color=color,
                   linewidth=0, alpha=0.999)
        for i, v in enumerate(vals):
            if v > 4:
                ax_tpt.text(i, bottom[i] + v / 2, f"{v:.0f}",
                            ha="center", va="center", fontsize=6,
                            color="white", fontweight="bold")
        bottom += vals

    ax_tpt.set_ylabel(f"norm to {tpt_norm_setup} (%)", labelpad=1)
    ax_tpt.set_xticks(x)
    ax_tpt.set_xticklabels(params)
    ax_tpt.tick_params(axis="x", which="major", pad=1)
    ax_tpt.tick_params(axis="y", which="major", pad=1)
    ax_tpt.grid(axis="y", linestyle="--", alpha=0.3)
    ax_tpt.spines["top"].set_visible(False)
    ax_tpt.spines["left"].set_visible(False)
    ax_tpt.yaxis.set_label_position("right")
    ax_tpt.yaxis.tick_right()

    # ── Legend ──
    handles = []
    handles.append(mpatches.Patch(color=colors[1], label="throughput"))
    for label, _, _, color in split_categories:
        handles.append(mpatches.Patch(color=color, label=label))
    for label, _, color in single_categories:
        handles.append(mpatches.Patch(color=color, label=label))
    # Add fwd/bwd indicator
    handles.append(mpatches.Patch(facecolor="gray", alpha=0.999, label="opaque=fwd"))
    handles.append(mpatches.Patch(facecolor="gray", alpha=0.5, label="transparent=bwd"))

    legend_kwargs = dict(
        handles=handles,
        loc="upper center",
        ncol=(len(handles) + 1) // 2,
        frameon=False,
        fontsize=7,
        handlelength=0.8,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    if paper_mode.legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = paper_mode.legend_bbox
    fig.legend(**legend_kwargs)



def main(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["b1s4"],
    norm_setup: str | None = None,
    ncol: int = 1,
    figsize_ratio: float = 2.5 / 7.16,
    left: float = 0.1, right: float = 0.9,
    bottom: float = 0.1, top: float = 0.9,
    wspace: float = 0.2, hspace: float = 0.3,
    legend_x: float = 0.5, legend_y: float = 1.02,
    figsize: tuple[float, float] = (7.16, 2.5),
    filename: str = "end_to_end.pdf",
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
    draw(fig, input_data, norm_setup, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
