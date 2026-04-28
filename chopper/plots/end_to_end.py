"""End-to-end throughput, iteration duration, and per-phase breakdown.

Per-config throughput (normalized) alongside iteration time decomposed by
training chunk, plus forward/backward duration broken out by operator type
(GEMM/FA/Vec). Includes launch-overhead bubbles per chunk and per operator
type. (Paper Figure: end_to_end.pdf)
"""

import re
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from chopper.common.colors import okabe_ito
from chopper.common.load import get_df
from chopper.common.annotations import Framework, PaperMode


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    variants: list[str] = ["default"],
    frameworks: list[Framework] = [Framework.FSDPv2],
):
    """Load and pre-aggregate trace data for end-to-end visualization.

    Loads each trace, removes communication-overlapped kernels, and aggregates
    by (gpu, chunk, iteration, layer, operator-type, operator-name, name) with
    summed durations and first/last timestamps.

    Args:
        ts_files: List of paths to ts.pkl trace files
        variants: Variant labels formatted as "<framework>-<config>" (e.g. "FSDPv1-b1s4")
        frameworks: Framework type for each ts_file

    Returns:
        Dict mapping variant name to aggregated DataFrame
    """
    dfs = {}
    for ts_file, variant, framework in zip(ts_files, variants, frameworks):
        df = get_df(
            ts_file,
            iter_idxs=None,
            assign_chunks=True,
            assign_optype=True,
            remove_nan_chunks=True,
            remove_overlap=True,
            fix_names=True,
            group_arr=[
                "gpu", "chunk", "iteration", "layer",
                "operator-type", "operator-name", "name",
            ],
            group_map={
                "dur": ["sum", "last"],
                "ts": ["first", "last"],
            },
            sort_value="ts_first",
            framework=framework,
        )
        dfs[variant] = df
    return dfs


def draw(
    fig: Figure,
    input_data,
    norm_setup: str = "FSDPv1-b1s4",
    paper_mode: PaperMode = PaperMode(),
):
    """Draw a 2x2 grid: throughput, iteration duration, fwd/bwd breakdown.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data() mapping variant -> DataFrame
        norm_setup: Variant name used to normalize throughput and durations
        paper_mode: PaperMode settings for publication-quality figures
    """
    dfs = input_data
    assert norm_setup in dfs.keys(), f"norm_setup '{norm_setup}' not in data"

    fig.clear()
    fig.patches.clear()

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    n_rows, n_cols = 2, 2
    axs = tuple(
        tuple(
            fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1)
            for c in range(n_cols)
        )
        for r in range(n_rows)
    )

    params = sorted(
        set(s.split("-")[1] for s in dfs.keys()),
        key=lambda x: re.findall(r"\d+", x)[::-1],
    )
    variants = sorted(set(s.split("-")[0] for s in dfs.keys()))

    chunks = None
    for df in dfs.values():
        cur = tuple(sorted(df["chunk"].unique()))
        if chunks is None:
            chunks = cur
        else:
            assert chunks == cur
    chunks = tuple(
        sorted(chunks, key=lambda x: 0 if x[0] == "f" else 1 if x[0] == "b" else 2)
    )

    def optype_key(x):
        return 0 if x == "FA" else 1 if x == "Vec" else 2 if x == "GEMM" else 3

    optypes = None
    for df in dfs.values():
        cur = tuple(sorted(df["operator-type"].unique(), key=optype_key))
        if optypes is None:
            optypes = cur
        else:
            assert optypes == cur

    bars = ("launch overhead", "throughput") + chunks + optypes
    colors = tuple(okabe_ito.values())
    bar_color = {bar: colors[i] for i, bar in enumerate(bars)}
    hatches = (None, r"\\\\\\\\")
    bar_hatch = {m: hatches[i] for i, m in enumerate(variants)}

    tok_p_sec = {}
    chunk_time = {}
    bubble_time = {}
    for setup in dfs.keys():
        df = dfs[setup]
        df = df.sort_values(["gpu", "iteration", "ts_first"]).reset_index(drop=True)
        prev_end_time = (
            df.groupby(["gpu", "iteration"])["ts_last"].shift(1)
            + df.groupby(["gpu", "iteration"])["dur_last"].shift(1)
        )
        df["launch_overhead"] = np.maximum(0, df["ts_first"] - prev_end_time)
        df.loc[df.groupby(["gpu", "iteration"]).head(1).index, "launch_overhead"] = 0
        dfs[setup] = df

        param = setup.split("-")[1]
        batch_size = int(re.findall(r"b(\d+)", param)[0])
        seq_len = int(re.findall(r"s(\d+)", param)[0]) << 10
        tokens = batch_size * seq_len

        iter_time = (
            (
                dfs[setup].groupby(["gpu", "iteration"])["dur"].sum()
                + dfs[setup].groupby(["gpu", "iteration"])["launch_overhead"].sum()
            )
            .groupby("iteration")
            .max()
            .median()
        )
        tok_p_sec[setup] = tokens / iter_time

        chunk_time_ = (
            dfs[setup]
            .groupby(["gpu", "iteration", "chunk"])
            .agg(
                ts_first=("ts_first", "first"),
                ts_last=("ts_last", "last"),
                dur_last=("dur_last", "last"),
            )
        )
        chunk_time[setup] = (
            (chunk_time_["ts_last"] + chunk_time_["dur_last"] - chunk_time_["ts_first"])
            .groupby("chunk")
            .median()
            .to_dict()
        )
        bubble_time[setup] = (
            dfs[setup]
            .groupby(["gpu", "iteration", "chunk"])
            .agg(launch_overhead=("launch_overhead", "sum"))["launch_overhead"]
            .groupby("chunk")
            .median()
            .to_dict()
        )

    chunk_time_norm = sum(
        chunk_time[norm_setup][chunk] + bubble_time[norm_setup][chunk]
        for chunk in chunks
    )

    for col_idx in range(2):
        for row_idx in range(2):
            ax = axs[row_idx][col_idx]
            ax.yaxis.set_major_locator(MaxNLocator(nbins=(4 if row_idx == 0 else 5)))
            for setup, df in dfs.items():
                variant = setup.split("-")[0]
                param = setup.split("-")[1]
                tick = params.index(param)
                bar_width = 0.9 / len(variants)
                offset = (
                    -bar_width / 2 * (len(variants) - 1)
                    + bar_width * variants.index(variant)
                )
                ax.set_xticks(range(len(params)))
                ax.set_xticklabels(params)
                ax.set_xlim(-0.5, len(params) - 1 + 0.5)
                ax.tick_params(axis="y", pad=2)

                if col_idx == 0 and row_idx == 0:
                    ax.set_title("throughput", pad=2, fontsize=8)
                    value = tok_p_sec[setup] / tok_p_sec[norm_setup]
                    ax.bar(
                        tick + offset,
                        value,
                        width=bar_width * 0.9,
                        bottom=0,
                        color=bar_color["throughput"],
                        alpha=0.999,
                        linewidth=0,
                        hatch=bar_hatch[variant],
                    )
                    ax.set_ylabel("norm", labelpad=1)
                    ax.grid(axis="y", linestyle="--", alpha=0.5)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_xticklabels([])
                    ax.tick_params(axis="x", length=0)

                elif col_idx == 1 and row_idx == 0:
                    ax.set_title("iteration duration", pad=2, fontsize=8)
                    bottom = 0
                    for chunk in chunks:
                        value = chunk_time[setup][chunk] / chunk_time_norm
                        ax.bar(
                            tick + offset, value,
                            width=bar_width * 0.9, bottom=bottom,
                            color=bar_color[chunk], linewidth=0, alpha=0.999,
                            hatch=bar_hatch[variant],
                        )
                        bottom += value
                        value = bubble_time[setup][chunk] / chunk_time_norm
                        ax.bar(
                            tick + offset, value,
                            width=bar_width * 0.9, bottom=bottom,
                            color=bar_color["launch overhead"],
                            linewidth=0, alpha=0.999, hatch=bar_hatch[variant],
                        )
                        bottom += value
                    ax.grid(axis="y", linestyle="--", alpha=0.5)
                    ax.spines["top"].set_visible(False)
                    ax.spines["left"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", length=0)
                    ax.set_xticklabels([])
                    ax.tick_params(axis="x", length=0)

                else:
                    is_fwd = row_idx == 1 and col_idx == 0
                    chunk_label = "fwd" if is_fwd else "bwd"
                    if col_idx == 0:
                        ax.set_ylabel("breakdown (%)", labelpad=1)
                    title = "forward duration" if is_fwd else "backward duration"
                    ax.set_title(title, pad=2, fontsize=8)

                    bottom = 0
                    ot_vals = {}
                    ot_bubbles = {}
                    for optype in optypes:
                        tmp_mask = (df["operator-type"] == optype) & (
                            df["chunk"] == chunk_label
                        )
                        ot_vals[optype] = (
                            df[tmp_mask].groupby(["gpu", "iteration"])["dur"].sum()
                        ).median()
                        ot_bubbles[optype] = (
                            df[tmp_mask]
                            .groupby(["gpu", "iteration"])["launch_overhead"]
                            .sum()
                        ).median()
                    stretch = sum(ot_vals.values()) + sum(ot_bubbles.values())
                    for optype in optypes:
                        bar_value = ot_vals[optype] / stretch * 100
                        ax.bar(
                            tick + offset, bar_value,
                            width=bar_width * 0.9, bottom=bottom,
                            color=bar_color[optype], linewidth=0, alpha=0.999,
                            hatch=bar_hatch[variant],
                        )
                        bottom += bar_value
                        bar_value = ot_bubbles[optype] / stretch * 100
                        ax.bar(
                            tick + offset, bar_value,
                            width=bar_width * 0.9, bottom=bottom,
                            color=bar_color["launch overhead"],
                            linewidth=0, alpha=0.999, hatch=bar_hatch[variant],
                        )
                        bottom += bar_value
                    ax.grid(axis="y", linestyle="--", alpha=0.5)
                    ax.tick_params(axis="x", pad=1)
                    ax.spines["top"].set_visible(False)
                    if col_idx == 1:
                        ax.spines["left"].set_visible(False)
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", length=0)
                    ax.spines["right"].set_visible(False)

    l0, u0 = axs[0][0].get_ylim()
    l1, u1 = axs[0][1].get_ylim()
    y_lim = (min(l0, l1), max(u0, u1))
    axs[0][0].set_ylim(y_lim)
    axs[0][1].set_ylim(y_lim)
    l0, u0 = axs[1][0].get_ylim()
    l1, u1 = axs[1][1].get_ylim()
    y_lim = (min(l0, l1), max(u0, u1))
    axs[1][0].set_ylim(y_lim)
    axs[1][1].set_ylim(y_lim)

    bar_handles = [mpatches.Patch(color=bar_color[bar], label=bar.lower()) for bar in bars]
    bar_handles.extend(
        [
            mpatches.Patch(
                facecolor="white", edgecolor="black", label=m, hatch=bar_hatch[m]
            )
            for m in variants
        ]
    )

    legend_kwargs = dict(
        handles=bar_handles,
        loc="upper center",
        ncol=(len(bars) + 2) // 2,
        labelspacing=0.2,
        handletextpad=0.4,
        columnspacing=1.4,
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
    norm_setup: str = "FSDPv1-b1s4",
    paper_mode: PaperMode = PaperMode(),
    filename: str = "end_to_end.png",
):
    fig = Figure()
    input_data = get_data(ts_files, variants, frameworks)
    draw(fig, input_data, norm_setup, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
