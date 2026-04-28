"""Overlap-ratio vs. duration correlation per operator across frameworks.

For a chosen set of backward operators, plots median overlap ratio (left) and
median normalized duration (right) over kernel samples, with min/max and
quartile bands. Reports Pearson correlation between the two.
(Paper Figure: overlap_correlation.pdf)
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from chopper.common.colors import rgb
from chopper.common.load import get_overlap_df
from chopper.common.annotations import Framework, PaperMode


def _no_digits_formatter(x, _):
    return f"{x:.0f}"


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    variants: list[str] = ["default"],
    frameworks: list[Framework] = [Framework.FSDPv2],
    operators: list[str] = [
        "b_attn_n", "b_mlp_n", "b_mlp_up", "b_mlp_gp", "b_mlp_dp",
    ],
    iter_idxs: range = range(-5, -2, 1),
):
    """Load overlap data for the requested operators across variants.

    Normalizes 'elapsed' per operator by the minimum elapsed across variants.

    Args:
        ts_files: List of ts.pkl trace files
        variants: Variant labels
        frameworks: Framework type per ts_file
        operators: Operator names to include
        iter_idxs: Range of iteration indices to select

    Returns:
        Dict mapping variant -> {operator -> overlap DataFrame}
    """
    data = {}
    for ts_file, variant, framework in zip(ts_files, variants, frameworks):
        data[variant] = get_overlap_df(
            ts_file,
            framework=framework,
            iter_idxs=list(iter_idxs),
        )

    overlap_data = {v: {} for v in variants}
    vendor_min_elapsed = {v: {} for v in variants}

    for vendor in variants:
        for op in operators:
            op_mask = data[vendor]["operator-name"] == op
            sub = data[vendor][op_mask].copy()
            sub["op_idx"] = sub.groupby("gpu").cumcount()
            overlap_data[vendor][op] = sub
            vendor_min_elapsed[vendor][op] = np.min(sub["elapsed"])

    for op in operators:
        global_min = min(vendor_min_elapsed[v][op] for v in variants)
        for vendor in variants:
            overlap_data[vendor][op]["elapsed"] /= global_min

    return overlap_data


def draw(
    fig: Figure,
    input_data,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw overlap-ratio vs. duration with Pearson correlation per operator.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Dict from get_data()
        paper_mode: PaperMode settings for publication-quality figures
    """
    data = input_data
    overlaps = None
    for vendor in data.keys():
        ops = tuple(data[vendor].keys())
        if overlaps is None:
            overlaps = ops
        else:
            assert overlaps == ops

    fig.clear()
    fig.patches.clear()
    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left, right=paper_mode.right,
            bottom=paper_mode.bottom, top=paper_mode.top,
            wspace=paper_mode.wspace, hspace=paper_mode.hspace,
        )

    n_rows = len(overlaps)
    n_cols = 2
    axses = tuple(
        tuple(fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1) for c in range(n_cols))
        for r in range(n_rows)
    )

    vendors = sorted(data.keys())
    rgb_colors = (rgb(0xD4, 0x11, 0x59), rgb(0x1A, 0x85, 0xFF))
    color_dict = {vc: rgb_colors[i] for i, vc in enumerate(vendors)}
    line_color_dict = {
        vc: (
            max(0, rgb_colors[i][0] - 0.3),
            max(0, rgb_colors[i][1] - 0.3),
            max(0, rgb_colors[i][2] - 0.3),
        )
        for i, vc in enumerate(vendors)
    }

    for j, ov in enumerate(overlaps):
        axs = axses[j]
        for i, vendor in enumerate(reversed(vendors)):
            overlap_ratio = (
                data[vendor][ov]
                .groupby("op_idx")["overlap_ratio"]
                .agg(["min", lambda x: x.quantile(0.25), "median",
                      lambda x: x.quantile(0.75), "max"])
                .reset_index()
            )
            elapsed = (
                data[vendor][ov]
                .groupby("op_idx")["elapsed"]
                .agg(["min", lambda x: x.quantile(0.25), "median",
                      lambda x: x.quantile(0.75), "max"])
                .reset_index()
            )
            overlap_ratio.columns = ["op_idx", "min", "q1", "median", "q3", "max"]
            elapsed.columns = ["op_idx", "min", "q1", "median", "q3", "max"]

            merged = overlap_ratio.merge(
                elapsed, on="op_idx", suffixes=("_overlap", "_elapsed")
            )
            merged["norm_overlap"] = (
                merged["median_overlap"] - merged["median_overlap"].min()
            ) / (merged["median_overlap"].max() - merged["median_overlap"].min())
            merged["norm_elapsed"] = (
                merged["median_elapsed"] - merged["median_elapsed"].min()
            ) / (merged["median_elapsed"].max() - merged["median_elapsed"].min())

            corr, _ = pearsonr(
                merged["norm_overlap"].to_numpy(), merged["norm_elapsed"].to_numpy()
            )

            axs[0].text(
                1.2 + i * 0.25, 0.98, f"{corr:.2f}",
                transform=axs[0].transAxes,
                ha="left", va="bottom",
                color=line_color_dict[vendor],
                fontsize=8, fontweight="bold",
            )

            marker = "." if (len(overlap_ratio) == 1 and len(elapsed) == 1) else None

            axs[0].plot(
                overlap_ratio["op_idx"], overlap_ratio["median"],
                linestyle="-", marker=marker, label=vendor,
                color=line_color_dict[vendor], linewidth=1.0,
            )
            axs[0].fill_between(
                overlap_ratio["op_idx"], overlap_ratio["min"], overlap_ratio["max"],
                alpha=0.2, color=color_dict[vendor], edgecolor=None,
            )
            axs[0].fill_between(
                overlap_ratio["op_idx"], overlap_ratio["q1"], overlap_ratio["q3"],
                alpha=0.5, color=color_dict[vendor], edgecolor=None,
            )

            axs[1].plot(
                elapsed["op_idx"], elapsed["median"],
                linestyle="-", marker=marker, label=vendor,
                color=line_color_dict[vendor], linewidth=1.0,
            )
            axs[1].fill_between(
                elapsed["op_idx"], elapsed["min"], elapsed["max"],
                alpha=0.3, color=color_dict[vendor], edgecolor=None,
            )
            axs[1].fill_between(
                elapsed["op_idx"], elapsed["q1"], elapsed["q3"],
                alpha=0.5, color=color_dict[vendor], edgecolor=None,
            )

        axs[0].yaxis.set_major_formatter(FuncFormatter(_no_digits_formatter))
        axs[0].grid(axis="y", linestyle="--", alpha=0.5)
        axs[0].grid(axis="x", linestyle="-", alpha=0.5)
        axs[1].grid(axis="y", linestyle="--", alpha=0.5)
        axs[1].grid(axis="x", linestyle="-", alpha=0.5)

        axs[0].tick_params(axis="y", pad=1)
        axs[1].tick_params(
            axis="y", which="both", left=False, right=True,
            labelright=True, labelleft=False, pad=1,
        )

        if j == len(overlaps) // 2:
            axs[0].set_ylabel("overlap ratio (%)", labelpad=1)
            axs[0].tick_params(axis="x", pad=1)
            axs[1].tick_params(axis="x", pad=1)
            axs[1].yaxis.set_label_position("right")
            axs[1].set_ylabel("norm duration", labelpad=6)
        if j == len(overlaps) - 1:
            axs[0].set_xlabel("kernel sample")
            axs[0].xaxis.set_label_coords(1.00, -0.525)
            axs[0].tick_params(axis="x", pad=1)
            axs[1].tick_params(axis="x", pad=1)
        else:
            axs[0].tick_params(axis="x", length=0)
            axs[1].tick_params(axis="x", length=0)
            axs[0].set_xticklabels([])
            axs[1].set_xticklabels([])

        axs[0].set_ylim(-5, 105)
        axs[0].text(
            0.25, 1.035, ov,
            transform=axs[0].transAxes,
            ha="center", va="bottom",
            fontsize=8, fontweight="bold",
        )

    legend_handles = [mpatches.Patch(color=color_dict[v], label=v) for v in vendors]
    legend_kwargs = dict(
        handles=legend_handles,
        loc="upper center",
        ncol=len(vendors),
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
    operators: list[str] = [
        "b_attn_n", "b_mlp_n", "b_mlp_up", "b_mlp_gp", "b_mlp_dp",
    ],
    iter_start: int = -5,
    iter_stop: int = -2,
    iter_step: int = 1,
    paper_mode: PaperMode = PaperMode(),
    filename: str = "overlap_correlation.png",
):
    fig = Figure()
    iter_idxs = range(iter_start, iter_stop, iter_step)
    input_data = get_data(ts_files, variants, frameworks, operators, iter_idxs)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
