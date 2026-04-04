"""Overlap ratio vs kernel duration with straggler GPU highlighted.

Visualizes the relationship between communication-computation overlap ratio
and kernel duration, with the straggler GPU highlighted to show correlation
between overlap and stragging behavior. (Paper Figure 4)
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from chopper.common.colors import okabe_ito
from chopper.common.load import get_overlap_df
from chopper.common.annotations import PaperMode, Framework


def no_digits_formatter(x, _):
    """Format tick labels without decimal digits."""
    return f"{x:.0f}"


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    variants: list[str] = ["default"],
    frameworks: list[Framework] = [Framework.FSDPv2],
    operators: list[str] = ["b_attn_fa", "f_attn_op", "b_mlp_up"],
    iter_idxs: range = range(-5, -2, 1),
):
    """Load and process overlap data for straggler analysis.

    Computes overlap ratios and kernel durations for specified operators,
    normalized by minimum duration to highlight relative slowdowns.

    Args:
        ts_files: List of paths to ts.pkl trace files
        variants: List of variant names corresponding to each file
        frameworks: List of Framework types for each variant
        operators: List of operator names to analyze
        iter_idxs: Range of iteration indices to select

    Returns:
        Tuple containing:
            - overlap_data: Dict mapping (variant, operator) to processed DataFrames
            - variants: List of variant names
            - operators: List of operator names
    """
    overlap_data = {}

    for ts_file, variant, framework in zip(ts_files, variants, frameworks):
        data = get_overlap_df(
            ts_file,
            framework=framework,
            iter_idxs=list(iter_idxs),
        )

        min_dur = {}
        for op in operators:
            op_mask = data["operator-name"] == op
            if not op_mask.any():
                continue
            op_data = data[op_mask].copy()
            op_data["op_idx"] = op_data.groupby("gpu").cumcount()
            min_dur[op] = np.min(op_data["dur"])
            overlap_data[(variant, op)] = op_data

        # Normalize durations by minimum
        for op in operators:
            if (variant, op) in overlap_data:
                overlap_data[(variant, op)]["dur"] /= min_dur[op]

    return overlap_data, variants, operators


def draw(
    fig: Figure,
    input_data,
    straggler_gpu: int = 4,
    tags: list[str] = ["(min overlap)", "(varying overlap)", "(max overlap)"],
    paper_mode: PaperMode = PaperMode(),
):
    """Draw overlap ratio and normalized duration plots with straggler highlighted.

    Creates a multi-panel plot showing overlap ratio (left) and normalized
    duration (right) for each operator, with the straggler GPU highlighted
    in orange while other GPUs are shown in semi-transparent black.

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() containing overlap data
        straggler_gpu: GPU index to highlight as straggler (shown in orange)
        tags: Descriptive tags for each operator
        paper_mode: PaperMode settings for publication-quality figures
    """
    overlap_data, variants, operators = input_data

    n_rows = len(operators)
    n_cols = 2

    fig.clear()
    fig.patches.clear()

    if paper_mode.enabled:
        fig.subplots_adjust(
            left=paper_mode.left,
            right=paper_mode.right,
            bottom=paper_mode.bottom,
            top=paper_mode.top,
            wspace=paper_mode.wspace,
            hspace=paper_mode.hspace,
        )

    # Create axes grid: axs[row][col]
    axs = tuple(
        tuple(
            fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            for col in range(n_cols)
        )
        for row in range(n_rows)
    )

    for vi, variant in enumerate(variants):
        for j, op in enumerate(operators):
            key = (variant, op)
            if key not in overlap_data:
                continue

            data = overlap_data[key]
            axs_row = axs[j]

            # Create color mapping: straggler is orange, others are black
            gpus = sorted(data["gpu"].unique())
            # Reorder so straggler is first (for drawing order)
            if straggler_gpu in gpus:
                gpus_reordered = [straggler_gpu] + [g for g in gpus if g != straggler_gpu]
            else:
                gpus_reordered = gpus

            rgb_colors = tuple(
                okabe_ito["Orange"] if gpu == straggler_gpu else okabe_ito["Black"]
                for gpu in gpus_reordered
            )
            color_dict = {gpu: rgb_colors[i] for i, gpu in enumerate(gpus_reordered)}

            # Draw in reverse order so straggler is on top
            for gpu in reversed(gpus_reordered):
                gpu_df = data[data["gpu"] == gpu]
                is_straggler = gpu == straggler_gpu
                alpha = 0.99 if is_straggler else 0.3

                # Left: overlap ratio
                axs_row[0].plot(
                    gpu_df["op_idx"],
                    gpu_df["overlap_ratio"],
                    linestyle="-",
                    color=color_dict[gpu],
                    linewidth=1.0,
                    alpha=alpha,
                )

                # Right: normalized duration
                axs_row[1].plot(
                    gpu_df["op_idx"],
                    gpu_df["dur"],
                    linestyle="-",
                    color=color_dict[gpu],
                    linewidth=1.0,
                    alpha=alpha,
                )

            # Format left axis (overlap ratio)
            axs_row[0].yaxis.set_major_formatter(FuncFormatter(no_digits_formatter))
            axs_row[0].grid(axis="y", linestyle="--", alpha=0.5)
            axs_row[0].grid(axis="x", linestyle="-", alpha=0.5)
            axs_row[0].set_ylim(-10, 110)
            axs_row[0].tick_params(axis="y", pad=1)

            # Format right axis (normalized duration)
            axs_row[1].grid(axis="y", linestyle="--", alpha=0.5)
            axs_row[1].grid(axis="x", linestyle="-", alpha=0.5)
            axs_row[1].tick_params(
                axis="y", which="both", left=False, right=True, labelright=True, labelleft=False
            )
            axs_row[1].tick_params(axis="y", pad=1)

            # Labels
            if j == len(operators) // 2:
                axs_row[0].set_ylabel("overlap ratio (%)", labelpad=1)
                axs_row[1].yaxis.set_label_position("right")
                axs_row[1].set_ylabel("norm duration", labelpad=4)

            if j == len(operators) - 1:
                axs_row[0].set_xlabel("kernel sample")
                axs_row[0].xaxis.set_label_coords(1.00, -0.3)
                axs_row[0].tick_params(axis="x", pad=1)
                axs_row[1].tick_params(axis="x", pad=1)
            else:
                axs_row[0].tick_params(axis="x", length=0)
                axs_row[1].tick_params(axis="x", length=0)
                axs_row[0].set_xticklabels([])
                axs_row[1].set_xticklabels([])

            # Title with tag
            tag = tags[j] if j < len(tags) else ""
            axs_row[0].text(
                0.50,
                1.025,
                f"{op} {tag}",
                transform=axs_row[0].transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Add border around figure in paper mode
    if paper_mode.enabled:
        fig.patches.append(
            mpatches.Rectangle(
                (0, 0),
                1,
                1,
                transform=fig.transFigure,
                fill=False,
                edgecolor="black",
                linewidth=1,
                zorder=1000,
            )
        )


def main(
    ts_files: list[str] = ["./ts.pkl"],
    variants: list[str] = ["default"],
    frameworks: list[Framework] = [Framework.FSDPv2],
    operators: list[str] = ["b_attn_fa", "f_attn_op", "b_mlp_up"],
    iter_start: int = -5,
    iter_stop: int = -2,
    iter_step: int = 1,
    straggler_gpu: int = 4,
    tags: list[str] = ["(min overlap)", "(varying overlap)", "(max overlap)"],
    filename: str = "overlap_straggler_leader.png",
):
    fig = Figure()
    iter_idxs = range(iter_start, iter_stop, iter_step)
    input_data = get_data(ts_files, variants, frameworks, operators, iter_idxs)
    draw(fig, input_data, straggler_gpu, tags, PaperMode())
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
