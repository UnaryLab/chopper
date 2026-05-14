"""Overlap-duration correlation: fwd/bwd side-by-side.

For each operator, shows overlap ratio and normalized duration as time
series across layers and iterations. Forward (pink) and backward (blue)
are shown in adjacent columns for direct comparison.
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from scipy.stats import pearsonr

from chopper.common.load import get_overlap_df
from chopper.common.annotations import PaperMode, apply_paper_rcparams, paper_figsize


def get_data(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    operators: list[str] = [
        "attn_n", "mlp_n", "mlp_gu", "qkv_re",
    ],
    iter_idxs: range | None = None,
):
    """Load overlap data for fwd/bwd variants of each operator.

    Args:
        ts_files: List of ts.pkl trace files
        configs: Config labels
        operators: Base operator names (without f_/b_ prefix)
        iter_idxs: Iteration indices to select (None = all)

    Returns:
        Dict mapping config -> {operator -> overlap DataFrame}
    """
    # Expand base names to f_/b_ variants
    all_ops = []
    for op in operators:
        all_ops.extend([f"f_{op}", f"b_{op}"])

    data = {}
    for ts_file, config in zip(ts_files, configs):
        data[config] = get_overlap_df(
            ts_file,
            iter_idxs=list(iter_idxs) if iter_idxs is not None else None,
        )

    overlap_data = {config: {} for config in configs}
    config_min_elapsed = {config: {} for config in configs}

    for config in configs:
        for op in all_ops:
            op_mask = data[config]["operator-name"] == op
            sub = data[config][op_mask].copy()
            sub = sub.sort_values(["gpu", "iteration", "layer"]).reset_index(drop=True)
            sub["op_idx"] = sub.groupby("gpu").cumcount()
            overlap_data[config][op] = sub
            config_min_elapsed[config][op] = np.min(sub["elapsed"]) if len(sub) > 0 else 1

    for op in all_ops:
        global_min = min(config_min_elapsed[config][op] for config in configs)
        for config in configs:
            if len(overlap_data[config][op]) > 0:
                overlap_data[config][op]["elapsed"] /= global_min

    return overlap_data, list(operators)


def draw(
    fig: Figure,
    input_data,
    paper_mode: PaperMode = PaperMode(),
):
    """Draw 4-column fwd/bwd overlap correlation.

    Columns: fwd_overlap, fwd_duration, bwd_overlap, bwd_duration.
    Rows: one per base operator.
    """
    data, base_ops = input_data
    params = list(data.keys())

    fig.clear()
    fig.patches.clear()

    fwd_color = (0.83, 0.07, 0.35)
    bwd_color = (0.20, 0.52, 1.0)

    n_rows = len(base_ops)
    gs_kwargs = dict(hspace=0.35, wspace=0.03)
    if paper_mode.enabled:
        gs_kwargs.update(left=paper_mode.left, right=paper_mode.right,
                         top=paper_mode.top, bottom=paper_mode.bottom)
    else:
        gs_kwargs.update(left=0.08, right=0.92, top=0.90, bottom=0.1)
    gs = fig.add_gridspec(n_rows, 4, **gs_kwargs)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(n_rows)]

    # First pass: compute global duration max for consistent y-axis
    dur_max = 1.0
    for base_op in base_ops:
        for prefix in ["f_", "b_"]:
            op = prefix + base_op
            for config in params:
                sub = data[config].get(op)
                if sub is None or len(sub) == 0:
                    continue
                dur_max = max(dur_max, sub.groupby("op_idx")["elapsed"].max().max())

    for ri, base_op in enumerate(base_ops):
        for prefix, color, col_offset in [("f_", fwd_color, 0), ("b_", bwd_color, 2)]:
            op = prefix + base_op
            dark = tuple(max(0, c - 0.3) for c in color)

            ax_ovr = axes[ri][col_offset]
            ax_dur = axes[ri][col_offset + 1]

            for ci, config in enumerate(reversed(params)):
                sub = data[config].get(op)
                if sub is None or len(sub) == 0:
                    continue

                overlap_stats = sub.groupby("op_idx")["overlap_ratio"].agg(
                    ["min", lambda x: x.quantile(0.25), "median",
                     lambda x: x.quantile(0.75), "max"]).reset_index()
                overlap_stats.columns = ["op_idx", "min", "q1", "median", "q3", "max"]

                elapsed_stats = sub.groupby("op_idx")["elapsed"].agg(
                    ["min", lambda x: x.quantile(0.25), "median",
                     lambda x: x.quantile(0.75), "max"]).reset_index()
                elapsed_stats.columns = ["op_idx", "min", "q1", "median", "q3", "max"]

                line_color = dark
                ax_ovr.plot(overlap_stats["op_idx"], overlap_stats["median"],
                           color=line_color, linewidth=1)
                ax_ovr.fill_between(overlap_stats["op_idx"], overlap_stats["min"],
                                   overlap_stats["max"], alpha=0.2, color=color)
                ax_ovr.fill_between(overlap_stats["op_idx"], overlap_stats["q1"],
                                   overlap_stats["q3"], alpha=0.5, color=color)

                ax_dur.plot(elapsed_stats["op_idx"], elapsed_stats["median"],
                           color=line_color, linewidth=1)
                ax_dur.fill_between(elapsed_stats["op_idx"], elapsed_stats["min"],
                                   elapsed_stats["max"], alpha=0.2, color=color)
                ax_dur.fill_between(elapsed_stats["op_idx"], elapsed_stats["q1"],
                                   elapsed_stats["q3"], alpha=0.5, color=color)

                # Correlation (nan if overlap is constant, e.g. all zeros)
                ovr_vals = overlap_stats["median"].values
                dur_vals = elapsed_stats["median"].values
                if np.std(ovr_vals) == 0 or np.std(dur_vals) == 0:
                    corr_str = "nan"
                else:
                    r = np.corrcoef(ovr_vals, dur_vals)[0, 1]
                    corr_str = f"{r:.2f}"
                ax_dur.set_title(corr_str, fontsize=8, color=dark, pad=1)

        # Op name above the row
        axes[ri][0].set_title(base_op, fontsize=8, pad=1)

        for c in range(4):
            ax = axes[ri][c]
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.tick_params(labelsize=8)
            if ri < n_rows - 1:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            if c in (0, 2):
                ax.set_ylim(-5, 105)
            else:
                ax.set_ylim(0.8, dur_max * 1.05)
            if c == 0:
                ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
            elif c == 3:
                ax.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
            else:
                ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

    # Y-axis labels centered vertically across all rows
    top_pos = axes[0][0].get_position()
    bot_pos = axes[-1][0].get_position()
    y_mid = (top_pos.y1 + bot_pos.y0) / 2
    left_x = axes[0][0].get_position().x0 - 0.09
    right_x = axes[0][3].get_position().x1 + 0.09
    fig.text(left_x, y_mid, "overlap (%)", ha="center", va="center",
             rotation=90, fontsize=8)
    fig.text(right_x, y_mid, "norm dur", ha="center", va="center",
             rotation=90, fontsize=8)

    fig.text(0.5, 0.02, "kernel sample", ha="center", fontsize=8)
    legend_kwargs = dict(
        handles=[
            mpatches.Patch(color=fwd_color, label="fwd"),
            mpatches.Patch(color=bwd_color, label="bwd"),
        ],
        loc="upper center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 0.99),
    )
    if paper_mode.legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = paper_mode.legend_bbox
    fig.legend(**legend_kwargs)



def main(
    ts_files: list[str] = ["./ts.pkl"],
    configs: list[str] = ["default"],
    operators: list[str] = [
        "attn_n", "mlp_n", "mlp_gu", "qkv_re",
    ],
    iter_start: int = 0,
    iter_stop: int = -1,
    iter_step: int = 1,
    ncol: int = 1,
    figsize_ratio: float = 4 / 7,
    left: float = 0.1, right: float = 0.9,
    bottom: float = 0.1, top: float = 0.9,
    wspace: float = 0.2, hspace: float = 0.3,
    legend_x: float = 0.5, legend_y: float = 0.99,
    figsize: tuple[float, float] = (7, 4),
    filename: str = "overlap_correlation.pdf",
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
    if iter_start == 0 and iter_stop == -1:
        iter_idxs = None
    else:
        iter_idxs = range(iter_start, iter_stop, iter_step)
    input_data = get_data(ts_files, configs, operators, iter_idxs)
    draw(fig, input_data, paper_mode)
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
