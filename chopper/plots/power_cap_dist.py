#!/usr/bin/env python3
"""Power cap distribution plot from sbatch log files.

This module parses sbatch job output files (.out) to extract GPU power cap
settings and visualize the distribution across GPUs and scenarios.
"""

import os
import re
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from chopper.common.colors import okabe_ito


def _get_final_powercaps(fn: str, config: str = 'b2s4') -> dict[int, int]:
    """Extract final power cap settings from sbatch log files.

    Parses log files to find power cap adjustment lines and extracts
    the final power cap value for each GPU.

    Args:
        fns: List of paths to sbatch output files
        config: Configuration string to search for (e.g., 'b2s4')

    Returns:
        Dict mapping GPU index to power cap value
    """
    lines: list[str] = []
    with open(fn, 'r') as fp:
        lines = fp.readlines()

    megaline = ''.join(lines)
    matches = re.findall(
        r'(\d+),(Successfully set power cap|Power cap is already set) to (\d+)', megaline)
    power_caps: dict[int, int] = {}
    for gpu, _, cap in matches:
        print(gpu, cap)
        power_caps[int(gpu)] = int(cap)
    return power_caps


def get_data(
    log_files: list[str] = ["training_log.txt"],
    scenarios: list[str] = ["default"],
    variants: list[int] = [750],
    config: str = "b2s4",
) -> tuple[dict[str, dict[int, dict[int, int]]], list[str], list[int]]:
    """Load power cap data from sbatch log directories.

    Scans directories for sbatch_fsdp_*.out files and extracts power cap
    settings for each GPU, scenario, and variant combination.

    Args:
        log_files: List of directory paths containing sbatch output files.
                  Format: one directory per scenario, each containing logs
                  for different variants (initial power cap values).
        scenarios: List of scenario names corresponding to each log_dir
        variants: List of variant values (e.g., initial power cap values)
        config: Configuration string to search for in logs (e.g., 'b2s4')

    Returns:
        Tuple containing:
            - powercaps: Dict[scenario][variant][gpu] -> power cap value
            - scenarios: List of scenario names
            - variants: List of variant values
    """
    powercaps: dict[str, dict[int, dict[int, int]]] = {}

    for scenario, file in zip(scenarios, log_files):
        powercaps[scenario] = {}

        for variant in variants:
            caps = _get_final_powercaps(file, config)
            powercaps[scenario][variant] = caps

    return powercaps, scenarios, variants


def draw(
    fig: Figure,
    input_data: tuple[dict[str, dict[int, dict[int, int]]], list[str], list[int]],
    normalize_to_local_max: bool = True,
    alpha: float = 1.0,
    point_size: float = 4.0,
    n_gpus: int = 8,
    y_max: list[float] = [float("inf")],
    y_min: list[float] = [float("-inf")],
) -> None:
    """Draw power cap distribution scatter plot.

    Creates a multi-panel scatter plot showing normalized power caps for each
    GPU across different scenarios and variants (initial power cap settings).

    Args:
        fig: Matplotlib Figure object to draw on
        input_data: Tuple from get_data() containing power cap data
        normalize_to_local_max: If True, normalize to per-variant max;
                                 if False, use global min
        alpha: Transparency of scatter points (0-1)
        point_size: Size of scatter points
        n_gpus: Number of GPUs to show in legend
        y_max: Maximum y-axis limits per row (default inf uses auto)
        y_min: Minimum y-axis limits per row (default -inf uses auto)
    """
    powercaps, scenarios, variants = input_data

    rgb_colors = tuple(okabe_ito.values())
    n_rows = 1
    n_cols = len(scenarios)

    fig.clear()

    if n_cols == 0:
        return

    axs = tuple(
        fig.add_subplot(n_rows, n_cols, j + 1) for j in range(n_cols)
    )
    if n_cols == 1:
        axs = (axs,) if not isinstance(axs, tuple) else axs

    gymin: float | None = None
    gymax: float | None = None

    for i, scenario in enumerate(scenarios):
        ax = axs[i]
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(scenario, fontsize=8, pad=1)
        ax.set_xticks(tuple(range(len(variants))))
        ax.set_xticklabels([str(v) for v in variants])
        ax.tick_params(axis='x', pad=1)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(axis='y', pad=1)

        if scenario not in powercaps:
            continue

        for j, variant in enumerate(variants):
            if variant not in powercaps[scenario]:
                continue
            caps = powercaps[scenario][variant]
            if not caps:
                continue

            local_norm = max(caps.values()) if normalize_to_local_max else min(caps.values())
            if local_norm == 0:
                continue

            tweak_off = 0.07
            total_tweak = tweak_off * len(caps) / 2
            for tweak, (gpu, cap) in enumerate(caps.items()):
                ax.scatter(
                    j + tweak * tweak_off - total_tweak,
                    cap / local_norm,
                    color=rgb_colors[gpu % len(rgb_colors)],
                    s=point_size,
                    alpha=alpha,
                )

        ylim = ax.get_ylim()
        if gymin is None:
            gymin = ylim[0]
        else:
            gymin = min(gymin, ylim[0])
        if gymax is None:
            gymax = ylim[1]
        else:
            gymax = max(gymax, ylim[1])

    # Apply consistent y-limits (single row, so use index 0)
    row_ymin = y_min[0] if len(y_min) > 0 else float("-inf")
    row_ymax = y_max[0] if len(y_max) > 0 else float("inf")
    if row_ymin != float("-inf") and row_ymax != float("inf"):
        for ax in axs:
            ax.set_ylim((row_ymin, row_ymax))
    elif gymin is not None and gymax is not None:
        for ax in axs:
            ax.set_ylim((gymin, gymax))

    # Hide y-tick labels for non-leftmost columns
    for col in range(1, n_cols):
        axs[col].tick_params(axis='y', length=0)
        axs[col].set_yticklabels([])

    # Labels
    axs[n_cols // 2].set_xlabel("initial power cap", labelpad=0)
    axs[0].set_ylabel("norm power cap", labelpad=2)

    # Legend
    legend_handles = [
        mpatches.Patch(color=rgb_colors[gpu % len(rgb_colors)], label=str(gpu))
        for gpu in range(n_gpus)
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=n_gpus,
        borderpad=0.17,
        handletextpad=0.4,
        columnspacing=0.6,
        handlelength=0.5,
        frameon=False,
    )
