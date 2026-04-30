"""Device timeline: instantaneous MFMA utilization and GPU frequency.

Reads a device_ts.pkl produced by:
  python -m chopper.profile.merge --device-dir <dir> -o device_ts.pkl

Shows median MFMA utilization and/or GPU frequency across all GPUs with
min/max fill, plus GEMM/NCCL kernel bars from the matching counter group.
Auto-detects steady-state training window.
"""

import pickle
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from chopper.common.rocm_metrics import derive_tensor_util_rocm

NOMINAL_FREQ_MHZ = 2100


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _prepare_samples(data, rank):
    """Sum across HW dims, pivot, diff, derive metrics for one rank."""
    dfs = []
    for (gi, r), df in data["counter_samples"].items():
        if r == rank:
            dfs.append(df)
    if not dfs:
        return None
    counters = pd.concat(dfs, ignore_index=True)

    totals = counters.groupby(
        ["timestamp_ns", "counter_name"]
    )["counter_value"].sum().reset_index()

    piv = totals.pivot(
        index="timestamp_ns", columns="counter_name", values="counter_value"
    ).reset_index()
    piv.columns.name = None
    piv = piv.sort_values("timestamp_ns").reset_index(drop=True)

    for col in [c for c in piv.columns if c != "timestamp_ns"]:
        piv[col] = piv[col].diff()
    piv["dt_ms"] = piv["timestamp_ns"].diff() / 1e6
    piv = piv.iloc[1:]

    if "SQ_VALU_MFMA_BUSY_CYCLES" in piv.columns and "GRBM_GUI_ACTIVE" in piv.columns:
        derive_tensor_util_rocm(piv)
        piv.rename(columns={"Tensor Util": "mfma_util"}, inplace=True)

    if "GRBM_GUI_ACTIVE" in piv.columns:
        piv["gpu_freq_mhz"] = (piv["GRBM_GUI_ACTIVE"] / 8) / (piv["dt_ms"] * 1000)

    return piv


def _get_kernels(data, rank, counter_names):
    """Get kernel traces from the group that collected the given counters."""
    counter_to_group = data["counter_to_group"]
    # Find which group has these counters
    group = None
    for ctr in counter_names:
        if ctr in counter_to_group:
            group = counter_to_group[ctr]
            break
    if group is None:
        group = 0

    key = (group, rank)
    assert key in data["kernel_traces"], f"No kernel traces for group={group}, rank={rank}"
    kdf = data["kernel_traces"][key].copy()
    kdf["type"] = "other"
    kdf.loc[kdf["kernel_name"].str.contains("Cijk", na=False), "type"] = "gemm"
    kdf.loc[kdf["kernel_name"].str.contains("nccl", na=False, case=False), "type"] = "nccl"
    return kdf


def get_data(
    device_files: list[str] = ["./device_ts.pkl"],
    target_gpu: int = -1,
):
    """Load and prepare device sampling data.

    target_gpu=-1 (default): aggregate all GPUs (median with min/max).
    target_gpu>=0: single GPU.
    """
    data = _load_pickle(device_files[0])

    all_ranks = sorted(set(r for _, r in data["counter_samples"].keys()))
    if target_gpu >= 0:
        all_ranks = [target_gpu]

    # Determine which counters we have for kernel trace matching
    counter_names = list(data["counter_to_group"].keys())

    per_gpu = {}
    for rank in all_ranks:
        samples = _prepare_samples(data, rank)
        if samples is not None:
            per_gpu[rank] = samples

    assert per_gpu, "No counter samples found"

    # Kernel traces from all ranks
    all_kernels = []
    for rank in all_ranks:
        kdf = _get_kernels(data, rank, counter_names)
        kdf["gpu"] = rank
        all_kernels.append(kdf)
    kernels = pd.concat(all_kernels, ignore_index=True)

    t0 = min(s["timestamp_ns"].iloc[0] for s in per_gpu.values())
    for rank, s in per_gpu.items():
        s["t_sec"] = (s["timestamp_ns"] - t0) / 1e9
    kernels["t_sec_start"] = (kernels["start_ns"] - t0) / 1e9
    kernels["t_sec_end"] = (kernels["end_ns"] - t0) / 1e9

    return {"per_gpu": per_gpu, "kernels": kernels, "n_gpus": len(all_ranks)}


def _auto_window(kernels, samples):
    gemm = kernels[kernels["type"] == "gemm"]
    if len(gemm) < 100:
        return samples["t_sec"].iloc[0], samples["t_sec"].iloc[-1]
    sorted_t = np.sort(gemm["t_sec_start"].values)
    gaps = np.diff(sorted_t)
    big_gap_idx = np.where(gaps > 5.0)[0]
    if len(big_gap_idx) >= 1:
        return sorted_t[big_gap_idx[-1] + 1] - 0.5, sorted_t[-1] + 1.0
    return sorted_t[0] - 0.5, sorted_t[-1] + 1.0


def _plot_metric(ax, per_gpu, t_start, t_end, metric, ylabel, color, ylim=None):
    """Plot median with min/max fill for a metric across GPUs."""
    # Resample all GPUs onto a common time grid
    all_t = []
    for s in per_gpu.values():
        if metric in s.columns:
            zoom = s[(s["t_sec"] >= t_start) & (s["t_sec"] <= t_end)]
            all_t.extend(zoom["t_sec"].values)
    if not all_t:
        return
    t_grid = np.sort(np.unique(np.round(np.array(all_t), 4)))

    values = []
    for s in per_gpu.values():
        if metric not in s.columns:
            continue
        zoom = s[(s["t_sec"] >= t_start) & (s["t_sec"] <= t_end)]
        interp = np.interp(t_grid, zoom["t_sec"].values, zoom[metric].values,
                           left=np.nan, right=np.nan)
        values.append(interp)

    if not values:
        return
    values = np.array(values)
    median = np.nanmedian(values, axis=0)
    vmin = np.nanmin(values, axis=0)
    vmax = np.nanmax(values, axis=0)

    ax.plot(t_grid, median, linewidth=0.5, color=color, alpha=0.9)
    ax.fill_between(t_grid, vmin, vmax, alpha=0.15, color=color)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)


def draw(
    fig: Figure,
    input_data,
    mode: str = "mfma",
    show_kernels: bool = True,
):
    per_gpu = input_data["per_gpu"]
    kernels = input_data["kernels"]

    first_samples = next(iter(per_gpu.values()))
    t_start, t_end = _auto_window(kernels, first_samples)

    nccl_z = kernels[(kernels["type"] == "nccl") &
                      (kernels["t_sec_end"] > t_start) &
                      (kernels["t_sec_start"] < t_end)]
    gemm_z = kernels[(kernels["type"] == "gemm") &
                      (kernels["t_sec_end"] > t_start) &
                      (kernels["t_sec_start"] < t_end)]

    panels = []
    if mode in ("mfma", "both"):
        panels.append("mfma")
    if mode in ("freq", "both"):
        panels.append("freq")

    n_panels = len(panels) + (1 if show_kernels else 0)
    ratios = [3] * len(panels) + ([1] if show_kernels else [])
    axes = fig.subplots(n_panels, 1, sharex=True,
                         gridspec_kw={"height_ratios": ratios})
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    for panel in panels:
        ax = axes[ax_idx]
        if panel == "mfma":
            _plot_metric(ax, per_gpu, t_start, t_end,
                         "mfma_util", "MFMA Util (%)", "steelblue", (-5, 105))
        elif panel == "freq":
            _plot_metric(ax, per_gpu, t_start, t_end,
                         "gpu_freq_mhz", "GPU Frequency (MHz)", "green",
                         (-50, NOMINAL_FREQ_MHZ * 1.2))
            ax.axhline(y=NOMINAL_FREQ_MHZ, color="gray", linestyle="--",
                       alpha=0.3, label=f"{NOMINAL_FREQ_MHZ} MHz")
            ax.legend()
        ax_idx += 1

    if show_kernels:
        ax = axes[-1]
        n_gpus = input_data["n_gpus"]
        # Alpha scales with GPU count so overlapping bars get denser
        bar_alpha = min(0.6, 3.0 / max(n_gpus, 1))
        for _, row in nccl_z.iterrows():
            ax.barh(0.75, row["t_sec_end"] - row["t_sec_start"],
                    left=row["t_sec_start"], height=0.4,
                    color="orange", alpha=bar_alpha)
        for _, row in gemm_z.iterrows():
            ax.barh(0.25, row["t_sec_end"] - row["t_sec_start"],
                    left=row["t_sec_start"], height=0.4,
                    color="steelblue", alpha=bar_alpha)
        ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(["GEMM", "NCCL"])
        ax.set_ylim(-0.1, 1.2)
        ax.grid(True, alpha=0.3)

    n_gpus = len(per_gpu)
    axes[-1].set_xlabel("Time (seconds)")
    if n_gpus == 1:
        gpu_id = next(iter(per_gpu.keys()))
        fig.suptitle(f"Device Timeline (GPU {gpu_id})")
    else:
        fig.suptitle(f"Device Timeline (median across {n_gpus} GPUs)")


def main(
    device_files: list[str] = ["./device_ts.pkl"],
    target_gpu: int = -1,
    mode: str = "mfma",
    show_kernels: bool = True,
    figsize: tuple[float, float] = (18, 7),
    filename: str = "device_timeline.png",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(device_files, target_gpu)
    draw(fig, input_data, mode, show_kernels)
    fig.savefig(filename, dpi=150)
    logger.info(f"Saved {filename}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
