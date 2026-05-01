"""Device timeline: instantaneous MFMA utilization, GPU frequency, HBM bandwidth.

Reads a device_merged.pkl produced by:
  python -m chopper.profile.merge --device-dir <dir> -p ts.pkl -o device_merged.pkl

Each counter group keeps its own runtime trace and samples. The plot
uses the group that collected the requested metric.
"""

import pickle
import re
import numpy as np
from loguru import logger
from matplotlib.figure import Figure

from chopper.common.rocm_metrics import (
    derive_tensor_util_rocm,
    derive_l2_fabric_read_bw,
    derive_l2_fabric_write_bw,
)

NOMINAL_FREQ_MHZ = 2100

# Map derived metric -> required raw counters
METRIC_COUNTERS = {
    "mfma_util": ["SQ_VALU_MFMA_BUSY_CYCLES", "GRBM_GUI_ACTIVE"],
    "gpu_freq_mhz": ["GRBM_GUI_ACTIVE"],
    "read_bw_gbs": ["TCC_BUBBLE", "TCC_EA0_RDREQ", "TCC_EA0_RDREQ_32B"],
    "write_bw_gbs": ["TCC_EA0_WRREQ_64B", "TCC_EA0_WRREQ"],
}

# Map panel name -> derived metric
PANEL_METRIC = {
    "mfma": "mfma_util",
    "freq": "gpu_freq_mhz",
    "read_bw": "read_bw_gbs",
    "write_bw": "write_bw_gbs",
}


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_group_for_metric(data, metric):
    """Find which group collected the counters needed for a metric."""
    needed = METRIC_COUNTERS[metric]
    c2g = data["counter_to_group"]
    groups = set()
    for c in needed:
        assert c in c2g, f"Counter {c} (needed for {metric}) not collected"
        groups.add(c2g[c])
    assert len(groups) == 1, (
        f"Metric {metric} requires counters from multiple groups {groups} -- not yet supported"
    )
    return groups.pop()


def _derive_metrics(piv, metric):
    """Derive a specific metric from raw counter deltas."""
    if metric == "mfma_util":
        derive_tensor_util_rocm(piv)
        piv.rename(columns={"Tensor Util": "mfma_util"}, inplace=True)
    elif metric == "gpu_freq_mhz":
        piv["gpu_freq_mhz"] = (piv["GRBM_GUI_ACTIVE"] / 8) / (piv["dt_ms"] * 1000)
    elif metric == "read_bw_gbs":
        piv["dur"] = piv["dt_ms"] * 1e6
        for c in ["TCC_BUBBLE", "TCC_EA0_RDREQ", "TCC_EA0_RDREQ_32B"]:
            piv[c + "_sum"] = piv[c]
        derive_l2_fabric_read_bw(piv)
        piv.rename(columns={"L2 Fabric Read Bandwidth": "read_bw_gbs"}, inplace=True)
        for c in ["TCC_BUBBLE", "TCC_EA0_RDREQ", "TCC_EA0_RDREQ_32B"]:
            piv.drop(columns=[c + "_sum"], inplace=True, errors="ignore")
        piv.drop(columns=["dur"], inplace=True, errors="ignore")
    elif metric == "write_bw_gbs":
        piv["dur"] = piv["dt_ms"] * 1e6
        for c in ["TCC_EA0_WRREQ_64B", "TCC_EA0_WRREQ"]:
            piv[c + "_sum"] = piv[c]
        derive_l2_fabric_write_bw(piv)
        piv.rename(columns={"L2 Fabric Write&Atomic Bandwidth": "write_bw_gbs"}, inplace=True)
        for c in ["TCC_EA0_WRREQ_64B", "TCC_EA0_WRREQ"]:
            piv.drop(columns=[c + "_sum"], inplace=True, errors="ignore")
        piv.drop(columns=["dur"], inplace=True, errors="ignore")


def _prepare_group(group_data, target_gpus, metric):
    """Build per-GPU sample time-series for one group."""
    samples = group_data["samples"]
    kernels = group_data["kernels"].copy()

    per_gpu = {}
    for gpu in target_gpus:
        gpu_samples = samples[samples["gpu"] == gpu].copy()
        gpu_samples = gpu_samples.sort_values("timestamp_ns").reset_index(drop=True)
        gpu_samples["dt_ms"] = gpu_samples["timestamp_ns"].diff() / 1e6
        gpu_samples = gpu_samples.iloc[1:].reset_index(drop=True)
        _derive_metrics(gpu_samples, metric)
        per_gpu[gpu] = gpu_samples

    # Classify kernels
    kernels["type"] = "other"
    kernels.loc[kernels["name"].str.contains("Cijk", na=False), "type"] = "gemm"
    kernels.loc[kernels["name"].str.contains("nccl", na=False, case=False), "type"] = "nccl"

    # Time normalization
    t0 = kernels["ts"].min()
    for gpu, s in per_gpu.items():
        s["t_sec"] = (s["timestamp_ns"] - t0) / 1e9
    kernels["t_sec_start"] = (kernels["ts"] - t0) / 1e9
    kernels["t_sec_end"] = ((kernels["ts"] + kernels["dur"]) - t0) / 1e9

    return per_gpu, kernels, t0


def get_data(
    device_files: list[str] = ["./device_merged.pkl"],
    target_gpu: int = -1,
    mode: str = "mfma",
):
    """Load merged device sampling data.

    target_gpu=-1 (default): aggregate all GPUs (median with min/max).
    target_gpu>=0: single GPU.
    """
    data = _load_pickle(device_files[0])

    # Determine which panels we need
    panels = []
    if mode in ("mfma", "both", "all"):
        panels.append("mfma")
    if mode in ("freq", "both", "all"):
        panels.append("freq")
    if mode in ("read_bw", "bw", "all"):
        panels.append("read_bw")
    if mode in ("write_bw", "bw", "all"):
        panels.append("write_bw")

    # Check all needed metrics come from the same group
    metrics = [PANEL_METRIC[p] for p in panels]
    group_ids = set()
    for m in metrics:
        group_ids.add(_find_group_for_metric(data, m))
    assert len(group_ids) == 1, (
        f"Metrics {metrics} span multiple counter groups {group_ids} -- not yet supported"
    )
    gi = group_ids.pop()

    group_data = data["groups"][gi]
    all_gpus = sorted(group_data["samples"]["gpu"].unique())
    if target_gpu >= 0:
        all_gpus = [target_gpu]

    per_gpu, kernels, t0 = _prepare_group(group_data, all_gpus, metrics[0])

    # Derive remaining metrics if multiple panels from same group
    for m in metrics[1:]:
        for gpu, s in per_gpu.items():
            _derive_metrics(s, m)

    return {
        "per_gpu": per_gpu,
        "kernels": kernels,
        "n_gpus": len(all_gpus),
        "panels": panels,
    }


def _get_window(kernels, samples, last_iters=0):
    """Get time window for plotting.

    last_iters > 0: use the last N annotated iterations.
    last_iters == 0: auto-detect steady-state from GEMM gaps.
    """
    if last_iters > 0:
        annotated = kernels[kernels["iteration"].notna()]
        assert len(annotated) > 0, "No annotated iterations in kernel trace"
        iters = sorted(annotated["iteration"].unique())
        selected = iters[-last_iters:]
        window = kernels[kernels["iteration"].isin(selected)]
        return window["t_sec_start"].min(), window["t_sec_end"].max()

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
    """Plot each GPU as a separate line with same color and alpha."""
    n_gpus = len(per_gpu)
    line_alpha = min(0.9, 3.0 / max(n_gpus, 1))
    for gpu, s in per_gpu.items():
        if metric not in s.columns:
            continue
        zoom = s[(s["t_sec"] >= t_start) & (s["t_sec"] <= t_end)]
        ax.plot(zoom["t_sec"], zoom[metric], linewidth=0.4,
                color=color, alpha=line_alpha)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)


def _classify_kernel_type(name, op_name):
    """Classify kernel into gemm/fa/nccl/vec."""
    if "nccl" in name.lower():
        return "nccl"
    if isinstance(op_name, str) and re.match(r".*_\wp$", op_name):
        return "gemm"
    if "fmha" in name.lower() or "mha_bwd" in name.lower():
        return "fa"
    return "vec"


def _classify_phase(op_name):
    """Classify operator-name into fwd/bwd/opt."""
    if not isinstance(op_name, str):
        return None
    if op_name.startswith("f_"):
        return "fwd"
    if op_name.startswith("b_"):
        return "bwd"
    if op_name.startswith("opt_") or op_name.startswith("Optimizer"):
        return "opt"
    return None


def draw(
    fig: Figure,
    input_data,
    mode: str = "mfma",
    show_kernels: bool = True,
    last_iters: int = 1,
):
    per_gpu = input_data["per_gpu"]
    kernels = input_data["kernels"]
    panels = input_data["panels"]

    first_samples = next(iter(per_gpu.values()))
    t_start, t_end = _get_window(kernels, first_samples, last_iters)

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
        elif panel == "read_bw":
            _plot_metric(ax, per_gpu, t_start, t_end,
                         "read_bw_gbs", "L2 Fabric Read BW (GB/s)", "purple")
        elif panel == "write_bw":
            _plot_metric(ax, per_gpu, t_start, t_end,
                         "write_bw_gbs", "L2 Fabric Write BW (GB/s)", "crimson")
        ax_idx += 1

    if show_kernels:
        n_gpus = input_data["n_gpus"]
        bar_alpha = min(0.6, 3.0 / max(n_gpus, 1))

        # Classify kernels in window
        wk = kernels[(kernels["t_sec_end"] > t_start) &
                      (kernels["t_sec_start"] < t_end)].copy()
        wk["ktype"] = [_classify_kernel_type(n, o) for n, o in zip(wk["name"], wk["operator-name"])]
        wk["phase"] = wk["operator-name"].apply(_classify_phase)

        # Phase background colors on ALL panels -- min/max span per phase
        phase_colors = {"fwd": "#2ecc71", "bwd": "#e74c3c", "opt": "#f39c12"}
        phase_labels = {"fwd": "Forward", "bwd": "Backward", "opt": "Optimizer"}
        for phase, color in phase_colors.items():
            phase_k = wk[wk["phase"] == phase]
            if len(phase_k) == 0:
                continue
            for ai, ax in enumerate(axes):
                ax.axvspan(phase_k["t_sec_start"].min(), phase_k["t_sec_end"].max(),
                           alpha=0.12, color=color,
                           label=phase_labels[phase] if ai == 0 else None)
        axes[0].legend(fontsize=7, loc="upper right")

        # Kernel type bars on bottom panel
        ax = axes[-1]
        ktype_config = {
            "gemm": (0.2, "steelblue"),
            "fa":   (0.4, "#2ca02c"),
            "vec":  (0.6, "#9467bd"),
            "nccl": (0.8, "orange"),
        }
        for ktype, (y, color) in ktype_config.items():
            kt = wk[wk["ktype"] == ktype]
            for _, row in kt.iterrows():
                ax.barh(y, row["t_sec_end"] - row["t_sec_start"],
                        left=row["t_sec_start"], height=0.15,
                        color=color, alpha=bar_alpha)

        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["GEMM", "FA", "Vec", "NCCL"], fontsize=8)
        ax.set_ylim(0.05, 0.95)
        ax.grid(True, alpha=0.3)

    n_gpus = len(per_gpu)
    axes[-1].set_xlabel("Time (seconds)")
    if n_gpus == 1:
        gpu_id = next(iter(per_gpu.keys()))
        fig.suptitle(f"Device Timeline (GPU {gpu_id})")
    else:
        fig.suptitle(f"Device Timeline ({n_gpus} GPUs)")


def main(
    device_files: list[str] = ["./device_merged.pkl"],
    target_gpu: int = -1,
    mode: str = "mfma",
    show_kernels: bool = True,
    last_iters: int = 1,
    figsize: tuple[float, float] = (12, 5),
    filename: str = "device_timeline.png",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(device_files, target_gpu, mode)
    draw(fig, input_data, mode, show_kernels, last_iters)
    fig.savefig(filename, dpi=150)
    logger.info(f"Saved {filename}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
