"""MFMA utilization CDF: GEMM-only vs overlapped with NCCL.

Reads a device_merged.pkl produced by:
  python -m chopper.profile.merge --device-dir <dir> -p ts.pkl -o device_merged.pkl

Classifies each 1ms sample interval during GEMM kernel execution as
solo (no concurrent NCCL) or overlapped (NCCL running simultaneously),
then plots the CDF of MFMA utilization for each group.
"""

import pickle
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from chopper.common.rocm_metrics import derive_tensor_util_rocm


def get_data(
    device_files: list[str] = ["./device_merged.pkl"],
    target_gpu: int = 0,
):
    with open(device_files[0], "rb") as f:
        data = pickle.load(f)

    # Find group with MFMA counters
    c2g = data["counter_to_group"]
    assert "SQ_VALU_MFMA_BUSY_CYCLES" in c2g, "No MFMA counters collected"
    gi = c2g["SQ_VALU_MFMA_BUSY_CYCLES"]
    group = data["groups"][gi]

    samples = group["samples"]
    kernels = group["kernels"]

    gpu_samples = samples[samples["gpu"] == target_gpu].copy()
    gpu_samples = gpu_samples.sort_values("timestamp_ns").reset_index(drop=True)
    gpu_kernels = kernels[kernels["gpu"] == target_gpu].copy()

    # Derive MFMA util
    derive_tensor_util_rocm(gpu_samples)
    gpu_samples.rename(columns={"Tensor Util": "mfma_util"}, inplace=True)

    # Classify kernels
    gpu_kernels["type"] = "other"
    gpu_kernels.loc[gpu_kernels["name"].str.contains("Cijk", na=False), "type"] = "gemm"
    gpu_kernels.loc[gpu_kernels["name"].str.contains("nccl", na=False, case=False), "type"] = "nccl"

    gemm = gpu_kernels[gpu_kernels["type"] == "gemm"]
    nccl = gpu_kernels[gpu_kernels["type"] == "nccl"]

    ts = gpu_samples["timestamp_ns"].values
    gemm_active = np.zeros(len(gpu_samples), dtype=bool)
    nccl_active = np.zeros(len(gpu_samples), dtype=bool)

    for _, row in gemm.iterrows():
        gemm_active |= (ts >= row["ts"]) & (ts <= row["ts"] + row["dur"])
    for _, row in nccl.iterrows():
        nccl_active |= (ts >= row["ts"]) & (ts <= row["ts"] + row["dur"])

    return {
        "gemm_only": gpu_samples.loc[gemm_active & ~nccl_active, "mfma_util"].values,
        "gemm_nccl": gpu_samples.loc[gemm_active & nccl_active, "mfma_util"].values,
    }


def draw(
    fig: Figure,
    input_data,
):
    gemm_only = input_data["gemm_only"]
    gemm_nccl = input_data["gemm_nccl"]

    ax = fig.subplots(1, 1)

    for data, label, color in [
        (gemm_only, f"GEMM only (n={len(gemm_only)})", "steelblue"),
        (gemm_nccl, f"GEMM + NCCL overlap (n={len(gemm_nccl)})", "red"),
    ]:
        if len(data) == 0:
            continue
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, linewidth=2, color=color, label=label)

    ax.set_xlabel("Instantaneous MFMA Utilization (%)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if len(gemm_only):
        med = np.median(gemm_only)
        ax.axvline(med, color="steelblue", linestyle="--", alpha=0.5)
        ax.text(med + 1, 0.45, f"median={med:.1f}%", color="steelblue", fontsize=10)
    if len(gemm_nccl):
        med = np.median(gemm_nccl)
        ax.axvline(med, color="red", linestyle="--", alpha=0.5)
        ax.text(med + 1, 0.55, f"median={med:.1f}%", color="red", fontsize=10)

    fig.suptitle("MFMA Utilization CDF: GEMM-only vs Overlapped with NCCL",
                 fontsize=13)


def main(
    device_files: list[str] = ["./device_merged.pkl"],
    target_gpu: int = 0,
    figsize: tuple[float, float] = (10, 6),
    filename: str = "device_cdf.png",
):
    fig = Figure(figsize=figsize)
    input_data = get_data(device_files, target_gpu)
    draw(fig, input_data)
    fig.savefig(filename, dpi=150)
    logger.info(f"Saved {filename}")

    gemm_only = input_data["gemm_only"]
    gemm_nccl = input_data["gemm_nccl"]
    if len(gemm_only) and len(gemm_nccl):
        deg = (1 - np.median(gemm_nccl) / np.median(gemm_only)) * 100
        logger.info(f"MFMA degradation during overlap: {deg:.1f}%")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
