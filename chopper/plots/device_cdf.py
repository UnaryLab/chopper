"""MFMA utilization CDF: GEMM-only vs overlapped with NCCL.

Reads a device_ts.pkl produced by:
  python -m chopper.profile.merge --device-dir <dir> -o device_ts.pkl

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


def _prepare(data, target_gpu):
    """Sum, pivot, diff, derive MFMA util for one GPU."""
    dfs = []
    for (gi, rank), df in data["counter_samples"].items():
        if rank == target_gpu:
            dfs.append(df)
    assert dfs, f"No counter samples for GPU {target_gpu}"
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
    piv = piv.iloc[1:]

    derive_tensor_util_rocm(piv)
    piv.rename(columns={"Tensor Util": "mfma_util"}, inplace=True)

    # Kernel traces from the group that collected MFMA counters
    counter_to_group = data["counter_to_group"]
    group = counter_to_group.get("SQ_VALU_MFMA_BUSY_CYCLES", 0)
    key = (group, target_gpu)
    assert key in data["kernel_traces"], f"No kernel traces for group={group}, rank={target_gpu}"
    kernels = data["kernel_traces"][key].copy()

    kernels["type"] = "other"
    kernels.loc[kernels["kernel_name"].str.contains("Cijk", na=False), "type"] = "gemm"
    kernels.loc[kernels["kernel_name"].str.contains("nccl", na=False, case=False), "type"] = "nccl"

    return piv, kernels


def get_data(
    device_files: list[str] = ["./device_ts.pkl"],
    target_gpu: int = 0,
):
    with open(device_files[0], "rb") as f:
        data = pickle.load(f)

    samples, kernels = _prepare(data, target_gpu)

    ts = samples["timestamp_ns"].values
    gemm = kernels[kernels["type"] == "gemm"]
    nccl = kernels[kernels["type"] == "nccl"]

    gemm_active = np.zeros(len(samples), dtype=bool)
    nccl_active = np.zeros(len(samples), dtype=bool)

    for _, row in gemm.iterrows():
        gemm_active |= (ts >= row["start_ns"]) & (ts <= row["end_ns"])
    for _, row in nccl.iterrows():
        nccl_active |= (ts >= row["start_ns"]) & (ts <= row["end_ns"])

    return {
        "gemm_only": samples.loc[gemm_active & ~nccl_active, "mfma_util"].values,
        "gemm_nccl": samples.loc[gemm_active & nccl_active, "mfma_util"].values,
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
    device_files: list[str] = ["./device_ts.pkl"],
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
