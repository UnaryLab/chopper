"""
Stroboscopic overlay of derived read/write bytes from device counters.

Works with output from: python -m chopper.profile.collect --device

Usage:
    python plot_counters.py --reads-dir output/reads --writes-dir output/writes [--save]
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MARGIN_NS = 1_000_000  # 1ms before and after kernel


def pivot_counters(counters):
    idx = ["timestamp_ns", "DIMENSION_INSTANCE", "DIMENSION_XCC", "agent_id"]
    piv = counters.pivot_table(
        index=idx, columns="counter_name", values="counter_value", aggfunc="first"
    ).reset_index()
    piv.columns.name = None
    return piv


def load_chopper_dir(d):
    """Load kernels + counters from a chopper --device output directory."""
    d = Path(d)
    groups = sorted(d.glob("chopper_device_counters*"))
    assert groups, f"No chopper_device_counters* dirs found in {d}"
    kernels = pd.read_csv(groups[0] / "kernel_traces.csv")
    counters = pd.concat(
        [pd.read_csv(g / "counter_samples.csv") for g in groups],
        ignore_index=True,
    )
    return kernels, pivot_counters(counters)


def stroboscopic_overlay(kernels, data, col, ax, color, skip_frac, algo_bytes):
    gemm = kernels[kernels["kernel_name"].str.contains("Cijk", na=False)]
    gemm = gemm.sort_values("start_ns").iloc[int(len(gemm) * skip_frac):]

    all_t, all_d, kernel_totals = [], [], []

    for _, inst in data.groupby(["DIMENSION_INSTANCE", "DIMENSION_XCC"]):
        inst = inst.sort_values("timestamp_ns")
        ts = inst["timestamp_ns"].values
        vals = inst[col].values.astype(float)

        for ki, (_, k) in enumerate(gemm.iterrows()):
            start, end, dur = k["start_ns"], k["end_ns"], k["end_ns"] - k["start_ns"]

            before = np.where(ts < start - MARGIN_NS)[0]
            if len(before) == 0:
                continue
            base = vals[before[-1]]

            win = np.where((ts >= start - MARGIN_NS) & (ts <= end + MARGIN_NS))[0]
            if len(win) == 0:
                continue

            all_t.extend((ts[win] - start) / dur)
            all_d.extend(vals[win] - base)

            while len(kernel_totals) <= ki:
                kernel_totals.append(0)
            kernel_totals[ki] += vals[win[-1]] - base

    ax.scatter(all_t, all_d, s=3, alpha=0.1, color=color)
    ax.set_ylabel(f"{col}\n(delta from baseline)", fontsize=10)
    ax.grid(True, alpha=0.3)

    if kernel_totals:
        med = np.median(kernel_totals)
        ratio = med / algo_bytes if algo_bytes else 0
        ax.set_title(f"Sum across dims: {med:,.0f} bytes "
                     f"({med / 1e6:,.1f} MB, {ratio:.2f}x algorithmic)",
                     fontsize=10, loc="left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reads-dir", required=True,
                   help="Chopper --device output dir for read counters")
    p.add_argument("--writes-dir", required=True,
                   help="Chopper --device output dir for write counters")
    p.add_argument("--skip-fraction", type=float, default=0.5)
    p.add_argument("-m", type=int, default=8192)
    p.add_argument("-n", type=int, default=8192)
    p.add_argument("-k", type=int, default=8192)
    p.add_argument("--save", action="store_true")
    a = p.parse_args()

    algo_rd = (a.m * a.k + a.n * a.k) * 2
    algo_wr = (a.m * a.n) * 2

    k1, g1 = load_chopper_dir(a.reads_dir)
    g1["read_bytes"] = (128 * g1["TCC_BUBBLE"]
                        + 64 * (g1["TCC_EA0_RDREQ"] - g1["TCC_BUBBLE"]
                                - g1["TCC_EA0_RDREQ_32B"])
                        + 32 * g1["TCC_EA0_RDREQ_32B"])

    k2, g2 = load_chopper_dir(a.writes_dir)
    g2["write_bytes"] = (64 * g2["TCC_EA0_WRREQ_64B"]
                         + 32 * (g2["TCC_EA0_WRREQ"] - g2["TCC_EA0_WRREQ_64B"]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    stroboscopic_overlay(k1, g1, "read_bytes", axes[0], "steelblue",
                         a.skip_fraction, algo_rd)
    stroboscopic_overlay(k2, g2, "write_bytes", axes[1], "darkorange",
                         a.skip_fraction, algo_wr)

    axes[-1].set_xlabel("Normalized time within kernel [0 = start, 1 = end]")
    fig.suptitle("Stroboscopic overlay: read/write bytes, all kernels on [0,1]",
                 fontsize=13)
    fig.tight_layout()

    if a.save:
        fig.savefig("plot_bytes_stitched.png", dpi=150)
        print("Saved to plot_bytes_stitched.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
