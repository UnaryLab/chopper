"""
CDF of instantaneous MFMA utilization: overlapped vs non-overlapped with NCCL.

Only includes samples where a GEMM kernel is active.

Usage:
    python plot_mfma_cdf.py --traces kernel_traces_rank0.csv \
                            --counters counter_samples_rank0.csv [--save]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

N_XCD = 8
N_CU = 304
N_SIMD = 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True)
    p.add_argument("--counters", required=True)
    p.add_argument("--save", action="store_true")
    a = p.parse_args()

    k = pd.read_csv(a.traces)
    c = pd.read_csv(a.counters)

    # Sum across HW dimensions per timestamp
    totals = c.groupby(['timestamp_ns', 'counter_name'])['counter_value'].sum().reset_index()
    piv = totals.pivot(index='timestamp_ns', columns='counter_name',
                       values='counter_value').reset_index()
    piv = piv.sort_values('timestamp_ns').reset_index(drop=True)

    # Instantaneous deltas
    for col in ['SQ_VALU_MFMA_BUSY_CYCLES', 'GRBM_GUI_ACTIVE']:
        piv[f'd_{col}'] = piv[col].diff()
    piv = piv.iloc[1:]

    piv['mfma_util'] = (100 * piv['d_SQ_VALU_MFMA_BUSY_CYCLES']
                        / (piv['d_GRBM_GUI_ACTIVE'] / N_XCD * N_CU * N_SIMD))
    piv['mfma_util'] = piv['mfma_util'].clip(0, 100)

    # Classify kernels
    nccl = k[k['kernel_name'].str.contains('nccl', na=False, case=False)]
    gemm_k = k[k['kernel_name'].str.contains('Cijk', na=False)]

    # Mark each sample as GEMM-active and/or NCCL-active
    ts = piv['timestamp_ns'].values
    gemm_active = np.zeros(len(piv), dtype=bool)
    nccl_active = np.zeros(len(piv), dtype=bool)

    for _, row in gemm_k.iterrows():
        gemm_active |= (ts >= row['start_ns']) & (ts <= row['end_ns'])
    for _, row in nccl.iterrows():
        nccl_active |= (ts >= row['start_ns']) & (ts <= row['end_ns'])

    gemm_only_util = piv.loc[gemm_active & ~nccl_active, 'mfma_util'].values
    gemm_nccl_util = piv.loc[gemm_active & nccl_active, 'mfma_util'].values

    # Build CDFs
    fig, ax = plt.subplots(figsize=(10, 6))

    for data, label, color in [
        (gemm_only_util, f'GEMM only (n={len(gemm_only_util)})', 'steelblue'),
        (gemm_nccl_util, f'GEMM + NCCL overlap (n={len(gemm_nccl_util)})', 'red'),
    ]:
        if len(data) == 0:
            continue
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, linewidth=2, color=color, label=label)

    ax.set_xlabel('Instantaneous MFMA Utilization (%)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add median lines
    if len(gemm_only_util):
        med = np.median(gemm_only_util)
        ax.axvline(med, color='steelblue', linestyle='--', alpha=0.5)
        ax.text(med + 1, 0.45, f'median={med:.1f}%', color='steelblue', fontsize=10)
    if len(gemm_nccl_util):
        med = np.median(gemm_nccl_util)
        ax.axvline(med, color='red', linestyle='--', alpha=0.5)
        ax.text(med + 1, 0.55, f'median={med:.1f}%', color='red', fontsize=10)

    ax.set_title('MFMA Utilization CDF: GEMM-only vs Overlapped with NCCL', fontsize=13)
    fig.tight_layout()

    if a.save:
        fig.savefig('mfma_cdf.png', dpi=150)
        print("Saved mfma_cdf.png")
    else:
        plt.show()

    # Summary
    print(f"\nMFMA util during GEMM kernels:")
    if len(gemm_only_util):
        print(f"  GEMM only:   median={np.median(gemm_only_util):.1f}%  "
              f"mean={np.mean(gemm_only_util):.1f}%")
    if len(gemm_nccl_util):
        print(f"  GEMM + NCCL: median={np.median(gemm_nccl_util):.1f}%  "
              f"mean={np.mean(gemm_nccl_util):.1f}%")
    if len(gemm_only_util) and len(gemm_nccl_util):
        deg = (1 - np.median(gemm_nccl_util) / np.median(gemm_only_util)) * 100
        print(f"  Degradation: {deg:.1f}%")


if __name__ == "__main__":
    main()
