"""
Per-kernel MFMA utilization: computes MFMA util for each kernel dispatch
by taking counter deltas across the kernel's duration.

Usage:
    python plot_mfma_per_kernel.py \
        --traces mfma_run/kernel_traces_rank0.csv \
        --counters mfma_run/counter_samples_rank0.csv [--save]
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

    print(f"Kernels: {len(k)}, Counter samples: {len(c)}")

    # Sum across HW dimensions per timestamp
    totals = c.groupby(['timestamp_ns', 'counter_name'])['counter_value'].sum().reset_index()
    piv = totals.pivot(index='timestamp_ns', columns='counter_name',
                       values='counter_value').reset_index()
    piv = piv.sort_values('timestamp_ns').reset_index(drop=True)
    ts = piv['timestamp_ns'].values
    mfma_vals = piv['SQ_VALU_MFMA_BUSY_CYCLES'].values
    grbm_vals = piv['GRBM_GUI_ACTIVE'].values

    # Classify kernels
    k['type'] = 'other'
    k.loc[k['kernel_name'].str.contains('Cijk', na=False), 'type'] = 'gemm'
    k.loc[k['kernel_name'].str.contains('nccl', na=False, case=False), 'type'] = 'nccl'

    # Check NCCL overlap for each kernel
    nccl = k[k['type'] == 'nccl']

    # Per-kernel MFMA util
    results = []
    for _, row in k.iterrows():
        start, end = row['start_ns'], row['end_ns']

        # Baseline: last sample before kernel start
        before = np.searchsorted(ts, start, side='left') - 1
        if before < 0:
            continue
        # After: first sample after kernel end
        after = np.searchsorted(ts, end, side='right')
        if after >= len(ts):
            continue

        d_mfma = mfma_vals[after] - mfma_vals[before]
        d_grbm = grbm_vals[after] - grbm_vals[before]

        denom = d_grbm / N_XCD * N_CU * N_SIMD
        util = 100 * d_mfma / denom if denom > 0 else 0

        # Check NCCL overlap
        has_nccl = ((nccl['start_ns'] < end) & (nccl['end_ns'] > start)).any()

        results.append({
            'start_ns': start,
            'end_ns': end,
            'dur_us': (end - start) / 1e3,
            'type': row['type'],
            'mfma_util': min(max(util, 0), 100),
            'overlapped': has_nccl,
            'name': row['kernel_name'][:60],
        })

    df = pd.DataFrame(results)
    t0 = df['start_ns'].min()
    df['t_sec'] = (df['start_ns'] - t0) / 1e9

    print(f"\nPer-kernel MFMA util computed for {len(df)} kernels")

    # Plot all kernels
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    gemm = df[df['type'] == 'gemm']
    gemm_solo = gemm[~gemm['overlapped']]
    gemm_overlap = gemm[gemm['overlapped']]
    nccl_k = df[df['type'] == 'nccl']
    other = df[df['type'] == 'other']

    axes[0].scatter(other['t_sec'], other['mfma_util'],
                    s=1, alpha=0.1, c='gray', label='other')
    axes[0].scatter(gemm_solo['t_sec'], gemm_solo['mfma_util'],
                    s=3, alpha=0.3, c='steelblue', label='GEMM (solo)')
    axes[0].scatter(gemm_overlap['t_sec'], gemm_overlap['mfma_util'],
                    s=3, alpha=0.3, c='red', label='GEMM (+ NCCL)')
    axes[0].scatter(nccl_k['t_sec'], nccl_k['mfma_util'],
                    s=2, alpha=0.2, c='orange', label='NCCL')
    axes[0].set_ylabel('Per-Kernel MFMA Util (%)')
    axes[0].set_ylim(-5, 105)
    axes[0].legend(markerscale=5, loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Bottom: kernel duration
    axes[1].scatter(gemm_solo['t_sec'], gemm_solo['dur_us'],
                    s=2, alpha=0.3, c='steelblue')
    axes[1].scatter(gemm_overlap['t_sec'], gemm_overlap['dur_us'],
                    s=2, alpha=0.3, c='red')
    axes[1].set_ylabel('Kernel Duration (us)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Per-Kernel MFMA Utilization', fontsize=13)
    fig.tight_layout()

    if a.save:
        fig.savefig('mfma_per_kernel.png', dpi=150)
        print("Saved mfma_per_kernel.png")
    else:
        plt.show()

    # Stats
    print("\nGEMM kernel MFMA util:")
    print(f"  Solo:       n={len(gemm_solo):5d}  "
          f"median={gemm_solo['mfma_util'].median():.1f}%  "
          f"mean={gemm_solo['mfma_util'].mean():.1f}%")
    print(f"  Overlapped: n={len(gemm_overlap):5d}  "
          f"median={gemm_overlap['mfma_util'].median():.1f}%  "
          f"mean={gemm_overlap['mfma_util'].mean():.1f}%")
    if len(gemm_solo) and len(gemm_overlap):
        deg = (1 - gemm_overlap['mfma_util'].median() / gemm_solo['mfma_util'].median()) * 100
        print(f"  Degradation: {deg:.1f}%")


if __name__ == "__main__":
    main()
