"""
Device timeline: instantaneous MFMA utilization and/or GPU frequency
during training, with NCCL overlap analysis.

Expects output from: python -m chopper.profile.collect --device
  --counters SQ_VALU_MFMA_BUSY_CYCLES GRBM_GUI_ACTIVE

Usage:
    python plot_device_timeline.py --traces kernel_traces_rank0.csv \
                                   --counters counter_samples_rank0.csv [--save]

    # Show GPU frequency instead of MFMA util:
    python plot_device_timeline.py ... --mode freq

    # Show both:
    python plot_device_timeline.py ... --mode both
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

N_XCD = 8
N_CU = 304
N_SIMD = 4
NOMINAL_FREQ_MHZ = 2100


def load_and_prepare(traces_file, counters_file):
    k = pd.read_csv(traces_file)
    c = pd.read_csv(counters_file)

    # Sum across all HW dimensions per timestamp
    totals = c.groupby(['timestamp_ns', 'counter_name'])['counter_value'].sum().reset_index()
    piv = totals.pivot(index='timestamp_ns', columns='counter_name',
                       values='counter_value').reset_index()
    piv = piv.sort_values('timestamp_ns').reset_index(drop=True)

    # Instantaneous deltas
    for col in ['SQ_VALU_MFMA_BUSY_CYCLES', 'GRBM_GUI_ACTIVE']:
        if col in piv.columns:
            piv[f'd_{col}'] = piv[col].diff()
    piv['dt_ms'] = piv['timestamp_ns'].diff() / 1e6
    piv = piv.iloc[1:]

    # MFMA util per interval
    if 'd_SQ_VALU_MFMA_BUSY_CYCLES' in piv.columns and 'd_GRBM_GUI_ACTIVE' in piv.columns:
        piv['mfma_util'] = (100 * piv['d_SQ_VALU_MFMA_BUSY_CYCLES']
                            / (piv['d_GRBM_GUI_ACTIVE'] / N_XCD * N_CU * N_SIMD))

    # GPU frequency: GRBM_GUI_ACTIVE summed across 8 XCDs
    if 'd_GRBM_GUI_ACTIVE' in piv.columns:
        piv['gpu_freq_mhz'] = (piv['d_GRBM_GUI_ACTIVE'] / N_XCD) / (piv['dt_ms'] * 1000)

    t0 = piv['timestamp_ns'].iloc[0]
    piv['t_sec'] = (piv['timestamp_ns'] - t0) / 1e9

    return k, piv, t0


def auto_window(gemm_k, piv):
    """Skip init, find steady-state training by looking for the last big gap."""
    gemm_times = (gemm_k['start_ns'] - piv['timestamp_ns'].iloc[0]).values / 1e9
    if len(gemm_times) > 100:
        sorted_t = np.sort(gemm_times)
        gaps = np.diff(sorted_t)
        big_gap_idx = np.where(gaps > 5.0)[0]
        if len(big_gap_idx) >= 1:
            return sorted_t[big_gap_idx[-1] + 1] - 0.5, sorted_t[-1] + 1.0
        return sorted_t[0] - 0.5, sorted_t[-1] + 1.0
    return piv['t_sec'].iloc[0], piv['t_sec'].iloc[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True)
    p.add_argument("--counters", required=True)
    p.add_argument("--t-start", type=float, default=None)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--mode", choices=["mfma", "freq", "both"], default="mfma")
    p.add_argument("--save", action="store_true")
    a = p.parse_args()

    k, piv, t0 = load_and_prepare(a.traces, a.counters)

    nccl = k[k['kernel_name'].str.contains('nccl', na=False, case=False)]
    gemm_k = k[k['kernel_name'].str.contains('Cijk', na=False)]

    if a.t_start is None or a.t_end is None:
        auto_start, auto_end = auto_window(gemm_k, piv)
        if a.t_start is None:
            a.t_start = auto_start
        if a.t_end is None:
            a.t_end = auto_end

    zoom = piv[(piv['t_sec'] >= a.t_start) & (piv['t_sec'] <= a.t_end)]

    # Kernel bars for the window
    z_start = zoom['timestamp_ns'].iloc[0]
    z_end = zoom['timestamp_ns'].iloc[-1]
    nccl_z = nccl[(nccl['end_ns'] > z_start) & (nccl['start_ns'] < z_end)]
    gemm_z = gemm_k[(gemm_k['end_ns'] > z_start) & (gemm_k['start_ns'] < z_end)]

    # Determine panel layout
    panels = []
    if a.mode in ("mfma", "both"):
        panels.append("mfma")
    if a.mode in ("freq", "both"):
        panels.append("freq")

    n_panels = len(panels) + 1  # +1 for kernel bar
    ratios = [3] * len(panels) + [1]
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 3 * n_panels + 1),
                              sharex=True, gridspec_kw={'height_ratios': ratios})

    ax_idx = 0
    for panel in panels:
        ax = axes[ax_idx]
        if panel == "mfma":
            ax.plot(zoom['t_sec'], zoom['mfma_util'],
                    linewidth=0.5, color='steelblue', alpha=0.7)
            ax.fill_between(zoom['t_sec'], zoom['mfma_util'],
                            alpha=0.3, color='steelblue')
            ax.set_ylabel('MFMA Util (%)')
            ax.set_ylim(-5, 105)
        elif panel == "freq":
            ax.plot(zoom['t_sec'], zoom['gpu_freq_mhz'],
                    linewidth=0.5, color='green', alpha=0.7)
            ax.fill_between(zoom['t_sec'], zoom['gpu_freq_mhz'],
                            alpha=0.2, color='green')
            ax.axhline(y=NOMINAL_FREQ_MHZ, color='gray', linestyle='--',
                       alpha=0.3, label=f'{NOMINAL_FREQ_MHZ} MHz')
            ax.set_ylabel('GPU Frequency (MHz)')
            ax.set_ylim(-50, NOMINAL_FREQ_MHZ * 1.2)
            ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    # Kernel bar panel
    ax = axes[-1]
    for _, row in nccl_z.iterrows():
        s = (row['start_ns'] - t0) / 1e9
        e = (row['end_ns'] - t0) / 1e9
        ax.barh(0.75, e-s, left=s, height=0.4, color='orange', alpha=0.6)
    for _, row in gemm_z.iterrows():
        s = (row['start_ns'] - t0) / 1e9
        e = (row['end_ns'] - t0) / 1e9
        ax.barh(0.25, e-s, left=s, height=0.4, color='steelblue', alpha=0.6)

    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(['GEMM', 'NCCL'])
    ax.set_xlabel('Time (seconds)')
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Device Timeline during Training', fontsize=13)
    fig.tight_layout()

    if a.save:
        fig.savefig('device_timeline.png', dpi=150)
        print("Saved device_timeline.png")
    else:
        plt.show()

    # Stats (only if MFMA data available)
    if 'mfma_util' in zoom.columns:
        gemm_active = np.zeros(len(zoom), dtype=bool)
        nccl_active = np.zeros(len(zoom), dtype=bool)
        zoom_ts = zoom['timestamp_ns'].values
        for _, row in gemm_z.iterrows():
            gemm_active |= (zoom_ts >= row['start_ns']) & (zoom_ts <= row['end_ns'])
        for _, row in nccl_z.iterrows():
            nccl_active |= (zoom_ts >= row['start_ns']) & (zoom_ts <= row['end_ns'])

        gemm_only = zoom.iloc[gemm_active & ~nccl_active]['mfma_util']
        gemm_nccl = zoom.iloc[gemm_active & nccl_active]['mfma_util']

        print(f"\nMFMA util during GEMM kernels:")
        print(f"  GEMM only:     n={len(gemm_only):4d}  "
              f"mean={gemm_only.mean():.1f}%  median={gemm_only.median():.1f}%")
        print(f"  GEMM + NCCL:   n={len(gemm_nccl):4d}  "
              f"mean={gemm_nccl.mean():.1f}%  median={gemm_nccl.median():.1f}%")

        if len(gemm_only) and len(gemm_nccl):
            ratio = gemm_nccl.median() / gemm_only.median()
            print(f"  Degradation:   {(1-ratio)*100:.1f}%")


if __name__ == "__main__":
    main()
