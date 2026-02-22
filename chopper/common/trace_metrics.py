"""Metric derivation functions for trace analysis."""

import numpy as np
from pandas import DataFrame


def derive_launch_overhead(df: DataFrame) -> DataFrame:
    """Compute kernel launch overhead (gap between kernel completions).
    
    Calculates the time gap between the end of one kernel and the start
    of the next kernel on the same GPU.
    
    Args:
        df: DataFrame with 'gpu', 'ts', and 'dur' columns
        
    Returns:
        DataFrame with added 'Launch Overhead' column in milliseconds
    """
    df = df.sort_values(['gpu', 'ts']).reset_index()

    prev_end_time = (df.groupby('gpu')['ts'].shift(1) +
                     df.groupby('gpu')['dur'].shift(1))

    df['Launch Overhead'] = np.maximum(
        0, df['ts'] - prev_end_time).astype(float) * 1e-6

    df.loc[df.groupby(['gpu']).head(
        1).index, 'Launch Overhead'] = 0

    return df


def derive_prep_overhead(df: DataFrame) -> DataFrame:
    """Compute kernel preparation overhead (CUDA runtime to kernel launch).
    
    Calculates the time from the end of the previous kernel to when the
    CUDA runtime is called for the next kernel.
    
    Args:
        df: DataFrame with 'gpu', 'ts', 'ts_cuda_runtime', and 'dur' columns
        
    Returns:
        DataFrame with added 'Prep Overhead' column in milliseconds
    """
    df = df.sort_values(['gpu', 'ts']).reset_index(drop=True)

    prev_end_time = df.groupby('gpu')['ts'].shift(
        1) + df.groupby('gpu')['dur'].shift(1)
    df['Prep Overhead'] = np.maximum(
        0, df['ts_cuda_runtime'] - prev_end_time).astype(float) * 1e-6

    first_idx = df.groupby('gpu').head(1).index
    df.loc[first_idx, 'Prep Overhead'] = 0

    return df


def derive_call_overhead(df: DataFrame) -> DataFrame:
    """Compute CUDA call overhead (runtime call to kernel launch).
    
    Calculates the time from when the CUDA runtime is called to when
    the kernel actually launches on the GPU.
    
    Args:
        df: DataFrame with 'gpu', 'ts', 'ts_cuda_runtime', and 'dur' columns
        
    Returns:
        DataFrame with added 'Call Overhead' column in milliseconds
    """
    df = df.sort_values(['gpu', 'ts']).reset_index(drop=True)

    prev_ts = df.groupby('gpu')['ts'].shift(1)
    prev_dur = df.groupby('gpu')['dur'].shift(1)
    prev_end_time = prev_ts + prev_dur

    tklqt = df['ts'] - df['ts_cuda_runtime']
    lo = np.maximum(
        0, df['ts'] - prev_end_time)
    df['Call Overhead'] = np.minimum(
        tklqt, lo).astype(float) * 1e-6

    first_idx = df.groupby('gpu').head(1).index
    df.loc[first_idx, 'Call Overhead'] = 0

    return df


def compute_overlap_cdf(
    kernel_df: DataFrame,
    overlap_df: DataFrame,
    sort_ratio: bool = True,
) -> DataFrame:
    """Compute cumulative distribution of communication overlap ratios.
    
    Analyzes what fraction of computation kernels have communication overlap,
    producing a CDF for visualization.
    
    Args:
        kernel_df: DataFrame containing computation kernels
        overlap_df: DataFrame containing communication operations
        sort_ratio: If True, sort by overlap ratio before computing CDF
        
    Returns:
        DataFrame with 'overlap_ratio', 'cdf', and 'op_idx' columns
    """
    cdf_df = kernel_df.copy().sort_values('ts_first').reset_index()
    cdf_df['end_ts'] = cdf_df['ts_last'] + cdf_df['dur_last']
    cdf_df['elapsed'] = cdf_df['end_ts'] - cdf_df['ts_first']

    ov_df = overlap_df.copy()
    ov_df['end_ts'] = ov_df['ts'] + ov_df['dur']

    def compute_overlap_ratio(group: DataFrame) -> DataFrame:
        gpu = group.name
        gpu_ov = ov_df[ov_df['gpu'] == gpu]
        ratios = []

        for _, row in group.iterrows():
            start = row['ts_first']
            end = row['end_ts']
            elapsed = row['elapsed']

            overlaps = gpu_ov[
                (gpu_ov['ts'] <= end) &
                (gpu_ov['end_ts'] >= start)
            ]

            total_overlap = 0
            for _, ov_row in overlaps.iterrows():
                overlap_start = max(start, ov_row['ts'])
                overlap_end = min(end, ov_row['end_ts'])
                overlap = max(0, overlap_end - overlap_start)
                total_overlap += overlap

            total_overlap = min(elapsed, total_overlap)
            ratio = 100 * total_overlap / elapsed if elapsed > 0 else 0
            ratios.append(ratio)

        group = group.copy()
        group['overlap_ratio'] = ratios
        sort_by = ['ts_first']
        if sort_ratio:
            sort_by = ['overlap_ratio'] + sort_by

        group = group.sort_values(
            sort_by).reset_index(drop=True)
        group['cdf'] = 100 * (group.index + 1) / len(group)
        group['op_idx'] = group.index
        return group

    return cdf_df.groupby('gpu').apply(
        compute_overlap_ratio
    ).reset_index(drop=True)
