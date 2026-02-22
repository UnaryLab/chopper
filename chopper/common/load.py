"""Data loading and preprocessing utilities for trace analysis."""

import pandas as pd
from typing import Optional, List, Dict, Sequence

from chopper.common.annotations import (
    Framework,
    no_overlap_mask,
    assign_operator_type,
    assign_chunks as do_assign_chunks,
    fix_names as do_fix_names,
)
from chopper.common.cache import load_pickle


def select_iters(df: pd.DataFrame, iters: List) -> pd.DataFrame:
    """Select specific iterations from trace data.
    
    Args:
        df: DataFrame containing trace data with 'iteration' column
        iters: List of iteration indices to select
        
    Returns:
        Filtered DataFrame containing only selected iterations
    """
    u_iters = df.loc[~df['iteration'].isna(), 'iteration'].unique()
    iters = [u_iters[i] for i in iters]
    return df[df['iteration'].isin(iters)]


def get_df(
    fn: str,
    iter_idxs: Optional[List] = None,
    assign_chunks: bool = False,
    assign_optype: bool = False,
    remove_nan_chunks: bool = False,
    remove_overlap: bool = False,
    fix_names: bool = False,
    group_arr: Optional[List] = None,
    group_map: Optional[Dict[str, List[str]]] = None,
    sort_value: Optional[str] = None,
    framework: Framework = Framework.FSDPv1,
) -> pd.DataFrame:
    """Load and preprocess trace data with optional transformations.
    
    Main entry point for loading trace files with flexible preprocessing options
    including filtering, grouping, and aggregation.
    
    Args:
        fn: Path to trace pickle file
        iter_idxs: Optional list of iteration indices to select
        assign_chunks: If True, assign training phase chunks (fwd/bwd/opt)
        assign_optype: If True, categorize operators by type (GEMM/FA/Vec)
        remove_nan_chunks: If True, remove rows with unassigned chunks
        remove_overlap: If True, filter out communication-overlapped kernels
        fix_names: If True, normalize operator names
        group_arr: Optional list of columns to group by for aggregation
        group_map: Optional dict mapping columns to aggregation functions
        sort_value: Optional column name to sort by after grouping
        framework: Framework enum for framework-specific processing
        
    Returns:
        Processed DataFrame with applied transformations
    """
    df = load_pickle(fn)
    df['layer'] = df['layer'].fillna(-1)
    df = df[df['name'] != 'Memcpy HtoD (Host -> Device)']

    if iter_idxs:
        df = select_iters(df, iter_idxs)

    if remove_overlap:
        df = df[no_overlap_mask(
            df, framework=framework)]
    if assign_optype:
        df = assign_operator_type(df)

    if assign_chunks:
        df = do_assign_chunks(df)
        if remove_nan_chunks:
            df = df[~df['chunk'].isna()]

    if fix_names:
        df = do_fix_names(df)

    if group_arr:
        assert group_map, f"Null group_map is invalid with non-null group_arr: {
            group_arr}"
        assert all(col in df.columns.tolist() for col in group_map.keys())

        missing_cols = tuple(col for col in group_map.keys()
                             if col not in df.columns.tolist())
        assert len(missing_cols) == 0, f"Missing: {missing_cols}"

        weight_metrics: dict[str, list[str]] = {}
        new_group_map = {}
        for metric, aggs in group_map.items():
            for agg in aggs:
                if agg in df.columns.tolist():
                    weight_metrics.setdefault(metric, []).append(agg)
                    new_group_map[agg] = (agg, 'sum')
                    new_group_map[metric] = (metric, 'sum')
                else:
                    new_group_map[metric if agg ==
                                  'sum' else f"{metric}_{agg}"] = (metric, agg)

        for metric, weights in weight_metrics.items():
            assert len(weights) == 1, "cannot weigh by multiple metrics"
            df[metric] *= df[weights[0]]

        df = df.groupby(group_arr, dropna=False).agg(**new_group_map)

        for metric, weights in weight_metrics.items():
            df[metric] /= df[weights[0]]

        if sort_value:
            df = df.sort_values(sort_value).reset_index()
        else:
            df = df.reset_index()

    return df


def get_straggler_df(
    fn: str,
    iter_idxs: Optional[List] = None,
    agg_meth: str = 'max',
    framework: Framework = Framework.FSDPv1,
    kernel_name: bool = False,
) -> pd.DataFrame:
    """Load and compute straggler metrics from trace data.
    
    Processes trace data to identify performance stragglers by computing
    how much each GPU lags behind the slowest GPU for each operation.
    
    Args:
        fn: Path to trace pickle file
        iter_idxs: Optional list of iteration indices to select
        agg_meth: Aggregation method ('max', 'min', 'mean') for straggler reference
        framework: Framework enum for framework-specific processing
        kernel_name: If True, include kernel names in grouping
        
    Returns:
        DataFrame with straggler metrics including 's-value' (lag time) and
        's-delta' (change in lag between operations)
    """
    group_arr = ['iteration', 'layer', 'operator-name',
                 'name'] if kernel_name else ['iteration', 'layer', 'operator-name']
    df = get_df(
        fn,
        iter_idxs=iter_idxs,
        assign_chunks=True,
        remove_nan_chunks=True,
        remove_overlap=True,
        fix_names=True,
        group_arr=['gpu'] + group_arr,
        group_map={
            'ts': ['first', 'last'],
            'dur': ['sum', 'last'],
        },
        sort_value='ts_first',
        framework=framework,
    )
    agg_df = df.groupby(
        group_arr,
        dropna=False
    ).agg(
        **{f'ts_first_{agg_meth}': ('ts_first', agg_meth)}
    ).sort_values(f'ts_first_{agg_meth}').reset_index()

    df = df.merge(
        agg_df,
        on=group_arr,
        how='left'
    )

    df['s-value'] = (
        df[f'ts_first_{agg_meth}'] - df['ts_first'])

    df['s-delta'] = df.groupby(
        'gpu'
    )['s-value'].transform(lambda x: x.shift(-1) - x)

    last_op_of_iter_mask = df.groupby(
        ['gpu', 'iteration']).cumcount(ascending=False) == 0
    df.loc[last_op_of_iter_mask, 's-delta'] = 0

    return df


def get_straggler_contributors(
    df: pd.DataFrame,
    group_arr: List[str] = ['gpu', 'operator-name'],
    delta: bool = False,
    agg_cols: List[str] = ['min', 'max', 'median', 'sum'],
):
    """Aggregate straggler contributions by operator or GPU.
    
    Args:
        df: DataFrame from get_straggler_df() containing straggler metrics
        group_arr: List of columns to group by (e.g., ['gpu'], ['operator-name'])
        delta: If True, analyze s-delta instead of s-value
        agg_cols: List of aggregation functions to apply
        
    Returns:
        DataFrame with aggregated straggler contributions
    """
    return df.groupby(group_arr)[
        's-delta' if delta else 's-value'
    ].agg(list(agg_cols)).reset_index()


def get_overlap_df(
    fn: str,
    iter_idxs: Optional[List] = None,
    framework: Framework = Framework.FSDPv1,
    kernel_name: bool = False,
    include_comm_df: bool = False,
):
    """Compute communication-computation overlap ratios.
    
    Analyzes how much computation overlaps with communication operations
    to assess pipeline efficiency.
    
    Args:
        fn: Path to trace pickle file
        iter_idxs: Optional list of iteration indices to select
        framework: Framework enum for framework-specific processing
        kernel_name: If True, include kernel names in grouping
        include_comm_df: If True, return both overlap and communication DataFrames
        
    Returns:
        If include_comm_df is False: DataFrame with overlap_ratio column
        If include_comm_df is True: Tuple of (overlap_df, comm_df)
    """
    comm_df = get_df(
        fn,
        iter_idxs=iter_idxs,
        sort_value='ts',
    )
    comm_df = comm_df[~no_overlap_mask(
        comm_df, framework=framework)]
    comm_df['end_ts'] = comm_df['ts'] + comm_df['dur']

    comp_df = get_df(
        fn,
        iter_idxs=iter_idxs,
        assign_chunks=True,
        remove_nan_chunks=True,
        remove_overlap=True,
        fix_names=True,
        group_arr=['gpu', 'iteration', 'layer', 'operator-name',
                   'name'] if kernel_name else ['gpu', 'iteration', 'layer', 'operator-name'],
        group_map={
            'ts': ['first', 'last'],
            'dur': ['sum', 'last'],
        },
        sort_value='ts_first',
        framework=framework,
    )
    comp_df['end_ts'] = comp_df['ts_last'] + comp_df['dur_last']
    comp_df['elapsed'] = comp_df['end_ts'] - comp_df['ts_first']

    def add_overlap(group):
        gpu = group.name
        gpu_comm_df = comm_df[comm_df['gpu'] == gpu]

        for op_idx, operation in group.iterrows():
            start = operation['ts_first']
            end = operation['end_ts']
            elapsed = operation['elapsed']
            overlapped_comm = gpu_comm_df[
                (gpu_comm_df['ts'] <= end) &
                (gpu_comm_df['end_ts'] >= start)
            ]

            # FIXME maybe double counts some overlap
            total_ovr = 0
            for _, comm_kern in overlapped_comm.iterrows():
                ovr_start = max(start, comm_kern['ts'])
                ovr_end = min(end, comm_kern['end_ts'])
                total_ovr += max(0, ovr_end - ovr_start)

            total_ovr = min(elapsed, total_ovr)
            ratio = 100 * total_ovr / elapsed
            group.loc[op_idx, "overlap_ratio"] = ratio
        return group

    ovr_df = comp_df.groupby('gpu').apply(add_overlap).droplevel(0)
    if include_comm_df:
        return ovr_df, comm_df
    else:
        return ovr_df


def get_slack_adv_df(
    fn: str,
    iter_idxs: Optional[List] = None,
    framework: Framework = Framework.FSDPv1,
    kernel_name: bool = False,
    agg_meth: str = 'max',
):
    """Compute slack advancement metrics for communication operations.
    
    Analyzes how much communication operations can be advanced (started earlier)
    based on available slack in the computation schedule.
    
    Args:
        fn: Path to trace pickle file
        iter_idxs: Optional list of iteration indices to select
        framework: Framework enum for framework-specific processing
        kernel_name: If True, include kernel names in grouping
        agg_meth: Aggregation method ('max', 'min', 'mean') for slack reference
        
    Returns:
        Tuple of (comm_df, comp_df) with timing and straggler information
    """
    group_arr = ['iteration', 'layer', 'operator-name',
                 'name'] if kernel_name else ['iteration', 'layer', 'operator-name']
    comm_df = get_df(
        fn,
        iter_idxs=iter_idxs,
        group_arr=['gpu'] + group_arr,
        group_map={
            'ts': ['first', 'last'],
            'dur': ['sum', 'last'],
        },
        sort_value='ts_first',
        framework=framework,
    )
    comm_df = comm_df[~no_overlap_mask(
        comm_df, framework=framework)]
    comm_df = comm_df[comm_df['name'] != 'Memcpy HtoD (Host -> Device)']
    comm_df['end_ts'] = comm_df['ts_last'] + comm_df['dur']
    comm_df['elapsed'] = comm_df['end_ts'] - comm_df['ts_first']

    comp_df = get_df(
        fn,
        iter_idxs=iter_idxs,
        assign_chunks=True,
        remove_nan_chunks=True,
        remove_overlap=True,
        fix_names=True,
        group_arr=['gpu'] + group_arr,
        group_map={
            'ts': ['first', 'last'],
            'dur': ['sum', 'last'],
        },
        sort_value='ts_first',
        framework=framework,
    )
    comp_df['end_ts'] = comp_df['ts_last'] + comp_df['dur_last']
    comp_df['elapsed'] = comp_df['end_ts'] - comp_df['ts_first']

    agg_df = comm_df.groupby(
        group_arr,
        dropna=False
    ).agg(
        **{f'ts_{agg_meth}': ('ts_first', agg_meth)}
    ).sort_values(f'ts_{agg_meth}').reset_index()

    comm_df = comm_df.merge(
        agg_df,
        on=group_arr,
        how='left'
    )

    comm_df['s-value'] = (
        comm_df[f'ts_{agg_meth}'] - comm_df['ts_first'])

    return comm_df, comp_df
