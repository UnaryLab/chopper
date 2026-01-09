import pandas as pd
from typing import Optional, List, Dict, Tuple
import chopper

from chopper.common.annotations import Framework


def select_iters(df: pd.DataFrame, iters: List) -> pd.DataFrame:
    u_iters = df.loc[~df['iteration'].isna(), 'iteration'].unique()
    iters = [u_iters[i] for i in iters]
    return df[df['iteration'].isin(iters)]


def get_df(
    fn: str | pd.DataFrame,
    iter_idxs: Optional[List] = None,
    assign_chunks: bool = False,
    assign_optype: bool = False,
    remove_nan_chunks: bool = False,
    remove_overlap: bool = False,
    fix_names: bool = False,
    group_arr: Optional[List] = None,
    group_map: Optional[Dict[str, Tuple[str, str]]] = None,
    sort_value: Optional[str] = None,
    framework: Framework = Framework.FSDPv1,
) -> pd.DataFrame:
    if isinstance(fn, pd.DataFrame):
        df = fn.copy()
    elif isinstance(fn, str):
        df = pd.read_pickle(fn)
    else:
        raise ValueError(f"{fn} was a {type(fn)}")
    df['layer'] = df['layer'].fillna(-1)
    df = df[df['name'] != 'Memcpy HtoD (Host -> Device)']

    if iter_idxs:
        df = select_iters(df, iter_idxs)

    if remove_overlap:
        df = df[chopper.common.annotations.no_overlap_mask(
            df, framework=framework)]
    if assign_optype:
        df = chopper.common.annotations.assign_operator_type(df)

    if assign_chunks:
        df = chopper.common.annotations.assign_chunks(df)
        if remove_nan_chunks:
            df = df[~df['chunk'].isna()]

    if fix_names:
        df = chopper.common.annotations.fix_names(df)

    if group_arr:
        assert group_map, f"Null group_map is invalid with non-null group_arr: {
            group_arr}"
        assert all(col in df.columns.tolist() for col in group_map.keys())

        missing_cols = tuple(col for col in group_map.keys()
                             if col not in df.columns.tolist())
        assert len(missing_cols) == 0, f"Missing: {missing_cols}"

        weight_metrics = {}
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
    fn: str | pd.DataFrame,
    iter_idxs: Optional[List] = None,
    agg_meth: str = 'max',
    framework: Framework = Framework.FSDPv1,
    kernel_name: bool = False,
) -> pd.DataFrame:
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
    return df.groupby(group_arr)[
        's-delta' if delta else 's-value'
    ].agg(agg_cols).reset_index()


def get_overlap_df(
    fn: str | pd.DataFrame,
    iter_idxs: Optional[List] = None,
    framework: Framework = Framework.FSDPv1,
    kernel_name: bool = False,
    include_comm_df: bool = False,
):
    comm_df = get_df(
        fn,
        iter_idxs=iter_idxs,
        sort_value='ts',
    )
    comm_df = comm_df[~chopper.common.annotations.no_overlap_mask(
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
    fn: str | pd.DataFrame,
    iter_idxs: Optional[List] = None,
    framework: Framework = Framework.FSDPv1,
    kernel_name: bool = False,
    agg_meth: str = 'max',
):
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
    comm_df = comm_df[~chopper.common.annotations.no_overlap_mask(
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
