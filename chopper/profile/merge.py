from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import ijson
import re
from argparse import ArgumentParser
from collections import Counter
from functools import partial
from decimal import Decimal
from chopper.common.printing import info, warn, err


def dtoi(ts):
    return np.int64(ts * Decimal('1000'))


def fwdbwd_link(line, data):
    ts = dtoi(line['ts'])
    id = line['id']
    bwd = "bp" in line

    data.append({
        'ts': ts,
        'id': id,
        'bwd': bwd,
    })


def cpu_op_link(line, data):
    args = line['args']

    if 'Collective name' in args:
        name = f"{line['name']}:{args['Collective name']}"
    else:
        name = line['name']
    data.append({
        'ext_id': args['External id'],
        'seq_num': args['Sequence number'] if 'Sequence number' in args else None,
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'name': name,
    })


def kernel_link(line, data):
    data.append({
        'ext_id': line['args']['External id'] if 'External id' in line['args'] else None,
        'name': line['name'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'correlation': line['args']['correlation'],
        # 'grid': line['args']['grid'],
        # 'block': line['args']['block'],
    })


def user_annotation_link(line, data):
    name = line['name']

    if re.fullmatch(r"Layer\d+", name):
        data.append({
            'type': 'layer',
            'ts': dtoi(line['ts']),
            'dur': dtoi(line['dur']),
            'value': int(name.split("Layer")[1]),
        })

    elif re.fullmatch(r"Iteration\d+", name):
        data.append({
            'type': 'iteration',
            'ts': dtoi(line['ts']),
            'dur': dtoi(line['dur']),
            'value': int(name.split("Iteration")[1]),
        })

    else:
        data.append({
            'type': 'operator-name',
            'ts': dtoi(line['ts']),
            'dur': dtoi(line['dur']),
            'value': name,
        })


def cuda_runtime_link(line, data):
    data.append({
        'name': line['name'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'correlation': line['args']['correlation'],
        'ext_id': line['args']['External id'] if 'External id' in line['args'] else None,
    })


def gpu_memcpy_link(line, data):
    data.append({
        'name': line['name'],
        'device': line['args']['device'],
        # 'kind': line['args']['kind'],
        'bytes': line['args']['bytes'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'correlation': line['args']['correlation'],
        'ext_id': line['args']['External id'] if 'External id' in line['args'] else None,
    })


def gpu_memset_link(line, data):
    data.append({
        'name': line['name'],
        'device': line['args']['device'],
        # 'kind': line['args']['kind'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'correlation': line['args']['correlation'],
        'ext_id': line['args']['External id'] if 'External id' in line['args'] else None,
    })


def json_to_pandas(rocprof_json_filename: str) -> pd.DataFrame:
    cat_func_map = {
        'fwdbwd': fwdbwd_link,
        'cpu_op': cpu_op_link,
        'kernel': kernel_link,
        'user_annotation': user_annotation_link,
        'cuda_runtime': cuda_runtime_link,
        'gpu_memcpy': gpu_memcpy_link,
        'gpu_memset': gpu_memset_link,
    }
    cat_func_data = {
        cat: [] for cat in cat_func_map.keys()
    }

    with open(rocprof_json_filename, 'r') as fp:
        info("Reading timestamp data...")
        ignore_cat = Counter()
        for line in ijson.items(fp, 'traceEvents.item'):
            if 'cat' in line:
                cat = line['cat']
            else:
                continue

            if cat in cat_func_map:
                cat_func_map[cat](
                    line,
                    cat_func_data[cat],
                )
            else:
                ignore_cat.update((cat,))

    for cat in ignore_cat:
        warn(f"Ignoring cat: {cat} ({ignore_cat[cat]}x)")

    for cat in cat_func_data.keys():
        cat_func_data[cat] = pd.DataFrame(cat_func_data[cat])

    return cat_func_data


def merge_df(left, right, on, suffixes, combine):
    # WARN merge introduces NaN when no match
    # which converts values to floats
    if isinstance(on, tuple):
        merged = pd.merge(
            left,
            right,
            left_on=on[0],
            right_on=on[1],
            how="left",
            suffixes=suffixes,
        )
    else:
        merged = pd.merge(
            left,
            right,
            on=on,
            how="left",
            suffixes=suffixes,
        )
    for c in combine:
        col0 = merged.get(f'{c}{suffixes[0]}')
        col1 = merged.get(f'{c}{suffixes[1]}')
        mask = ((col0 == col1) | (col0.isna() & col1.isna()))
        assert mask.all(), f'combine column {c} does not match'

        merged.drop(
            columns=[f'{c}{suffixes[0]}'],
            inplace=True,
        )
        merged.rename(columns={f'{c}{suffixes[1]}': c}, inplace=True)
    return merged


def add_cuda_runtime(json_data):
    info("adding cuda runtime to gpu kernels...")
    runtime_merge = partial(
        merge_df,
        right=json_data['cuda_runtime'],
        on='correlation',
        suffixes=('', '_cuda_runtime'),
        # combine=('ext_id',)
        combine=()
    )
    for key in ('gpu_memcpy', 'gpu_memset', 'kernel'):
        json_data[key] = runtime_merge(json_data[key])


def assign_annotation(json_data):
    # TODO optimize this
    ann_df = json_data['user_annotation']
    cpu_df = json_data['cpu_op']

    ann_df['end_ts'] = ann_df['ts'] + ann_df['dur']

    for df_type in ('layer', 'iteration', 'operator-name'):
        info(f"assigning {df_type} to cpu ops...")
        to_merge = ann_df[ann_df['type'] == df_type][['ts', 'end_ts', 'value']].sort_values(
            ['ts', 'end_ts'],
            ascending=(True, False),
        )

        for _, row in to_merge.iterrows():
            mask = (cpu_df['ts'] >= row['ts']) & (
                cpu_df['ts'] <= row['end_ts'])
            cpu_df.loc[mask, df_type] = row['value']

    json_data['cpu_op'] = cpu_df


def subordinate_cpu(json_data):
    info("propagating operator-name to subordinate cpu ops...")

    cpu_df = json_data['cpu_op']
    cpu_df['end_ts'] = cpu_df['ts'] + cpu_df['dur']

    is_root = ~cpu_df['id'].isna()
    root_df = cpu_df[is_root].sort_values('ts').reset_index()
    sub_df = cpu_df[~is_root].reset_index()

    root_ts = root_df['ts'].values
    root_end = root_ts + root_df['dur'].values
    root_op = root_df['operator-name'].values
    root_layer = root_df['layer'].values

    sub_ts = sub_df['ts'].values
    sub_indices = sub_df['index'].values

    idx = np.searchsorted(root_ts, sub_ts, side='right') - 1

    for i, root_idx in enumerate(idx):
        ts = sub_ts[i]
        while root_idx >= 0:
            if ts <= root_end[root_idx]:
                cpu_df.at[sub_indices[i], 'operator-name'] = root_op[root_idx]
                cpu_df.at[sub_indices[i], 'layer'] = root_layer[root_idx]
                break
            root_idx -= 1

    json_data['cpu_op'] = cpu_df


def add_fwdbwd(json_data):
    info("setting fwd and bwd cpu op names...")
    cpu_df = json_data['cpu_op'].copy()
    cpu_df = merge_df(
        cpu_df,
        json_data['fwdbwd'],
        on='ts',
        suffixes=('', '_fwdbwd'),
        combine=()
    )

    has_seq_mask = ~cpu_df['seq_num'].isna()
    fwd_mask = (cpu_df['bwd'] == False)
    missing_mask = has_seq_mask & (~fwd_mask)

    seq_with_name = cpu_df.loc[
        has_seq_mask & fwd_mask, [
            'seq_num', 'operator-name', 'layer', 'id']
    ].drop_duplicates()

    missing_df = cpu_df[missing_mask].copy()
    missing_df['orig_index'] = missing_df.index

    missing_df = missing_df.merge(
        seq_with_name,
        on='seq_num',
        how='left',
        suffixes=('', '_child'),
    )
    missing_df['operator-name_child'] = ('b_' +
                                         missing_df['operator-name_child'])
    cpu_df.loc[~missing_mask,
               'operator-name'] = ('f_' + cpu_df.loc[~missing_mask, 'operator-name'])
    cpu_df.loc[
        missing_df['orig_index'],
        ['operator-name', 'layer', 'id']
    ] = missing_df[['operator-name_child', 'layer_child', 'id_child']].values
    json_data['cpu_op'] = cpu_df


def add_cpu(json_data):
    cpu_merge = partial(
        merge_df,
        right=json_data['cpu_op'],
        on='ext_id',
        suffixes=('', '_cpu_op'),
        combine=()
    )

    memcpy_df = cpu_merge(json_data['gpu_memcpy'])
    memset_df = cpu_merge(json_data['gpu_memset'])
    kernel_df = cpu_merge(json_data['kernel'])

    return pd.concat((
        df for df in (memset_df, memcpy_df, kernel_df)
    )).sort_values('ts').reset_index(drop=True)


def parse_trace(trace_fn):
    json_data = json_to_pandas(trace_fn)

    add_cuda_runtime(json_data)
    assign_annotation(json_data)
    add_fwdbwd(json_data)
    subordinate_cpu(json_data)
    return add_cpu(json_data)


def kern_name_short(name, start=0, length=80):
    return f"{name[start:length]}{'...' if len(name) > length else ''}"


def get_pivoted(rocprof_csv_filename):
    rocprof_df = pd.read_csv(rocprof_csv_filename)

    unique_counters = list(
        c for c in rocprof_df.columns.to_list() if
        c != "Counter_Name" and c != "Counter_Value"
    )

    pivoted_df = rocprof_df.pivot_table(
        index=unique_counters,
        columns="Counter_Name",
        values="Counter_Value",
    ).sort_values("Start_Timestamp", ascending=True).reset_index()
    return pivoted_df


def get_combined_counters(
    counter_filenames,
):
    kname = 'Kernel_Name'
    df_combined = None
    info("Getting raw counter data...")
    for cur_csv in counter_filenames:
        df_cur = get_pivoted(cur_csv)

        df_cur["_mi"] = df_cur.groupby(kname).cumcount()
        if df_combined is None:
            df_combined = df_cur
        else:
            df_combined = df_combined.merge(
                df_cur,
                on=[kname, "_mi"],
                how="left",
                suffixes=("", "_new")
            )
            df_combined = df_combined.loc[
                :, ~df_combined.columns.str.endswith("_new")]

    assert df_combined is not None
    df_combined = df_combined.drop(columns=["_mi"])
    return df_combined


def get_merged(
    counter_filenames,
    df_ts,
    iterations,
) -> pd.DataFrame:

    gpus = sorted(set(df_ts['gpu']))
    n_gpus = len(gpus)

    counter_filenames = tuple(
        sorted(fns)
        for fns in counter_filenames
    )
    # invert counters per GPU
    counter_filenames = tuple(list(fns) for fns in zip(*counter_filenames))
    n_counters = len(counter_filenames)

    if n_gpus != n_counters:
        warn(f"Number of counter files doesn't match {n_gpus} gpus:")
        warn(
            f"    Only the first {n_counters} counters will be used")

    df_cntr = pd.concat(
        [df.assign(gpu=i) for i, df in enumerate(
            map(partial(get_combined_counters), counter_filenames))],
        ignore_index=True
    )
    info("Loaded counter data")

    kname = 'Kernel_Name'

    # sanity check GPU against agent id
    agent_id = 'Agent_Id'
    aids = tuple(set(df_cntr[df_cntr['gpu'] == gpus[0]][agent_id]))
    assert len(aids) == 1, f'More than one Agent ID was present: {aids}'
    agent_id_inc = aids[0]
    for gpu in gpus[1:n_counters]:
        agent_id_inc += 1
        aids = tuple(set(df_cntr[df_cntr['gpu'] == gpu][agent_id]))
        assert len(aids) == 1, 'More than one Agent ID was present'
        assert aids[0] == agent_id_inc, 'Agent ID was unexpected'

    if iterations is not None:
        ts_mask = (
            (df_ts['gpu'] < n_counters) &
            (df_ts['iteration'].isin(iterations))
        )
    else:
        ts_mask = (
            (df_ts['gpu'] < n_counters)
        )

    ts_names = df_ts.loc[ts_mask, "name"]
    assert ts_names is not None

    cntr_names = df_cntr.get(kname)
    assert cntr_names is not None
    ts_name_set = set(ts_names)
    cntr_name_set = set(cntr_names)

    cntr_diff = cntr_name_set - ts_name_set
    ts_diff = ts_name_set - cntr_name_set

    if len(cntr_diff):
        warn("These kernel names are missing from Pytorch Profiler data:")
        for d in cntr_diff:
            warn(f"    {kern_name_short(d)}")

    if len(ts_diff):
        warn("These kernel names are missing from Counter data:")
        for d in ts_diff:
            warn(f"    {kern_name_short(d)}")

    ts_names = df_ts.loc[ts_mask, "name"]
    ts_name_set = set(ts_names)

    ignore_kernels = []
    ignore_kernels.extend(cntr_diff)
    ts_names = df_ts["name"]

    remove_ts_mask = ts_names.isin(ignore_kernels)
    remove_cntr_mask = cntr_names.isin(ignore_kernels)

    df_ts.drop(df_ts[remove_ts_mask].index, inplace=True)
    df_cntr.drop(df_cntr[remove_cntr_mask].index, inplace=True)

    left_group_arrs = ['name', 'gpu']
    right_group_arrs = [kname, 'gpu']

    df_ts["_mi"] = (
        df_ts
        .iloc[::-1]
        .groupby(left_group_arrs)
        .cumcount()
        .iloc[::-1]
    )
    df_cntr["_mi"] = (
        df_cntr
        .iloc[::-1]
        .groupby(right_group_arrs)
        .cumcount()
        .iloc[::-1]
    )

    info("Merging timestamps and counters...")

    if iterations is not None:
        info("Selecting iterations:")
        info(f"  {iterations}")
        info("from iterations:")
        unique_iters = set(df_ts['iteration'].unique())
        info(f"  {unique_iters}")

        if any(it not in unique_iters for it in iterations):
            err("Invalid iterations selected")
            exit(-1)

        mask = df_ts["iteration"].isin(iterations)
        df_ts_merge = df_ts[mask]
        df_ts_rest = df_ts[~mask]
    else:
        df_ts_merge = df_ts

    left_group_arrs += ["_mi"]
    right_group_arrs += ["_mi"]
    df_merged = df_ts_merge.merge(
        df_cntr,
        left_on=left_group_arrs,
        right_on=right_group_arrs,
        how="left",
        suffixes=('', '_y'),
    )

    df_merged.drop(columns=['_mi', kname], inplace=True)
    if iterations is not None:
        df_ts_rest.drop(columns=['_mi'], inplace=True)
        df_merged = pd.concat([df_merged, df_ts_rest], ignore_index=True)
    df_merged = df_merged.sort_values("ts")

    assert 'gpu_y' not in df_merged.columns

    return df_merged


def main(
        pytorch_trace,
        duration_pickle_fns,
        counters,
        iterations,
        output_filename,
):
    # TODO double check what this does
    pd.set_option('future.no_silent_downcasting', True)

    if pytorch_trace is None and duration_pickle_fns is None and counters is None:
        err("Nothing was passed to the script, what are you doing? :|")
        return -1
    if pytorch_trace is None and duration_pickle_fns is None:
        err("Please pass either raw pytorch traces or a duration pickle to merge with counters")
        return -1
    if pytorch_trace is not None and duration_pickle_fns is not None:
        err("Please only pass raw pytorch traces or a duration pickle not both")
        return -1

    if pytorch_trace is not None:
        with ProcessPoolExecutor(max_workers=8) as ex:
            gpu_dfs = tuple(ex.map(parse_trace, pytorch_trace))
        for i, gpu_df in enumerate(gpu_dfs):
            gpu_df["gpu"] = i
        duration_pickle = pd.concat(gpu_dfs)
        if counters is None:
            duration_pickle.to_pickle(output_filename)
            return 0
    elif len(duration_pickle_fns) == 1:
        duration_pickle = pd.read_pickle(duration_pickle_fns[0])
    else:
        if not (pytorch_trace is None and counters is None):
            err("Trying to merge duration pickles, but was passed trace file or counters")
            return -1
        info(f"merging pickles: {duration_pickle_fns}")
        pickles = [pd.read_pickle(dpf) for dpf in duration_pickle_fns]
        merged = pd.concat(pickles)
        merged.to_pickle(output_filename)
        return 0

    if counters is not None:
        merged = get_merged(counters, duration_pickle, iterations)
        merged.to_pickle(output_filename)
    else:
        duration_pickle.to_pickle(output_filename)

    return 0


if __name__ == "__main__":
    desc = (
        "There are three ways to use this script. "
        "To merge with hardware counters perform each step below in order "
        "(skip step two if only merging one trace per GPU). "
        "Otherwise, only step one and optionally step two are needed.\n"
        "1) Pass just pytorch trace files to create a duration pickle\n"
        "2) Pass multiple duration pickles to merge duration pickles\n"
        "3) Pass a duration pickle with counters to create a counter pickle\n"
    )
    parser = ArgumentParser(
        description=desc,
    )
    parser.add_argument(
        "--pytorch-trace",
        "-t",
        type=str,
        required=False,
        nargs="+",
        help="Filenames of pytorch trace json files"
    )
    parser.add_argument(
        "--duration_pickle",
        "-p",
        type=str,
        required=False,
        nargs='+',
        help="Pickle of pytorch traces without performance counters"
    )
    parser.add_argument(
        "--counters",
        "-c",
        action="append",
        required=False,
        nargs="+",
        type=str,
        help="List of counters csv files, pass additional lists to merge counter files together"
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        required=False,
        nargs='+',
        help="Pass to select iteration(s) to assign to timestamps (since counters may collect fewer iterations)"
    )
    parser.add_argument(
        "-o",
        "--output-filename",
        type=str,
        required=True,
        help="Output pkl name",
    )
    args = parser.parse_args()
    exit(main(
        sorted(args.pytorch_trace) if args.pytorch_trace else None,
        args.duration_pickle,
        args.counters,
        args.iterations,
        args.output_filename,
    ))
