"""Full trace merge: parse PyTorch Chrome traces into a kernel DataFrame."""
import json
import time
import re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from loguru import logger


def assign_ranges(timestamps, ranges):
    """Assign labels from (start, end, label) ranges to timestamps.
    Ranges must be sorted longest-first so innermost overwrites."""
    ts_arr = np.array(timestamps)
    sort_idx = ts_arr.argsort()
    sorted_ts = ts_arr[sort_idx]
    result = np.empty(len(sorted_ts), dtype=object)
    for start, end, label in ranges:
        lo = np.searchsorted(sorted_ts, start, side='left')
        hi = np.searchsorted(sorted_ts, end, side='right')
        result[lo:hi] = label
    final = np.empty_like(result)
    final[sort_idx] = result
    return final


def parse(filename):
    with open(filename) as f:
        trace = json.load(f)

    cpu_ops = {}
    annotations = []
    fwdbwd = {}
    kernels = []
    runtime = {}

    for e in trace['traceEvents']:
        cat = e.get('cat', '')
        if cat == 'cpu_op':
            cpu_ops[e['args']['External id']] = {
                'name': e['name'],
                'ts': int(e['ts'] * 1000),
                'dur': int(e['dur'] * 1000),
                'seq': e['args'].get('Sequence number'),
            }
        elif cat == 'user_annotation':
            ts = int(e['ts'] * 1000)
            dur = int(e['dur'] * 1000)
            annotations.append({'name': e['name'], 'ts': ts, 'end_ts': ts + dur})
        elif cat == 'fwdbwd':
            fid = e['id']
            if fid not in fwdbwd:
                fwdbwd[fid] = {}
            fwdbwd[fid]['bwd' if 'bp' in e else 'fwd'] = int(e['ts'] * 1000)
        elif cat == 'kernel':
            kernels.append({
                'name': e['name'],
                'ts': int(e['ts'] * 1000),
                'dur': int(e['dur'] * 1000),
                'correlation': e['args']['correlation'],
            })
        elif cat == 'cuda_runtime':
            runtime[e['args']['correlation']] = {
                'ext_id': e['args'].get('External id'),
                'ts': int(e['ts'] * 1000),
            }

    return cpu_ops, annotations, fwdbwd, kernels, runtime


def classify_annotations(annotations):
    """Split annotations into layers, iterations, and operator annotations."""
    layers = []
    iterations = []
    ops = []
    for a in annotations:
        if (m := re.fullmatch(r'Layer(\d+)', a['name'])):
            layers.append((a['ts'], a['end_ts'], int(m.group(1))))
            continue
        if (m := re.fullmatch(r'Iteration(\d+)', a['name'])):
            iterations.append((a['ts'], a['end_ts'], int(m.group(1))))
            continue
        ops.append((a['ts'], a['end_ts'], a['name']))

    # Sort longest-first so innermost overwrites
    layers.sort(key=lambda r: r[1] - r[0], reverse=True)
    iterations.sort(key=lambda r: r[1] - r[0], reverse=True)
    ops.sort(key=lambda r: r[1] - r[0], reverse=True)
    return layers, iterations, ops


def link_fwdbwd(cpu_ops, op_ranges, fwdbwd, layer_ranges):
    """Link fwd<->bwd cpu_ops.
    Returns (ext_id -> f_/b_ annotation name, ext_id -> layer)."""
    ts_to_ext = {op['ts']: eid for eid, op in cpu_ops.items()}
    fwd_data = []
    fwd_timestamps = []

    for fid, ep in fwdbwd.items():
        assert 'fwd' in ep and 'bwd' in ep, f"fwdbwd {fid} missing endpoint: {ep}"
        fwd_ext = ts_to_ext.get(ep['fwd'])
        bwd_ext = ts_to_ext.get(ep['bwd'])
        assert fwd_ext is not None, f"fwdbwd {fid} fwd ts {ep['fwd']} not in cpu_ops"
        assert bwd_ext is not None, f"fwdbwd {fid} bwd ts {ep['bwd']} not in cpu_ops"
        fwd_data.append((fwd_ext, bwd_ext))
        fwd_timestamps.append(cpu_ops[fwd_ext]['ts'])

    anns = assign_ranges(fwd_timestamps, op_ranges)
    fwd_layers = assign_ranges(fwd_timestamps, layer_ranges)

    results = {}
    fwdbwd_layers = {}
    for i, (fwd_ext, bwd_ext) in enumerate(fwd_data):
        ann = anns[i]
        assert ann is not None, f"cpu_op {fwd_ext} ({cpu_ops[fwd_ext]['name']}) has no annotation"
        results[fwd_ext] = 'f_' + ann
        results[bwd_ext] = 'b_' + ann
        fwdbwd_layers[fwd_ext] = fwd_layers[i]
        fwdbwd_layers[bwd_ext] = fwd_layers[i]

    return results, fwdbwd_layers


def promote(cpu_ops, labeled):
    """Promote fwdbwd labels up to parent cpu_ops with the same sequence number.
    This ensures siblings of the linked op also get covered when we propagate down."""
    seq_to_label = {}
    for eid, label in labeled.items():
        seq = cpu_ops[eid].get('seq')
        if seq is not None:
            seq_to_label[seq] = label

    result = dict(labeled)
    for eid, op in cpu_ops.items():
        if eid in result:
            continue
        seq = op.get('seq')
        if seq is not None and seq in seq_to_label:
            result[eid] = seq_to_label[seq]
    return result


def propagate(cpu_ops, labeled):
    """Propagate labels to child cpu_ops. Returns ext_id -> label (expanded)."""
    parents = [(cpu_ops[eid]['ts'], cpu_ops[eid]['ts'] + cpu_ops[eid]['dur'], label)
               for eid, label in labeled.items()]
    parents.sort(key=lambda p: p[1] - p[0], reverse=True)

    unlabeled_eids = [eid for eid in cpu_ops if eid not in labeled]
    unlabeled_ts = [cpu_ops[eid]['ts'] for eid in unlabeled_eids]
    assigned = assign_ranges(unlabeled_ts, parents)

    result = dict(labeled)
    for i, eid in enumerate(unlabeled_eids):
        # None is expected: some autograd backward ops (e.g. DivBackward0) aren't
        # contained by any fwdbwd-labeled parent. These get picked up by assign_unlabeled.
        if assigned[i] is not None:
            result[eid] = assigned[i]
    return result


def assign_unlabeled(cpu_ops, labeled, op_ranges):
    """Assign annotations to cpu_ops not covered by fwdbwd (opt, FSDP, etc)."""
    unlabeled_eids = [eid for eid in cpu_ops if eid not in labeled]
    unlabeled_ts = [cpu_ops[eid]['ts'] for eid in unlabeled_eids]
    assigned = assign_ranges(unlabeled_ts, op_ranges)

    result = dict(labeled)
    for i, eid in enumerate(unlabeled_eids):
        # None is expected: some cpu_ops occur outside any user_annotation range
        # (e.g. during profiler init before training starts). These remain unlabeled.
        if assigned[i] is not None:
            result[eid] = assigned[i]
    return result


def build_kernel_df(cpu_ops, kernels, runtime, labels, layer_ranges, iter_ranges, fwdbwd_layers):
    """Build final kernel DataFrame with all context."""
    # Assign layer and iteration to all cpu_ops via timestamp ranges
    all_eids = list(cpu_ops.keys())
    all_ts = [cpu_ops[eid]['ts'] for eid in all_eids]
    layers = assign_ranges(all_ts, layer_ranges)
    iterations = assign_ranges(all_ts, iter_ranges)
    eid_to_layer = {eid: layers[i] for i, eid in enumerate(all_eids)}
    eid_to_iter = {eid: iterations[i] for i, eid in enumerate(all_eids)}

    # fwdbwd_layers covers both fwd and bwd ops (bwd gets fwd's layer).
    # Overwrite timestamp-based layers with these authoritative values.
    eid_to_layer.update(fwdbwd_layers)

    rows = []
    for k in kernels:
        rt = runtime.get(k['correlation'])
        assert rt is not None, f"kernel {k['correlation']} has no runtime match"
        ext_id = rt['ext_id']
        cpu_op = cpu_ops.get(ext_id)
        assert cpu_op is not None, f"ext_id {ext_id} not in cpu_ops"

        rows.append({
            'name': k['name'],
            'ts': k['ts'],
            'dur': k['dur'],
            'ts_cuda_runtime': rt['ts'],
            'name_cpu_op': cpu_op['name'],
            'operator-name': labels.get(ext_id),
            'layer': eid_to_layer.get(ext_id),
            'iteration': eid_to_iter.get(ext_id),
        })

    return pd.DataFrame(rows)


def parse_trace(filename):
    """Full pipeline: parse trace -> kernel DataFrame."""
    cpu_ops, annotations, fwdbwd, kernels, runtime = parse(filename)
    layer_ranges, iter_ranges, op_ranges = classify_annotations(annotations)

    # Step 1: fwd/bwd linking (also assigns layers to both fwd and bwd ops)
    labeled, fwdbwd_layers = link_fwdbwd(cpu_ops, op_ranges, fwdbwd, layer_ranges)

    # Step 2: promote labels up to parent cpu_ops with same sequence number
    labeled = promote(cpu_ops, labeled)
    fwdbwd_layers = promote(cpu_ops, fwdbwd_layers)

    # Step 3: propagate labels and layers to children
    labeled = propagate(cpu_ops, labeled)
    fwdbwd_layers = propagate(cpu_ops, fwdbwd_layers)

    # Step 4: assign annotations to remaining unlabeled ops
    labeled = assign_unlabeled(cpu_ops, labeled, op_ranges)

    # Step 5: build kernel DataFrame
    return build_kernel_df(cpu_ops, kernels, runtime, labeled, layer_ranges, iter_ranges, fwdbwd_layers)


def get_pivoted(csv_filename):
    """Pivot rocprofv3 CSV from long format to wide format.

    rocprofv3 --pmc outputs one row per (kernel, counter). This pivots so
    each kernel dispatch is one row with counter names as columns.
    """
    df = pd.read_csv(csv_filename)
    meta_cols = [c for c in df.columns if c not in ("Counter_Name", "Counter_Value")]
    pivoted = df.pivot_table(
        index=meta_cols, columns="Counter_Name", values="Counter_Value",
    ).sort_values("Start_Timestamp", ascending=True).reset_index()
    return pivoted


def get_combined_counters(csv_list):
    """Merge multiple counter CSV batches for one GPU.

    Each batch collected different counters from the same workload re-run.
    Joins on [Kernel_Name, _mi] where _mi is the instance index within
    each kernel name.
    """
    kname = "Kernel_Name"
    df_combined = None
    for cur_csv in csv_list:
        df_cur = get_pivoted(cur_csv)
        df_cur["_mi"] = df_cur.groupby(kname).cumcount()
        if df_combined is None:
            df_combined = df_cur
        else:
            df_combined = df_combined.merge(
                df_cur, on=[kname, "_mi"], how="left", suffixes=("", "_new"),
            )
            df_combined = df_combined.loc[
                :, ~df_combined.columns.str.endswith("_new")]
    assert df_combined is not None, "no counter CSV files provided"
    df_combined = df_combined.drop(columns=["_mi"])
    return df_combined


def merge_counters(df_ts, counter_batches):
    """Join hardware counter data with trace pickle.

    Args:
        df_ts: Trace DataFrame (from ts.pkl)
        counter_batches: List of lists -- each inner list is one batch of
            CSV files (sorted = GPU order). E.g.:
            [["batch0/gpu0.csv", "batch0/gpu1.csv"],
             ["batch1/gpu0.csv", "batch1/gpu1.csv"]]
    """
    # Transpose from per-batch to per-GPU
    per_gpu = [list(fns) for fns in zip(*[sorted(b) for b in counter_batches])]
    n_counter_gpus = len(per_gpu)

    gpus = sorted(df_ts["gpu"].unique())
    if len(gpus) != n_counter_gpus:
        logger.warning(f"{len(gpus)} GPUs in trace but {n_counter_gpus} counter files")

    # Load and combine counters per GPU
    df_cntr = pd.concat(
        [df.assign(gpu=i) for i, df in enumerate(
            map(get_combined_counters, per_gpu))],
        ignore_index=True,
    )
    logger.info(f"Loaded {len(df_cntr)} counter rows across {n_counter_gpus} GPUs")

    kname = "Kernel_Name"

    # Select first iteration from trace for matching
    first_iter = df_ts["iteration"].dropna().unique().min()
    logger.info(f"Matching counters against iteration {first_iter}")
    match_mask = df_ts["iteration"] == first_iter
    df_match = df_ts[match_mask].copy()
    df_rest = df_ts[~match_mask].copy()

    # Warn about kernel name mismatches
    ts_names = set(df_match["name"])
    cntr_names = set(df_cntr[kname])
    cntr_only = cntr_names - ts_names
    ts_only = ts_names - cntr_names
    if cntr_only:
        logger.warning(f"{len(cntr_only)} kernels in counters but not trace")
    if ts_only:
        logger.warning(f"{len(ts_only)} kernels in trace but not counters")

    # Remove mismatched kernels from counter side
    df_cntr = df_cntr[~df_cntr[kname].isin(cntr_only)]

    # Assign reverse-cumcount _mi (count from end so late dispatches align)
    left_keys = ["name", "gpu"]
    right_keys = [kname, "gpu"]

    df_match["_mi"] = (
        df_match.iloc[::-1].groupby(left_keys).cumcount().iloc[::-1]
    )
    df_cntr["_mi"] = (
        df_cntr.iloc[::-1].groupby(right_keys).cumcount().iloc[::-1]
    )

    # Merge
    df_merged = df_match.merge(
        df_cntr,
        left_on=left_keys + ["_mi"],
        right_on=right_keys + ["_mi"],
        how="left",
        suffixes=("", "_y"),
    )

    # Drop temp and duplicate columns
    drop_cols = ["_mi", kname]
    drop_cols += [c for c in df_merged.columns if c.endswith("_y")]
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

    # Rejoin with non-matched iterations
    df_result = pd.concat([df_merged, df_rest], ignore_index=True)
    df_result = df_result.sort_values("ts").reset_index(drop=True)

    # Report
    counter_cols = [c for c in df_merged.columns if c not in df_ts.columns and c != "_mi"]
    n_matched = df_merged[counter_cols[0]].notna().sum() if counter_cols else 0
    logger.info(f"Merged {n_matched}/{len(df_match)} kernels with counters")
    logger.info(f"Counter columns: {counter_cols}")

    return df_result


def merge_device_counters(device_dir, output):
    """Store device sampling CSVs into a pickle. No transformations.

    Reads counter_samples_rank*.csv and kernel_traces_rank*.csv from
    chopper_device_counters*/ subdirs.
    """
    from pathlib import Path

    device_dir = Path(device_dir)
    groups = sorted(device_dir.glob("chopper_device_counters*"))
    assert groups, f"No chopper_device_counters* dirs in {device_dir}"

    counter_samples = {}
    kernel_traces = {}
    counter_to_group = {}  # {counter_name: group_index}

    for gi, group_dir in enumerate(groups):
        counter_files = sorted(group_dir.glob("counter_samples_rank*.csv"))
        trace_files = sorted(group_dir.glob("kernel_traces_rank*.csv"))
        assert counter_files, f"No counter_samples_rank*.csv in {group_dir}"
        assert trace_files, f"No kernel_traces_rank*.csv in {group_dir}"

        for cf in counter_files:
            rank = int(cf.stem.split("_rank")[1])
            df = pd.read_csv(cf)
            assert len(df) > 0, f"Empty counter file: {cf}"
            counter_samples[(gi, rank)] = df
            for name in df["counter_name"].unique():
                counter_to_group[name] = gi

        for tf in trace_files:
            rank = int(tf.stem.split("_rank")[1])
            df = pd.read_csv(tf)
            if len(df) > 0:
                kernel_traces[(gi, rank)] = df

    result = {
        "counter_samples": counter_samples,
        "kernel_traces": kernel_traces,
        "counter_to_group": counter_to_group,
    }

    import pickle
    with open(output, "wb") as f:
        pickle.dump(result, f)

    ranks = sorted(set(r for _, r in counter_samples.keys()))
    total_samples = sum(len(df) for df in counter_samples.values())
    total_kernels = sum(len(df) for df in kernel_traces.values())
    logger.info(f"Wrote {output}: {total_samples} counter samples, {total_kernels} kernel records")
    logger.info(f"  Groups: {len(groups)}, Ranks: {ranks}")
    logger.info(f"  Counter -> group: {counter_to_group}")


def main(traces, pickles, counters, device_dir, output):
    if device_dir:
        merge_device_counters(device_dir, output)
        return

    assert not (traces and pickles), "pass either -t or -p, not both"
    assert traces or pickles, "pass -t (traces) or -p (pickles)"
    assert not (traces and counters), "cannot use -c with -t; merge traces first, then add counters with -p"

    t0 = time.time()

    if pickles and counters:
        assert len(pickles) == 1, "pass exactly one pickle with -c"
        df = pd.read_pickle(pickles[0])
        df = merge_counters(df, counters)
        df.to_pickle(output)
        t1 = time.time()
        logger.info(f"Merged counters into {len(df)} kernels -> {output} in {t1-t0:.2f}s")
        return

    if pickles:
        dfs = [pd.read_pickle(p) for p in pickles]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('ts').reset_index(drop=True)
        df.to_pickle(output)
        t1 = time.time()
        logger.info(f"Merged {len(pickles)} pickles, {len(df)} kernels -> {output} in {t1-t0:.2f}s")
        return

    if len(traces) == 1:
        df = parse_trace(traces[0])
        df['gpu'] = 0
    else:
        with ProcessPoolExecutor(max_workers=len(traces)) as ex:
            dfs = list(ex.map(parse_trace, traces))
        for i, d in enumerate(dfs):
            d['gpu'] = i
        df = pd.concat(dfs, ignore_index=True)

    df = df.sort_values('ts').reset_index(drop=True)
    df.to_pickle(output)
    t1 = time.time()

    logger.info(f"Wrote {len(df)} kernels to {output} in {t1-t0:.2f}s")
    logger.info(f"Columns: {list(df.columns)}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=(
        "Merge PyTorch traces into a kernel pickle, combine pickles, "
        "join hardware counters, or merge device sampling data.\n"
        "  1) -t trace*.json -o ts.pkl              (parse traces)\n"
        "  2) -p iter*.pkl -o ts.pkl                 (combine pickles)\n"
        "  3) -p ts.pkl -c batch0/*.csv -o out.pkl   (add counters)\n"
        "  4) --device-dir outputs/run -o device.pkl  (device sampling)\n"
    ))
    parser.add_argument('-t', '--traces', nargs='+')
    parser.add_argument('-p', '--pickles', nargs='+')
    parser.add_argument('-c', '--counters', action='append', nargs='+',
                        help='Counter CSV files (one -c per batch, sorted = GPU order)')
    parser.add_argument('--device-dir',
                        help='Device sampling output directory (chopper --device)')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    main(sorted(args.traces) if args.traces else None,
         sorted(args.pickles) if args.pickles else None,
         args.counters,
         args.device_dir,
         args.output)
