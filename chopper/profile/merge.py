"""Full trace merge: parse PyTorch Chrome traces into a kernel DataFrame."""
import json
import time
import re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


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
            if fid not in fwdbwd: fwdbwd[fid] = {}
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


def main(traces, pickles, output):
    assert not (traces and pickles), "pass either -t or -p, not both"
    assert traces or pickles, "pass -t (traces) or -p (pickles)"

    t0 = time.time()

    if pickles:
        dfs = [pd.read_pickle(p) for p in pickles]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('ts').reset_index(drop=True)
        df.to_pickle(output)
        t1 = time.time()
        print(f"Merged {len(pickles)} pickles, {len(df)} kernels -> {output} in {t1-t0:.2f}s")
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

    print(f"Wrote {len(df)} kernels to {output} in {t1-t0:.2f}s")
    print(f"Columns: {list(df.columns)}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-t', '--traces', nargs='+')
    parser.add_argument('-p', '--pickles', nargs='+')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    main(sorted(args.traces) if args.traces else None,
         sorted(args.pickles) if args.pickles else None,
         args.output)
