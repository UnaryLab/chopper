"""Compare kernel durations between device-counter and trace-only runs.

Reads kernel_traces.csv from both runs, finds the most common kernel,
and compares its timing between the two runs.
"""

import argparse
import glob

import pandas as pd


def load_traces(directory: str) -> pd.DataFrame:
    """Load and concatenate all kernel_traces CSVs under a directory."""
    pattern = f"{directory}/**/kernel_traces*.csv"
    files = glob.glob(pattern, recursive=True)
    assert len(files) > 0, f"No kernel_traces*.csv found under {directory}"
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare kernel times: device counters vs trace-only"
    )
    parser.add_argument("--counters-dir", required=True,
                        help="Directory with counter-sampling run output")
    parser.add_argument("--trace-only-dir", required=True,
                        help="Directory with trace-only run output")
    args = parser.parse_args()

    counters_df = load_traces(args.counters_dir)
    trace_df = load_traces(args.trace_only_dir)

    # Show all kernel types with counts (use trace-only as reference)
    counts = trace_df["kernel_name"].value_counts()
    print("=" * 70)
    print("KERNEL TYPES (trace-only run)")
    print("=" * 70)
    for name, count in counts.items():
        print(f"  {count:6d}x  {name[:40]}")
    print()

    # Compare the most common kernel
    top_kernel = counts.index[0]
    top_count = counts.iloc[0]
    print("=" * 70)
    print(f"COMPARING: {top_kernel[:40]}")
    print(f"  ({top_count} dispatches per run)")
    print("=" * 70)

    ck = counters_df[counters_df["kernel_name"] == top_kernel]["duration_ns"]
    tk = trace_df[trace_df["kernel_name"] == top_kernel]["duration_ns"]

    print(f"  {'':20s} {'counters':>14s} {'trace_only':>14s} {'ratio':>14s}")
    print(f"  {'count':20s} {len(ck):14d} {len(tk):14d}")

    for label, func in [("mean", "mean"), ("median", "median"),
                         ("std", "std"), ("min", "min"), ("max", "max")]:
        cv = getattr(ck, func)()
        tv = getattr(tk, func)()
        ratio = tv / cv if cv > 0 else float('nan')
        print(f"  {label + ' (ns)':20s} {cv:14.1f} {tv:14.1f} {ratio:14.4f}")
    print()


if __name__ == "__main__":
    main()
