#!/bin/bash
# Compare kernel times with and without device counter sampling.
#
# Runs the same workload twice:
#   1. With device counters (one group)
#   2. Trace-only (CHOPPER_TRACE_ONLY=1, no PMC sampling)
#
# Then uses pandas to compare per-kernel durations.
#
# Usage:
#   ./compare_overhead.sh                 # PyTorch GEMM (default)
#   ./compare_overhead.sh --hipblaslt     # Native hipBLASLt GEMM
#   ./compare_overhead.sh --sleep-ms 4    # Adjust idle gap

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

MODE="pytorch"
ITERS=200
WARMUP=10
SLEEP_MS=4
SAMPLE_MS=1
M=8192
N=8192
K=8192
OUTDIR=overhead_compare
COUNTERS="SQ_WAVES"

while [[ $# -gt 0 ]]; do
    case $1 in
        --hipblaslt) MODE="hipblaslt"; shift ;;
        --iters) ITERS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --sleep-ms) SLEEP_MS="$2"; shift 2 ;;
        --sample-ms) SAMPLE_MS="$2"; shift 2 ;;
        -m) M="$2"; shift 2 ;;
        -n) N="$2"; shift 2 ;;
        -k) K="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --counters) COUNTERS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

rm -rf "$OUTDIR"

if [[ "$MODE" == "hipblaslt" ]]; then
    if [[ ! -f hipblaslt_gemm ]]; then
        echo "Building hipblaslt_gemm..."
        hipcc -std=c++17 -O2 -DROCM_USE_FLOAT16 \
            -D__HIP_PLATFORM_AMD__ -o hipblaslt_gemm hipblaslt_gemm.cpp \
            -lhipblaslt
    fi
    BENCH_CMD="./hipblaslt_gemm $M $N $K"
else
    BENCH_CMD="$PYTHON bench.py --mode gemm \
        --iters $ITERS --warmup $WARMUP --sleep-ms $SLEEP_MS \
        --m $M --n $N --k $K"
fi

echo "=== Run 1: with device counters ($COUNTERS) ==="
$PYTHON -m chopper.profile.collect --device \
    --counters $COUNTERS \
    --sample-ms "$SAMPLE_MS" \
    --output-dir "$OUTDIR/counters" \
    -- $BENCH_CMD

echo ""
echo "=== Run 2: trace-only (no device counters) ==="
CHOPPER_TRACE_ONLY=1 \
$PYTHON -m chopper.profile.collect --device \
    --counters $COUNTERS \
    --sample-ms "$SAMPLE_MS" \
    --output-dir "$OUTDIR/trace_only" \
    -- $BENCH_CMD

echo ""
echo "=== Comparing kernel times ==="
$PYTHON compare_overhead.py \
    --counters-dir "$OUTDIR/counters" \
    --trace-only-dir "$OUTDIR/trace_only"

echo "Done. Raw data in $OUTDIR/"
