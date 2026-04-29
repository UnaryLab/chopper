#!/bin/bash
# Stroboscopic device counter sampling for GEMM benchmarks.
#
# Usage:
#   ./go.sh                    # PyTorch GEMM (default)
#   ./go.sh --hipblaslt        # Native hipBLASLt GEMM
#   ./go.sh --sleep-ms 4       # Adjust idle gap between kernels

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
OUTDIR=output

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
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

READ_COUNTERS="TCC_BUBBLE TCC_EA0_RDREQ_32B TCC_EA0_RDREQ"
WRITE_COUNTERS="TCC_EA0_WRREQ_64B TCC_EA0_WRREQ"

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

echo "=== Collecting read counters (${M}x${N}x${K} GEMM) ==="
$PYTHON -m chopper.profile.collect --device \
    --counters $READ_COUNTERS \
    --sample-ms "$SAMPLE_MS" \
    --output-dir "$OUTDIR/reads" \
    -- $BENCH_CMD

echo "=== Collecting write counters ==="
$PYTHON -m chopper.profile.collect --device \
    --counters $WRITE_COUNTERS \
    --sample-ms "$SAMPLE_MS" \
    --output-dir "$OUTDIR/writes" \
    -- $BENCH_CMD

echo ""
echo "=== Plotting ==="
$PYTHON plot_counters.py \
    --reads-dir "$OUTDIR/reads" \
    --writes-dir "$OUTDIR/writes" \
    -m "$M" -n "$N" -k "$K" --save

echo "Done. Output in $OUTDIR/, plot in plot_bytes_stitched.png"
