#!/usr/bin/env bash
# Run flodl (Rust) and PyTorch (Python) benchmarks, then compare.
#
# Usage: benchmarks/run.sh [--cpu] [--tier1|--tier2|--bench NAME]
#
# By default runs CUDA benchmarks. Pass --cpu for CPU-only mode.
# Expects to run inside the bench Docker container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

# Parse --cpu flag (pass the rest through to benchmarks)
CPU_MODE=0
PASS_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--cpu" ]; then
        CPU_MODE=1
    else
        PASS_ARGS+=("$arg")
    fi
done

if [ "$CPU_MODE" -eq 1 ]; then
    MODE="cpu"
    CARGO_FEATURES=""
    echo "=== Benchmark mode: CPU ==="
else
    MODE="cuda"
    CARGO_FEATURES="--features cuda"
    echo "=== Benchmark mode: CUDA ==="
fi
echo ""

echo "=== Building flodl benchmarks (release, $MODE) ==="
cargo build --manifest-path "$SCRIPT_DIR/Cargo.toml" --release $CARGO_FEATURES 2>&1
echo ""

echo "=== Running flodl (Rust) benchmarks ==="
"$SCRIPT_DIR/target/release/flodl-bench" --json "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}" > /tmp/flodl_bench.json 2>/dev/stderr
echo ""

echo "=== Running PyTorch (Python) benchmarks ==="
cd "$SCRIPT_DIR/python"

# Force CPU mode in Python if requested
if [ "$CPU_MODE" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="" python3 run_all.py --json "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}" > /tmp/pytorch_bench.json 2>/dev/stderr
else
    python3 run_all.py --json "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}" > /tmp/pytorch_bench.json 2>/dev/stderr
fi
echo ""

echo "=== Comparison ($MODE) ==="
python3 compare.py /tmp/flodl_bench.json /tmp/pytorch_bench.json
