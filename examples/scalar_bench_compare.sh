#!/usr/bin/env bash
# Run examples/scalar_bench.m under numbl's three --opt levels and under
# MATLAB, extracting the wall time from each run so the speedup across
# configurations is directly visible.
#
# Usage:  bash examples/scalar_bench_compare.sh
#         MATLAB=/path/to/matlab bash examples/scalar_bench_compare.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="$SCRIPT_DIR/scalar_bench.m"

MATLAB_BIN="${MATLAB:-matlab}"

run_numbl() {
    local opt="$1"
    local label="$2"
    echo "=== numbl --opt $opt ($label) ==="
    ( cd "$REPO_ROOT" && \
      npx tsx src/cli.ts run "$BENCH" --opt "$opt" 2>&1 ) | \
      grep -E '^(result|elapsed|throughput|C-JIT: cc)'
    echo ""
}

run_matlab() {
    if ! command -v "$MATLAB_BIN" >/dev/null 2>&1; then
        echo "=== MATLAB (skipped: '$MATLAB_BIN' not on PATH) ==="
        echo ""
        return
    fi
    echo "=== MATLAB -batch ==="
    ( cd "$REPO_ROOT" && \
      "$MATLAB_BIN" -batch "run('$BENCH')" 2>&1 ) | \
      grep -E '^(result|elapsed|throughput|warmup|N=)'
    echo ""
}

run_numbl 0 "interpreter"
run_numbl 1 "JS-JIT"
run_numbl 2 "C-JIT"
run_matlab
