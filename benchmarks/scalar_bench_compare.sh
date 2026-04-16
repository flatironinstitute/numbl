#!/usr/bin/env bash
# Run benchmarks/scalar_bench.m under numbl's three --opt levels and under
# MATLAB, extracting the wall time from each run so the speedup across
# configurations is directly visible.
#
# Usage:  bash benchmarks/scalar_bench_compare.sh
#         MATLAB=/path/to/matlab bash benchmarks/scalar_bench_compare.sh
#
# See benchmarks/scalar_bench.md for full description + typical results.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="$SCRIPT_DIR/scalar_bench.m"

MATLAB_BIN="${MATLAB:-matlab}"
OCTAVE_BIN="${OCTAVE:-octave}"

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

run_octave() {
    if ! command -v "$OCTAVE_BIN" >/dev/null 2>&1; then
        echo "=== Octave (skipped: '$OCTAVE_BIN' not on PATH) ==="
        echo ""
        return
    fi
    echo "=== Octave --eval scalar_bench ==="
    # Octave doesn't auto-invoke a function file passed as a positional
    # arg the way MATLAB -batch / numbl run do. cd into the benchmarks
    # dir so the function is on the path, then --eval it by name.
    ( cd "$SCRIPT_DIR" && \
      "$OCTAVE_BIN" --no-gui --quiet --eval "scalar_bench" 2>&1 ) | \
      grep -E '^(result|elapsed|throughput|warmup|N=)'
    echo ""
}

run_numbl 0 "interpreter"
run_numbl 1 "JS-JIT"
run_numbl 2 "C-JIT"
run_matlab
run_octave
