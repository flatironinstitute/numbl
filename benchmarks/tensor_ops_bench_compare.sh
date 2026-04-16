#!/usr/bin/env bash
# Run benchmarks/tensor_ops_bench.m under numbl's three --opt levels and
# under MATLAB and Octave, extracting the wall time and check values
# from each run so timing and correctness across configurations are
# directly visible.
#
# Usage:  bash benchmarks/tensor_ops_bench_compare.sh
#         MATLAB=/path/to/matlab bash benchmarks/tensor_ops_bench_compare.sh
#
# See benchmarks/tensor_ops_bench.md for full description + typical results.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="$SCRIPT_DIR/tensor_ops_bench.m"

MATLAB_BIN="${MATLAB:-matlab}"
OCTAVE_BIN="${OCTAVE:-octave}"

run_numbl() {
    local opt="$1"
    local label="$2"
    echo "=== numbl --opt $opt ($label) ==="
    ( cd "$REPO_ROOT" && \
      npx tsx src/cli.ts run "$BENCH" --opt "$opt" 2>&1 ) | \
      grep -E '^(Real|Comparisons|Reductions|Chained|elapsed|sum\(r\)|sum\(u\)|cmp_acc|red_acc|chain_acc|C-JIT: cc|N=)'
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
      grep -E '^(Real|Comparisons|Reductions|Chained|elapsed|sum\(r\)|sum\(u\)|cmp_acc|red_acc|chain_acc|warmup|N=)'
    echo ""
}

run_octave() {
    if ! command -v "$OCTAVE_BIN" >/dev/null 2>&1; then
        echo "=== Octave (skipped: '$OCTAVE_BIN' not on PATH) ==="
        echo ""
        return
    fi
    echo "=== Octave --eval tensor_ops_bench ==="
    ( cd "$SCRIPT_DIR" && \
      "$OCTAVE_BIN" --no-gui --quiet --eval "tensor_ops_bench" 2>&1 ) | \
      grep -E '^(Real|Comparisons|Reductions|Chained|elapsed|sum\(r\)|sum\(u\)|cmp_acc|red_acc|chain_acc|warmup|N=)'
    echo ""
}

run_numbl 0 "interpreter"
run_numbl 1 "JS-JIT"
run_numbl 2 "C-JIT"
run_matlab
run_octave
