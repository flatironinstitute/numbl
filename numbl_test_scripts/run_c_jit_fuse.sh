#!/usr/bin/env bash
# Run all .m test scripts under --opt 2 --fuse to verify that the fused
# per-element C-JIT codegen preserves correctness.
#
# Requirements: a C compiler (cc or $NUMBL_CC) and Node API headers.
#
# Exit code: 0 = all passed, 1 = one or more failed.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running tests (--opt 2 --fuse) ==="
npx tsx "$SCRIPT_DIR/../src/cli.ts" run-tests "$SCRIPT_DIR" --opt 2 --fuse
