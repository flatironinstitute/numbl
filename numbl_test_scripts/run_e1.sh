#!/usr/bin/env bash
# Run all .m test scripts under --opt e1 to verify that the JS-JIT-with-
# inline-C-kernels path preserves correctness. Scripts whose IR escapes
# the e1 kernel whitelist transparently run under plain JS-JIT — that's
# expected; the goal here is end-to-end parity with --opt 1.
#
# Requirements: a C compiler (cc or $NUMBL_CC).
#
# Exit code: 0 = all passed, 1 = one or more failed.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running tests (--opt e1: JS-JIT with inline C kernels) ==="
npx tsx "$SCRIPT_DIR/../src/cli.ts" run-tests "$SCRIPT_DIR" --opt e1
