#!/usr/bin/env bash
# Run all .m test scripts under --opt 2 with --check-c-jit-parity, so any
# case where the C-JIT declines but JS-JIT would have compiled becomes a
# hard failure. Use this to enumerate C-JIT parity gaps as an actionable
# punch list; expect many failures until coverage catches up.
#
# Requirements: a C compiler (cc or $NUMBL_CC) and Node API headers.
#
# Exit code: 0 = all passed (full parity), 1 = one or more gaps/failures.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running tests (--opt 2 --check-c-jit-parity) ==="
npx tsx "$SCRIPT_DIR/../src/cli.ts" run-tests "$SCRIPT_DIR" --opt 2 --check-c-jit-parity
