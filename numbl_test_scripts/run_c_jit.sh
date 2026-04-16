#!/usr/bin/env bash
# Run all .m test scripts under --opt 2 (C-JIT) to verify that the C-JIT
# path preserves correctness. Most scripts exercise IR the C path can't
# emit and transparently fall back to JS-JIT — that's expected; the goal
# here is end-to-end parity with --opt 1, not full C coverage.
#
# Requirements: a C compiler (cc or $NUMBL_CC) and Node API headers.
# Headers are installed automatically on first run via `npx node-gyp install`.
#
# Exit code: 0 = all passed, 1 = one or more failed.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running tests (--opt 2: C-JIT with JS-JIT fallback) ==="
npx tsx "$SCRIPT_DIR/../src/cli.ts" run-tests "$SCRIPT_DIR" --opt 2
