#!/usr/bin/env bash
# Run all .m test scripts via the internal test runner.
# Exit code: 0 = all passed, 1 = one or more failed.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running tests (no optimization) ==="
npx tsx "$SCRIPT_DIR/../src/cli.ts" run-tests "$SCRIPT_DIR"

echo "=== Running tests (--opt 1) ==="
npx tsx "$SCRIPT_DIR/../src/cli.ts" run-tests "$SCRIPT_DIR" --opt 1
