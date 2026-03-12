#!/usr/bin/env bash
# Run all .m test scripts via the internal test runner.
# Exit code: 0 = all passed, 1 = one or more failed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
npx tsx "$SCRIPT_DIR/../src/cli.ts" run-tests "$SCRIPT_DIR"
