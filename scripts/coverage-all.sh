#!/usr/bin/env bash
# Generate a combined coverage report from both unit tests and integration
# test scripts.  Both suites run under vitest's native V8 coverage provider
# (via test-scripts.test.ts which wraps the .m scripts), so a single
# `vitest run --coverage` produces one consistent report with no merging.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

rm -rf coverage coverage-all

npx vitest run --coverage

# Copy into coverage-all/ for anything that reads that dir
cp -r coverage coverage-all

echo "=== Coverage report written to coverage-all/ ==="
