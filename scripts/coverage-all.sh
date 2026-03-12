#!/usr/bin/env bash
# Collect coverage from both vitest and test scripts, then merge into a
# single report under coverage-all/.
#
# Both suites run under c8 so the coverage maps use identical
# instrumentation and can be merged cleanly by nyc.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COVERAGE_EXCLUDE=(
  --exclude 'node_modules/**'
  --exclude 'dist/**'
  --exclude '**/ts-lapack/**'
  --exclude 'src/cli*.ts'
  --exclude 'src/components/**'
  --exclude 'src/db/**'
  --exclude 'src/hooks/**'
  --exclude 'src/mip/**'
  --exclude 'src/mip-directives*.ts'
)

NYX_EXCLUDE=(
  -x 'node_modules/**'
  -x 'dist/**'
  -x '**/ts-lapack/**'
  -x 'src/cli*.ts'
  -x 'src/components/**'
  -x 'src/db/**'
  -x 'src/hooks/**'
  -x 'src/mip/**'
  -x 'src/mip-directives*.ts'
)

# Clean previous artifacts
rm -rf coverage coverage-scripts .nyc_output coverage-all

# 1. Vitest coverage via c8 (produces coverage/coverage-final.json)
echo "=== Running vitest with coverage (c8) ==="
npx c8 --reporter=json --report-dir=coverage "${COVERAGE_EXCLUDE[@]}" \
  npx vitest run

# 2. Script tests coverage via c8 (produces coverage-scripts/coverage-final.json)
echo "=== Running test scripts with coverage ==="
npx c8 --reporter=json --report-dir=coverage-scripts "${COVERAGE_EXCLUDE[@]}" \
  npx tsx src/cli.ts run-tests numbl_test_scripts

# 3. Merge
echo "=== Merging coverage ==="
mkdir -p .nyc_output
cp coverage/coverage-final.json .nyc_output/vitest.json
cp coverage-scripts/coverage-final.json .nyc_output/scripts.json
npx nyc merge .nyc_output coverage-all/coverage-final.json

# 4. Generate merged report
npx nyc report --temp-dir .nyc_output \
  "${NYX_EXCLUDE[@]}" \
  --reporter=text --reporter=json --report-dir=coverage-all

# Clean up intermediates
rm -rf .nyc_output coverage-scripts

echo "=== Merged coverage report written to coverage-all/ ==="
