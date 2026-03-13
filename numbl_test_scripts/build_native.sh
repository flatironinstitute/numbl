#!/bin/bash
# Build native shared libraries (.so) from .c sources in numbl_test_scripts.
# Only compiles .c files whose matching .js file contains a "// native:" directive.
#
# Usage: bash numbl_test_scripts/build_native.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v gcc &> /dev/null; then
  echo "Error: gcc not found on PATH." >&2
  exit 1
fi

count=0
while IFS= read -r -d '' cfile; do
  dir="$(dirname "$cfile")"
  base="$(basename "$cfile" .c)"
  jsfile="$dir/$base.js"

  # Only compile if a matching .js file with a native directive exists
  if [ ! -f "$jsfile" ] || ! grep -q "^// native:" "$jsfile"; then
    continue
  fi

  sofile="$dir/$base.so"
  echo "Compiling $cfile -> $sofile"
  gcc -shared -fPIC -O2 -o "$sofile" "$cfile"
  count=$((count + 1))
done < <(find "$SCRIPT_DIR" -name '*.c' -print0)

echo "Built $count shared library file(s)."
