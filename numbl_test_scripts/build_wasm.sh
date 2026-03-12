#!/bin/bash
# Build all .wasm files from .c sources in numbl_test_scripts.
# Requires Emscripten (emcc) to be installed and on PATH.
#
# Usage: bash numbl_test_scripts/build_wasm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v emcc &> /dev/null; then
  echo "Error: emcc (Emscripten) not found on PATH." >&2
  echo "Install from https://emscripten.org/docs/getting_started/downloads.html" >&2
  exit 1
fi

count=0
while IFS= read -r -d '' cfile; do
  dir="$(dirname "$cfile")"
  base="$(basename "$cfile" .c)"
  wasmfile="$dir/$base.wasm"

  # Only compile if a matching .js file exists (this is a wasm companion)
  jsfile="$dir/$base.js"
  if [ ! -f "$jsfile" ]; then
    continue
  fi

  echo "Compiling $cfile -> $wasmfile"
  emcc "$cfile" -O2 \
    -s STANDALONE_WASM \
    --no-entry \
    -o "$wasmfile"
  count=$((count + 1))
done < <(find "$SCRIPT_DIR" -name '*.c' -print0)

echo "Built $count .wasm file(s)."
