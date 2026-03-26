#!/usr/bin/env bash
set -euo pipefail

output=${NUMBL_BROWSER_WASM_OUTPUT:?NUMBL_BROWSER_WASM_OUTPUT is required}
opt_level=${NUMBL_BROWSER_WASM_OPT_LEVEL:--O3}
lto=${NUMBL_BROWSER_WASM_LTO:-1}
simd=${NUMBL_BROWSER_WASM_SIMD:-1}
fast_math=${NUMBL_BROWSER_WASM_FAST_MATH:-0}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
wrapper="${repo_root}/browser-wasm/wrappers/fftw-fft/fftw_fft_wrapper.c"

compile_flags=(
  "${opt_level}"
  -DNDEBUG
  -fno-fast-math
  -fno-math-errno
  -ffp-contract=on
)
if [[ "${lto}" == "1" ]]; then
  compile_flags+=(-flto)
fi
if [[ "${simd}" == "1" ]]; then
  compile_flags+=(-msimd128)
fi
if [[ "${fast_math}" == "1" ]]; then
  compile_flags+=(-ffast-math)
fi

mkdir -p "$(dirname "${output}")"

emcc \
  "${wrapper}" \
  "${compile_flags[@]}" \
  -s STANDALONE_WASM \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s FILESYSTEM=0 \
  -s ERROR_ON_UNDEFINED_SYMBOLS=1 \
  -s USE_FFTW=3 \
  -s INITIAL_MEMORY=16777216 \
  -s 'EXPORTED_FUNCTIONS=["_malloc","_free","_numbl_fft1d_f64"]' \
  --no-entry \
  -o "${output}"
