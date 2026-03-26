#!/usr/bin/env bash
set -euo pipefail

source_root=${NUMBL_BROWSER_WASM_SOURCE_ROOT:?NUMBL_BROWSER_WASM_SOURCE_ROOT is required}
output=${NUMBL_BROWSER_WASM_OUTPUT:?NUMBL_BROWSER_WASM_OUTPUT is required}
opt_level=${NUMBL_BROWSER_WASM_OPT_LEVEL:--O3}
lto=${NUMBL_BROWSER_WASM_LTO:-1}
simd=${NUMBL_BROWSER_WASM_SIMD:-1}
fast_math=${NUMBL_BROWSER_WASM_FAST_MATH:-0}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
wrapper="${repo_root}/browser-wasm/wrappers/ducc0-fft/ducc0_fft_wrapper.cpp"

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

if [[ ! -f "${source_root}/src/ducc0/fft/fft.h" ]]; then
  echo "DUCC0 source root does not contain src/ducc0/fft/fft.h: ${source_root}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"

em++ \
  "${wrapper}" \
  "${source_root}/src/ducc0/infra/threading.cc" \
  "${source_root}/src/ducc0/infra/mav.cc" \
  "${source_root}/src/ducc0/infra/system.cc" \
  "${source_root}/src/ducc0/infra/types.cc" \
  "${source_root}/src/ducc0/infra/string_utils.cc" \
  "${source_root}/src/ducc0/infra/communication.cc" \
  "${source_root}/src/ducc0/fft/fft_inst1.cc" \
  "${source_root}/src/ducc0/fft/fft_inst2.cc" \
  -I"${source_root}/src" \
  -std=c++17 \
  "${compile_flags[@]}" \
  -s STANDALONE_WASM \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s FILESYSTEM=0 \
  -s ERROR_ON_UNDEFINED_SYMBOLS=1 \
  -s INITIAL_MEMORY=16777216 \
  -s 'EXPORTED_FUNCTIONS=["_malloc","_free","_numbl_fft1d_f64","_numbl_fft_along_dim_f64"]' \
  --no-entry \
  -o "${output}"
