#!/usr/bin/env bash
set -euo pipefail

source_root=${NUMBL_BROWSER_WASM_SOURCE_ROOT:?NUMBL_BROWSER_WASM_SOURCE_ROOT is required}
output=${NUMBL_BROWSER_WASM_OUTPUT:?NUMBL_BROWSER_WASM_OUTPUT is required}
jobs=${NUMBL_BROWSER_WASM_JOBS:-4}
opt_level=${NUMBL_BROWSER_WASM_OPT_LEVEL:--O3}
lto=${NUMBL_BROWSER_WASM_LTO:-1}
simd=${NUMBL_BROWSER_WASM_SIMD:-1}
fast_math=${NUMBL_BROWSER_WASM_FAST_MATH:-0}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
wrapper="${repo_root}/browser-wasm/wrappers/openblas-linalg/openblas_linalg_wrapper.c"
lapacke_root="${source_root}/lapack-netlib/LAPACKE"

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
printf -v compile_flags_str '%s ' "${compile_flags[@]}"
compile_flags_str=${compile_flags_str% }

if [[ ! -f "${source_root}/cblas.h" ]]; then
  echo "OpenBLAS source root does not contain cblas.h: ${source_root}" >&2
  exit 1
fi

if [[ "${NUMBL_BROWSER_WASM_CLEAN:-0}" == "1" ]]; then
  make -C "${source_root}" clean >/dev/null 2>&1 || true
fi

make -C "${source_root}" -j"${jobs}" \
  libs \
  netlib \
  TARGET=WASM128_GENERIC \
  NO_SHARED=1 \
  NUM_THREADS=1 \
  USE_THREAD=0 \
  USE_OPENMP=0 \
  USE_SIMPLE_THREADED_LEVEL3=1 \
  NOFORTRAN=1 \
  NO_LAPACKE=1 \
  BUILD_SINGLE=0 \
  BUILD_DOUBLE=1 \
  BUILD_COMPLEX=0 \
  BUILD_COMPLEX16=0 \
  COMMON_OPT="${compile_flags_str}" \
  LAPACK_CFLAGS="${compile_flags_str}" \
  CC=emcc \
  AR=emar \
  RANLIB=emranlib \
  HOSTCC=cc

libopenblas=$(find "${source_root}" -maxdepth 1 -name 'libopenblas*.a' | head -n 1)
if [[ -z "${libopenblas}" ]]; then
  echo "OpenBLAS build did not produce a static library in ${source_root}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"

lapacke_sources=(
  "${lapacke_root}/src/lapacke_dgetrf_work.c"
  "${lapacke_root}/src/lapacke_dgetri_work.c"
  "${lapacke_root}/src/lapacke_dgesv_work.c"
  "${lapacke_root}/utils/lapacke_xerbla.c"
  "${lapacke_root}/utils/lapacke_dge_trans.c"
)

emcc \
  "${wrapper}" \
  "${lapacke_sources[@]}" \
  "${libopenblas}" \
  -I"${source_root}" \
  -I"${source_root}/lapack-netlib/LAPACKE/include" \
  "${compile_flags[@]}" \
  -s STANDALONE_WASM \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s FILESYSTEM=0 \
  -s ERROR_ON_UNDEFINED_SYMBOLS=1 \
  -s INITIAL_MEMORY=67108864 \
  -s 'EXPORTED_FUNCTIONS=["_malloc","_free","_numbl_matmul_f64","_numbl_inv_f64","_numbl_linsolve_f64"]' \
  --no-entry \
  -o "${output}"
