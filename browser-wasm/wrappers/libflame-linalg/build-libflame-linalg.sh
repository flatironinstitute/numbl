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
wrapper="${repo_root}/browser-wasm/wrappers/libflame-linalg/libflame_linalg_wrapper.c"
build_dir=${NUMBL_BROWSER_WASM_BUILD_DIR:-"${source_root}/wasm-build-release"}
expected_host=wasm32-unknown-linux-gnu
expected_archive="${build_dir}/lib/${expected_host}/libflame.a"

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

configure_args=(
  --host="${expected_host}"
  --disable-dynamic-build
  --enable-static-build
  --disable-autodetect-f77-ldflags
  --disable-autodetect-f77-name-mangling
  --disable-multithreading
  --disable-supermatrix
  --disable-gpu
  --disable-hip
  --disable-vector-intrinsics
  --disable-warnings
  --disable-debug
  --disable-memory-leak-counter
  --enable-builtin-blas
  --enable-lapack2flame
  --enable-legacy-lapack
  --disable-external-lapack-for-subproblems
  --disable-external-lapack-interfaces
  --disable-cblas-interfaces
)
if [[ "${lto}" == "1" ]]; then
  configure_args+=(--enable-lto)
fi

if [[ ! -f "${source_root}/configure" ]]; then
  echo "libFLAME source root does not contain configure: ${source_root}" >&2
  exit 1
fi

if [[ "${NUMBL_BROWSER_WASM_CLEAN:-0}" == "1" ]]; then
  rm -rf "${build_dir}"
fi

mkdir -p "${build_dir}"

if [[ -f "${build_dir}/config.sys_type" ]]; then
  current_host=$(tr -d '\n' < "${build_dir}/config.sys_type")
  if [[ "${current_host}" != "${expected_host}" ]]; then
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
  fi
fi

if [[ ! -f "${expected_archive}" ]]; then
  (
    cd "${build_dir}"
    if [[ ! -f Makefile ]]; then
      ac_cv_prog_cc_cross=yes \
      FC=true \
      F77=true \
      CFLAGS="${compile_flags_str}" \
      CXXFLAGS="${compile_flags_str}" \
      LDFLAGS="${compile_flags_str}" \
      emconfigure ../configure \
        "${configure_args[@]}" \
        CC=emcc \
        AR=emar \
        RANLIB=emranlib
    fi
    if [[ -f "config/${expected_host}/config.mk" ]]; then
      sed -i 's/ -fno-semantic-interposition//g' "config/${expected_host}/config.mk"
    fi
    emmake make -f ../Makefile PYTHON=python3 -j"${jobs}"
  )
fi

if [[ ! -f "${expected_archive}" ]]; then
  echo "libFLAME build did not produce ${expected_archive}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"

emcc \
  "${wrapper}" \
  "${expected_archive}" \
  -I"${source_root}" \
  "${compile_flags[@]}" \
  -s STANDALONE_WASM \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s FILESYSTEM=0 \
  -s ERROR_ON_UNDEFINED_SYMBOLS=1 \
  -s INITIAL_MEMORY=33554432 \
  -s 'EXPORTED_FUNCTIONS=["_malloc","_free","_numbl_matmul_f64","_numbl_inv_f64","_numbl_linsolve_f64"]' \
  --no-entry \
  -o "${output}"
