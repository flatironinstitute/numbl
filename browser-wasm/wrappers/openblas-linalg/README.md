This wrapper builds a standalone browser Wasm module around an OpenBLAS
source tree compiled with Emscripten.

The exported C ABI is:

- `numbl_matmul_f64(const double *a, size_t m, size_t k, const double *b, size_t n, double *out)`
- `numbl_inv_f64(const double *data, size_t n, double *out)`
- `numbl_linsolve_f64(const double *a, size_t m, size_t n, const double *b, size_t nrhs, double *out)`

The browser `linsolve` kernel currently accelerates only the square `dgesv`
path. Non-square systems fall back to Numbl's TypeScript implementation in the
browser bridge so the Wasm module does not need to pull in the much larger
`dgels` dependency tree.

`build-openblas-linalg.sh` expects:

- `NUMBL_BROWSER_WASM_SOURCE_ROOT` to point at an OpenBLAS checkout
- `NUMBL_BROWSER_WASM_OUTPUT` to point at the destination `.wasm`
- Optional: `NUMBL_BROWSER_WASM_CLEAN=1` to force a clean OpenBLAS rebuild
- Optional: `NUMBL_BROWSER_WASM_OPT_LEVEL=-O3` (default) to override the
  release optimization level
- Optional: `NUMBL_BROWSER_WASM_LTO=0` to disable link-time optimization
- Optional: `NUMBL_BROWSER_WASM_SIMD=0` to disable Wasm SIMD/autovectorization
- Optional: `NUMBL_BROWSER_WASM_FAST_MATH=1` to opt into `-ffast-math`

The build uses the OpenBLAS `WASM128_GENERIC` target with `NOFORTRAN=1`, which
enables the C-translated LAPACK subset (`C_LAPACK`) needed for `inv` and
`linsolve`. It also forces `NUM_THREADS=1` and
`USE_SIMPLE_THREADED_LEVEL3=1` so the Wasm build avoids OpenBLAS threaded
kernel paths that do not fit this browser target. To keep iteration practical,
the wrapper builds the OpenBLAS `libs` and `netlib` targets directly instead of
the default `all` target, which would also build OpenBLAS test binaries, and
links only the narrow LAPACKE work shims needed by the exported ABI. Release
builds default to `-O3 -flto -msimd128 -DNDEBUG` plus the safe floating-point
profile `-fno-fast-math -fno-math-errno -ffp-contract=on`. `-ffast-math` is
left opt-in because it can relax floating-point semantics in ways that are
risky for linear algebra code.
