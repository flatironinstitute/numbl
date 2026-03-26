# Browser Wasm Kernels

Numbl already supports loading `.wasm` files as workspace assets for JS user
functions. This document defines a separate, first-class build path for browser
performance kernels such as FFT and BLAS/LAPACK wrappers.

## Goals

- Keep vendored third-party sources out of `src/`
- Give browser Wasm builds a stable output location
- Make build targets explicit and repeatable

## Layout

- Target manifests: `browser-wasm/targets/*.json`
- Local source mapping: `browser-wasm/local-sources.json` or per-target env vars
- Built artifacts: `public/wasm-kernels/`

## Build

List configured targets:

```bash
npm run build:browser-wasm -- --list
```

Build every configured target whose source tree exists:

```bash
npm run build:browser-wasm
```

The default build only includes targets whose manifest leaves
`enabledByDefault` unset or sets it to `true`. Experimental targets such as
`flame-blas-lapack`, `blas-lapack`, and `fftw-fft` are kept available for
explicit builds:

```bash
npm run build:browser-wasm -- flame-blas-lapack
npm run build:browser-wasm -- blas-lapack
npm run build:browser-wasm -- fftw-fft
```

Prepare the source roots used by the currently supported targets and export
them for the current shell:

```bash
while IFS= read -r line; do export "$line"; done < <(
  node scripts/prepare-browser-wasm-sources.mjs ducc0-fft flame-blas-lapack
)
npm run build:browser-wasm
```

Point a target at an external source tree with environment variables:

```bash
NUMBL_DUCC0_FFT_SRC_ROOT=/abs/path/to/ducc0 \
NUMBL_FLAME_BLAS_LAPACK_SRC_ROOT=/abs/path/to/libflame \
NUMBL_BLAS_LAPACK_SRC_ROOT=/abs/path/to/OpenBLAS \
npm run build:browser-wasm
```

Or create an untracked `browser-wasm/local-sources.json`:

```json
{
  "targets": {
    "ducc0-fft": {
      "sourceRoot": "/abs/path/to/ducc0"
    },
    "flame-blas-lapack": {
      "sourceRoot": "/abs/path/to/libflame"
    },
    "blas-lapack": {
      "sourceRoot": "/abs/path/to/OpenBLAS"
    }
  }
}
```

Build a specific target:

```bash
npm run build:browser-wasm -- ducc0-fft
```

The wrapper build scripts default to a release-oriented Emscripten profile:

- `-O3`
- `-flto`
- `-msimd128`
- `-DNDEBUG`
- `-fno-fast-math`
- `-fno-math-errno`
- `-ffp-contract=on`

That keeps the browser kernels on the fastest safe default path Emscripten
documents for release builds while still preserving normal floating-point
semantics. `NUMBL_BROWSER_WASM_FAST_MATH=1` is available if you explicitly want
`-ffast-math`, but it is not enabled by default because it can change
numerical behavior in FFT and LAPACK-style code.

## Storage

Built `.wasm` files are written into `public/wasm-kernels/` so the browser can
fetch them as static assets. A generated `manifest.json` is written alongside
them to give runtime code a single place to discover available kernels.
Successful builds replace the runtime manifest by default so optional kernels
do not stick around accidentally across incremental builds. Use `--merge` only
when you explicitly want to add targets on top of the existing manifest.

```bash
npm run build:browser-wasm -- --merge flame-blas-lapack
```

## ABI contract

The browser bridge loads operation-level exports, not raw BLAS/LAPACK symbols.
Current expected exports are:

- FFT kernels: `numbl_fft1d_f64`, `numbl_fft_along_dim_f64`
- Linear algebra kernels: `numbl_matmul_f64`, `numbl_inv_f64`, `numbl_linsolve_f64`
- Memory helpers: `malloc`, `free` or `_malloc`, `_free`

Expected signatures are C-style and status-oriented:

```c
int numbl_fft1d_f64(
  const double* re, const double* im, int n, int inverse,
  double* out_re, double* out_im
);

int numbl_fft_along_dim_f64(
  const double* re, const double* im,
  const int32_t* shape, int ndim, int dim, int n, int inverse,
  double* out_re, double* out_im
);

int numbl_matmul_f64(
  const double* A, int m, int k,
  const double* B, int n,
  double* out_C
);

int numbl_inv_f64(
  const double* A, int n,
  double* out_Ainv
);

int numbl_linsolve_f64(
  const double* A, int m, int n,
  const double* B, int nrhs,
  double* out_X
);
```

The wrapper is free to use `OpenBLAS`, `LAPACK`, `libflame`, `FFTW`, `ducc0`,
or another backend internally as long as it preserves this exported ABI.

Current in-repo wrappers use:

- `ducc0-fft`: DUCC0 C++ FFT kernels
- `flame-blas-lapack`: libFLAME builtin-BLAS browser linear algebra target; the runtime uses it for rectangular `linsolve` when present
- `fftw-fft`: placeholder/manual FFTW integration path; current Emscripten releases do not ship an official FFTW port
- `blas-lapack`: OpenBLAS-based browser linear algebra target; the runtime prefers it for `matmul`, `inv`, and square `linsolve`

## CI

The Wasm workflow in [`.github/workflows/wasm.yml`](../.github/workflows/wasm.yml)
runs `scripts/prepare-browser-wasm-sources.mjs` for `ducc0-fft`,
`blas-lapack`, and `flame-blas-lapack`, then builds all three before the
existing simple Wasm test build and smoke tests.

## Benchmarks

Use the local Tinybench-backed backend harness for side-by-side comparisons:

```bash
npm run bench:backends -- --quick
npm run bench:backends -- --backend wasm:blas-lapack,wasm:flame-blas-lapack --markdown bench/results/wasm-report.md
```

For actual browser timings, run the web app and open `/bench`. The browser
page uses the same quick scenarios as the CLI harness, validates outputs
before timing, and reports backend load failures instead of silently dropping
them.

## Act

When running the workflow locally with `act`, the first run may need network
access for the Emscripten setup action, Docker image pulls, and the temporary
source clones used by `scripts/prepare-browser-wasm-sources.mjs`.

## Current targets

- `ducc0-fft`
- `flame-blas-lapack`
- `fftw-fft`
- `blas-lapack`

`ducc0-fft`, `flame-blas-lapack`, and `blas-lapack` resolve third-party source
trees externally at build time. The source preparation helper pins libFLAME
`5.2.0`, OpenBLAS `v0.3.32`, and the `ducc0` branch by default unless you
override the corresponding `NUMBL_*_GIT_REF` variables. `fftw-fft` remains a
build-script-only manual integration path.
