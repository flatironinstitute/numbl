# Native Addon

An optional Node-API C++ module that provides fast implementations of numerical kernels. When it is absent, numbl falls back to pure-JS implementations for nearly every operation (see below for the exceptions).

## What it provides

- **LAPACK / BLAS** — eigendecomposition (general and symmetric/Hermitian), SVD, LU, QR, QZ, Cholesky, general linear solve, matrix inverse, matrix-matrix and matrix-vector multiply, GMRES. Built against OpenBLAS (or a system LAPACK/BLAS).
- **FFTW** — batched FFT.
- **Element-wise ops** — the kernels referenced by the ops-layer op-code table (real and complex binary and unary ops, reductions, comparisons, Bessel functions).
- **Random number generation** — for large draws where the JS PRNG is a bottleneck.

## The bridge

The addon and its pure-TypeScript fallback implement one interface, `LapackBridge` (`src/numbl-core/native/lapack-bridge.ts`). `setLapackBridge` installs an accelerated bridge (the addon, a wasm module, or a test stub; any of them may be partial), and `getLapackBridge()` returns a complete bridge in which each entry comes from the installed bridge when it provides it and from the TypeScript implementation (`ts-lapack-bridge.ts`) otherwise. Call sites are plain `getLapackBridge().X(...)`, with no availability checks and no inline fallbacks. The ops layer (`src/numbl-core/ops/dispatch.ts`) reaches the element-wise `tensorOp*` kernels the same way.

The TypeScript side is correct but slower. Its LAPACK routines come from a lightweight in-tree port (`src/ts-lapack/`) plus a few hand translations in `src/numbl-core/native/`: the complex eigensolvers (`ts-lapack-eig-complex.ts`, `ts-lapack-eig-hermitian.ts`), real column-pivoted QR and complex QR/LU (`ts-lapack-qr-lu.ts`), and the complex helpers they share, including `zlarfg` (`ts-lapack-complex.ts`). This is the path that runs in the browser and wherever the addon is not built.

Operations the TypeScript side does not implement are listed in one place, `TS_UNSUPPORTED` in `ts-lapack-bridge.ts`, and throw an error saying they require LAPACK:

- complex `svd` (`svdComplex`);
- QZ, real and complex (`qz`, `qzComplex`), and hence `qz` itself and `eig(A, B)` for a pencil that is not symmetric/Hermitian-definite.

A few addon entry points are native-only accelerators with no TypeScript counterpart in the bridge: the FFTW transforms (`fft1d`, `fft1dComplex`, `fftAlongDim`) and `fillRandn`. Their callers (`fft.ts`, `prng.ts`) keep their own JS paths, because the JS FFT and PRNG differ from the native ones in post-processing and state handling.

## Eigensolver dispatch

`eig` picks a bridge entry point from the shape of the problem, and the choice is made once, in `eigApply` (`src/numbl-core/interpreter/builtins/linear-algebra.ts`), so that direct `bridge.eig` callers such as `roots` and `poly` are unaffected.

- **Exactly symmetric / Hermitian input → `eigSym` (dsyev) / `eigHerm` (zheev).** The test is exact element equality, matching MATLAB's `isequal(A, A')`; a tolerance would make the algorithm depend on how nearly symmetric the input happens to be. These entry points give real eigenvalues in ascending order and an orthonormal (unitary) eigenvector basis, which `dgeev` cannot: it back-substitutes eigenvectors out of the Schur form and normalizes each column separately, so across a degenerate eigenspace the columns are individually unit-length but not mutually orthogonal, and a tight cluster can even deflate as a complex pair. Both bridges provide them.
- **Everything else → `eig` (dgeev) / `eigComplex` (zgeev)**, unchanged.
- **`eig(A, B)`** solves the generalized problem. A symmetric/Hermitian-definite pencil is reduced through the Cholesky factor of `B` (`R = chol(B)`, solve the symmetric problem on `R^-H A R^-1`, map back with `V = R^-1 S`), which also yields MATLAB's `V'*B*V = I` normalization. Any other pencil goes through `qz`, which is native-only; without the addon the bridge raises an error there rather than answering a different problem.

The pure-TypeScript side of this lives in `src/ts-lapack/src/` (`dsyev` = `dsytd2` + `dorgtr` + `dsteqr`, with `dsymv`/`dsyr2`/`dlae2`/`dlaev2`/`dlanst`/`dlansy` underneath) and, for the Hermitian case, in `src/numbl-core/native/ts-lapack-eig-hermitian.ts`, which does the complex tridiagonal reduction and reuses the same `dsteqr` for the tridiagonal QL/QR.

## Optional WASM accelerator bridge (browser)

The native addon is Node-only. In the browser the LAPACK fallbacks are the pure-JS ts-lapack port (and, for matrix multiply under the JS-JIT, a naive triple-loop runtime snippet). An **optional, externally-hosted WebAssembly bridge** (e.g. libFLAME/BLIS compiled to wasm, served from GitHub Pages) can be loaded to accelerate heavy kernels — v1 covers real f64 `matmul`.

- **Core (`src/numbl-core/native/wasm-lapack-bridge.ts`)** — platform-agnostic. Defines the versioned manifest (`AcceleratorManifest`, `ACCELERATOR_ABI`) and the wasm ABI (`WasmLapackExports`: `numbl_malloc`/`numbl_free`/`numbl_matmul_f64_colmajor` + exported `memory`, all column-major f64 pointers into `memory`). `installWasmLapackBridge` wires **two seams** from one instantiated module: (1) `setLapackBridge` for the interpreter path (opt 0), where `getLapackBridge` takes ts-lapack for any op the module omits; and (2) `globalThis.$matmulAccel`, the hook the JS-JIT matmul runtime snippet (`tensor_mtimes_real.js`) consults before its loop (opt 1). The `$matmulAccel` wrapper declines small matmuls (below a work threshold) and any error, so the JS loop stays the fallback.
- **Loader (`src/numbl-core/native/wasm-lapack-browser.ts`)** — `ensureWasmLapackBridge(url)` fetches the manifest, fetches and instantiates the wasm (numbl instantiates it itself with a numbl-controlled import object — no remote JS is executed, so the accelerator is sandboxed), and installs it. Best-effort and idempotent per URL, mirroring `qhull-browser.ts`.
- **Wiring** — the IDE worker (`numbl-worker.ts`) loads it lazily on a `set_wasm_bridge` message; the main thread reads the settings (`src/utils/wasmLapackBridge.ts`, localStorage, enabled by default) and posts the effective URL on worker creation and when the Execution Settings dialog is saved. Downloaded wasm bytes are cached in IndexedDB (`src/utils/wasmByteCache.ts`).

Node bundling (using the wasm bridge as a fallback when the native addon isn't built) is intended but not yet wired.

## Op-code contract

The element-wise kernels are dispatched by integer op code. The TypeScript side (used by the ops layer) and the C side (in the addon's header) must agree on these values. A dedicated unit test loads the addon, asks it for its op-code table, and compares against the TypeScript enum — drift fails CI rather than silently producing wrong results.

## Build and installation

The addon is built with `numbl build-addon` (or `npx tsx src/cli.ts build-addon`). It requires a C++ compiler and the LAPACK/BLAS and FFTW development headers. The build is copied to `build/Release/numbl_addon.nofastmath.node` (or `.fastmath.node` with `--fast-math`).

The CLI loads the addon eagerly, before any code runs, in `run`, `eval`, `run-tests` and the REPL (via `initNodePlatform` in `src/node-platform.ts`, which Node library hosts can also call through `numbl/node`), unless `--no-native` or `NUMBL_NO_NATIVE` is set. If the file is missing, or its `addonVersion()` differs from `NATIVE_ADDON_EXPECTED_VERSION` (`src/numbl-core/native/lapack-bridge.ts`), the CLI warns and uses the JS fallbacks. Changing the addon's exported API means bumping `ADDON_VERSION` in `native/numbl_addon.cpp` and the TS constant together; users then need to rebuild.

## When to reach for it

- New tensor kernels that need to match interpreter behavior exactly should be added on both sides (JS fallback and native) and given an op code. Add them to the shared op-code list.
- New LAPACK-backed operations need C++ wrapper code in the addon, a `LapackBridge` entry, and a TypeScript implementation in `ts-lapack-bridge.ts` (or, failing that, an entry in `TS_UNSUPPORTED`).
- Pure-scalar math that already runs fast in V8 generally does not need an addon implementation.
