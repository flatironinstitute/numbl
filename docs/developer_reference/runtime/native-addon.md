# Native Addon

An optional Node-API C++ module that provides fast implementations of numerical kernels. Numbl detects it at runtime; when absent, it falls back to pure-JS implementations.

## What it provides

- **LAPACK / BLAS** — eigendecomposition, SVD, LU, QR, QZ, Cholesky, general linear solve, matrix inverse, matrix-matrix and matrix-vector multiply, GMRES. Built against OpenBLAS (or a system LAPACK/BLAS).
- **FFTW** — batched FFT.
- **Element-wise ops** — the kernels referenced by the ops-layer op-code table (real and complex binary and unary ops, reductions, comparisons, Bessel functions).
- **Random number generation** — for large draws where the JS PRNG is a bottleneck.

## JS fallbacks

Every addon-backed operation has a pure-JS fallback. The fallbacks are correct but slower — in particular, LAPACK fallbacks rely on a lightweight in-tree port. The fallback path is what runs in the browser (where loading a native addon is not possible) and on systems where the addon is not installed.

## Optional WASM accelerator bridge (browser)

The native addon is Node-only. In the browser the LAPACK fallbacks are the pure-JS ts-lapack port (and, for matrix multiply under the JS-JIT, a naive triple-loop runtime snippet). An **optional, externally-hosted WebAssembly bridge** (e.g. libFLAME/BLIS compiled to wasm, served from GitHub Pages) can be loaded to accelerate heavy kernels — v1 covers real f64 `matmul`.

- **Core (`src/numbl-core/native/wasm-lapack-bridge.ts`)** — platform-agnostic. Defines the versioned manifest (`AcceleratorManifest`, `ACCELERATOR_ABI`) and the wasm ABI (`WasmLapackExports`: `numbl_malloc`/`numbl_free`/`numbl_matmul_f64_colmajor` + exported `memory`, all column-major f64 pointers into `memory`). `installWasmLapackBridge` wires **two seams** from one instantiated module: (1) `setLapackBridge` for the interpreter path (opt 0), which already resolves per-op via `getEffectiveBridge` and falls back to ts-lapack for any op the bridge omits; and (2) `globalThis.$matmulAccel`, the hook the JS-JIT matmul runtime snippet (`tensor_mtimes_real.js`) consults before its loop (opt 1). The `$matmulAccel` wrapper declines small matmuls (below a work threshold) and any error, so the JS loop stays the fallback.
- **Loader (`src/numbl-core/native/wasm-lapack-browser.ts`)** — `ensureWasmLapackBridge(url)` fetches the manifest, fetches and instantiates the wasm (numbl instantiates it itself with a numbl-controlled import object — no remote JS is executed, so the accelerator is sandboxed), and installs it. Best-effort and idempotent per URL, mirroring `qhull-browser.ts`.
- **Wiring** — the IDE worker (`numbl-worker.ts`) loads it lazily on a `set_wasm_bridge` message; the main thread reads the settings (`src/utils/wasmLapackBridge.ts`, localStorage, enabled by default) and posts the effective URL on worker creation and when the Execution Settings dialog is saved. Downloaded wasm bytes are cached in IndexedDB (`src/utils/wasmByteCache.ts`).

Node bundling (using the wasm bridge as a fallback when the native addon isn't built) is intended but not yet wired.

## Op-code contract

The element-wise kernels are dispatched by integer op code. The TypeScript side (used by the ops layer) and the C side (in the addon's header) must agree on these values. A dedicated unit test loads the addon, asks it for its op-code table, and compares against the TypeScript enum — drift fails CI rather than silently producing wrong results.

## Build and installation

The addon is built with `numbl build-addon` (or `npx tsx src/cli.ts build-addon`). It requires a C++ compiler and the LAPACK/BLAS and FFTW development headers. The compiled artifact is loaded lazily on first use of an addon-backed operation; a rebuild is required after upgrading the numbl package.

## When to reach for it

- New tensor kernels that need to match interpreter behavior exactly should be added on both sides (JS fallback and native) and given an op code. Add them to the shared op-code list.
- New LAPACK-backed operations need C++ wrapper code in the addon and a JS fallback for browser parity.
- Pure-scalar math that already runs fast in V8 generally does not need an addon implementation.
