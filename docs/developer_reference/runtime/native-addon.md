# Native Addon

An optional Node-API C++ module that provides fast implementations of numerical kernels. Numbl detects it at runtime; when absent, it falls back to pure-JS implementations.

## What it provides

- **LAPACK / BLAS** — eigendecomposition, SVD, LU, QR, QZ, Cholesky, general linear solve, matrix inverse, matrix-matrix and matrix-vector multiply, GMRES. Built against OpenBLAS (or a system LAPACK/BLAS).
- **FFTW** — batched FFT.
- **Element-wise ops** — the kernels referenced by the ops-layer op-code table (real and complex binary and unary ops, reductions, comparisons, Bessel functions).
- **Random number generation** — for large draws where the JS PRNG is a bottleneck.

## JS fallbacks

Every addon-backed operation has a pure-JS fallback. The fallbacks are correct but slower — in particular, LAPACK fallbacks rely on a lightweight in-tree port. The fallback path is what runs in the browser (where loading a native addon is not possible) and on systems where the addon is not installed.

## Op-code contract

The element-wise kernels are dispatched by integer op code. The TypeScript side (used by the ops layer and the C-JIT backend) and the C side (in the addon's header) must agree on these values. A dedicated unit test loads the addon, asks it for its op-code table, and compares against the TypeScript enum — drift fails CI rather than silently producing wrong results.

## Build and installation

The addon is built with `numbl build-addon` (or `npx tsx src/cli.ts build-addon`). It requires a C++ compiler and the LAPACK/BLAS and FFTW development headers. The compiled artifact is loaded lazily on first use of an addon-backed operation; a rebuild is required after upgrading the numbl package.

## When to reach for it

- New tensor kernels that need to match interpreter behavior exactly should be added on both sides (JS fallback and native) and given an op code. Add them to the shared op-code list.
- New LAPACK-backed operations need C++ wrapper code in the addon and a JS fallback for browser parity.
- Pure-scalar math that already runs fast in V8 generally does not need an addon implementation.
