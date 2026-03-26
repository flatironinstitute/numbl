# libFLAME browser linalg wrapper

This target builds an experimental standalone browser Wasm module on top of
`libFLAME`'s builtin BLAS plus legacy LAPACK interfaces.

Exported ABI:

- `numbl_matmul_f64`
- `numbl_inv_f64`
- `numbl_linsolve_f64`

Build defaults:

- static-only `libFLAME`
- builtin BLAS enabled with `--enable-builtin-blas`
- legacy LAPACK enabled with `--enable-legacy-lapack`
- browser release flags `-O3 -flto -msimd128 -DNDEBUG`
- safe floating-point flags `-fno-fast-math -fno-math-errno -ffp-contract=on`

`NUMBL_BROWSER_WASM_FAST_MATH=1` can be used to opt into `-ffast-math`, but it
is not enabled by default because it can relax floating-point semantics in
ways that are risky for LAPACK-style kernels.
