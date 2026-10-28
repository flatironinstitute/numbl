# Differences from MATLAB

numbl aims for MATLAB compatibility but currently has some behavioral differences. The list below reflects the present state of numbl; some of these gaps may be closed in future releases.

## Performance

Browser mode is designed for development, teaching, and sharing. The CLI with the native addon is significantly faster but still generally slower than MATLAB. Performance is actively improving.

## Numeric Types

All numeric values are double precision. The conversion functions `single`, `int8`, ..., `uint64` exist so that ported code runs: `single` rounds to single precision and the integer functions round and saturate to the type's range, but the result is still a double and `class` reports `'double'`. Integer type names are otherwise recognized only as precision strings in `fread`/`fwrite`.

## Path Management

numbl has no notion of a saved path, so `savepath` is a no-op. `addpath` does work, but only for the duration of the current session. To make directories available persistently, use [mip](https://mip.sh) to install reusable libraries, or extend the search path via the CLI `--path` flag or the `NUMBL_PATH` environment variable.

## Stubbed Commands

A handful of MATLAB commands are accepted by numbl but currently do nothing meaningful. They exist so that ported code keeps running without modification:

- `clc`: no-op
- `lastwarn`: always returns `''` (warnings are not tracked)
- `pathdef`: returns `''`
- `listfonts`: returns `{}`

These may gain real implementations as the relevant subsystems land.

## Currently Unsupported Features

The following MATLAB features are not implemented today. Some may be added in future releases; others are unlikely to ever be supported.

- Single-precision and integer numeric types
- Parallel computing (`parfor`, `spmd`)
- GPU arrays
- MEX interface (numbl provides its own equivalent: `.js` user functions can bind to compiled `.wasm` modules or native shared libraries via `// wasm:` and `// native:` directives)
- Simulink and toolboxes
- Java / .NET integration
- App Designer / GUI
- Parts of class introspection: `?ClassName` and `metaclass` return `matlab.metadata.Class` objects, but `meta.class.fromName`, `findobj` on meta objects and property `GetMethod`/`SetMethod` handles are not modelled

## `eig`

`eig(A)` follows MATLAB's dispatch: exactly symmetric (or Hermitian) input goes to the symmetric eigensolver, giving real eigenvalues in ascending order and an orthonormal eigenvector basis; everything else goes to the general solver. Symmetry is tested with exact equality, as MATLAB's `isequal(A, A')` does, so a matrix that is symmetric only to within rounding takes the general path in both.

`eig(A, B)` solves the generalized problem `A*v = lambda*B*v`. A symmetric/Hermitian-definite pencil is reduced through the Cholesky factor of `B` and normalized so that `V'*B*V = I`, matching MATLAB. Any other pencil is solved by QZ, which requires the native LAPACK addon; without it (in the browser, or where the addon is not built) `eig(A, B)` raises an error rather than falling back to a different algorithm. The `'chol'`, `'qz'`, `'vector'`, `'matrix'`, `'balance'` and `'nobalance'` options are supported.

## Sparse Direct Solvers

Sparse matrices are factored without converting to full storage. `chol(S)` is an up-looking sparse Cholesky that, like MATLAB, reads only the upper triangle and does not check symmetry; `[R, p] = chol(S)` reports the failing column and returns the first `p-1` rows of the factor. The three-output form `[R, p, P] = chol(S)` applies a fill-reducing ordering. `lu(S)` is a left-looking LU with threshold partial pivoting (`[L, U, P]`, `[L, U, P, Q]`, the `'vector'` form, and a numeric pivot threshold are supported; the threshold defaults to 1 for three outputs and 0.1 for four). `S \ b` detects triangular matrices structurally and solves them by substitution, tries a permuted Cholesky when `S` is symmetric with a positive diagonal, and otherwise uses LU with a column ordering; a nearly full sparse matrix (more than half nonzeros) is handed to the dense LAPACK path. `b / S` is solved through the transpose, and `inv(S)` returns a sparse result. `kron` with a sparse operand returns sparse. Complex sparse matrices and non-square (least-squares) sparse systems still go through the dense path. The orderings differ from MATLAB's: `amd` is an exact minimum degree ordering (MATLAB uses approximate minimum degree) and `colamd` is minimum degree on the pattern of `A'*A`, so the permutations are valid and fill-reducing but not identical to MATLAB's. `decomposition(A)` is a builtin object supporting the `'auto'`, `'chol'`, `'lu'` and `'triangular'` types (other MATLAB types fall back to LU); it requires a real square matrix, and for dense input the `'auto'` type is `'lu'` unless the matrix is triangular.
