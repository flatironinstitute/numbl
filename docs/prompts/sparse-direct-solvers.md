# Sparse direct solvers for implicit time stepping

## Why

The plate mechanics lab (`~/src/plate-mechanics-sim`) runs MATLAB-syntax solvers in numbl in the browser. Its explicit finite-difference scheme for the Kirchhoff-Love plate is limited by a stability bound dt <= dx^2 / (4 kappa), so 20 ms of simulated time on a 61 x 41 grid costs about 10 s. An implicit scheme removes the bound. Each step then solves one sparse linear system with a fixed matrix, so the right implementation is: factor the matrix once, do two sparse triangular solves per step.

numbl already stores sparse matrices in CSC form and has sparse arithmetic (`sparse-arithmetic.ts`: add, negate, transpose, matrix product, `spdiags`, `speye`). What it lacks is everything a direct solver needs. Today `kron`, `chol`, `lu`, `mldivide` and `inv` convert sparse inputs to dense (`sparseToDense` in `linear-algebra.ts` and `toSplitSquare`), so a sparse `A \ b` at n = 2000 densifies a 4 million entry matrix and runs an O(n^3) factorization every step. That makes an implicit scheme slower than the explicit one it is meant to replace.

## The concrete workload

The lab assembles, on the interior nodes of an nx x ny grid, with m = (nx - 2)(ny - 2):

```matlab
e = ones(mx, 1); D2x = spdiags([e -2*e e], -1:1, mx, mx) / dx^2;   % 1-D second difference, Dirichlet
e = ones(my, 1); D2y = spdiags([e -2*e e], -1:1, my, my) / dy^2;
L = kron(speye(my), D2x) + kron(D2y, speye(mx));                   % Laplacian, 5 nonzeros per row
B = kron(speye(my), D2x*D2x) + 2*kron(D2y, D2x) + kron(D2y*D2y, speye(mx));   % biharmonic, 13 per row
A = (1 + sigma0*dt) * speye(m) - sigma1*dt*L + theta*kappa^2*dt^2 * B;        % SPD, 13 nonzeros per row
R = chol(A);                       % once per run
for n = 1:nsteps
    rhs = ...;                     % dense m x 1, built from sparse times dense products L*w, B*w
    w = R \ (R' \ rhs);            % two triangular solves per step
end
```

Sizes that must be fast: m from about 1,000 (41 x 27 grid) to about 10,000 (121 x 81), with nsteps from 900 to 5,000. Target: the whole loop under about 3 s in the browser at m = 2,300, nsteps = 900, which means a triangular solve pair well under 1 ms and a factorization under 1 s. The matrix is banded with half-bandwidth about mx (column-major node ordering), so even a plain banded Cholesky with no reordering has fill of order m \* mx, about 200k entries at m = 2,300. That is enough; fill-reducing orderings are welcome but not required for v1.

## What must be implemented

All of it inside `src/numbl-core`, working on the existing `RuntimeSparseMatrix` CSC representation, pure TypeScript (the browser has no native addon), with a Node path that may optionally use the native bridge later.

1. **`kron(A, B)` for sparse operands returns sparse.** Output has nnz(A) \* nnz(B) entries. Mixed sparse and dense: the result is sparse (MATLAB semantics). Today it densifies.

2. **`chol(A)` for sparse SPD A returns a sparse upper-triangular `R` with `R' * R = A`.** Support the `[R, p] = chol(A)` form (p = 0 on success, otherwise the failing column index, no error thrown) and `chol(A, 'lower')`. Implementation: an up-looking or left-looking sparse Cholesky with a symbolic phase (elimination tree, column counts) so storage is allocated once. A simpler banded Cholesky is acceptable as a first step if it detects the bandwidth and falls back to the general algorithm when the bandwidth exceeds, say, m / 4. Must throw the MATLAB error message for non-SPD input when called with one output.

3. **`lu(A)` for sparse A returns sparse factors.** `[L, U, P] = lu(A)` with a sparse permutation matrix P, and `[L, U, P, Q] = lu(A)` with column permutation Q for fill reduction. Partial pivoting is required for correctness on nonsymmetric matrices (the plate's clamped-edge biharmonic is symmetric, but a nonsymmetric damping operator or a shell model will not be). A left-looking sparse LU (Gilbert-Peierls) is the standard choice.

4. **Sparse triangular solves inside `mldivide` and `mrdivide`.** `R \ b` and `R' \ b` where R is sparse and triangular must be detected (a structural check of the CSC pattern, O(nnz)) and solved by forward or back substitution in O(nnz(R)), for dense right-hand sides with one or many columns. `R'` must not be materialized densely: either transpose sparsely (already available) or solve with the transposed factor directly.

5. **`mldivide` for general sparse square A** (`A \ b` without a stored factorization) should factor with the sparse LU above, or sparse Cholesky when A is symmetric with positive diagonal and the Cholesky succeeds, then solve. Never densify a sparse matrix in `mldivide` unless nnz is a large fraction of m \* n (MATLAB's own heuristic is fine). Least-squares for non-square sparse A may keep densifying for now, with a clear error or warning.

6. **Permutation matrices.** `P * b` and `P' * b` with P a sparse permutation (as returned by `lu`) must stay O(m). Also useful, and cheap once the pieces exist: `symrcm(A)` (reverse Cuthill-McKee) and `amd(A)` orderings, so a user can reduce bandwidth themselves as in MATLAB.

7. **`decomposition(A)` object, optional but valuable.** `dA = decomposition(A, 'chol')`, then `dA \ b` reuses the factors. If this is out of scope, the `R = chol(A); x = R \ (R' \ b)` pattern above must be efficient, which items 2 and 4 already guarantee.

8. **Sparse times dense vector must be O(nnz).** `L * w` and `B * w` with w a dense column already exist; confirm they do not densify and that they are reachable from JIT-compiled loops (if the JIT declines on sparse operands, the interpreter path is fine, but the per-call overhead should be small).

9. **`issparse`, `nnz`, `nonzeros`, `full`, `spy`-free diagnostics** are present; add `istriu`, `istril`, `isbanded`, `bandwidth` for sparse inputs (dense versions exist).

## Acceptance tests

Add integration scripts under `numbl_test_scripts/sparse/` ending in `disp('SUCCESS')`, and a benchmark under `benchmarks/`.

Correctness:

```matlab
% kron stays sparse and matches dense
A = sprandn(20, 20, 0.2); B = speye(7);
K = kron(A, B); assert(issparse(K)); assert(norm(full(K) - kron(full(A), full(B)), 1) < 1e-12);

% chol of a 2-D Laplacian-based SPD matrix
n = 40; e = ones(n, 1); D = spdiags([e -2*e e], -1:1, n, n);
L2 = kron(speye(n), D) + kron(D, speye(n));
A = 4*speye(n^2) - L2;                     % SPD, m = 1600
R = chol(A); assert(issparse(R)); assert(istriu(R));
assert(norm(R'*R - A, 1) / norm(A, 1) < 1e-12);
b = randn(n^2, 1); x = R \ (R' \ b);
assert(norm(A*x - b) / norm(b) < 1e-10);
[R2, p] = chol(-A); assert(p > 0);         % not SPD, no throw with two outputs

% lu with pivoting on a nonsymmetric sparse matrix
N = A + 0.3*spdiags(ones(n^2, 1), 1, n^2, n^2);
[Lf, Uf, P] = lu(N); assert(issparse(Lf) && issparse(Uf) && issparse(P));
assert(norm(P*N - Lf*Uf, 1) / norm(N, 1) < 1e-12);
y = Uf \ (Lf \ (P*b)); assert(norm(N*y - b) / norm(b) < 1e-10);

% mldivide on sparse square does not densify and is exact
z = A \ b; assert(norm(A*z - b) / norm(b) < 1e-10);
disp('SUCCESS')
```

Performance (Node, pure TypeScript, no native addon, `--no-native`): for the m = 2,301 plate matrix (61 x 41 grid, 13 nonzeros per row, half-bandwidth 59):

- `chol(A)`: under 0.5 s.
- `R \ (R' \ b)`: under 1 ms per call, so 900 steps under 1 s.
- `kron` assembly of B: under 50 ms.
- Peak memory for the factor: proportional to m times the bandwidth, not m^2.

For m = 10,000 (121 x 81 grid) the same operations should stay under 5 s, 5 ms and 0.5 s respectively.

## Notes for the implementer

- Keep CSC as the single sparse storage format; the factors are ordinary sparse matrices so users can inspect them with `nnz`, `full`, and indexing.
- Column-major node ordering of a rectangular grid gives a banded matrix; a first implementation that exploits bandwidth alone will meet the targets above. General fill-reducing orderings (AMD) can follow.
- Detecting triangularity in `mldivide` must be structural (scan `ir` against column index), not numerical, so that an explicitly stored zero does not defeat it.
- `chol` on a matrix that is symmetric only to rounding: MATLAB uses the upper triangle and does not check symmetry; do the same, and document it in `docs/matlab-compatibility.md`.
- Update `docs/matlab-compatibility.md` (Sparse matrices row and Linear algebra row) and `docs/developer_reference/builtins.md` when done, per `docs/agents.md`.
- Once these land, `~/src/plate-mechanics-sim/packages/core/src/m/plate_implicit.m` (to be written) will use exactly the code pattern in "The concrete workload" above, so that pattern is the one to make fast.
