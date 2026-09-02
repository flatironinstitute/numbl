# sparse_direct_bench — sparse direct solvers on the implicit plate step

Times the pieces an implicit time-stepping scheme needs when the system
matrix is fixed: assembling the operator with `kron`, factoring it once
with `chol`, and then two sparse triangular solves per step. The matrix is
the Kirchhoff-Love plate operator on the interior nodes of an `nx x ny`
grid, with 13 nonzeros per row and half-bandwidth `2*(nx-2)` in the
column-major node ordering (the biharmonic term reaches two node rows away).

The benchmark also times `A \ b` without a stored factorization (which
internally uses a minimum degree ordering and Cholesky), `lu(A)`, and a
`decomposition(A, 'chol')` object reused across the steps.

## How to run

```bash
npx tsx src/cli.ts run benchmarks/sparse_direct_bench.m --no-native   # pure TypeScript, as in the browser
npx tsx src/cli.ts run benchmarks/sparse_direct_bench.m               # with the native addon (only affects dense fallbacks)
matlab -batch "run('benchmarks/sparse_direct_bench.m')"
```

## Results (pure TypeScript, `--no-native`, Node 24, 2026-09-02)

Grid 61 x 41, m = 2301, 900 steps:

| Kernel                                 | Time   | Notes                         |
| -------------------------------------- | ------ | ----------------------------- |
| assembly (kron, spdiags, sparse add)   | 22 ms  | nnz(A) = 28,937               |
| `chol(A)`                              | 41 ms  | nnz(R) = 263,549              |
| time loop (2 triangular solves + B\*w) | 886 ms | 0.98 ms per step              |
| `A \ b` (no stored factor)             | 38 ms  | RCM ordering + chol + solve   |
| `lu(A)`                                | 124 ms | nnz(L) + nnz(U) = 527,098     |
| `decomposition` + 900 solves           | 404 ms | 0.45 ms per solve (reordered) |

Grid 121 x 81, m = 9401, 200 steps:

| Kernel                       | Time    | Notes                       |
| ---------------------------- | ------- | --------------------------- |
| assembly                     | 41 ms   | nnz(A) = 120,237            |
| `chol(A)`                    | 679 ms  | nnz(R) = 2,204,709          |
| time loop                    | 1580 ms | 7.9 ms per step             |
| `A \ b`                      | 321 ms  |                             |
| `lu(A)`                      | 1445 ms | nnz(L) + nnz(U) = 4,409,418 |
| `decomposition` + 200 solves | 1138 ms | 5.7 ms per solve            |

For reference, MATLAB R2025b on the same machine: m = 2301, chol 64 ms,
time loop 0.63 ms per step, `A \ b` 19 ms, lu 63 ms, decomposition solve
0.25 ms; m = 9401, chol 256 ms, time loop 7.5 ms per step, `A \ b` 91 ms,
lu 630 ms, decomposition solve 2.1 ms. The per-step cost of the plain
`R \ (R' \ rhs)` pattern is within a factor of 1.6 of MATLAB's; the gaps are
larger where MATLAB's orderings (AMD, COLAMD) and supernodal kernels pay off.

The natural (column-major) ordering keeps the factor banded, so the cost of
a solve pair is proportional to `m` times the half-bandwidth, and the
substitution loops are memory-bound scatters (about 1 ns per stored entry).
A fill-reducing ordering is the main lever: `decomposition(A, 'chol')` and
the three-output `[R, p, P] = chol(A)` form use a minimum degree ordering
(factor about 3x smaller at m = 9401, but the ordering itself costs about
0.6 s there), while `A \ b` uses reverse Cuthill-McKee, which is O(nnz) to
compute and roughly halves the factor.
