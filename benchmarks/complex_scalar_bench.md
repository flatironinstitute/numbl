# complex_scalar_bench — numbl vs MATLAB on a complex-scalar hot loop

Compute-bound benchmark for the C-JIT's pair-of-doubles complex scalar
codegen (Phase 1 of full-complex support). Compares numbl's three
optimization levels (`--opt 0/1/2`) against MATLAB.

## What the benchmark does

`run_bench(N, M)` iterates the map `z -> z^2 + c` with `c = 0.001 + 0.001i`
for `M` inner steps from `N` distinct starting points, then sums
`real(z_final)` across all starts. Default sizes `N = 50000`, `M = 400`
→ **20M** complex squared+add ops per run.

The inner step is `z = z*z + c` — one complex mul + one complex add,
each lowered to real-only C via the pair-of-doubles emitter:

```
double z_re2 = z_re*z_re - z_im*z_im;
double z_im2 = z_re*z_im + z_im*z_re;
z_re = z_re2 + 0.001;
z_im = z_im2 + 0.001;
```

No branches, no divisions — a tight loop the compiler can keep entirely
in registers.

## How to run

```bash
npx tsx src/cli.ts run benchmarks/complex_scalar_bench.m --opt 0   # interpreter
npx tsx src/cli.ts run benchmarks/complex_scalar_bench.m --opt 1   # JS-JIT
npx tsx src/cli.ts run benchmarks/complex_scalar_bench.m --opt 2   # C-JIT
matlab -batch "run('benchmarks/complex_scalar_bench.m')"
```

All runs produce the same `result = 49.999798997206` (to FP rounding).

## Results (Linux, N=50 000, M=400, 20M z^2+c ops)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0
- **MATLAB:** R2025b Update 5

Single-run numbers (these are short runs — variance between invocations
is a few %).

| Mode                    |  Wall time |         Throughput | Speedup vs `--opt 0` |
| ----------------------- | ---------: | -----------------: | -------------------: |
| `--opt 0` (interpreter) |     6.85 s |        2.92 Mops/s |                   1× |
| `--opt 1` (JS-JIT)      |     0.94 s |       21.34 Mops/s |                  ~7× |
| `--opt 2` (C-JIT)       | **0.04 s** | **524–596 Mops/s** |            **~180×** |
| MATLAB R2025b `-batch`  |     0.11 s |      189.06 Mops/s |                 ~65× |

The C-JIT wins by a wide margin here because the complex arithmetic
lowers to straight-line `double` math that `-O2 -march=native -ffast-math`
can keep resident in registers through the entire inner loop. The
JS-JIT pays boxing costs on every `cMul` / `cAdd` call, and MATLAB's
complex scalars go through its heavier dispatch path.

## Caveats

- Phase 1 only supports `+`, `-`, `*`, `/`, unary `-`, `ImagLiteral`,
  `real`/`imag`/`conj`. Anything more (comparisons, `exp`, `log`, `sin`
  on complex) still bails to JS-JIT.
- This benchmark exercises the best case (tight scalar loop, no
  transcendentals, no branching). Mixed workloads will show less
  dramatic speedups.
