# tensor_ops_bench — numbl vs MATLAB on whole-tensor element-wise ops

Tensor element-wise + reduction benchmark. Five kernels on real Float64
vectors (N=2M elements), 50 trials each. The loop body is inlined so the
loop-JIT can specialize the full iteration as a single unit under
`--opt e1`.

## Kernels

| Kernel  | Pattern                                                                               |
| ------- | ------------------------------------------------------------------------------------- |
| Binary  | `r = x+y; r = r-0.5.*x; r = r.*y+3; r = r./(1+abs(y))`                                |
| Unary   | `u = exp(-x.*x); u = u.*cos(5.*x); u = u+sin(x+1).*sin(x+1); u = abs(u); u = tanh(u)` |
| Cmp+Red | `s += sum((x>0).*(y<0.5))`                                                            |
| Reduce  | `s += sum(x)+mean(x)+max(x)+min(x)`                                                   |
| Chain   | `r2 = x.*y+0.5; r2 = exp(-r2.*r2); r2 = r2.*x; s += sum(r2)`                          |

## How to run

```bash
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 0          # interpreter
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 1          # JS-JIT
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt e1         # JS-JIT + inline C kernels
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt e1 --par   # + OpenMP
matlab -batch "run('benchmarks/tensor_ops_bench.m')"
(cd benchmarks && octave --no-gui --quiet --eval tensor_ops_bench)
```

All runs produce the same check values (to FP rounding).

## Results (Linux, N=2 000 000, trials=50)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.2.0
- **MATLAB:** R2025b Update 5
- **Measured:** 2026-04-23 16:10 UTC

Median of 3 runs for non-interpreter modes; `--opt 0` is a single run.

| Mode                      |  Total | Binary |  Unary | Cmp+Red | Reduce |  Chain |
| ------------------------- | -----: | -----: | -----: | ------: | -----: | -----: |
| `--opt 0` (interpreter)   | 5.02 s | 1.11 s | 2.30 s |  0.41 s | 0.26 s | 0.94 s |
| `--opt 1` (JS-JIT)        | 3.25 s | 0.66 s | 1.40 s |  0.32 s | 0.30 s | 0.58 s |
| `--opt e1`                | 1.32 s | 0.09 s | 0.62 s |  0.20 s | 0.29 s | 0.12 s |
| `--opt e1 --par`          | 0.96 s | 0.09 s | 0.26 s |  0.20 s | 0.29 s | 0.12 s |
| `--opt e2`                | 1.42 s | 0.10 s | 0.66 s |  0.29 s | 0.18 s | 0.18 s |
| `--opt e2 --par`          | 0.97 s | 0.09 s | 0.30 s |  0.28 s | 0.14 s | 0.19 s |
| MATLAB R2025b (1 thread)  | 4.90 s | 0.32 s | 3.45 s |  0.32 s | 0.21 s | 0.60 s |
| MATLAB R2025b (8 threads) | 1.81 s | 0.26 s | 0.84 s |  0.13 s | 0.25 s | 0.33 s |

`--opt e1 --par` auto-parallelizes chain kernels whose per-element
body has transcendentals (`exp`, `sin`, ...); arithmetic-only bodies
(Binary) stick to serial `#pragma omp simd` since thread-spawn
overhead exceeds the memory-bandwidth-bound compute. Unary drops 2.3×
with `--par` (0.61 → 0.27 s). Reduction kernels (no tensor output)
can't be parallelized without a `reduction(...)` clause, which e1
doesn't emit yet.

### macOS (N=2 000 000, trials=50)

- **CPU:** Apple M4 Max (16 threads)
- **OS:** macOS 15.7.5 (Darwin 24.6.0)
- **Toolchain:** Node v25.9.0, Apple clang 17.0.0, gcc-15 15.2.0 (Homebrew), numbl 0.2.0
- **MATLAB:** R2025b
- **Measured:** 2026-04-23

Single run per mode. `--opt e1 --par` compiled with `NUMBL_CC=gcc-15`.

| Mode                     |      Total |     Binary |      Unary |    Cmp+Red |     Reduce |      Chain |
| ------------------------ | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| `--opt 1` (JS-JIT)       |     3.01 s |     0.17 s |     1.26 s |     0.47 s |     0.73 s |     0.39 s |
| `--opt e1`               |     1.94 s |     0.02 s |     0.94 s |     0.06 s |     0.73 s |     0.19 s |
| `--opt e1 --par`         |     1.17 s |     0.04 s | **0.12 s** |     0.07 s |     0.73 s |     0.21 s |
| MATLAB R2025b (1 thread) |     2.85 s |     0.20 s |     1.83 s |     0.10 s |     0.34 s |     0.38 s |
| MATLAB R2025b (multi)    | **0.47 s** | **0.06 s** |     0.22 s | **0.07 s** | **0.06 s** | **0.07 s** |

MATLAB's threaded reductions dominate on macOS because numbl's `--par`
doesn't yet emit OpenMP `reduction(...)` clauses — the Reduce row stays
at single-thread speed (0.73 s) under `--par`. numbl's `--par` still
takes the Unary column (0.12 s vs MATLAB's 0.22 s) thanks to OMP
threading across the transcendental-heavy body.

## Architecture notes

`--opt e1` detects fusible runs of consecutive tensor element-wise
assigns and emits each as a single per-element C loop. Trailing
reductions (`acc += sum(r)`) are absorbed as inline accumulators.
The JS wrapper dispatches to the compiled kernel when `N` exceeds a
threshold (default 100 000); below that, the plain inline JS fused
loop runs instead. See [docs/developer_reference/jit/e1-kernels.md](../docs/developer_reference/jit/e1-kernels.md).

Generated C for the binary kernel:

```c
for (double __t1 = 1.0; __t1 <= v_trials; __t1 += 1.0) {
    #pragma omp simd
    for (int64_t __i = 0; __i < v_x_len; __i++) {
      double __f_r = (v_x_data[__i] + v_y_data[__i]);
      __f_r = (__f_r - (0.5 * v_x_data[__i]));
      __f_r = ((__f_r * v_y_data[__i]) + 3.0);
      __f_r = (__f_r / (1.0 + fabs(v_y_data[__i])));
      v_r_data[__i] = __f_r;
    }
}
```

Generated C for the chain kernel (with fused reduction):

```c
for (double __t1 = 1.0; __t1 <= v_trials; __t1 += 1.0) {
    double __f_reduce_acc = 0.0;
    for (int64_t __i = 0; __i < v_x_len; __i++) {
      double __f_r2 = ((v_x_data[__i] * v_y_data[__i]) + 0.5);
      __f_r2 = exp((-__f_r2 * __f_r2));
      __f_r2 = (__f_r2 * v_x_data[__i]);
      __f_reduce_acc += __f_r2;
    }
    v_chain_acc += __f_reduce_acc;
}
```

## Caveats

- **Domain-restricted unaries** (`sqrt`, `log`, `log2`, `log10`, `asin`,
  `acos`) bail to the interpreter — MATLAB promotes out-of-domain inputs
  to complex, libnumbl_ops returns NaN. Same behavior as JS-JIT.
- **Compiled `.so` cache** lives at `~/.cache/numbl/c-jit/<sha>.so`,
  keyed by source + compiler/platform/numbl versions + git HEAD +
  `libnumbl_ops.a` contents.
- **`-ffast-math`** is enabled for the C kernel compile. This allows SIMD
  vectorization of transcendentals but changes FP rounding — check
  values may differ slightly between modes.
