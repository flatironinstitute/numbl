# tensor_ops_bench — numbl vs MATLAB on whole-tensor element-wise ops

Tensor element-wise + reduction benchmark. Five kernels on real Float64
vectors (N=2M elements), 50 trials each. The loop body is inlined so the
loop-JIT can specialize the full iteration as a single C function under
`--opt 2`.

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
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 2          # C-JIT (per-op)
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 2 --fuse        # C-JIT (fused)
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 2 --fuse --par  # C-JIT (fused+parallel)
matlab -batch "run('benchmarks/tensor_ops_bench.m')"
(cd benchmarks && octave --no-gui --quiet --eval tensor_ops_bench)
bash benchmarks/tensor_ops_bench_compare.sh                           # all of the above
```

All runs produce the same check values (to FP rounding).

## Results (Linux, N=2 000 000, trials=50)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.1.7
- **MATLAB:** R2025b Update 5 · **Octave:** 9.4.0

Median of 3 runs for all modes.

| Mode                           |  Total | Binary |  Unary | Cmp+Red | Reduce |  Chain |
| ------------------------------ | -----: | -----: | -----: | ------: | -----: | -----: |
| `--opt 0` (interpreter)        | 5.50 s | 1.32 s | 2.38 s |  0.44 s | 0.28 s | 1.08 s |
| `--opt 1` (JS-JIT)             | 3.23 s | 0.62 s | 1.37 s |  0.34 s | 0.30 s | 0.60 s |
| `--opt 2` (C-JIT)              | 3.16 s | 0.66 s | 1.33 s |  0.36 s | 0.27 s | 0.54 s |
| `--opt 2 --fuse` (C-JIT)       | 1.65 s | 0.08 s | 0.63 s |  0.27 s | 0.26 s | 0.40 s |
| `--opt 2 --fuse --par` (C-JIT) | 1.44 s | 0.12 s | 0.29 s |  0.34 s | 0.29 s | 0.41 s |
| MATLAB R2025b (1 thread)       | 4.95 s | 0.30 s | 3.42 s |  0.32 s | 0.21 s | 0.70 s |
| MATLAB R2025b `-batch`         | 3.02 s | 0.24 s | 2.00 s |  0.20 s | 0.19 s | 0.39 s |
| MATLAB R2025b (8 threads)      | 2.10 s | 0.24 s | 1.16 s |  0.13 s | 0.28 s | 0.29 s |
| Octave 9.4 `--eval`            | 7.75 s | 0.98 s | 4.41 s |  0.75 s | 0.30 s | 1.40 s |
| C baseline, per-op             | 2.63 s | 0.58 s | 1.22 s |  0.24 s | 0.11 s | 0.49 s |
| C baseline, fused              | 1.03 s | 0.10 s | 0.68 s |  0.07 s | 0.04 s | 0.13 s |

### macOS (N=2 000 000, trials=50)

- **CPU:** Apple M4 Max (16 threads)
- **OS:** macOS 15.7.3 (Darwin 24.6.0)
- **Toolchain:** Node v25.9.0, Apple clang 17.0.0, numbl 0.1.7
- **MATLAB:** R2026a (26.1.0)

| Mode                       |      Total |     Binary |  Unary | Cmp+Red |     Reduce |      Chain |
| -------------------------- | ---------: | ---------: | -----: | ------: | ---------: | ---------: |
| `--opt 0` (interpreter)    |     3.58 s |     0.48 s | 1.55 s |  0.46 s |     0.67 s |     0.42 s |
| `--opt 1` (JS-JIT)         |     2.87 s |     0.15 s | 1.23 s |  0.44 s |     0.67 s |     0.38 s |
| `--opt 2` (C-JIT)          |     2.80 s |     0.16 s | 1.20 s |  0.43 s |     0.68 s |     0.34 s |
| `--opt 2 --fuse` (C-JIT)   |     1.93 s |     0.03 s | 0.93 s |  0.11 s |     0.67 s |     0.20 s |
| MATLAB R2026a (1 thread)   |     2.83 s |     0.20 s | 1.81 s |  0.12 s |     0.31 s |     0.40 s |
| MATLAB R2026a (16 threads) | **0.46 s** | **0.05 s** | 0.22 s |  0.07 s | **0.05 s** | **0.07 s** |

## C-JIT architecture

### Per-op mode (`--opt 2`)

The C-JIT emits pure C functions with raw `double*` / `int64_t`
parameters — no N-API, no `napi_value`, no `napi_env`. Each compiled
loop body is a self-contained `.so` loaded via [koffi](https://koffi.dev)
(`dlopen`/`dlsym`, no module registration).

Generated C for the binary kernel (simplified):

```c
for (double __t1 = 1.0; __t1 <= v_trials; __t1 += 1.0) {
    numbl_real_binary_elemwise(ADD, v_x_len, v_x_data, v_y_data, v_r_data);
    if (!__s1_data) __s1_data = malloc(v_x_len * sizeof(double));
    numbl_real_scalar_binary_elemwise(MUL, v_x_len, 0.5, v_x_data, 1, __s1_data);
    numbl_real_binary_elemwise(SUB, v_x_len, v_r_data, __s1_data, v_r_data);
    ...
}
```

### Fused mode (`--opt 2 --fuse`)

The fusion pass (`cFusion.ts`) scans the loop body for runs of
consecutive tensor element-wise assigns and collapses them into a single
per-element `for` loop (`cFusedCodegen.ts`). Trailing reductions
(`acc += sum(r)`) are absorbed as inline accumulators.

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
- **Complex tensors** are not in the C-JIT path yet. Real-only.
- **Compiled `.so` cache** lives at `~/.cache/numbl/c-jit/<sha>.so`,
  keyed by source + compiler/platform/numbl versions + git HEAD +
  `libnumbl_ops.a` contents.
- **`-ffast-math`** is enabled for the C-JIT compile. This allows SIMD
  vectorization of transcendentals but changes FP rounding — check
  values may differ slightly between modes.
