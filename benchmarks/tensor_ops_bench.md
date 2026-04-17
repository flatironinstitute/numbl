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
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 2 --fuse   # C-JIT (fused)
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

| Mode                      |  Total | Binary |  Unary | Cmp+Red | Reduce |  Chain |
| ------------------------- | -----: | -----: | -----: | ------: | -----: | -----: |
| `--opt 0` (interpreter)   | 5.29 s | 1.25 s | 2.42 s |  0.52 s | 0.27 s | 0.96 s |
| `--opt 1` (JS-JIT)        | 3.27 s | 0.66 s | 1.36 s |  0.34 s | 0.30 s | 0.61 s |
| `--opt 2` (C-JIT)         | 3.09 s | 0.66 s | 1.34 s |  0.30 s | 0.25 s | 0.55 s |
| `--opt 2 --fuse` (C-JIT)  | 1.56 s | 0.10 s | 0.57 s |  0.28 s | 0.24 s | 0.38 s |
| MATLAB R2025b `-batch`    | 2.86 s | 0.29 s | 1.82 s |  0.17 s | 0.18 s | 0.37 s |
| MATLAB R2025b (8 threads) | 2.18 s | 0.40 s | 0.95 s |  0.16 s | 0.30 s | 0.35 s |
| Octave 9.4 `--eval`       | 7.75 s | 0.98 s | 4.41 s |  0.75 s | 0.30 s | 1.40 s |
| C baseline, per-op        | 2.63 s | 0.58 s | 1.22 s |  0.24 s | 0.11 s | 0.49 s |
| C baseline, fused         | 1.03 s | 0.10 s | 0.68 s |  0.07 s | 0.04 s | 0.13 s |

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

On a single thread the fused C-JIT is 1.5× faster than MATLAB. With 16 threads
MATLAB auto-parallelizes the element-wise ops and wins wall time by ~4×.
numbl's fused loops are per-core faster than MATLAB's (compare the
`--opt 2 --fuse` row to MATLAB 1 thread); parallelizing them with
`#pragma omp parallel for` would close the multi-threaded gap.

## Reading the table

- **`--opt 2 --fuse` is 1.8× faster than MATLAB overall** and 2× faster
  than per-op C-JIT. The fused codegen collapses consecutive tensor
  element-wise assigns into a single `for` loop with inline scalar
  expressions per element — no libnumbl_ops calls, no intermediate
  buffers, one memory pass. The compiler auto-vectorizes via
  `#pragma omp simd` and `-fopenmp-simd -ffast-math`.
- **Binary fusion matches the hand-written C baseline** (0.10 s vs
  0.10 s). All four ops run in a single SIMD-vectorized loop with one
  read of `x[i]`/`y[i]` and one write of `r[i]`.
- **Unary fusion beats the C baseline** (0.57 s vs 0.68 s). The fused
  loop keeps all transcendentals (`exp`, `cos`, `sin`, `tanh`) in
  registers; the baseline was compiled separately and may not benefit
  from the same cross-function inlining.
- **MATLAB is faster on reductions and comparisons** (0.17 s vs 0.28 s).
  The comparison kernel's `sum(c1 .* c2)` pattern is not yet fused into
  the comparison chain — `c1` and `c2` are fused, but the subsequent
  `sum(c1.*c2)` falls back to per-op. Fusing reduction of arbitrary
  tensor expressions (not just the last chain variable) is future work.
- **Per-op C-JIT (`--opt 2`) is ~5% faster than JS-JIT** on the same
  per-statement architecture. The C-JIT eliminates per-statement JS
  overhead; the main wins are on reductions and comparisons.
- **Octave** is ~5× slower than numbl's fused C-JIT. Octave's
  experimental JIT is off by default in 9.x.

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
