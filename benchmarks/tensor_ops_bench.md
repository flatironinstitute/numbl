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
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 0   # interpreter
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 1   # JS-JIT
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 2   # C-JIT
matlab -batch "run('benchmarks/tensor_ops_bench.m')"
(cd benchmarks && octave --no-gui --quiet --eval tensor_ops_bench)
bash benchmarks/tensor_ops_bench_compare.sh                    # all of the above
```

All runs produce the same check values (to FP rounding).

## Results (Linux, N=2 000 000, trials=50)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.1.7
- **MATLAB:** R2025b Update 5 · **Octave:** 9.4.0

Median of 3 runs per numbl mode; single run for MATLAB/Octave.

| Mode                    |  Total | Binary |  Unary | Cmp+Red | Reduce |  Chain |
| ----------------------- | -----: | -----: | -----: | ------: | -----: | -----: |
| `--opt 0` (interpreter) | 6.10 s | 1.10 s | 2.99 s |  0.54 s | 0.33 s | 1.13 s |
| `--opt 1` (JS-JIT)      | 3.49 s | 0.71 s | 1.48 s |  0.36 s | 0.32 s | 0.63 s |
| `--opt 2` (C-JIT)       | 3.35 s | 0.72 s | 1.46 s |  0.32 s | 0.27 s | 0.60 s |
| MATLAB R2025b `-batch`  | 3.19 s | 0.28 s | 2.10 s |  0.20 s | 0.20 s | 0.42 s |
| Octave 9.4 `--eval`     | 8.77 s | 1.03 s | 4.87 s |  0.88 s | 0.37 s | 1.62 s |
| C baseline, per-op      | 2.63 s | 0.58 s | 1.22 s |  0.24 s | 0.11 s | 0.49 s |
| C baseline, fused       | 1.03 s | 0.10 s | 0.68 s |  0.07 s | 0.04 s | 0.13 s |

### macOS — TBD

| Mode                    | Total | Binary | Unary | Cmp+Red | Reduce | Chain |
| ----------------------- | ----: | -----: | ----: | ------: | -----: | ----: |
| `--opt 0` (interpreter) |       |        |       |         |        |       |
| `--opt 1` (JS-JIT)      |       |        |       |         |        |       |
| `--opt 2` (C-JIT)       |       |        |       |         |        |       |
| MATLAB `-batch`         |       |        |       |         |        |       |
| Octave `--eval`         |       |        |       |         |        |       |

## Reading the table

- **`--opt 2` (C-JIT) is ~4% faster overall than `--opt 1` (JS-JIT).**
  The C-JIT compiles each loop body to a single `.so` loaded via koffi,
  with direct `numbl_real_*` calls and lazy-allocated scratch buffers
  reused across iterations. The main wins are on reductions (−16%) and
  comparisons (−11%), where eliminating per-statement JS overhead matters
  most.
- **MATLAB is ~5% faster overall** than `--opt 2`, but slower on unary
  ops (2.10 s vs 1.46 s). MATLAB's binary dispatch and BLAS-backed
  element-wise ops are highly optimized; numbl's `libnumbl_ops` kernels
  are competitive on transcendentals (exp/sin/cos/tanh).
- **The C baselines bracket future optimization headroom.** `per-op`
  (2.63 s) is the ceiling for the current per-statement architecture —
  any mode that calls libnumbl_ops per op can approach but not beat it.
  `fused` (1.03 s) is the ceiling if the C-JIT ever emits fused per-
  element loops instead of chained libnumbl_ops calls.
- **Octave** is ~2.5× slower than numbl's C-JIT. Octave's experimental
  JIT is off by default in 9.x.
- **No exit hangs.** The koffi-based `.so` loading has no Node module
  registration or libuv teardown hooks, eliminating the process-exit
  hangs that affected the previous N-API `.node` approach.

## C-JIT architecture (koffi)

The C-JIT emits pure C functions with raw `double*` / `int64_t`
parameters — no N-API, no `napi_value`, no `napi_env`. Each compiled
loop body is a self-contained `.so` loaded via [koffi](https://koffi.dev)
(`dlopen`/`dlsym`, no module registration).

Generated C for the binary kernel (simplified):

```c
void jit__loop_for(
    double v_trials,
    const double *v_x_data, int64_t v_x_len,
    const double *v_y_data, int64_t v_y_len,
    const double *v_r_data_in, int64_t v_r_len_in,
    double *v_k_out,
    double *v_r_buf, int64_t *v_r_out_len)
{
  double *v_r_data = v_r_buf;
  int64_t v_r_len = v_r_len_in;
  double *__s1_data = NULL; /* scratch, lazily allocated */
  ...
  for (double __t1 = 1.0; __t1 <= v_trials; __t1 += 1.0) {
    v_r_len = v_x_len;
    numbl_real_binary_elemwise(NUMBL_REAL_BIN_ADD, v_r_len, v_x_data, v_y_data, v_r_data);
    if (!__s1_data) __s1_data = malloc(v_r_len * sizeof(double));
    numbl_real_scalar_binary_elemwise(NUMBL_REAL_BIN_MUL, v_r_len, 0.5, v_x_data, 1, __s1_data);
    numbl_real_binary_elemwise(NUMBL_REAL_BIN_SUB, v_r_len, v_r_data, __s1_data, v_r_data);
    ...
  }
  *v_k_out = v_k;
  *v_r_out_len = v_r_len;
  free(__s1_data); ...
}
```

## Caveats

- **Domain-restricted unaries** (`sqrt`, `log`, `log2`, `log10`, `asin`,
  `acos`) bail to the interpreter — MATLAB promotes out-of-domain inputs
  to complex, libnumbl_ops returns NaN. Same behavior as JS-JIT.
- **Complex tensors** are not in the C-JIT path yet. Real-only.
- **Compiled `.so` cache** lives at `~/.cache/numbl/c-jit/<sha>.so`,
  keyed by source + compiler/platform/numbl versions + `libnumbl_ops.a`
  contents.
