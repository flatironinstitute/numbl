# tensor_ops_bench2 — extended fusion benchmark

Extended fusion benchmark exercising single-expression fusion, inline
reductions, and two-argument element-wise builtins. Companion to
`tensor_ops_bench.m`. Six kernels on real Float64 vectors (N=2M), 50
trials each.

## Kernels

| Kernel          | Pattern                                             |
| --------------- | --------------------------------------------------- |
| Gaussian        | `u = exp(-x .* x)`                                  |
| Nested          | `u = tanh(abs(sin(x+1).*sin(x+1) + cos(y.*2)))`     |
| Inline red      | `s = sum(x .* y + 0.5)`                             |
| Accum red       | `s = s + sum(exp(-x .* x))`                         |
| Binary builtins | `u = max(x,y); u = u+atan2(y,x); u = u.*hypot(x,y)` |
| Clamp+dist      | `u = max(min(x,0.5),-0.5); u = hypot(u-y, x.*y)`    |

## How to run

```bash
npx tsx src/cli.ts run benchmarks/tensor_ops_bench2.m --opt 0          # interpreter
npx tsx src/cli.ts run benchmarks/tensor_ops_bench2.m --opt 1          # JS-JIT
npx tsx src/cli.ts run benchmarks/tensor_ops_bench2.m --opt 2          # C-JIT (per-op)
npx tsx src/cli.ts run benchmarks/tensor_ops_bench2.m --opt 2 --fuse        # C-JIT (fused)
npx tsx src/cli.ts run benchmarks/tensor_ops_bench2.m --opt 2 --fuse --par  # C-JIT (fused+parallel)
matlab -batch "run('benchmarks/tensor_ops_bench2.m')"
```

All runs produce the same check values (to FP rounding).

## Results (Linux, N=2 000 000, trials=50)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.1.7
- **MATLAB:** R2025b Update 5

Median of 3 runs for all modes.

| Mode                           |  Total |  Gauss | Nested | Inl.red | Acc.red | BinOps |  Clamp |
| ------------------------------ | -----: | -----: | -----: | ------: | ------: | -----: | -----: |
| `--opt 0` (interpreter)        | 15.4 s | 0.57 s | 1.87 s |  0.35 s |  0.52 s | 5.64 s | 6.45 s |
| `--opt 1` (JS-JIT)             | 13.3 s | 0.28 s | 1.05 s |  0.24 s |  0.37 s | 5.25 s | 6.08 s |
| `--opt 2` (C-JIT)              | 12.2 s | 0.27 s | 1.33 s |  0.20 s |  0.29 s | 4.63 s | 5.48 s |
| `--opt 2 --fuse` (C-JIT)       | 1.39 s | 0.10 s | 0.51 s |  0.07 s |  0.32 s | 0.27 s | 0.12 s |
| `--opt 2 --fuse --par` (C-JIT) | 0.95 s | 0.07 s | 0.19 s |  0.08 s |  0.36 s | 0.14 s | 0.11 s |
| MATLAB R2025b (1 thread)       | 6.28 s | 0.43 s | 3.27 s |  0.12 s |  0.46 s | 1.41 s | 0.58 s |
| MATLAB R2025b `-batch`         | 3.95 s | 0.24 s | 2.04 s |  0.09 s |  0.27 s | 0.93 s | 0.39 s |
| MATLAB R2025b (8 threads)      | 2.43 s | 0.12 s | 1.16 s |  0.11 s |  0.16 s | 0.61 s | 0.28 s |

### macOS (N=2 000 000, trials=50)

- **CPU:** Apple M4 Max (16 threads)
- **OS:** macOS 15.7.3 (Darwin 24.6.0)
- **Toolchain:** Node v25.9.0, Apple clang 17.0.0, numbl 0.1.7
- **MATLAB:** R2026a (26.1.0)

| Mode                       |      Total |      Gauss |     Nested |    Inl.red |    Acc.red |     BinOps |      Clamp |
| -------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| `--opt 0` (interpreter)    |     8.01 s |     0.34 s |     1.37 s |     0.14 s |     0.35 s |     2.70 s |     3.12 s |
| `--opt 1` (JS-JIT)         |     7.36 s |     0.26 s |     0.92 s |     0.09 s |     0.32 s |     2.66 s |     3.12 s |
| `--opt 2` (C-JIT)          |     7.37 s |     0.23 s |     0.93 s |     0.09 s |     0.28 s |     2.67 s |     3.18 s |
| `--opt 2 --fuse` (C-JIT)   |     1.90 s |     0.20 s |     0.76 s | **0.02 s** |     0.19 s |     0.61 s |     0.14 s |
| MATLAB R2026a (1 thread)   |     3.48 s |     0.31 s |     1.49 s |     0.06 s |     0.33 s |     0.91 s |     0.37 s |
| MATLAB R2026a (16 threads) | **0.47 s** | **0.05 s** | **0.17 s** | **0.02 s** | **0.04 s** | **0.13 s** | **0.06 s** |

## Generated C examples

Single-expression Gaussian (`u = exp(-x .* x)`):

```c
#pragma omp simd
for (int64_t __i = 0; __i < v_x_len; __i++) {
  double __f_u1 = exp(((-v_x_data[__i]) * v_x_data[__i]));
  v_u1_data[__i] = __f_u1;
}
```

Inline reduction (`s = sum(x .* y + 0.5)`):

```c
double __f_reduce_acc = 0.0;
for (int64_t __i = 0; __i < v_x_len; __i++) {
  double __f___red_tmp = ((v_x_data[__i] * v_y_data[__i]) + 0.5);
  __f_reduce_acc += __f___red_tmp;
}
v_ir_acc = __f_reduce_acc;
```

Clamp + distance (`u = max(min(x,0.5),-0.5); u = hypot(u-y, x.*y)`):

```c
#pragma omp simd
for (int64_t __i = 0; __i < v_x_len; __i++) {
  double __f_u3 = fmax(fmin(v_x_data[__i], 0.5), (-0.5));
  __f_u3 = hypot((__f_u3 - v_y_data[__i]), (v_x_data[__i] * v_y_data[__i]));
  v_u3_data[__i] = __f_u3;
}
```

## New fusion features exercised

1. **Single-assign fusion**: a single tensor assignment with 2+ tensor ops
   is now fusible (previously required 2+ consecutive assignments).
2. **Inline-reduction fusion**: `s = sum(expr)` and `s = s + sum(expr)`
   where `expr` is a purely element-wise tensor expression — the
   reduction is absorbed with no intermediate buffer.
3. **Two-arg element-wise builtins**: `max`, `min`, `mod`, `rem`,
   `atan2`, `hypot` are now fusible in both JS-JIT and C-JIT paths.
