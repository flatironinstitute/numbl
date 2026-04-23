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
npx tsx src/cli.ts run benchmarks/tensor_ops_bench2.m --opt e1         # JS-JIT + inline C kernels
npx tsx src/cli.ts run benchmarks/tensor_ops_bench2.m --opt e1 --par   # + OpenMP
matlab -batch "run('benchmarks/tensor_ops_bench2.m')"
```

All runs produce the same check values (to FP rounding).

## Results (Linux, N=2 000 000, trials=50)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.2.0
- **MATLAB:** R2025b Update 5
- **Measured:** 2026-04-23 16:10 UTC

Median of 3 runs for non-interpreter modes; `--opt 0` is a single run.

| Mode                      |   Total |  Gauss | Nested | Inl.red | Acc.red | BinOps |  Clamp |
| ------------------------- | ------: | -----: | -----: | ------: | ------: | -----: | -----: |
| `--opt 0` (interpreter)   | 14.58 s | 0.50 s | 2.00 s |  0.36 s |  0.53 s | 5.15 s | 6.05 s |
| `--opt 1` (JS-JIT)        | 13.98 s | 0.31 s | 1.09 s |  0.24 s |  0.35 s | 5.51 s | 6.43 s |
| `--opt e1`                |  1.23 s | 0.10 s | 0.51 s |  0.05 s |  0.10 s | 0.29 s | 0.18 s |
| `--opt e1 --par`          |  0.87 s | 0.09 s | 0.22 s |  0.06 s |  0.12 s | 0.18 s | 0.21 s |
| MATLAB R2025b (1 thread)  |  6.72 s | 0.48 s | 3.43 s |  0.13 s |  0.50 s | 1.46 s | 0.72 s |
| MATLAB R2025b (8 threads) |  2.63 s | 0.12 s | 1.30 s |  0.13 s |  0.20 s | 0.62 s | 0.27 s |

`--opt e1 --par` auto-parallelizes chain kernels whose per-element
body has transcendentals — Nested (2.3×: 0.51→0.22 s), BinOps (1.6×:
0.29→0.18 s), and Gauss (1.1×). Kernels without heavy ops (Inl.red,
Acc.red) skip the parallel-for pragma because thread-spawn overhead
would exceed the memory-bandwidth-bound compute.

### macOS (N=2 000 000, trials=50)

- **CPU:** Apple M4 Max (16 threads)
- **OS:** macOS 15.7.3 (Darwin 24.6.0)
- **Toolchain:** Node v25.9.0, Apple clang 17.0.0, numbl 0.1.7
- **MATLAB:** R2026a (26.1.0)

| Mode                       |  Total |  Gauss | Nested | Inl.red | Acc.red | BinOps |  Clamp |
| -------------------------- | -----: | -----: | -----: | ------: | ------: | -----: | -----: |
| `--opt 0` (interpreter)    | 8.01 s | 0.34 s | 1.37 s |  0.14 s |  0.35 s | 2.70 s | 3.12 s |
| `--opt 1` (JS-JIT)         | 7.36 s | 0.26 s | 0.92 s |  0.09 s |  0.32 s | 2.66 s | 3.12 s |
| MATLAB R2026a (1 thread)   | 3.48 s | 0.31 s | 1.49 s |  0.06 s |  0.33 s | 0.91 s | 0.37 s |
| MATLAB R2026a (16 threads) | 0.47 s | 0.05 s | 0.17 s |  0.02 s |  0.04 s | 0.13 s | 0.06 s |

(macOS `--opt e1 --par` not yet re-collected post-cleanup; on Linux,
e1 + par captures most of the win.)

> **Caveat — Inl.red is a LICM artifact, not a real speedup.** Kernel 3
> is `ir_acc = sum(x.*y + 0.5)` assigned fresh every trial iteration,
> so the whole trials-loop body is loop-invariant. gcc at
> `-O2 -ffast-math` hoists it and runs the kernel once instead of 50
> times, collapsing to ~0 ms. Apple clang doesn't hoist as aggressively,
> and MATLAB doesn't either. The closely-related kernel 4
> (`ir_acc = ir_acc + sum(...)`) has a real carried dependency and is
> not affected.

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

## Fusion features exercised

1. **Single-assign fusion**: a single tensor assignment with 2+ tensor ops
   is fusible (not just 2+ consecutive assignments).
2. **Inline-reduction fusion**: `s = sum(expr)` and `s = s + sum(expr)`
   where `expr` is a purely element-wise tensor expression — the
   reduction is absorbed with no intermediate buffer.
3. **Two-arg element-wise builtins**: `max`, `min`, `mod`, `rem`,
   `atan2`, `hypot` are fusible in both the JS and e1 kernel paths.
