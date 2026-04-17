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
npx tsx src/cli.ts run benchmarks/tensor_ops_bench2.m --opt 2 --fuse   # C-JIT (fused)
matlab -batch "run('benchmarks/tensor_ops_bench2.m')"
```

All runs produce the same check values (to FP rounding).

## Results (Linux, N=2 000 000, trials=50)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.1.7
- **MATLAB:** R2025b Update 5

Median of 3 runs for all modes.

| Mode                     |  Total |  Gauss | Nested | Inl.red | Acc.red | BinOps |  Clamp |
| ------------------------ | -----: | -----: | -----: | ------: | ------: | -----: | -----: |
| `--opt 0` (interpreter)  | 15.3 s | 0.53 s | 2.10 s |  0.35 s |  0.58 s | 5.21 s | 6.29 s |
| `--opt 1` (JS-JIT)       | 15.1 s | 0.37 s | 1.20 s |  0.27 s |  0.41 s | 5.84 s | 6.81 s |
| `--opt 2` (C-JIT)        | 15.0 s | 0.30 s | 1.22 s |  0.24 s |  0.34 s | 5.91 s | 6.89 s |
| `--opt 2 --fuse` (C-JIT) |  1.6 s | 0.12 s | 0.56 s |  0.07 s |  0.37 s | 0.31 s | 0.13 s |
| MATLAB R2025b `-batch`   |  4.3 s | 0.30 s | 2.21 s |  0.10 s |  0.29 s | 0.94 s | 0.39 s |

## Reading the table

- **`--opt 2 --fuse` is 2.6× faster than MATLAB overall.** The fused
  codegen collapses each kernel into a single per-element `for` loop —
  no intermediate buffers, one memory pass.
- **Binary-builtin fusion is the biggest win** (0.31 s vs 5.91 s per-op,
  19× speedup). Without fusion, `max(x,y)` / `atan2` / `hypot` each
  allocate a tensor and iterate separately. Fused, they become one loop
  with `fmax` / `atan2` / `hypot` calls per element.
- **Clamp+distance** shows a similar effect (0.13 s vs 6.89 s). The
  fused loop inlines `fmax(fmin(x[i], 0.5), -0.5)` and `hypot(...)` in
  a single SIMD-vectorized pass.
- **Single-expression Gaussian** (`exp(-x.*x)`) benefits from the new
  single-assign fusion (0.12 s vs 0.30 s). Previously a single assign
  couldn't form a fusible chain; now any assignment with 2+ tensor ops
  qualifies.
- **Inline reduction** (`sum(x.*y + 0.5)`) fuses the element-wise
  expression into the reduction loop — no intermediate tensor at all
  (0.07 s vs 0.24 s).
- **Accumulate reduction** (`s += sum(exp(-x.*x))`) is roughly break-even
  with per-op. The `exp()` inside the reduction loop can't be
  SIMD-vectorized (no `#pragma omp simd` on reduction loops), so the
  per-element overhead offsets the saved allocation.
- **JS-JIT and per-op C-JIT are similar** on kernels 5–6 because two-arg
  tensor builtins weren't in the C-JIT path until this change. Both now
  emit per-element loops in the per-op path, but the real win is fusion.

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
