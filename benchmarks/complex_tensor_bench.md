# complex_tensor_bench — numbl vs MATLAB on complex-tensor element-wise ops

Six-kernel complex-tensor benchmark. Companion to `complex_scalar_bench`
(which measures per-scalar complex arithmetic) and `tensor_ops_bench`
(which measures real-tensor fusion). Kernels are chosen to exercise both
the fused per-element path and the per-op libnumbl_ops kernels, so the
total captures numbl's coverage of the full complex-tensor surface.

## Kernels

| #   | Pattern                 | Fuses?   | What it exercises                                 |
| --- | ----------------------- | -------- | ------------------------------------------------- |
| 1   | `u1 = z .* z + c`       | ✅ fused | scalar broadcast complex mul + add                |
| 2   | `u1 = z .* w + y`       | ✅ fused | two-tensor complex mul + real tensor add          |
| 3   | `u1 = conj(z) .* z + y` | ✅ fused | `conj` in a chain (result's imag cancels)         |
| 4   | `u1 = x + y * 1i`       | ✅ fused | real→complex widening via `ImagLiteral`           |
| 5   | `u2 = z ./ w`           | per-op   | complex `./` (Smith's method branches break SIMD) |
| 6   | `acc += sum(abs(z))`    | per-op   | `abs(complex)` is complex→real mid-chain          |

Kernels 1–4 collapse into a single `#pragma omp simd for` loop with the
re/im arithmetic inlined in registers. Kernels 5–6 fall back to the
libnumbl_ops kernel-call path (`numbl_complex_binary_elemwise`,
`numbl_complex_abs`, `numbl_complex_flat_reduce`).

## How to run

```bash
npx tsx src/cli.ts run benchmarks/complex_tensor_bench.m --opt 1           # JS-JIT
npx tsx src/cli.ts run benchmarks/complex_tensor_bench.m --opt 2           # C-JIT, per-op
npx tsx src/cli.ts run benchmarks/complex_tensor_bench.m --opt 2 --fuse    # C-JIT, fused
matlab -batch "run('benchmarks/complex_tensor_bench.m')"
```

All runs produce the same check values to FP rounding:

```
real(sum(u1))  = 12500.025
imag(sum(u1))  = 25000.05
real(sum(u2))  = 473072.2695
imag(sum(u2))  = 6857.423421
abs_acc        = 2795090.562
```

## Results (Linux, N=500 000, trials=100)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0
- **MATLAB:** R2025b Update 5

Median of 3 runs. Times in seconds.

| Kernel                    | `--opt 1` | `--opt 2` | `--opt 2 --fuse` | MATLAB |
| ------------------------- | --------: | --------: | ---------------: | -----: |
| 1. Mandelbrot z.\*z+c     |     0.112 |     0.177 |        **0.044** |  0.138 |
| 2. Tensor chain z.\*w+y   |     0.147 |     0.217 |        **0.082** |  0.132 |
| 3. Conj chain conj(z).\*z |     0.324 |     0.279 |        **0.063** |  0.190 |
| 4. Widening x+y\*1i       |     0.103 |     0.159 |        **0.048** |  0.110 |
| 5. Divide z./w            |     0.097 |     0.151 |            0.153 |  0.124 |
| 6. abs + sum reduction    |     0.077 |     0.052 |            0.053 |  0.178 |
| **Total**                 |     0.874 |     1.069 |        **0.449** |  0.884 |

Per-kernel notes:

- **Kernels 1–4 (fused):** the C-JIT runs 2–4× faster than MATLAB or
  JS-JIT because the whole chain lowers to one AVX2-vectorizable loop
  instead of two or three memory passes through libnumbl_ops kernels.
  The conj chain (kernel 3) is the most striking — 3× MATLAB, 5× JS-JIT.
- **Kernel 5 (divide):** all three modes land in roughly the same place
  (≈ 0.1–0.15 s). The per-op C kernel uses Smith's method with a single
  branch per element; branch prediction and the scalar loop itself
  dominate over op dispatch.
- **Kernel 6 (abs + sum):** the C-JIT and JS-JIT both use a streaming
  `numbl_complex_abs` + `numbl_complex_flat_reduce` pipeline which MATLAB
  beats on some runs and loses on others — near the noise floor.
- **Total:** `--opt 2 --fuse` finishes in ~half MATLAB's time. Most of
  the win comes from the fused kernels; the non-fused kernels are
  already competitive per-op.

## Fused complex-tensor codegen

With `--fuse`, `z = z .* z + c` becomes a single SIMD loop like this:

```c
#pragma omp simd
for (int64_t __i = 0; __i < v_z_len; __i++) {
  double __f_u1_re = (v_z_data[__i] * v_z_data[__i]
                      - __im_v_z_data[__i] * __im_v_z_data[__i]) + v_c;
  double __f_u1_im = (v_z_data[__i] * __im_v_z_data[__i]
                      + __im_v_z_data[__i] * v_z_data[__i]) + __im_v_c;
  v_u1_data[__i] = __f_u1_re;
  __im_v_u1_data[__i] = __f_u1_im;
}
```

The re/im arithmetic stays in registers; the compiler CSEs the repeated
`v_z_data[__i]` loads and vectorizes the straight-line body. Without
`--fuse`, the same step emits two back-to-back libnumbl_ops kernel calls
(`numbl_complex_binary_elemwise` for `z.*z`, then
`numbl_complex_scalar_binary_elemwise` for `+c`) — half the memory
bandwidth is spent re-reading the intermediate.

## Scratch-allocation fast-path (`--opt 2`, no fuse)

A prior change to the per-op path (same commit that landed the
benchmark) guards every scratch / destination alloc with
`__need != current_len`. Hot-loop sizes are invariant, so the first
call allocates and subsequent calls short-circuit. Before that fix the
per-op C-JIT ran at ≈ 185 Mops/s on the single-kernel `z.*z+c` bench;
with the fix it's 416 Mops/s, matching MATLAB before `--fuse`.

## What fuses, what doesn't (complex chains)

The complex per-element emitter supports:

- Binary: `+`, `-`, `*` / `.*`
- Unary: `+`, `-`
- Call: `conj`, `real`, `imag`
- Operand widening: real tensor or real scalar in a complex chain is
  read with im = 0 implicitly.

Chains with `./` (complex divide — Smith's method branches break SIMD),
`abs(complex)` (type transition: complex → real mid-chain), or
transcendentals (`exp`, `sin`, ... on complex) fall back to per-op
libnumbl_ops kernel calls. Trailing complex reductions (`sum(z)` →
complex scalar) are also not absorbed, since the fused-loop accumulator
is a single `double`; the reduction runs post-loop via the normal path.

## Caveats

- Phase 2 supports complex tensor `+ - .* ./`, unary minus, `conj`,
  `real`, `imag`, `abs`, and flat reductions (`sum`, `prod`, `any`,
  `all`). Anything more (transcendentals, index read/write, range
  slices) still bails to JS-JIT.
- `-ffast-math` is enabled for the C-JIT compile; reduction results
  may differ from `--opt 0/1` by a few ULP.
- `--opt 0` (interpreter) is omitted because a single kernel at
  N = 500 000 × 100 trials would take tens of minutes.
