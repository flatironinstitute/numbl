# complex_tensor_bench — numbl vs MATLAB on complex-tensor element-wise ops

Six-kernel complex-tensor benchmark. Companion to `complex_scalar_bench`
and `tensor_ops_bench`.

## Kernels

| #   | Pattern                 | Fuses?   | What it exercises                                 |
| --- | ----------------------- | -------- | ------------------------------------------------- |
| 1   | `u1 = z .* z + c`       | ✅ fused | scalar broadcast complex mul + add                |
| 2   | `u1 = z .* w + y`       | ✅ fused | two-tensor complex mul + real tensor add          |
| 3   | `u1 = conj(z) .* z + y` | ✅ fused | `conj` in a chain (result's imag cancels)         |
| 4   | `u1 = x + y * 1i`       | ✅ fused | real→complex widening via `ImagLiteral`           |
| 5   | `u2 = z ./ w`           | per-op   | complex `./` (Smith's method branches break SIMD) |
| 6   | `acc += sum(abs(z))`    | per-op   | `abs(complex)` is complex→real mid-chain          |

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
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.2.0
- **MATLAB:** R2025b Update 5
- **Measured:** 2026-04-23 16:10 UTC

Median of 3 runs. Times in seconds. `--opt 0` is omitted (a single
kernel at N=500 000 × 100 trials takes tens of minutes).

| Kernel                    | `--opt 1` | `--opt e1` | `--opt 2 --fuse --par` | MATLAB (1 thread) | MATLAB (8 threads) |
| ------------------------- | --------: | ---------: | ---------------------: | ----------------: | -----------------: |
| 1. Mandelbrot z.\*z+c     |     0.119 |      0.130 |                  0.047 |             0.272 |              0.056 |
| 2. Tensor chain z.\*w+y   |     0.151 |      0.158 |                  0.088 |             0.278 |              0.076 |
| 3. Conj chain conj(z).\*z |     0.340 |      0.361 |                  0.064 |             0.354 |              0.140 |
| 4. Widening x+y\*1i       |     0.111 |      0.118 |                  0.055 |             0.225 |              0.060 |
| 5. Divide z./w            |     0.079 |      0.084 |                  0.078 |             0.249 |              0.070 |
| 6. abs + sum reduction    |     0.081 |      0.085 |                  0.053 |             0.266 |              0.071 |
| **Total**                 |     0.913 |      0.947 |                  0.396 |             1.643 |              0.475 |

On this benchmark `--opt e1` behaves like `--opt 1` — the per-chain
C-kernel path in e1 currently handles real-tensor chains only, so
the four fusion-eligible complex chains (kernels 1–4) fall through
to the JS-JIT per-op complex path. `--opt 2 --fuse --par` uses its
dedicated complex fused emitter (paired re/im buffers, 6-flop
per-element inner body) and wins. Adding complex support to the e1
chain kernel would close most of this gap; it's tracked in the e1
follow-ups (see `src/numbl-core/jit/e1/`).

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
