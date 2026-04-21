# complex_tensor_bench — numbl vs MATLAB on a complex-tensor hot loop

Compute + memory bound benchmark for the C-JIT's paired-buffer complex
tensor codegen (Phase 2 of full-complex support). Compares numbl's three
optimization levels (`--opt 0/1/2`) against MATLAB.

## What the benchmark does

`run_bench(x, M)` iterates the map `z -> z.*z + c` on a length-N complex
tensor for `M` steps (seed `z = x + 2i*x`, `c = 0.001 + 0.001i`), then
reduces via `real(sum(z))`. Default sizes `N = 200 000`, `M = 500` →
**100M** complex-element `z.*z + c` operations per run.

Each inner step is one `.*` (tensor × tensor) plus one `+ c` (tensor +
complex scalar) — both lowered via paired re/im buffer kernels in
`numbl_ops`:

```
numbl_complex_binary_elemwise(MUL, len, z_re, z_im, z_re, z_im, s_re, s_im)
numbl_complex_scalar_binary_elemwise(ADD, len, c_re, c_im, s_re, s_im, 0, dst_re, dst_im)
```

## How to run

```bash
npx tsx src/cli.ts run benchmarks/complex_tensor_bench.m --opt 0   # interpreter
npx tsx src/cli.ts run benchmarks/complex_tensor_bench.m --opt 1   # JS-JIT
npx tsx src/cli.ts run benchmarks/complex_tensor_bench.m --opt 2   # C-JIT
matlab -batch "run('benchmarks/complex_tensor_bench.m')"
```

All runs produce the same `result = 199.999195988823` (to FP rounding).

## Results (Linux, N=200 000, M=500, 100M z.\*z+c ops)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0
- **MATLAB:** R2025b Update 5

Median of 5 runs. Variance between invocations is ±5 %.

| Mode                     | Wall time |      Throughput | vs MATLAB |
| ------------------------ | --------: | --------------: | --------: |
| `--opt 0` (interpreter)  |     >60 s |         < 2 M/s |      skip |
| `--opt 1` (JS-JIT)       |   0.176 s |      568 Mops/s |      1.4× |
| `--opt 2` (C-JIT)        |   0.240 s |      416 Mops/s |      1.0× |
| `--opt 2 --fuse` (C-JIT) |   0.041 s | **2430 Mops/s** |  **6.0×** |
| MATLAB R2025b `-batch`   |   0.249 s |      402 Mops/s |      1.0× |

## Fused complex-tensor codegen (`--fuse`)

With `--fuse`, the C-JIT collapses `z = z .* z + c` into a single fused
per-element loop instead of two back-to-back libnumbl_ops kernel calls.
That's the 6× speedup vs both unfused C-JIT and MATLAB:

```c
#pragma omp simd
for (int64_t __i = 0; __i < v_z_len; __i++) {
  double __f_z_re = (v_z_data[__i] * v_z_data[__i]
                     - __im_v_z_data[__i] * __im_v_z_data[__i]) + v_c;
  double __f_z_im = (v_z_data[__i] * __im_v_z_data[__i]
                     + __im_v_z_data[__i] * v_z_data[__i]) + __im_v_c;
  v_z_data[__i] = __f_z_re;
  __im_v_z_data[__i] = __f_z_im;
}
```

Per element the inner body does 6 real flops (one complex multiply +
one complex add) against 4 memory ops (2 reads + 2 writes of re/im).
`#pragma omp simd` vectorizes the straight-line body to AVX2 FMAs; the
compiler CSE's the repeated `v_z_data[__i]` / `__im_v_z_data[__i]`
loads into registers. The data still streams through memory on every
iteration (3.2 MB hot set at N = 200 000), so peak throughput is
roughly `mem_bandwidth / 3.2MB ≈ 2–3 Gops/s` — the fused emitter
saturates that, where the per-op path spent half its bandwidth
re-reading the scratch tensor between the two kernel calls.

## Scratch-allocation fast-path (`--opt 2`, no fuse)

Before landing the bench, the C-JIT was emitting an unconditional
`free(); malloc();` pair for every scratch / destination tensor on
every hot-loop iteration. At N = 200 000 that was ~12 MB of allocator
churn per step, and the C-JIT ran at **185 Mops/s** — slower than
JS-JIT and MATLAB.

Fix (same commit as this benchmark): guard every scratch + destination
alloc with a `__need != current_len` check. Hot-loop sizes are invariant
across iterations, so the first call allocates and subsequent calls
short-circuit. The fast-path brought per-op C-JIT to **416 Mops/s**
(2.25× speedup, on par with MATLAB before `--fuse`).

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
