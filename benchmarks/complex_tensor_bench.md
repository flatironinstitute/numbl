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

| Mode                    | Wall time |     Throughput | Notes                       |
| ----------------------- | --------: | -------------: | --------------------------- |
| `--opt 0` (interpreter) |     >60 s |        < 2 M/s | omitted; too slow to finish |
| `--opt 1` (JS-JIT)      |   0.176 s | **568 Mops/s** | fastest                     |
| `--opt 2` (C-JIT)       |   0.240 s |     416 Mops/s | on par with MATLAB          |
| MATLAB R2025b `-batch`  |   0.249 s |     402 Mops/s |                             |

## Why C-JIT doesn't dominate here

Unlike `complex_scalar_bench` where the hot loop keeps both components of
`z` resident in registers (180× speedup over the interpreter), this
benchmark is **memory bound**:

- Each inner step reads and writes `2 * N * 8 = 3.2 MB` of data (re + im
  Float64 buffers). That's well past L1 / L2 and streams through memory
  on every iteration.
- Per element the kernel does ~6 flops (one complex multiply) + 2 flops
  (scalar add) = 8 flops, against 12 memory ops. Flops-to-mem ratio ≈
  0.66, so effective throughput is bounded by memory bandwidth, not
  arithmetic.

JS-JIT wins this particular workload because V8 inlines its Float64Array
element-wise helpers and the whole `z.*z + c` chain becomes one tight
loop that the CPU can execute at near-streaming bandwidth. The C-JIT
currently emits **two separate kernel calls** per step (elemwise MUL
then scalar ADD), each doing its own pass over memory — so it moves the
data through cache twice where JS-JIT moves it once.

Once `--fuse` grows complex-tensor support, the C-JIT can collapse the
two kernels into a single fused loop and recover the JS-JIT's advantage.

## Scratch-allocation fast-path

Before this benchmark landed, the C-JIT was emitting an unconditional
`free(); malloc();` pair for every scratch/destination tensor at every
hot-loop iteration. At N = 200 000, that was ~12 MB of allocator churn
per step and the C-JIT ran at **185 Mops/s** — slower than JS-JIT and
MATLAB.

The fix (same commit as this benchmark): guard every scratch + destination
alloc with a `__need != current_len` check. Hot-loop sizes are invariant
across iterations, so the first call allocates and subsequent calls
short-circuit. The fast-path brought C-JIT up to the **416 Mops/s**
reported above (2.25× speedup, now on par with MATLAB).

## Caveats

- Phase 2 supports complex tensor `+ - .* ./`, unary minus, `conj`,
  `real`, `imag`, `abs`, and flat reductions (`sum`, `prod`, `any`,
  `all`). Anything more (transcendentals, index read/write, range
  slices) still bails to JS-JIT.
- Fusion (`--fuse`) doesn't yet cover complex-tensor chains — that's
  the next optimization knob for closing the gap with JS-JIT.
- `-ffast-math` is enabled for the C-JIT compile; reduction results
  may differ from `--opt 0/1` by a few ULP.
