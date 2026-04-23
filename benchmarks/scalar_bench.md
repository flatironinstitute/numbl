# scalar_bench — numbl vs MATLAB on a scalar hot loop

A small compute-bound benchmark that stresses the scalar fast-path in
numbl's JIT. Compares numbl's optimization levels (`--opt 0/1/e1`)
against MATLAB and GNU Octave on the same machine.

## What the benchmark does

`run_bench(N, M)` evaluates

```
  sum_{i=1..N} sum_{k=1..M}  sin(x_i * k) / k²     where  x_i = i · 0.001
```

with all arithmetic held in scalar registers (a carried-dependency
accumulator plus one `sin` and one division per innermost iteration).
Default sizes are `N = 60000`, `M = 500` → **30M** sin+div ops per run.

The whole inner loop stays inside the e1 scalar-kernel whitelist — no
tensor ops, no complex, no struct — so `--opt e1` compiles the whole
`run_bench` body to a single C kernel and calls it from JS via koffi.

## How to run

```bash
npx tsx src/cli.ts run benchmarks/scalar_bench.m --opt 0   # interpreter
npx tsx src/cli.ts run benchmarks/scalar_bench.m --opt 1   # JS-JIT
npx tsx src/cli.ts run benchmarks/scalar_bench.m --opt e1  # JS-JIT + whole-fn C kernel
matlab -batch "run('benchmarks/scalar_bench.m')"
(cd benchmarks && octave --no-gui --quiet --eval scalar_bench)
```

All runs produce the same `result = 2070.336478567545` (to FP rounding).

## Results (Linux, N=60 000, M=500, 30M sin+div)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.2.0
- **MATLAB:** R2025b Update 5
- **Measured:** 2026-04-23 16:10 UTC

Median of 3 runs for all non-interpreter modes (`--opt 0` is a single
run since it's slow enough to make repeats uninteresting).

| Mode                      |  Wall time |       Throughput | Speedup vs `--opt 0` |
| ------------------------- | ---------: | ---------------: | -------------------: |
| `--opt 0` (interpreter)   |    31.36 s |    0.96 Mcalls/s |                   1× |
| `--opt 1` (JS-JIT)        |     0.31 s |      98 Mcalls/s |                ~102× |
| `--opt e1`                | **0.23 s** | **132 Mcalls/s** |            **~138×** |
| MATLAB R2025b (1 thread)  |     0.32 s |      94 Mcalls/s |                 ~99× |
| MATLAB R2025b (8 threads) |     0.32 s |      95 Mcalls/s |                 ~99× |

### macOS (N=60 000, M=500, 30M sin+div)

- **CPU:** Apple M4 Max
- **OS:** macOS 15.7.5 (Darwin 24.6.0)
- **Toolchain:** Node v25.9.0, Apple clang 17.0.0, gcc-15 15.2.0 (Homebrew), numbl 0.2.0
- **MATLAB:** R2025b
- **Measured:** 2026-04-23

Single run per mode. `--opt e1 --par` compiled with `NUMBL_CC=gcc-15`
(Apple clang ships without OpenMP threading).

| Mode                     |   Wall time |       Throughput |
| ------------------------ | ----------: | ---------------: |
| `--opt 1` (JS-JIT)       |     0.232 s |     129 Mcalls/s |
| `--opt e1`               | **0.086 s** | **349 Mcalls/s** |
| `--opt e1 --par`         |     0.086 s |     348 Mcalls/s |
| MATLAB R2025b (1 thread) |     0.208 s |     144 Mcalls/s |
| MATLAB R2025b (multi)    |     0.208 s |     144 Mcalls/s |

`--par` and MATLAB multi-thread are no-ops here: the inner loop is a
serial carried-dependency accumulator (`acc += sin(x·k)/k²`) with no
parallelism to exploit.

## Notes on timing methodology

- **Warm-up.** A `run_bench(100, 10)` call before `tic` lands the JIT
  specialization in cache. Compiled `.so` modules live in
  `~/.cache/numbl/c-jit/<sha>.so`, so second runs skip the `cc` cost.
- **Compile flags.** `-O2 -fPIC -shared -std=c11 -march=native` (printed
  on first `--opt e1` run as the `C-JIT:` banner).

## Caveats

- The inner loop is a carried-dependency accumulator, which prevents
  SIMD vectorization. `-march=native` mostly buys tighter scalar code,
  not auto-vectorization.
- **No exit hangs.** The koffi-based `.so` loading has no Node module
  registration, eliminating the process-exit hangs that affected the
  previous N-API `.node` approach.
