# scalar_bench — numbl vs MATLAB on a scalar hot loop

A small compute-bound benchmark that stresses the scalar fast-path in
numbl's JIT. Compares numbl's three optimization levels (`--opt 0/1/2`)
against MATLAB and GNU Octave on the same machine.

## What the benchmark does

`run_bench(N, M)` evaluates

```
  sum_{i=1..N} sum_{k=1..M}  sin(x_i * k) / k²     where  x_i = i · 0.001
```

with all arithmetic held in scalar registers (a carried-dependency
accumulator plus one `sin` and one division per innermost iteration).
Default sizes are `N = 60000`, `M = 500` → **30M** sin+div ops per run.

The whole inner loop stays inside the C-JIT whitelist — no tensor ops,
no complex, no struct — so every step runs on the scalar fast path when
`--opt 2` is active.

## How to run

```bash
npx tsx src/cli.ts run benchmarks/scalar_bench.m --opt 0   # interpreter
npx tsx src/cli.ts run benchmarks/scalar_bench.m --opt 1   # JS-JIT
npx tsx src/cli.ts run benchmarks/scalar_bench.m --opt 2   # C-JIT
matlab -batch "run('benchmarks/scalar_bench.m')"
(cd benchmarks && octave --no-gui --quiet --eval scalar_bench)
bash benchmarks/scalar_bench_compare.sh                    # all of the above
```

All runs produce the same `result = 2070.336478567545` (to FP rounding).

## Results (Linux, N=60 000, M=500, 30M sin+div)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.1.7
- **MATLAB:** R2025b Update 5 · **Octave:** 9.4.0

Median of 3 runs for all modes.

| Mode                      |  Wall time |       Throughput | Speedup vs `--opt 0` |
| ------------------------- | ---------: | ---------------: | -------------------: |
| `--opt 0` (interpreter)   |    30.71 s |    0.98 Mcalls/s |                   1× |
| `--opt 1` (JS-JIT)        |     0.30 s |     101 Mcalls/s |                ~103× |
| `--opt 2` (C-JIT)         | **0.22 s** | **135 Mcalls/s** |            **~139×** |
| `--opt 2 --fuse` (C-JIT)  |     0.23 s |     133 Mcalls/s |                ~136× |
| `--opt 2 --fuse --par`    |     0.22 s |     135 Mcalls/s |                ~139× |
| MATLAB R2025b (1 thread)  |     0.30 s |     101 Mcalls/s |                ~103× |
| MATLAB R2025b `-batch`    |     0.30 s |     100 Mcalls/s |                ~103× |
| MATLAB R2025b (8 threads) |     0.30 s |     100 Mcalls/s |                ~102× |
| Octave 9.4 `--eval`       |    65.03 s |    0.46 Mcalls/s |               ~0.53× |

### macOS (N=60 000, M=500, 30M sin+div)

- **CPU:** Apple M4 Max
- **OS:** macOS 15.7.3 (Darwin 24.6.0)
- **Toolchain:** Node v25.9.0, Apple clang 17.0.0, numbl 0.1.7
- **MATLAB:** R2026a (26.1.0) — single-threaded (`maxNumCompThreads(1)`; multi-threaded identical for this scalar loop)

| Mode                    |  Wall time |       Throughput | Speedup vs `--opt 0` |
| ----------------------- | ---------: | ---------------: | -------------------: |
| `--opt 0` (interpreter) |    17.83 s |    1.68 Mcalls/s |                   1× |
| `--opt 1` (JS-JIT)      |     0.23 s |     131 Mcalls/s |                 ~78× |
| `--opt 2` (C-JIT)       | **0.09 s** | **351 Mcalls/s** |            **~210×** |
| MATLAB R2026a `-batch`  |     0.20 s |     148 Mcalls/s |                 ~88× |

`--fuse` and `--par` don't apply here: the inner loop is a serial
carried-dependency accumulator (`acc += sin(x·k)/k²`), so there's
nothing to fuse and nothing to parallelize. Running with `--fuse --par`
(under `NUMBL_CC=gcc-15`) measures 0.09 s / 327 Mcalls/s — effectively
the same as plain `--opt 2`.

## Notes on timing methodology

- **Warm-up.** A `run_bench(100, 10)` call before `tic` lands the JIT
  specialization in cache. Compiled `.so` modules live in
  `~/.cache/numbl/c-jit/<sha>.so`, so second runs skip the `cc` cost.
- **Compile flags.** `-O2 -fPIC -shared -std=c11 -march=native` (printed
  on first `--opt 2` run as the `C-JIT:` banner).

## Caveats

- The inner loop is a carried-dependency accumulator, which prevents
  SIMD vectorization. `-march=native` mostly buys tighter scalar code,
  not auto-vectorization.
- **No exit hangs.** The koffi-based `.so` loading has no Node module
  registration, eliminating the process-exit hangs that affected the
  previous N-API `.node` approach.
