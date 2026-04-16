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

Median of 3 runs per numbl mode; single run for MATLAB/Octave.

| Mode                    |  Wall time |       Throughput | Speedup vs `--opt 0` |
| ----------------------- | ---------: | ---------------: | -------------------: |
| `--opt 0` (interpreter) |    34.32 s |    0.87 Mcalls/s |                   1× |
| `--opt 1` (JS-JIT)      |     0.31 s |      97 Mcalls/s |                ~111× |
| `--opt 2` (C-JIT)       | **0.23 s** | **129 Mcalls/s** |            **~149×** |
| MATLAB R2025b `-batch`  |     0.30 s |      98 Mcalls/s |                ~114× |
| Octave 9.4 `--eval`     |    65.03 s |    0.46 Mcalls/s |               ~0.53× |

### macOS — TBD

| Mode                    | Wall time | Throughput | Speedup vs `--opt 0` |
| ----------------------- | --------: | ---------: | -------------------: |
| `--opt 0` (interpreter) |           |            |                      |
| `--opt 1` (JS-JIT)      |           |            |                      |
| `--opt 2` (C-JIT)       |           |            |                      |
| MATLAB `-batch`         |           |            |                      |
| Octave `--eval`         |           |            |                      |

## Reading the table

- **`--opt 2` (C-JIT) is ~33% faster than `--opt 1` (JS-JIT)** and
  ~23% faster than MATLAB. The C compiler (`-O2 -march=native`) produces
  tighter scalar code than V8's TurboFan for this carried-dependency
  `sin` loop.
- **V8's TurboFan is remarkably good** — `--opt 1` matches MATLAB's
  performance on this benchmark. The larger structural win for the C-JIT
  surfaces on tensor workloads (see [tensor_ops_bench.md](tensor_ops_bench.md)).
- **Octave** is ~2× slower than numbl's _interpreter_. Octave's
  experimental JIT is off by default in 9.x.

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
