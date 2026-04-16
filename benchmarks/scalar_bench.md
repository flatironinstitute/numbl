# scalar_bench — numbl vs MATLAB on a scalar hot loop

A small compute-bound benchmark that stresses the scalar fast-path in
numbl's JIT. Used to compare numbl's three optimization levels
(`--opt 0`, `--opt 1`, `--opt 2`) against MATLAB and GNU Octave on the
same machine.

## What the benchmark does

`run_bench(N, M)` evaluates

```
  sum_{i=1..N} sum_{k=1..M}  sin(x_i * k) / k²     where  x_i = i · 0.001
```

with all arithmetic held in scalar registers (a carried-dependency
accumulator plus one `sin` and one division per innermost iteration).
Default sizes are `N = 60000`, `M = 500` → **30M** sin+div ops per run.

The whole inner loop stays inside the C-JIT MVP whitelist — no tensor
ops, no complex, no struct, no `ibcall` — so every step runs on the
scalar fast path when `--opt 2` is active.

## How to run

Single config (matches the command printed by the `C-JIT:` stderr
banner under `--opt 2`):

```bash
npx tsx src/cli.ts run benchmarks/scalar_bench.m --opt 0
npx tsx src/cli.ts run benchmarks/scalar_bench.m --opt 1
npx tsx src/cli.ts run benchmarks/scalar_bench.m --opt 2
matlab -batch "run('benchmarks/scalar_bench.m')"
(cd benchmarks && octave --no-gui --quiet --eval scalar_bench)
```

The benchmark is written as a function file (`function scalar_bench()`)
rather than a script so local function definitions at the bottom are
legal under Octave too (Octave 9 doesn't accept `function` following
top-level script statements). numbl and MATLAB both auto-invoke the
first function when a function file is run; Octave needs the explicit
`--eval scalar_bench` shown above.

One-shot cross-runtime comparison (runs all five configs and extracts
`elapsed` / `throughput` from each):

```bash
bash benchmarks/scalar_bench_compare.sh
# override the binaries if needed:
MATLAB=/opt/MATLAB/R2025b/bin/matlab \
  OCTAVE=/usr/local/bin/octave \
  bash benchmarks/scalar_bench_compare.sh
```

All runs produce the same `result = 2070.336478567545` (to
floating-point rounding).

## Notes on the timing methodology

- **Warm-up.** A single `run_bench(100, 10)` call before `tic` lands
  the JIT specialization in the in-memory cache; subsequent calls with
  different argument values hit the same cache entry (numeric `exact`
  field is stripped from param types on entry, see
  [src/numbl-core/interpreter/jit/index.ts](../src/numbl-core/interpreter/jit/index.ts)).
  This keeps ~50ms of `cc` invocation out of the timed section.
- **Disk cache.** Compiled `.node` modules live in
  `~/.cache/numbl/c-jit/<sha256>.node`, keyed by the C source plus
  compiler/node/numbl/platform versions. Second and later runs on the
  same machine only pay the N-API `createRequire` cost (~1ms).
- **Compile flags.** Printed once to stderr on the first `--opt 2` run
  in a process (the `C-JIT: cc ...` banner). Default set on Linux is
  `-O2 -fPIC -shared -std=c11 -march=native`.

## Typical results

### Linux

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0 (Debian), numbl 0.1.7
- **MATLAB:** R2025b Update 5
- **Octave:** 9.4.0
- **Size:** N=60000, M=500 (30M sin+div)

| Mode                    |  Wall time |       Throughput | Speedup vs `--opt 0` |
| ----------------------- | ---------: | ---------------: | -------------------: |
| `--opt 0` (interpreter) |    29.91 s |    1.00 Mcalls/s |                   1× |
| `--opt 1` (JS-JIT)      |     0.28 s |     108 Mcalls/s |                ~107× |
| `--opt 2` (C-JIT)       | **0.22 s** | **136 Mcalls/s** |            **~136×** |
| MATLAB R2025b `-batch`  |     0.31 s |      96 Mcalls/s |                 ~96× |
| Octave 9.4 `--eval`     |    64.93 s |    0.46 Mcalls/s |               ~0.46× |

### macOS — TBD

- **CPU:**
- **OS:**
- **Toolchain:** Node vX, cc (Apple clang), numbl 0.1.7
- **MATLAB:**
- **Octave:**
- **Size:** N=60000, M=500 (30M sin+div)

| Mode                    | Wall time | Throughput | Speedup vs `--opt 0` |
| ----------------------- | --------: | ---------: | -------------------: |
| `--opt 0` (interpreter) |           |            |                      |
| `--opt 1` (JS-JIT)      |           |            |                      |
| `--opt 2` (C-JIT)       |           |            |                      |
| MATLAB `-batch`         |           |            |                      |
| Octave `--eval`         |           |            |                      |

## Caveats

- The inner loop is a carried-dependency accumulator, which prevents
  SIMD vectorization. `-march=native` mostly buys tighter scalar code,
  not auto-vectorization. A non-serial variant (e.g. four independent
  partial sums) would give the C compiler more headroom vs. V8.
- V8's TurboFan is remarkably good on scalar `Math.sin` loops — hence
  `--opt 1` and `--opt 2` come in close. The larger structural win for
  the C-JIT will surface in later phases that cover tensor/struct IR
  beyond what V8 specializes well.
- Octave's timing is roughly 2× slower than numbl's _interpreter_ on
  this workload. Octave's experimental JIT is off by default in 9.x;
  the baseline interpreter pays a higher per-operation cost than
  numbl's because of overhead in its dispatch/typing layer. The
  comparison that matters for everyday Octave users is MATLAB ≈ numbl
  `--opt 1` ≈ ~100 Mcalls/s, Octave ≈ ~0.5 Mcalls/s.
