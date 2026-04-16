# tensor_ops_bench — numbl vs MATLAB on whole-tensor element-wise ops

Tensor element-wise + reduction benchmark. Five small kernels on real
Float64 vectors, 50 trials each.

Primary purpose: validate that `--opt 2` (C-JIT) achieves _parity_ with
`--opt 1` (JS-JIT) on tensor code — same results, comparable timings.
The two C baselines show where future optimization could take us.

Companion to [scalar_bench.md](scalar_bench.md) (scalar hot-loop, no
tensor allocations).

## Kernels

| Kernel                 | Pattern                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------- |
| `kernel_binary(x, y)`  | `r = x + y; r = r - 0.5.*x; r = r.*y + 3.0; r = r./(1+abs(y))`                          |
| `kernel_unary(x)`      | `u = exp(-x.*x); u = u.*cos(5.*x); u = u + sin(x+1).*sin(x+1); u = abs(u); u = tanh(u)` |
| `kernel_compare(x, y)` | `s = sum((x>0) .* (y<0.5))`                                                             |
| `kernel_reduce(x)`     | `s = sum(x)+mean(x)+max(x)+min(x)`                                                      |
| `kernel_chain(x, y)`   | `r = x.*y+0.5; r = exp(-r.*r); r = r.*x; s = sum(r)`                                    |

## How to run

```bash
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 0   # interpreter
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 1   # JS-JIT
npx tsx src/cli.ts run benchmarks/tensor_ops_bench.m --opt 2   # C-JIT
matlab -batch "run('benchmarks/tensor_ops_bench.m')"
(cd benchmarks && octave --no-gui --quiet --eval tensor_ops_bench)
bash benchmarks/tensor_ops_bench_compare.sh                    # all of the above
```

All runs produce the same check values (to FP rounding).

## Results (Linux, N=2 000 000, trials=50)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.1.7
- **MATLAB:** R2025b Update 5 · **Octave:** 9.4.0

| Mode                    |  Total | Binary |  Unary | Cmp+Red | Reduce |  Chain |
| ----------------------- | -----: | -----: | -----: | ------: | -----: | -----: |
| `--opt 0` (interpreter) | 5.05 s | 1.14 s | 2.04 s |  0.44 s | 0.27 s | 1.15 s |
| `--opt 1` (JS-JIT)      | 4.33 s | 0.83 s | 1.89 s |  0.50 s | 0.31 s | 0.80 s |
| `--opt 2` (C-JIT)       | 4.13 s | 0.99 s | 1.76 s |  0.36 s | 0.29 s | 0.73 s |
| MATLAB R2025b `-batch`  | 2.67 s | 0.27 s | 1.75 s |  0.16 s | 0.15 s | 0.34 s |
| Octave 9.4 `--eval`     | 8.39 s | 1.14 s | 4.93 s |  0.72 s | 0.29 s | 1.31 s |
| C baseline, per-op      | 2.55 s | 0.57 s | 1.16 s |  0.22 s | 0.12 s | 0.46 s |
| C baseline, fused       | 0.91 s | 0.10 s | 0.60 s |  0.06 s | 0.03 s | 0.12 s |

### macOS — TBD

| Mode                    | Total | Binary | Unary | Cmp+Red | Reduce | Chain |
| ----------------------- | ----: | -----: | ----: | ------: | -----: | ----: |
| `--opt 0` (interpreter) |       |        |       |         |        |       |
| `--opt 1` (JS-JIT)      |       |        |       |         |        |       |
| `--opt 2` (C-JIT)       |       |        |       |         |        |       |
| MATLAB `-batch`         |       |        |       |         |        |       |
| Octave `--eval`         |       |        |       |         |        |       |

## Reading the table

- **`--opt 1` and `--opt 2` are at parity.** Phase 2 of the C-JIT mirrors
  the JS-JIT's per-statement tensor model line-for-line (same helpers,
  same buffer-reuse rules, same bail-to-interpreter behavior) — so the
  two rows track closely, as designed. Any meaningful divergence would
  indicate a parity bug.
- **MATLAB is ~35% faster** than either numbl mode. Tighter dispatch
  and in-place destination reuse likely account for most of the gap.
- **The C baselines bracket future optimization headroom.** `per-op`
  (2.55 s) is the ceiling for the current per-statement architecture —
  any mode that calls libnumbl_ops per op can approach but not beat it.
  `fused` (0.91 s) is the ceiling if the C-JIT ever emits fused per-
  element loops instead of chained libnumbl_ops calls.
- **Octave** is ~2-3× slower than numbl's interpreter on binary/unary.
  Octave's experimental JIT is off by default in 9.x.

## Caveats

- **Domain-restricted unaries** (`sqrt`, `log`, `log2`, `log10`, `asin`,
  `acos`) bail to the interpreter — MATLAB promotes out-of-domain inputs
  to complex, libnumbl_ops returns NaN. Same behavior as JS-JIT.
- **Complex tensors** are not in the C-JIT path yet. Real-only.
- **Compiled `.node` cache** lives at `~/.cache/numbl/c-jit/<sha>.node`,
  keyed by source + compiler/node/numbl versions + `libnumbl_ops.a`
  contents. Rebuilding the addon invalidates cached modules.
