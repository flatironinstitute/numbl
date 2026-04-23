# chunkie_helmholtz_starfish — numbl vs MATLAB on a Helmholtz BIE

Exterior Helmholtz scattering from a starfish-shaped obstacle via the
combined-field integral equation (CFIE), built on top of the
[chunkie](https://github.com/flatironinstitute/chunkie) boundary-integral
toolbox. Based on the first example in the chunkie documentation, minus
the plotting.

## What the benchmark does

Per call to `run_bench(narms, amp, kvec, grid_n)`:

1. **discretize** — `chunkerfunc` builds a starfish chunker with
   `maxchunklen = 4/|kvec|`.
2. **build_matrix** — construct the `(1, −iκ)` combined-field helm
   kernel, form the CFIE system matrix via `chunkermat`, add `0.5·I`.
3. **solve** — apply the plane-wave RHS, solve via `gmres` to `1e-13`.
4. **interior** — classify target grid points (`chunkerinterior`), keep
   the exterior ones.
5. **eval** — evaluate the scattered field at the exterior targets via
   `chunkerkerneval` (FMM).

Hot-run parameters: `narms=5`, `amp=0.5`, `kvec=20·[1;−1.5]`
(so `zk ≈ 36.06`), grid of `200×200` targets → 172 chunks, `k=16`
Gauss-Legendre nodes per chunk, a `2752×2752` complex system matrix,
GMRES converges in 46 iterations, 36 110 exterior targets. All
configurations produce matching check values to FP rounding.

## How to run

```bash
npx tsx src/cli.ts run benchmarks/chunkie_helmholtz_starfish.m --opt 1
npx tsx src/cli.ts run benchmarks/chunkie_helmholtz_starfish.m --opt e1
npx tsx src/cli.ts run benchmarks/chunkie_helmholtz_starfish.m --opt e1 --par
matlab -batch "maxNumCompThreads(1); run('benchmarks/chunkie_helmholtz_starfish.m')"
matlab -batch "maxNumCompThreads(8); run('benchmarks/chunkie_helmholtz_starfish.m')"
```

The script is a function file with a small-problem warmup call before
the timed run, so JIT specialization / C kernel compile cost and the
native `fmm2d` / LAPACK bridge loads don't pollute the measurement.

## Results (Linux, hot run only, median of 3)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.1.7
- **MATLAB:** R2025b Update 5
- **Measured (numbl):** 2026-04-23 11:11 UTC
- **Measured (MATLAB):** 2026-04-22 19:42 UTC

Times in seconds.

| Mode                      | Total | Discretize | Build | Solve | Interior |  Eval |
| ------------------------- | ----: | ---------: | ----: | ----: | -------: | ----: |
| `--opt 1` (JS-JIT)        | 6.72s |      0.35s | 2.93s | 0.41s |    0.59s | 2.14s |
| `--opt e1`                | 6.53s |      0.35s | 2.99s | 0.39s |    0.59s | 2.22s |
| `--opt e1 --par`          | 7.19s |      0.37s | 3.05s | 0.40s |    0.62s | 2.40s |
| MATLAB R2025b (1 thread)  | 6.06s |      0.03s | 4.89s | 0.29s |    0.14s | 0.74s |
| MATLAB R2025b (8 threads) | 3.12s |      0.03s | 1.77s | 0.54s |    0.15s | 0.65s |

`--opt e1` is the fastest numbl mode on this workload, narrowly beating
`--opt 1`. It keeps the JS-JIT outer and splices in compiled C kernels
only where they help (fusible tensor chains), capturing small inner-
kernel wins without paying whole-function compile cost. `--par` doesn't
help: most of this workload's compute is inside `fmm2d` / LAPACK native
bridges and GMRES dot products, not in the fusible chains that `--par`
could thread.
