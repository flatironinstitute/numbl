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
npx tsx src/cli.ts run benchmarks/chunkie_helmholtz_starfish.m --opt 2
npx tsx src/cli.ts run benchmarks/chunkie_helmholtz_starfish.m --opt 2 --fuse
npx tsx src/cli.ts run benchmarks/chunkie_helmholtz_starfish.m --opt 2 --fuse --par
matlab -batch "maxNumCompThreads(1); run('benchmarks/chunkie_helmholtz_starfish.m')"
matlab -batch "maxNumCompThreads(8); run('benchmarks/chunkie_helmholtz_starfish.m')"
```

The script is a function file with a small-problem warmup call before
the timed run, so JIT specialization / C-JIT compile cost and the
native `fmm2d` / LAPACK bridge loads don't pollute the measurement.

## Results (Linux, hot run only, median of 3)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, numbl 0.1.7
- **MATLAB:** R2025b Update 5
- **Measured:** 2026-04-22 19:42 UTC

Times in seconds.

| Mode                           | Total | Discretize | Build | Solve | Interior |  Eval |
| ------------------------------ | ----: | ---------: | ----: | ----: | -------: | ----: |
| `--opt 1` (JS-JIT)             | 7.91s |      0.37s | 3.75s | 0.42s |    0.64s | 2.46s |
| `--opt 2` (C-JIT)              | 7.73s |      0.36s | 3.88s | 0.45s |    0.63s | 2.22s |
| `--opt 2 --fuse` (C-JIT)       | 7.43s |      0.38s | 3.58s | 0.41s |    0.59s | 2.32s |
| `--opt 2 --fuse --par` (C-JIT) | 8.37s |      0.37s | 4.00s | 0.40s |    0.81s | 2.38s |
| MATLAB R2025b (1 thread)       | 6.06s |      0.03s | 4.89s | 0.29s |    0.14s | 0.74s |
| MATLAB R2025b (8 threads)      | 3.12s |      0.03s | 1.77s | 0.54s |    0.15s | 0.65s |
