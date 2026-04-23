# koffi_overhead_bench — JS↔C crossover for a fused tensor op

Focused benchmark for answering: **when is it worth JIT-ing to C and
making a koffi call, vs. just running the loop in JS?**

Kernel: `y = exp(1 + sqrt(x))` (fused into a single C loop). `x` and `y`
are JS-owned `Float64Array`s passed as `double *` via koffi — the same
zero-copy contract numbl's C-JIT uses for tensor params, so there is no
input/output copy.

This deliberately sidesteps the numbl pipeline. The JS reference is a
plain `Float64Array` loop, which is what the JS-JIT lowers this kernel
to; the C reference is a direct koffi call into a standalone .so. That
lets us see the raw JS↔C overhead without the interpreter/JIT machinery
contributing noise.

## How to run

```bash
npx tsx benchmarks/koffi_overhead_bench.ts
npx tsx benchmarks/koffi_overhead_bench.ts --target-ms=500
npx tsx benchmarks/koffi_overhead_bench.ts --sizes=100,1000,10000
npx tsx benchmarks/koffi_overhead_bench.ts --no-fast-math
```

The first run compiles `koffi_overhead_bench.c` with
`-O2 -march=native -fopenmp-simd -fno-math-errno -ffast-math` and caches
the `.so` under `/tmp/numbl-koffi-overhead-bench/`.

## What it measures

For each `N`:

| Column  | Meaning                                                  |
| ------- | -------------------------------------------------------- |
| JS/call | `for i: y[i] = Math.exp(1 + Math.sqrt(x[i]))`            |
| C/call  | one `koffi` call into the fused C loop                   |
| speedup | `JS/call ÷ C/call`                                       |
| JS/elem | `JS/call ÷ N`                                            |
| C/elem  | `C/call ÷ N`                                             |
| C−noop  | `C/call − noop(n,x,y)` — time in C above the koffi floor |

Plus two N-independent probes run once at startup:

- `noop()` — empty C function, no args. Pure koffi call cost.
- `noop(n, x, y)` — empty C function with the same three-arg signature
  as the fused kernel. Call cost **including** argument marshalling.

## Results (Linux)

- **CPU:** 13th Gen Intel Core i7-1355U (12 threads)
- **OS:** Debian 13 (trixie), kernel 6.12.74
- **Toolchain:** Node v24.14.1, cc 14.2.0, koffi 2.15.2
- **Measured:** 2026-04-23

```
parity check (N=10000): max|JS-C| = 2.66e-15   (-ffast-math: small diffs expected)
koffi call overhead:   noop() = 35.9 ns, noop(n,x,y) = 366.1 ns

         N       JS/call        C/call   speedup     JS/elem      C/elem      C−noop
------------------------------------------------------------------------------------
         1       14.2 ns      406.1 ns    0.035x     14.2 ns    406.1 ns     39.9 ns
        10      110.0 ns      430.1 ns    0.256x     11.0 ns     43.0 ns     64.0 ns
       100       1.10 µs      546.3 ns     2.01x     11.0 ns      5.5 ns    180.2 ns
      1000      11.17 µs       1.57 µs     7.10x     11.2 ns      1.6 ns     1.21 µs
     10000     110.31 µs      11.16 µs     9.89x     11.0 ns      1.1 ns    10.79 µs
    100000       1.10 ms     119.24 µs     9.23x     11.0 ns      1.2 ns   118.87 µs
   1000000      10.22 ms       1.16 ms     8.85x     10.2 ns      1.2 ns     1.15 ms
  10000000      91.52 ms      12.11 ms     7.56x      9.2 ns      1.2 ns    12.11 ms

fit @ N=10000000:  C ≈ 366.1 ns + 1.2 ns·N,  JS ≈ 10.5 ns·N
break-even N (C-fused beats JS):  ~39
```

## Reading the table

- **Pure koffi call (no args)** ≈ **36 ns**. Argument marshalling for
  `(int64_t, double*, double*)` where the pointers are JS typed arrays
  adds ~330 ns, for a total per-call fixed cost of **~370 ns**.
- **Per-element cost** at large `N`:
  - JS loop: **~10 ns/elem** (one `sqrt` + one `exp` + a Float64Array
    load/store, V8-optimized).
  - C fused loop: **~1.2 ns/elem** — ~8× faster, driven by SIMD
    vectorization of `sqrt`/`exp` under `-ffast-math`.
- **Break-even** is at **N ≈ 40**. Below that, the koffi overhead
  dominates and the JS loop wins; above it, the SIMD-vectorized C kernel
  wins, approaching a ~9× asymptotic speedup.
- **Throughput ceiling** at N=10 M is ~12 ms for the C kernel — memory
  bandwidth becomes the bound, not FLOPs.

## Caveats

- **`-ffast-math`** enables SIMD vectorization of `sqrt`/`exp`. Without
  it, the C per-element cost jumps roughly 5× (run with `--no-fast-math`
  to see). numbl's C-JIT uses `-ffast-math`, so this matches production.
- **The noop-with-args overhead (~366 ns)** reflects koffi's cost to
  translate JS typed arrays into `double *` on every call. That cost is
  paid per call regardless of `N`, so it's the main driver of the
  crossover point.
- **JS per-element cost is V8-dependent**. The `Math.exp(1 + Math.sqrt(x[i]))`
  loop on `Float64Array` is friendly to V8's TurboFan; more complex JS
  (object allocations, non-typed arrays, non-monomorphic call sites)
  would shift the crossover down.
- **Single kernel only.** A program that can fuse _multiple_ statements
  into one koffi call amortizes the 370 ns fixed cost across more work,
  so numbl's fused C-JIT wins at smaller `N` than 40 for longer
  pipelines.
