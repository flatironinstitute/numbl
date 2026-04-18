# Tensor Operations Layer

A pointer-based, op-code-dispatched tensor math layer with side-by-side
C (native) and TypeScript (fallback) implementations. Callers allocate
input AND output buffers; the layer writes into them without ever copying
or allocating. Native is chosen when the addon is loaded; the pure-TS
mirror is used transparently in the browser, in unit tests that don't
build the addon, or on platforms where the native build fails.

## Motivation

Before this work, tensor math in numbl was split inconsistently across
three places:

- Element-wise binary/unary ops lived in [native/elemwise.cpp](../native/elemwise.cpp)
  and [native/unary_elemwise.cpp](../native/unary_elemwise.cpp) with integer op-code
  `switch` dispatch — but their JS fallbacks were scattered across builtin
  files (e.g. [src/numbl-core/jit/js/jitHelpersTensor.ts](../src/numbl-core/jit/js/jitHelpersTensor.ts))
  with no shared interface.
- LAPACK ops lived in `native/lapack_*.cpp` with one N-API export per
  function; their JS fallback was the separate
  [src/ts-lapack/](../src/ts-lapack/) library, dispatched via
  [src/numbl-core/native/bridge-resolve.ts](../src/numbl-core/native/bridge-resolve.ts).
- Every native call allocated a fresh `Float64Array` for the output.
  Inputs were zero-copy (`Napi::Float64Array::Data()` is a direct pointer
  into the V8 ArrayBuffer), but outputs were not.

Goal: one uniform layer where

1. Every tensor op goes through one dispatch table.
2. Each op has **two parity implementations** with identical signatures —
   native C and pure TypeScript. Native is selected when available.
3. Operations work on **caller-allocated input AND output pointers** —
   no copies, no per-call allocation in the math path itself.
4. Ops within a category dispatch by **integer op-code** (extending the
   existing `elemwise.cpp` pattern).
5. The future C JIT can `dlopen` the same C library and call ops
   directly, bypassing JS entirely.

## Architecture

### Why N-API, not koffi

- `Napi::Float64Array::Data()` is already zero-copy on inputs. Per-call
  overhead is tens of nanoseconds.
- koffi has higher per-call overhead (dynamic marshalling) and cannot
  pass `ArrayBuffer` storage as a true pointer without copying.
- koffi is already an optional dep used only for loading user-provided
  `.so`/`.dll` files — that role stays unchanged.

### Layered layout

```
native/ops/                       # Pure C library (extern "C" ABI)
  numbl_ops.h                     # Public ABI: op-code enums, function decls
  numbl_ops.c                     # Error codes, op-code dump (drift detection)
  real_binary_elemwise.c          # ADD, SUB, MUL, DIV (real + scalar variants)
  complex_binary_elemwise.c       # same, split re/im storage
  real_unary_elemwise.c           # 20 ops: exp, log, sqrt, sin, cos, ..., sign
  complex_unary_elemwise.c        # 19 ops (ABS excluded), uses C99 <complex.h>
  comparison.c                    # EQ, NE, LT, LE, GT, GE (real + complex)
  reduce.c                        # flat: SUM, PROD, MAX, MIN, ANY, ALL, MEAN

native/ops_napi.cpp               # Thin N-API wrappers (no business logic)
native/numbl_addon.cpp            # Registers all tensorOp* exports

src/numbl-core/ops/               # TS mirror (identical signatures)
  opCodes.ts                      # OpRealBin, OpComplexBin, OpUnary, OpCmp, OpReduce
  dispatch.ts                     # tensorOps: routes native-or-TS
  index.ts                        # public re-exports
  realBinaryElemwise.ts           # tsRealBinaryElemwise, tsRealScalarBinaryElemwise
  complexBinaryElemwise.ts        # tsComplexBinaryElemwise, tsComplexScalarBinaryElemwise
  realUnaryElemwise.ts            # tsRealUnaryElemwise
  complexUnaryElemwise.ts         # tsComplexUnaryElemwise, tsComplexAbs
  comparison.ts                   # tsRealComparison, tsRealScalarComparison, etc.
  reduce.ts                       # tsRealFlatReduce, tsComplexFlatReduce
```

The dispatcher ([src/numbl-core/ops/dispatch.ts](../src/numbl-core/ops/dispatch.ts))
is a thin pass-through:

```ts
export const tensorOps = {
  realBinaryElemwise(op, n, a, b, out): void {
    const bridge = getLapackBridge();
    if (bridge?.tensorOpRealBinary) bridge.tensorOpRealBinary(op, n, a, b, out);
    else tsRealBinaryElemwise(op, n, a, b, out);
  },
  // ... one entry per category/variant, same shape
};
```

### Caller-allocated output convention

All ops are `void` — caller owns input AND output memory. Only the
immediate caller (a builtin, a JIT fast path, or a helper) allocates the
output once via `new Float64Array(n)`. The C core never allocates; the
N-API wrapper forwards `out.Data()` straight through.

Implications:

- Complex outputs need pre-allocated `outRe` AND `outIm`. The "is
  purely real?" check that used to live in `native/elemwise.cpp` now
  runs in the caller (see `finalizeSplit` in `jitHelpersTensor.ts`).
- Multi-output ops (QR, SVD, ...) need shape helpers before they can
  be migrated; not done yet.
- LAPACK workspace memory stays internal to the C layer for now.

### Op-code numbering and drift detection

Per-category enums (`numbl_real_bin_op_t`, `numbl_unary_op_t`,
`numbl_cmp_op_t`, `numbl_reduce_op_t`, etc.). Real and complex binary
op codes are **intentionally aligned** (`ADD=0, SUB=1, MUL=2, DIV=3`)
so a single enum value works for both.

A drift-detection test ([src/**tests**/op-codes-sync.test.ts](../src/__tests__/op-codes-sync.test.ts))
calls `numbl_dump_op_codes()` in the native addon and compares the
result to `src/numbl-core/ops/opCodes.ts`. The test skips automatically
when the addon isn't built.

### Error handling

C entry points return negative codes (`NUMBL_ERR_BAD_OP`,
`NUMBL_ERR_NULL_PTR`). `numbl_strerror(int)` in C provides matching
messages. The N-API wrapper translates to `Napi::Error`; the TS fallback
throws `Error` with the same string.

### Precision

Float64 only. `USE_FLOAT32` code paths keep using the legacy per-op
closures for now.

## Categories Implemented

| Category                             | Variants                                    | Op count |
| ------------------------------------ | ------------------------------------------- | -------- |
| real_binary_elemwise                 | tensor-tensor, scalar-tensor                | 4        |
| complex_binary_elemwise              | tensor-tensor, scalar-tensor (split re/im)  | 4        |
| real_unary_elemwise                  | tensor                                      | 20       |
| complex_unary_elemwise + complex_abs | tensor (ABS returns real)                   | 19 + 1   |
| comparison                           | real/complex × tensor-tensor, scalar-tensor | 6        |
| flat_reduce                          | real: 7 ops; complex: SUM, PROD, ANY, ALL   | 7 + 4    |

## Wiring into numbl

Three consumer paths now hit the tensor-ops layer:

**1. Interpreter helpers** ([src/numbl-core/helpers/arithmetic.ts](../src/numbl-core/helpers/arithmetic.ts)):

- `tryNativeElemwiseReal`, `tryNativeElemwiseScalar`,
  `tryNativeComplexScalarTensor`, `tensorElemwiseComplex` —
  binary elemwise through `tensorOps.realBinaryElemwise` /
  `complexBinaryElemwise` / `realScalarBinaryElemwise` /
  `complexScalarBinaryElemwise`.
- `comparisonOp`, `complexComparisonOp` — all six op codes
  routed.

**2. Builtin helpers** ([src/numbl-core/interpreter/builtins/types.ts](../src/numbl-core/interpreter/builtins/types.ts)):

- `applyUnaryElemwise` — real-tensor and complex-tensor fast paths
  through `tensorOps.realUnaryElemwise` / `complexUnaryElemwise`.
- `applyUnaryElemwiseMaybeComplex` — same, with NaN → complex
  fallback preserved.
- `applyUnaryRealResult` — `abs` fast path via
  `realUnaryElemwise(ABS)` / `complexAbs`.

**3. JIT helpers** ([src/numbl-core/jit/js/jitHelpersTensor.ts](../src/numbl-core/jit/js/jitHelpersTensor.ts)):

- `tAdd`, `tSub`, `tMul`, `tDiv` (`fastBinaryOp`) — real AND complex
  tensor-tensor, tensor-scalar, and tensor-complex-scalar paths all
  route through tensorOps. Falls back to closure path only for
  non-Float64 or complex-imag Float32 edges.
- `tEq`, `tNeq`, `tLt`, `tLe`, `tGt`, `tGe` (`fastCompareOp`) — same
  treatment for comparisons.
- `tensorUnary` — op-code map translates `Math.exp` / `Math.sqrt` /
  etc. to OpUnary codes and dispatches through `realUnaryElemwise`.

**4. Reduction helpers** ([src/numbl-core/helpers/reduction-helpers.ts](../src/numbl-core/helpers/reduction-helpers.ts),
[src/numbl-core/helpers/reduction/min-max.ts](../src/numbl-core/helpers/reduction/min-max.ts)):

- `scanLogical` — Float64 path via `tensorOps.realFlatReduce(ANY|ALL)`
  or `complexFlatReduce` for split-storage imag.
- `accumKernel` — takes optional op-code; `sumKernel`, `prodKernel`
  (inline), and `meanKernel` pass `OpReduce.SUM` / `PROD` so
  `reduceAll` on Float64 tensors runs through `realFlatReduce`.
- `minMaxImpl` — `nargout === 1` real Float64 vector scan goes through
  `realFlatReduce(MIN|MAX)`. Two-output (value + index) case still
  uses the existing scan.

Axis-wise reductions (`dimReduce`, `sliceDimReduce`) are **not**
migrated — they keep the closure-based kernels because the axis
iteration logic is orthogonal to flat reduction. Future work.

## Correctness

- All 1436 unit + integration tests pass.
- Check values in [examples/tensor_ops_benchmark.m](../examples/tensor_ops_benchmark.m) match
  MATLAB R2025b to 10 digits (10-digit prints across sum(r), sum(u),
  acc, count, sum(real(w)), sum(imag(w))). The benchmark uses
  deterministic inputs (sin/cos of linspace) rather than RNG so it
  doubles as a numerical-correctness regression test.
- One MATLAB-semantics bug fixed along the way: `max`/`min` flat
  reductions now omit NaN by default (they previously propagated).

## Performance Investigation

On examples/tensor_ops_benchmark.m (N=2M, 50 trials, 5 op categories):

| Stage                                          | Total      | vs MATLAB (8.0 s) |
| ---------------------------------------------- | ---------- | ----------------- |
| Before any migration                           | 53.8 s     | 6.7×              |
| JIT helpers routed (real binary/unary/compare) | 25.0 s     | 3.1×              |
| Full coverage (complex, reductions, min/max)   | **15.2 s** | **1.9×**          |

### How the investigation unfolded

1. **Pure-C baseline** (`/tmp/bench_c.c`): two variants of the real
   binary elemwise loop, compiled with `-O3 -march=native`:
   - NAIVE (one C loop per op, intermediate buffers): 705 ms
   - FUSED (all 4 ops combined in one loop): 105 ms
   - MATLAB: 440 ms — sits **between** NAIVE and FUSED, showing
     MATLAB's tempo JIT does _partial_ loop fusion.
   - Both pure-C variants are memory-bandwidth-limited:
     - NAIVE moves ~16 GB total at ~20 GB/s DDR4 → ~0.8 s.
     - FUSED moves ~2.4 GB → ~120 ms.

   **Conclusion: the math isn't the problem. Pure-C-NAIVE (doing
   exactly what numbl does) is only 1.6× slower than MATLAB.**

2. **JS micro-bench** (bench_js.ts at the repo root — not checked in):
   - A (addon direct, preallocated out): 1844 µs/op — matches raw
     kernel cost.
   - B (addon direct, `new Float64Array(N)` each call): 3770 µs/op.
   - **alloc cost = B − A ≈ 1.9 ms per 16 MB buffer** (50% of per-op
     time).
   - C (via `tensorOps` dispatcher): same as A — **dispatcher is
     free**.
   - F (via full `mAdd(tensor, tensor)` path): same as B — **`mAdd`
     overhead is essentially zero**.

3. **Missing time traced to the JIT.** The .m benchmark took 7.3 s
   for the real binary block, but 400 tensor ops × 3.5 ms = 1.4 s
   worst-case. The 5.9 s gap lived in the JIT-generated code: the
   original `tAdd`/`tSub`/`tMul`/`tDiv` called `tensorBinaryOp(a, b,
(x,y)=>x+y, cAdd)` — **a JS closure invoked per element, 400
   million closure calls for the real binary block alone**.

4. **Fix: JIT fast paths.** Added `fastBinaryOp`, `fastCompareOp`,
   and an op-code map in `tensorUnary` that route Float64-real and
   Float64-complex cases straight to tensorOps. Closure path remains
   the fallback for exotic cases (float32, size mismatch, complex
   imag as non-Float64).

5. **Follow-up (reductions + complex):** `accumKernel` gained an
   opCode parameter so sum/prod/mean reduce-all run through
   `realFlatReduce`; `minMaxImpl` added a `realFlatReduce(MIN|MAX)`
   fast path for nargout=1; `applyUnaryElemwise` complex branch and
   `applyUnaryRealResult` (abs) now route through
   `complexUnaryElemwise` / `complexAbs`.

### Remaining gap vs MATLAB (final ~2×)

Per-category at this point:

| Category    | numbl      | MATLAB    | Ratio    |
| ----------- | ---------- | --------- | -------- |
| Real binary | 1.2 s      | 0.44 s    | 2.7×     |
| Real unary  | 5.4 s      | 3.4 s     | 1.6×     |
| Comparisons | 1.8 s      | 0.97 s    | 1.9×     |
| Reductions  | 0.8 s      | 0.35 s    | 2.3×     |
| Complex     | 6.0 s      | 2.9 s     | 2.1×     |
| **Total**   | **15.2 s** | **8.0 s** | **1.9×** |

The remaining gap is now dominated by:

- **Per-op allocation** — every tensor op still does
  `new Float64Array(N)` for its output. At N=2M that's ~1.9 ms per
  call. Pooling experiments (see below) confirm pool-based
  allocation eliminates the cost completely:

  | Strategy                     | Per op  |
  | ---------------------------- | ------- |
  | new Float64Array each call   | 3770 µs |
  | LIFO free-list pool          | 1815 µs |
  | Shared ArrayBuffer arena     | 1957 µs |
  | Single reused buffer (ideal) | 1850 µs |

- **No loop fusion.** MATLAB's tempo JIT fuses consecutive elemwise
  ops into a single pass over the data (one memory-bandwidth traversal
  per group of ops, not per op). numbl's JIT still generates one
  tensorOps call per source-level operator.

## What was learned

1. **Zero-copy through N-API is trivial.** `Napi::Float64Array::Data()`
   returns a stable pointer for the duration of a synchronous call.
   Caller-allocated output makes the whole path copy-free.

2. **Dispatcher overhead is negligible.** Adding
   `tensorOps.realBinaryElemwise` on top of the raw addon call costs
   nothing measurable (~15 µs/op out of ~1850 µs — noise). The
   pattern scales cleanly to any number of categories.

3. **Op-code switch dispatch in C is also essentially free.** The
   per-op branch lives outside the hot loop, so CPU branch prediction
   and code-cache hits make it zero-cost in practice.

4. **The expensive parts of "calling into native from JS" at large
   N are, in order:**
   1. Memory bandwidth of the kernel itself (1.8 ms per 16 MB op,
      bounded by DDR).
   2. `new Float64Array(N)` allocation (~1.9 ms per 16 MB) —
      V8 zero-inits the buffer, pressures large-object heap, and
      drives GC.
   3. Everything else combined (N-API marshalling, dispatcher,
      JS wrapper objects): single-digit percent.

5. **The interpreter-vs-JIT split matters a lot.** Before the JIT
   fast-path migration, 78% of the real-binary-elemwise benchmark
   time was inside the JIT helper calling JS closures per element
   instead of going through native. Once the JIT routes through
   `tensorOps`, the JIT'd and raw-JS paths perform identically.

6. **Semantics differ subtly between environments and need tests.**
   MATLAB `max`/`min` default to omitnan; C99 `max`/`min` and our
   initial implementation propagated NaN. The regression-test strategy
   of "run the same .m file in numbl and MATLAB and assert bitwise-
   identical check values" caught this and gives confidence for
   future changes.

7. **Loop fusion is the irreducible remaining gap with MATLAB.**
   Once allocation is out of the way, memory bandwidth is the bound,
   and you can only shrink that by doing more arithmetic per memory
   pass — which means fusing consecutive elemwise ops at the JIT
   level. That's a bigger structural change than the tensor-ops
   layer itself.

## File reference (quick map)

### Native (C + C++)

- [binding.gyp](../binding.gyp) — build rule for all ops sources
- [native/ops/numbl_ops.h](../native/ops/numbl_ops.h) — public C ABI
- [native/ops/numbl_ops.c](../native/ops/numbl_ops.c) — error strings + op-code dump
- [native/ops/real_binary_elemwise.c](../native/ops/real_binary_elemwise.c)
- [native/ops/complex_binary_elemwise.c](../native/ops/complex_binary_elemwise.c)
- [native/ops/real_unary_elemwise.c](../native/ops/real_unary_elemwise.c)
- [native/ops/complex_unary_elemwise.c](../native/ops/complex_unary_elemwise.c)
- [native/ops/comparison.c](../native/ops/comparison.c)
- [native/ops/reduce.c](../native/ops/reduce.c)
- [native/ops_napi.cpp](../native/ops_napi.cpp) — all N-API shims
- [native/numbl_addon.cpp](../native/numbl_addon.cpp) — registers exports
  (ADDON_VERSION = 7 at time of writing)

### TypeScript

- [src/numbl-core/ops/](../src/numbl-core/ops/) — TS mirror (see layout above)
- [src/numbl-core/native/lapack-bridge.ts](../src/numbl-core/native/lapack-bridge.ts)
  — `LapackBridge` interface includes `tensorOp*` entries (optional
  methods so missing ones fall back to TS)

### Tests and examples

- [src/**tests**/op-codes-sync.test.ts](../src/__tests__/op-codes-sync.test.ts)
  — drift detection
- [examples/tensor_ops_benchmark.m](../examples/tensor_ops_benchmark.m)
  — runs in both numbl and MATLAB; check values match bitwise

### Investigation scratch (not committed)

- `/tmp/bench_c.c` — pure-C baseline (NAIVE and FUSED variants)
- `bench_js.ts` in repo root — JS micro-bench (addon-direct vs
  tensorOps-dispatch vs pure-JS vs pooling strategies)

## Possible next steps

Roughly in order of effort-to-payoff ratio:

### 1. In-place ops in the JIT (biggest single per-op win)

When the JIT sees `r = f(r, y)`, it can write the result back into
`r`'s buffer instead of allocating a new one. Most C kernels are
already aliasing-safe (the output index `i` is written after the
reads at the same index). This eliminates allocation AND keeps the
buffer hot in cache. Likely cuts 1–2 seconds off the benchmark.

Concrete: in `jitCodegen.ts`, detect the pattern
`<var> = $h.tAdd(<var>, ...)` and emit a new helper
`$h.tAddInPlace(var, other)` that allocates-or-reuses. Needs one
variant per `tAdd`/`tSub`/`tMul`/`tDiv`/`tUnary*`.

### 2. Float64Array pool

Size-keyed LIFO pool behind the JIT fast paths. Needs a release
mechanism — two options:

- **Ref-counted via existing `_rc`:** hook into the JIT's variable
  reassignment path to decrement `_rc` and release when it hits
  zero. Easier than option (1) because no codegen changes are
  needed — the JIT-generated code looks the same, only the helper
  behavior changes.
- **JIT-emitted `poolRelease(oldBuf)` at reassignment.** More
  explicit but requires codegen tracking of tensor variable
  liveness.

Best case for the pool on our benchmark: ~1.9 ms saved per tensor
op × ~1500 ops = ~3 s off the total. Would put numbl very close to
MATLAB.

### 3. Axis-wise reductions through tensorOps

Currently only flat reductions are migrated. For `sum(A, 2)`,
`mean(A, 1)` etc., `dimReduce` and `sliceDimReduce` still use
closure kernels. Options:

- Add a "strided reduction" primitive to the C layer:
  `numbl_real_strided_reduce(op, fiber_len, stride, a, out_scalar)`.
  Call once per fiber from JS.
- Add a full "axis-reduction" primitive that takes shape + dim and
  does the whole multi-fiber walk in C. Less JS-side work per call
  but a more complex C API.

The first option is a smaller API change and gets most of the
benefit.

### 4. Extract `libnumbl_ops.{so,dylib,dll}` as a separate shared lib

Currently the C files are compiled directly into `numbl_addon.node`.
When the C JIT lands, it needs `dlopen`-able ops. The change is
mechanical: add a second binding.gyp target and link the addon
against it.

### 5. Loop fusion in the JIT (largest structural win, largest effort)

Scan consecutive JIT-IR elemwise ops and fuse them into a single
C-emitted pass. For example, `r = x + y; r = r .* z;` becomes one
pass that computes `r[i] = (x[i] + y[i]) * z[i]`. This requires:

- Building a small elemwise-op IR (we already have JIT IR — just
  needs to expose op signatures in a fusion-friendly form).
- Emitting either fused JS (V8 can vectorize some of these) or
  fused C (requires the C JIT path to be alive).

This is what MATLAB's tempo does, and it's the structural reason
MATLAB is still ~2× ahead after the work so far.

### 6. Matmul and linear algebra categories

Currently handled by the legacy `LapackBridge` methods (matmul, inv,
qr, svd, eig, chol, linsolve, lu, qz, fft, gmres). Migrating them
to the tensor-ops layout buys less than elemwise — LAPACK does its
own heavy lifting and the per-call overhead is a smaller fraction
of the kernel time — but it's the right step for uniformity and
for the future C JIT.

### 7. Further categories (opportunistic)

- Scalar-output-from-complex ops beyond abs: `angle`, `real`, `imag`,
  `conj` (at least the elemwise ones).
- `xor`, `and`, `or` logical ops on tensors (currently not in the
  layer).
- `pow` binary (only `mElemPow` uses a closure today).
