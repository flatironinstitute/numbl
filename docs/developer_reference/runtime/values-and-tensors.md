# Runtime Values and Tensors

## `RuntimeValue`

The runtime universe. A union of primitives plus refcounted classes:

- `number` — JavaScript number (real scalar). Not refcounted.
- `boolean` — logical scalar. Not refcounted.
- `string` — MATLAB string (distinct from char). Not refcounted.
- `RuntimeChar` — MATLAB char array.
- `RuntimeTensor` — real or complex, any rank, any shape.
- `RuntimeComplexNumber` — complex scalar.
- `RuntimeCell` — heterogeneous cell array.
- `RuntimeStruct` / struct array — record values.
- `RuntimeClassInstance` — `classdef` instance.
- `RuntimeFunction` / function handle — callable reference.
- Sparse matrix, dictionary, and a few others.

Every container kind is a class extending `Refcounted` (`runtime/refcount.ts`). Each instance carries a `_rc` field (starts at 0) and `incref()` / `decref(rt)` methods enforced by a strict slot API; see [refcount.md](refcount.md) for the lifecycle.

Type-guard predicates follow the pattern `isRuntimeTensor(v)`, `isRuntimeChar(v)`, etc. — they check `value.kind === "..."` and continue to work on the class instances. Use these instead of examining shape fields directly or using `instanceof`.

## `RTV` constructors

The `RTV` namespace exposes construction helpers for runtime values: `RTV.num`, `RTV.tensor(data, shape)`, `RTV.complex(re, im)`, `RTV.logical`, `RTV.char`, and so on. Prefer them to building objects by hand — they keep shape, precision, and flag invariants consistent.

Extract a scalar with `toNumber(v)`.

## Tensor representation

A `RuntimeTensor` holds:

- `data` — a typed array for the real part.
- `imag` — an optional typed array for the imaginary part. Absent means the tensor is real.
- `shape` — integer dimensions, column-major (Fortran) order.
- Flags — e.g., `_isLogical` for boolean arrays from comparisons.

**Column-major layout.** Element `(i, j)` of an `[m, n]` matrix is at flat index `j * m + i`. All tensor code assumes this; any new operation must preserve it.

**Copy-on-write.** Assigning a tensor variable or passing it as an argument shares the same underlying buffer. Before an indexed store mutates a tensor, the runtime decides whether to clone by running an on-demand reachability sweep from a small set of roots (active env chain, caller-env stack `rt._envStack`, globals, persistents) — see [`runtime/aliasing.ts`](../../../src/numbl-core/runtime/aliasing.ts). The LHS slot is excluded from the sweep, so a tensor that's only reachable through the slot being overwritten is treated as uniquely owned and mutated in place. A hard visit budget caps worst-case cost; on exhaustion the sweep returns true (conservative copy). The refcount system runs alongside the sweep but does not drive COW; see [refcount.md](refcount.md).

**No buffer sharing across wrappers.** `reshape` and `squeeze` previously returned a new `RuntimeTensor` whose `data` aliased the source's buffer (zero-copy). With refcount-driven pool reclamation in place, two wrappers cannot co-own one buffer — the second `_destroy` would release an already-pooled buffer. These ops now copy the data instead. Trade: one extra `Float64Array` copy per reshape/squeeze, in exchange for a single owner per buffer.

## Tensor ops

Tensor-level kernels (element-wise binary and unary, reductions, comparisons, Bessel, etc.) live in the ops layer. Each op has an integer op-code. The ops layer dispatches each op to the native addon when available or to a pure-JS fallback otherwise. The op-code list is shared between TypeScript and the native C; a unit test verifies they stay in sync.

Complex tensors follow the same dispatch with separate real/complex op-code sets. A complex-tensor operation that has no complex-aware kernel either falls back to a JS complex path or is handled by the interpreter.

## What the JIT does with tensors

For tensor-valued operations, JIT codegen either:

- calls a helper on the `$h` object that wraps the ops-layer dispatch, or
- when the operation is in the shared fast subset with a known shape and real data, emits an inline loop (possibly fused with neighboring ops — see [../jit/fusion.md](../jit/fusion.md)).

Anything outside that subset (unusual shapes, complex, sparse, mixed types) goes through the ops-layer helper.
