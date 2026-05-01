# Runtime Values and Tensors

## `RuntimeValue`

The runtime universe. A discriminated union of all values a numbl program can hold:

- `number` — JavaScript number (real scalar).
- `boolean` — logical scalar.
- `string` — MATLAB string (distinct from char).
- `RuntimeChar` — MATLAB char array.
- `RuntimeTensor` — real or complex, any rank, any shape.
- `RuntimeComplexNumber` — complex scalar.
- `RuntimeCell` — heterogeneous cell array.
- `RuntimeStruct` / struct array — record values.
- `RuntimeClassInstance` — `classdef` instance.
- `RuntimeFunction` / function handle — callable reference.
- Sparse matrix, dictionary, and a few others.

Type-guard predicates follow the pattern `isRuntimeTensor(v)`, `isRuntimeChar(v)`, etc. Use these instead of examining shape fields directly.

## `RTV` constructors

The `RTV` namespace exposes construction helpers for runtime values: `RTV.num`, `RTV.tensor(data, shape)`, `RTV.complex(re, im)`, `RTV.logical`, `RTV.char`, and so on. Prefer them to building objects by hand — they keep shape, precision, and flag invariants consistent.

Extract a scalar with `toNumber(v)`.

## Tensor representation

A `RuntimeTensor` holds:

- `data` — a typed array for the real part.
- `imag` — an optional typed array for the imaginary part. Absent means the tensor is real.
- `shape` — integer dimensions, column-major (Fortran) order.
- `_isLogical` — flag set on the result of comparison/logical ops.

**Column-major layout.** Element `(i, j)` of an `[m, n]` matrix is at flat index `j * m + i`. All tensor code assumes this; any new operation must preserve it.

**Precision.** The typed-array class is configurable globally: `Float64Array` by default, `Float32Array` when `NUMBL_USE_FLOAT32` is set. A tensor constructed in float64 mode is not portable to float32 mode and vice-versa.

**Value semantics via deep-clone.** Numbl previously implemented MATLAB's pass-by-value semantics with a refcount + copy-on-write scheme. That layer was removed in favor of a simpler "always deep-clone" model: at every function-call boundary, each non-handle argument is deep-cloned (`deepCloneValue` in [`runtime/utils.ts`](../../../src/numbl-core/runtime/utils.ts)). Plain `a = b` assignments where the RHS is a variable / member / index reference also deep-clone. Fresh expressions (`a + b`, function-call returns, etc.) are not cloned because deep-clone-on-call already guarantees the producer's inputs were independent. Mutating ops (indexed store, in-place updates) can therefore mutate buffers directly — no aliasing exists to worry about.

The trade-off is correctness for cost: the implementation is straightforward but every assignment of a tensor variable copies the full data buffer. Reintroducing a sharing scheme is a planned follow-up; the current shape is the safe baseline to build it back on top of.

Handle classes (`RuntimeClassInstance.isHandleClass`), graphics handles, dummy handles, and function handles (closures) are exempt from the deep-clone — they pass by reference, matching MATLAB's `handle` semantics.

## Tensor ops

Tensor-level kernels (element-wise binary and unary, reductions, comparisons, Bessel, etc.) live in the ops layer. Each op has an integer op-code. The ops layer dispatches each op to the native addon when available or to a pure-JS fallback otherwise. The op-code list is shared between TypeScript and the native C; a unit test verifies they stay in sync.

Complex tensors follow the same dispatch with separate real/complex op-code sets. A complex-tensor operation that has no complex-aware kernel either falls back to a JS complex path or is handled by the interpreter.

## What the JIT does with tensors

For tensor-valued operations, JIT codegen either:

- calls a helper on the `$h` object that wraps the ops-layer dispatch, or
- when the operation is in the shared fast subset with a known shape and real data, emits an inline loop (possibly fused with neighboring ops — see [../jit/fusion.md](../jit/fusion.md)).

Anything outside that subset (unusual shapes, complex, sparse, mixed types) goes through the ops-layer helper.
