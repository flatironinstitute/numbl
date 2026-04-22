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
- Flags — e.g., logical, integer, reference count.

**Column-major layout.** Element `(i, j)` of an `[m, n]` matrix is at flat index `j * m + i`. All tensor code assumes this; any new operation must preserve it.

**Precision.** The typed-array class is configurable globally: `Float64Array` by default, `Float32Array` when `NUMBL_USE_FLOAT32` is set. A tensor constructed in float64 mode is not portable to float32 mode and vice-versa.

**Copy-on-write.** Tensors carry a reference count. Assigning a tensor variable or passing it as an argument shares the same underlying buffer. Writes to a shared buffer clone first. This is transparent to user code but matters for anyone writing a new tensor op — use the provided COW helpers rather than mutating buffers in place.

## Tensor ops

Tensor-level kernels (element-wise binary and unary, reductions, comparisons, Bessel, etc.) live in the ops layer. Each op has an integer op-code. The ops layer dispatches each op to the native addon when available or to a pure-JS fallback otherwise. The op-code list is shared between TypeScript and the native C; a unit test verifies they stay in sync.

Complex tensors follow the same dispatch with separate real/complex op-code sets. A complex-tensor operation that has no complex-aware kernel either falls back to a JS complex path or is handled by the interpreter.

## What the JIT does with tensors

For tensor-valued operations, JIT codegen either:

- calls a helper on the `$h` object that wraps the ops-layer dispatch, or
- when the operation is in the shared fast subset with a known shape and real data, emits an inline loop (possibly fused with neighboring ops — see [../jit/fusion.md](../jit/fusion.md)).

Anything outside that subset (unusual shapes, complex, sparse, mixed types) goes through the ops-layer helper.
