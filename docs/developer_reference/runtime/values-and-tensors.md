# Runtime Values and Tensors

## `RuntimeValue`

The runtime universe. A union of primitives plus refcounted classes:

- `number`: JavaScript number (real scalar). Not refcounted.
- `boolean`: logical scalar. Not refcounted.
- `string`: MATLAB string _scalar_ (distinct from char). Not refcounted.
- `RuntimeStringArray`: MATLAB string array (`kind: "string_array"`): column-major `data: string[]` plus a 2-D `shape`. A 1×1 string is always represented as the primitive, never a 1×1 array: build results with `stringArrayValue(data, shape)`, which collapses 1×1 to the primitive (empties stay arrays, e.g. `string([])` is 0×0).
- `RuntimeChar`: MATLAB char array.
- `RuntimeTensor`: real or complex, any rank, any shape.
- `RuntimeComplexNumber`: complex scalar.
- `RuntimeCell`: heterogeneous cell array.
- `RuntimeStruct`: scalar struct.
- `RuntimeStructArray`: struct array (separate type from scalar struct): column-major `elements` (each carrying every field, in `fieldNames` order) plus an N-d `shape`. Results that may be 1×1 are built with `structArrayValue(fieldNames, elements, shape)`, which returns the scalar struct.
- `RuntimeClassInstance`: scalar `classdef` instance. A `classdef Foo < <builtin>` instance keeps its underlying built-in value in `_builtinData`; an enumeration member additionally sets `_enumMember` (the member name), with `_builtinData` holding the superclass value. Enumeration semantics (member access, `ClassName(x)` conversion, `==`/`~=`/`switch`/`isequal`) live in `runtime/runtimeEnum.ts` and run on the interpreter only: the JIT declines enumeration classes.
- `RuntimeClassInstanceArray`: class-instance array (separate type from scalar instance).
- `RuntimeFunction` / function handle: callable reference.
- `RuntimeSparseMatrix`: sparse 2-D matrix.
- `RuntimeDictionary`: MATLAB dictionary.
- `RuntimeDummyHandle`: placeholder handle for graphics-style APIs.
- `RuntimeGraphicsHandle`: graphics object handle (figure, axes, etc.).

Every container kind is a class extending `Refcounted` (`runtime/refcount.ts`). Each instance carries a `_rc` field (starts at 0) and `incref()` / `decref(rt)` methods enforced by a strict slot API; see [refcount.md](refcount.md) for the lifecycle.

Type-guard predicates follow the pattern `isRuntimeTensor(v)`, `isRuntimeChar(v)`, etc.: they check `value.kind === "..."` and continue to work on the class instances. Use these instead of examining shape fields directly or using `instanceof`.

## `RTV` constructors

The `RTV` namespace exposes construction helpers for runtime values: `RTV.num`, `RTV.tensor(data, shape)`, `RTV.complex(re, im)`, `RTV.logical`, `RTV.char`, and so on. Prefer them to building objects by hand: they keep shape, precision, and flag invariants consistent.

Extract a scalar with `toNumber(v)`.

## Tensor representation

A `RuntimeTensor` holds:

- `data`: a typed array for the real part.
- `imag`: an optional typed array for the imaginary part. Absent means the tensor is real.
- `shape`: integer dimensions, column-major (Fortran) order.
- Flags: e.g., `_isLogical` for boolean arrays from comparisons.

**Column-major layout.** Element `(i, j)` of an `[m, n]` matrix is at flat index `j * m + i`. All tensor code assumes this; any new operation must preserve it.

**Copy-on-write.** Assigning a tensor variable or passing it as an argument shares the same wrapper. An indexed store copies a container only when its refcount shows another holder, and each buffer has a single wrapper owner (`reshape` and `squeeze` copy). See [refcount.md](refcount.md).

## Tensor ops

Tensor-level kernels (element-wise binary and unary, reductions, comparisons, Bessel, etc.) live in the ops layer (`src/numbl-core/ops/`). Each op has an integer op-code (`ops/opCodes.ts`). The ops layer dispatches each op to the native addon when available or to a pure-JS fallback otherwise. The op-code list is shared between TypeScript and the native C; `op-codes-sync.test.ts` checks that the lists agree, and `native-ops-parity.test.ts` that the two implementations give the same results (when the addon is built).

Complex tensors follow the same dispatch with separate real/complex op-code sets. A complex-tensor operation that has no complex-aware kernel either falls back to a JS complex path or is handled by the interpreter.

## What the JIT does with tensors

For tensor-valued operations, JIT codegen either:

- calls a helper on the `$h` object that wraps the ops-layer dispatch, or
- when the operation is in the shared fast subset with a known shape and real data, emits an inline loop (possibly fused with neighboring ops: see [../jit/fusion.md](../jit/fusion.md)).

Anything outside that subset (unusual shapes, complex, sparse, mixed types) goes through the ops-layer helper.
