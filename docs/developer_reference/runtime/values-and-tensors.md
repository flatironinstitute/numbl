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

## Kind descriptors

`runtime/kinds.ts` holds one descriptor per kind (`KINDS`, keyed by `kindOf(v)`: `number`, `boolean`, `string`, then each class's `kind`): the MATLAB class name (`class`), the size, `numel`, whether the kind is an object for `isobject`, `isreal`, and a short label for internal error messages. Code that needs these facts calls the helpers built on it (`valueClass`, `valueShape`, `valueNumel`, `isEmptyValue`, `isObjectValue`, `isRealValue`, `kstr`) rather than switching on the kind. `dispatchClassOf(v)` gives the class whose methods calls and operators on `v` dispatch to (see [function-resolution.md](../compiler/function-resolution.md#object-arguments-and-operators)). `isaValue(v, name, ancestors)` implements `isa`: the value's class, its superclasses (for objects, through `Runtime.classAncestors`), and the categories `numeric`, `float` and `integer`; `isnumeric`, `isfloat` and `isinteger` are `isa` with a category, so a `classdef Foo < double` object or an integer-based enumeration member counts.

### Adding a value kind

1. Add the class to `types.ts` and to the `RuntimeValue` union, with an `isRuntimeX` guard.
2. Add its entry to `KINDS`. The table is a mapped type over every kind, so the build fails until the entry exists.
3. Handle it in the switches that end in `assertNever` (indexing in `indexIntoRTValue`, `valuesAreEqual`) and in `displayValue`, whose return type makes it exhaustive. The build fails for these too.
4. Still per kind, with no compile-time check: indexed assignment (`storeIntoRTValueIndex`), concatenation (`tensor-construction.ts`), `isequal` (`builtins/utility.ts`), the conversions in `convert.ts`, `whos` byte counts, refcounting of the values it holds (see [refcount.md](refcount.md)), and JIT input typing (`typeOfValue` in `executors/inputTypes.ts`), which types an unhandled kind as unknown.

## `RTV` constructors

The `RTV` namespace exposes construction helpers for runtime values: `RTV.num`, `RTV.tensor(data, shape, imag?, cls?)`, `RTV.logicalTensor(data, shape)`, `RTV.complex(re, im)`, `RTV.logical` (a scalar), `RTV.char`, and so on. Prefer them to building objects by hand: they keep shape, precision, and class invariants consistent. `RTV.tensor` takes its buffers without copying and throws an internal error unless `data` and `imag` are `Float64Array`s, so other storage is never silently widened to double; `RTV.tensorFromArray` copies from plain number arrays.

Extract a scalar with `toNumber(v)`.

## Tensor representation

A `RuntimeTensor` holds:

- `data`: a typed array for the real part.
- `imag`: an optional typed array for the imaginary part. Absent means the tensor is real.
- `shape`: integer dimensions, column-major (Fortran) order.
- `cls`: the element class (`ElementClass`, today `"double"` or `"logical"`), set at construction and `readonly`. See [Element classes](#element-classes).

**Column-major layout.** Element `(i, j)` of an `[m, n]` matrix is at flat index `j * m + i`. All tensor code assumes this; any new operation must preserve it.

**Copy-on-write.** Assigning a tensor variable or passing it as an argument shares the same wrapper. An indexed store copies a container only when its refcount shows another holder, and each buffer has a single wrapper owner (`reshape` and `squeeze` copy). See [refcount.md](refcount.md).

## Element classes

A tensor's MATLAB class lives in its `cls` field (`ElementClass` in `runtime/types.ts`). Every class currently stores its elements in a `Float64Array`; the class fixes their meaning (a logical tensor holds 0/1 and has no imaginary part). Scalars are not tensors: a logical scalar is a JS `boolean`, a double scalar a `number`. Char arrays are a separate kind (`RuntimeChar`), not an element class.

The class is chosen when the tensor is built (`RTV.tensor(..., cls)`, `RTV.logicalTensor`, `new RuntimeTensor(..., cls)`), and `shallowClone` keeps it, so copy-on-write stores keep it too. Operations that preserve their input's class (indexing, indexed assignment and deletion, reshape, permute, transpose, repmat, squeeze, sort, ...) pass `v.cls` to the constructor of their result. A result of another class is a new tensor; the one exception is `withElementClass`, which sets the class of a fresh result that nothing else holds yet.

Adding an element class (e.g. `single` or an integer class) touches:

1. `ElementClass` and the constructors; a class with narrower storage also changes the type of `data` and the `Float64Array` check in `RTV.tensor`.
2. `class`, `isa`, `islogical`, `isnumeric`, `isfloat`, `isinteger`: `KINDS.tensor.className` and the category table (`CATEGORIES`) in `runtime/kinds.ts`.
3. Display (`runtime/display.ts`).
4. Arithmetic result-class rules: element-wise and matrix ops in `helpers/arithmetic.ts` and `runtime/runtimeOperators.ts` (today every arithmetic result is double and every comparison or logical op is logical), and reductions whose result class follows the input (`min`/`max`, `cummax`/`cummin`) or is fixed (`any`/`all` give logical).
5. Concatenation rules: `catAlongDim` in `runtime/tensor-construction.ts` and `cat` in `builtins/array-manipulation.ts` (today: logical only when every operand is logical, empties included).
6. Indexed assignment of a value of another class into a tensor (`storeIntoTensor` in `runtime/indexing.ts`), and the conversion builtins (`double`, `logical`, ... in `builtins/type-constructors.ts`, `runtime/convert.ts`).
7. The JIT: the `NumericElem` lattice in `jit/lowering/types.ts`, input typing (`typeOfValue` in `executors/inputTypes.ts`), and both value adapters (`executors/jit/valueAdapter.ts`, `valueAdapterC.ts`), which carry the class across the boundary in each direction.
8. Serialization and text conversion: `jsondecode` (`runtime/jsonDecode.ts`), `jsonencode`, `mat2str`, `num2str`, `typecast`, and the argument types seen by JS user functions (`jsUserFunctions.ts`).

## Tensor ops

Tensor-level kernels (element-wise binary and unary, reductions, comparisons, Bessel, etc.) live in the ops layer (`src/numbl-core/ops/`). Each op has an integer op-code (`ops/opCodes.ts`). The ops layer dispatches each op to the native addon when available or to a pure-JS fallback otherwise. The op-code list is shared between TypeScript and the native C; `op-codes-sync.test.ts` checks that the lists agree, and `native-ops-parity.test.ts` that the two implementations give the same results (when the addon is built).

Complex tensors follow the same dispatch with separate real/complex op-code sets. A complex-tensor operation that has no complex-aware kernel either falls back to a JS complex path or is handled by the interpreter.

## What the JIT does with tensors

For tensor-valued operations, JIT codegen either:

- calls a helper on the `$h` object that wraps the ops-layer dispatch, or
- when the operation is in the shared fast subset with a known shape and real data, emits an inline loop (possibly fused with neighboring ops: see [../jit/fusion.md](../jit/fusion.md)).

Anything outside that subset (unusual shapes, complex, sparse, mixed types) goes through the ops-layer helper.
