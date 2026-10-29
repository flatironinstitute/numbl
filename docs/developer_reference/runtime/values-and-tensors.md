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
5. If the kind is a storage variant of an array that most builtins have no code for (as sparse is), give it a rule in `runtime/argKinds.ts`, so builtins receive it converted (or rejected) unless they declare `accepts` for it. See [builtins.md](../builtins.md#argument-kinds-sparse-and-other-non-dense-values).

## `RTV` constructors

The `RTV` namespace exposes construction helpers for runtime values: `RTV.num`, `RTV.tensor(data, shape, imag?, cls?)`, `RTV.logicalTensor(data, shape)`, `RTV.complex(re, im)`, `RTV.logical` (a scalar), `RTV.char`, and so on. Prefer them to building objects by hand: they keep shape, precision, and class invariants consistent. `RTV.tensor` takes its buffers without copying and throws an internal error unless `data` and `imag` are `Float64Array`s, so other storage is never silently widened to double; `RTV.tensorFromArray` copies from plain number arrays.

Extract a scalar with `toNumber(v)`.

## Tensor representation

A `RuntimeTensor` holds:

- `data`: a typed array for the real part.
- `imag`: an optional typed array for the imaginary part. Absent means the tensor is real.
- `shape`: integer dimensions, column-major (Fortran) order.
- `cls`: the element class (`ElementClass`: `"double"`, `"single"` or `"logical"`), set at construction and `readonly`. See [Element classes](#element-classes).

**Column-major layout.** Element `(i, j)` of an `[m, n]` matrix is at flat index `j * m + i`. All tensor code assumes this; any new operation must preserve it.

**Copy-on-write.** Assigning a tensor variable or passing it as an argument shares the same wrapper. An indexed store copies a container only when its refcount shows another holder, and each buffer has a single wrapper owner (`reshape` and `squeeze` copy). See [refcount.md](refcount.md).

## Element classes

A tensor's MATLAB class lives in its `cls` field (`ElementClass` in `runtime/types.ts`). Every class currently stores its elements in a `Float64Array`; the class fixes their meaning (a logical tensor holds 0/1 and has no imaginary part, a single tensor holds values exactly representable in single precision). Double and logical scalars are not tensors: a logical scalar is a JS `boolean`, a double scalar a `number`. A single scalar is a 1x1 tensor, since a JS number always means double. Char arrays are a separate kind (`RuntimeChar`), not an element class.

The class is chosen when the tensor is built (`RTV.tensor(..., cls)`, `RTV.logicalTensor`, `new RuntimeTensor(..., cls)`), and `shallowClone` keeps it, so copy-on-write stores keep it too. Operations that preserve their input's class (indexing, indexed assignment and deletion, reshape, permute, transpose, repmat, squeeze, sort, ...) pass `v.cls` to the constructor of their result. A result of another class is a new tensor; the one exception is `withElementClass`, which sets the class of a fresh result that nothing else holds yet.

**Single.** `runtime/single.ts` holds the helpers. A single result is computed in double and rounded with `Math.fround` (`toSingleResult`), which gives the same bits as computing in single for `+ - * /` and `sqrt` (double has more than twice single's precision); for other functions and for matrix products the result can differ from MATLAB's in the last place. The rules, as in MATLAB: an arithmetic operator with a single operand gives single (`binop`, `uminus`), comparisons and logical operators give logical, concatenation gives single when any operand is single (empties included) unless one is char, a colon with a single endpoint gives single, and an indexed store keeps the target's class (a single target rounds the assigned values; a double target stays double). Indexing a single array, and iterating one in a `for` loop, give single 1x1 values (`tensorElement`). The JIT declines single values (`typeOfValue` gives `unknown`, and the JIT has no `single` builtin), so code that handles them runs on the interpreter.

Adding an element class (e.g. an integer class) touches:

1. `ElementClass` and the constructors; a class with narrower storage also changes the type of `data` and the `Float64Array` check in `RTV.tensor`.
2. `class`, `isa`, `islogical`, `isnumeric`, `isfloat`, `isinteger`: `KINDS.tensor.className` and the category table (`CATEGORIES`) in `runtime/kinds.ts`.
3. Display (`runtime/display.ts`).
4. Arithmetic result-class rules: element-wise and matrix ops in `helpers/arithmetic.ts` and `runtime/runtimeOperators.ts`, which classify operands before calling the kernel layer (`ops/elementwise.ts`, whose kernels assume `Float64Array` storage) (today every arithmetic result is double and every comparison or logical op is logical), and reductions whose result class follows the input (`min`/`max`, `cummax`/`cummin`) or is fixed (`any`/`all` give logical).
5. Concatenation rules: `catAlongDim` in `runtime/tensor-construction.ts` and `cat` in `builtins/array-manipulation.ts` (today: logical only when every operand is logical, empties included).
6. Indexed assignment of a value of another class into a tensor (`storeIntoTensor` in `runtime/indexing.ts`), and the conversion builtins (`double`, `logical`, ... in `builtins/type-constructors.ts`, `runtime/convert.ts`).
7. The JIT: the `NumericElem` lattice in `jit/lowering/types.ts`, input typing (`typeOfValue` in `executors/inputTypes.ts`), and both value adapters (`executors/jit/valueAdapter.ts`, `valueAdapterC.ts`), which carry the class across the boundary in each direction.
8. Serialization and text conversion: `jsondecode` (`runtime/jsonDecode.ts`), `jsonencode`, `mat2str`, `num2str`, `typecast`, and the argument types seen by JS user functions (`jsUserFunctions.ts`).

## Tensor ops

Tensor-level kernels (element-wise binary and unary, reductions, comparisons, Bessel, etc.) live in the ops layer (`src/numbl-core/ops/`). Each op has an integer op-code (`ops/opCodes.ts`). `tensorOps` (`ops/dispatch.ts`) dispatches each op to the native addon when available or to a pure-JS fallback otherwise, over caller-allocated flat buffers. The op-code list is shared between TypeScript and the native C; `op-codes-sync.test.ts` checks that the lists agree, and `native-ops-parity.test.ts` that the two implementations give the same results (when the addon is built).

### Element-wise kernel layer

The element-wise operators and builtins (`+ - .* ./ .\ .^`, scalar `*` and `/`, the comparisons, `& | xor`, two-argument `min`/`max`, `mod`, `rem`, `atan2`, `hypot`, and unary math such as `sin` or `abs`) compute through one layer, `ops/elementwise.ts`. Its callers classify their values first and then make one call:

- `elementwiseBinary(op, a, b)` for an `OpBin` code, `elementwiseCompare(op, a, b)` for an `OpCmp` code (a 0/1 result the caller makes logical), `elementwiseUnary(op, a)` for an `OpUnary` code, and `mapBinary(fn, a, b)` for a per-element function with no op code. `binaryScalar` and `compareScalar` apply the same rules to two scalars without allocating arrays.
- An operand is a JS number (a real scalar) or a dense array `{data, imag?, shape}`; a `RuntimeTensor` is one as it is. A result is a fresh dense array whose `imag` is absent when the result is real; a complex result whose imaginary parts are all zero is returned as real.
- Broadcasting (implicit expansion) is in `ops/broadcast.ts` (`broadcastShape`, `forEachBroadcastRow`). Incompatible sizes raise MATLAB's error, `Arrays have incompatible sizes for this operation.` with identifier `MATLAB:sizeDimensionsMustMatch` (`sizeMismatchError`), for every operation; the JIT's runtime raises the same one.
- The kernel layer owns the numeric rules: the complex formulas (Smith's algorithm for a scalar or broadcast complex division), real times complex scaling each part without promotion (so `Inf * (0+1i)` has a 0 real part), `.*` and `./` treating an all-zero imaginary part as real, a negative base with a non-integer exponent making `.^` complex, and NaN-skipping `min`/`max` (complex values by magnitude, then angle).
- Ops with a native kernel (`OpBin` codes below `NATIVE_BIN_OP_COUNT`, comparisons below `NATIVE_CMP_OP_COUNT`, and the unary ops) go through `tensorOps` for same-shape and scalar-array operands, so the addon serves them when it is loaded. Broadcast operands and the other ops (power, `mod`, `atan2`, ...) run the loops in `elementwise.ts`. The JIT's JS runtime reaches the same `tensorOps` kernels for large tensors through `$elemwise` (`executors/jit/hostHelpers.ts`), so the `--opt` levels compute the same bits.

The callers (`helpers/arithmetic.ts`, `helpers/reduction/min-max.ts`, and `applyBinaryElemwise` and the unary applies in `interpreter/builtins/types.ts`) only classify their operands: string `+` and string comparison, sparse operands (the sparse-result paths, sparse + sparse, sparse .\* dense and sparse ./ scalar, stay in `helpers/sparse-arithmetic.ts`; other sparse operands are densified), char as character codes, logical as 0/1, a complex scalar as a 1-element array, and the result class (logical for comparisons, and for `min`/`max` of two logical operands). Two scalars give a scalar; any array operand, including a 1×1 tensor, gives a tensor. An alternative compute backend would sit behind this interface, not in the operators.

## What the JIT does with tensors

For tensor-valued operations, JIT codegen either:

- calls a helper on the `$h` object that wraps the ops-layer dispatch, or
- when the operation is in the shared fast subset with a known shape and real data, emits an inline loop (possibly fused with neighboring ops: see [../jit/fusion.md](../jit/fusion.md)).

Anything outside that subset (unusual shapes, complex, sparse, mixed types) goes through the ops-layer helper.
