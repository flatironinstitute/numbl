# Type System (`JitType`)

`JitType` is the static type language used by the JIT for specialization. It does not exist at parse time; the interpreter itself is untyped. `JitType` values are produced by the JIT's lowering pass from argument values and propagated by each builtin's `resolve` function.

## Kinds

The `JitType` discriminated union covers:

- `number` — scalar real (possibly integer, possibly logical, possibly non-negative).
- `boolean` — logical scalar.
- `complex_or_number` — may be complex at runtime.
- `tensor` — real or complex array with optional known shape.
- `string`, `char` — text values (MATLAB distinguishes these).
- `struct`, `struct_array` — record types.
- `cell` — heterogeneous container.
- `class_instance` — instance of a `classdef` class.
- `sparse_matrix` — sparse 2-D matrix.
- `dictionary` — MATLAB dictionary.
- `function_handle` — first-class function reference.
- `unknown` — fallback; prevents most JIT optimizations.

## Refinements

Each kind can carry additional information the JIT uses for sharper specialization:

- exact known value (for small constants);
- integer-ness, logical-ness, non-negativity;
- complex-ness (real vs guaranteed-complex);
- known shape (scalar, row/column vector, fixed dimensions, rank).

Refinements are monotonic: they narrow the type. Operations that produce values outside the refinement widen it (or bail).

## Unification

At join points — loop back-edges, branches that reconverge, function-return merges — types are unified by `unifyJitTypes`. The result is the least upper bound: if two arms produce `number(integer)` and `number(non-integer)`, the join is `number` without the integer refinement. Loss of a refinement can force a recompile on the next pass.

## How builtins participate

Each `IBuiltin` receives `argTypes: JitType[]` in `resolve`. It returns the output `JitType` for each output position, plus the `apply` function for the interpreter. The JIT trusts these output types when lowering callers. A builtin that returns the wrong type is a correctness bug that surfaces as a JIT bail (best case) or wrong results (worst case).

See [builtins.md](../builtins.md) for the resolution interface.
