# Type System (`JitType`)

`JitType` (`src/numbl-core/jitTypes.ts`) is the type language the executors use to classify values for JIT specialization. It does not exist at parse time; the interpreter itself is untyped. `inferJitType` produces a `JitType` from a runtime value, the executors key their caches on it (`jitTypeKey`), and `executors/jit/typeAdapter.ts` translates it into the JIT compiler's own `Type` (`jit/lowering/types.ts`).

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

Interpreter builtins do not see `JitType`s: they take runtime values (see [builtins.md](../builtins.md)). The JIT compiler's own builtins (`jit/builtins`) carry type-transfer rules over its `Type`.
