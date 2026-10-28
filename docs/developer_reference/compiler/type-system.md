# Type System

The interpreter is untyped. Types exist only in the JIT, which has a single type lattice: the compiler's `Type` (`jit/lowering/types.ts`). The lowerer infers it for every expression, and the executors classify runtime inputs straight into it when they specialize a unit, so a unit's input types, its cache key and the types the compiler specializes on are one and the same.

## Kinds

`Type` is a discriminated union:

- `Numeric`: a scalar or array of double, logical or char elements, real or complex. It carries per-axis dimensions (`dims`, each exact or unknown), the full `shape` when every axis is exact, a sign (`positive`, `nonneg`, `negative`, `nonpositive`, `zero`, `nonzero` or `unknown`), and optionally an exact value (a scalar, or a small array of at most 256 elements). A 1×1 `Numeric` is a scalar; anything else is stored as a tensor.
- `String`, `Char`: text, with an optional exact value.
- `Struct`: fields in insertion order, each with its own `Type`.
- `Class`, `Cell`, `Handle`: class instances, cells (tuple or uniform) and function handles, used inside compiled code.
- `Void` (a call with no outputs) and `Unknown`.

## Unification

At control-flow joins (branches that reconverge, loop back-edges, function-return merges) the lowerer merges types with `unify`. The result keeps what both sides agree on: an exact value survives only when both sides carry the same one, signs widen through the sign lattice (`positive` and `zero` give `nonneg`), and an axis whose sizes differ becomes unknown. Types that cannot share a representation (real and complex, double and logical, different ranks, different kinds) unify to `Unknown`.

## Executor inputs

The executors (see [executors.md](../executors.md)) type each input of a unit with `typeOfValue`:

- A real scalar is a `Numeric` scalar with its sign (zero counts as `nonneg`) but no exact value, so `f(1); f(2); f(3)` share one specialization. A logical scalar is `nonneg` logical; a complex scalar is complex with unknown sign.
- A dense tensor carries its shape and its real, complex or logical element type, but not its sign, which would take a pass over the data on every dispatch.
- Strings and char arrays map to `String` and `Char` without a value.
- A struct maps to a `Struct` whose fields are typed the same way.
- Anything else (cells, objects, sparse matrices, dictionaries, function handles, string arrays) is `Unknown`, and the executors decline the unit. A capture-free function handle input to a loop is inlined instead (see [executors.md](../executors.md#lowering)).

Each backend then decides which input types it accepts: the JS backend takes everything but `Unknown`, the C backend only the types it can marshal across the koffi ABI (numeric scalars and tensors).

Specializations are keyed on `canonicalizeType`, the same canonical string the compiler uses for its own spec cache. To keep a unit called with shifting argument types from compiling one spec per call, each input type is widened against the type seen on the unit's previous dispatch with `widenArgType`. It follows `unify`, except that a logical widens to double and a real to complex (the boundary adapters box such values), and it declines to widen a scalar against a non-scalar, across ranks, between structs with different fields, or to `Unknown`; the input then keeps its own type and gets its own spec. A unit called first with `x = 1` and then with `x = -1` thus compiles a spec for a positive scalar and then one for a scalar of unknown sign, which serves every later call.

This runs on every interpreted call of a user function, so the input types are interned: a value maps through a short signature string to a shared, frozen `Type`, whose canonical key is computed once, and widening is memoized on pairs of interned types. In the steady state a dispatch builds no types.

## How builtins participate

Interpreter builtins do not see types: they take runtime values (see [builtins.md](../builtins.md)). The JIT compiler's own builtins (`jit/builtins`) carry type-transfer rules over `Type`.
