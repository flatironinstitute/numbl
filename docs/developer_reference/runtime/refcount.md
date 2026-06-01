# Refcount and COW

The runtime tracks reference counts on every container kind. The count drives both lifecycle (`decref` → `_destroy` chain) and copy-on-write decisions on indexed stores. Strict ownership is enforced by an API on the container classes; the surrounding interpreter, JIT, and store paths thread the runtime through so every binding mutation goes through the API.

There is no buffer-pool reclamation — V8 GC reclaims wrappers and their backing `Float64Array` buffers automatically once they're unreachable.

## `Refcounted`

Defined in [`runtime/refcount.ts`](../../../src/numbl-core/runtime/refcount.ts). Every container class (`RuntimeTensor`, `RuntimeStruct`, `RuntimeCell`, …) extends `Refcounted`:

- `_rc: number` — total ref count. Starts at 0. Newly-constructed wrapper is owned by no slot.
- `_scopeHolds: number` — subset of `_rc` contributed by active per-statement scopes. The COW check subtracts this so transient scope holds don't force unnecessary copies.
- `incref()` — bumps `_rc`.
- `decref(rt)` — decrements; if the count hits 0, runs `_destroy(rt)`.
- `_destroy(rt)` — kind-specific. Containers with child refs decref each child. Tensors and sparse matrices have no children; their default `_destroy` is a no-op (V8 GC reclaims the buffer).

The free helpers `incref(v)` and `decref(rt, v)` are noops on primitives (`number | boolean | string`).

A constructor finalizer auto-adopts the new wrapper into the active runtime's `currentScope` (see [Transients](#transients) below). Tests that build `RuntimeValue`s without an active runtime are unaffected (`getCurrentRuntime()` returns null and adoption is skipped).

`isShared(v)` returns `true` iff `v._rc - v._scopeHolds > 1`. This is the COW predicate.

## Slot ownership

Every "slot" — env binding, struct field, cell element, function capture, dictionary entry, persistent / global — increfs the value when it takes ownership and decrefs the previous occupant when it's overwritten.

- **`Environment.set` / `setLocal` / `clearLocals` / `delete`** ([`interpreter/types.ts`](../../../src/numbl-core/interpreter/types.ts)) — incref new before decref old, so self-rebind is safe.
- **`Environment.snapshot()`** — increfs every value copied into the new flat env so anonymous-function closures survive the source frame's exit.
- **`Runtime.setPersistent`** ([`runtime/runtime.ts`](../../../src/numbl-core/runtime/runtime.ts)) — same pattern for persistent variables.
- **Container constructors** ([`runtime/types.ts`](../../../src/numbl-core/runtime/types.ts)) — `RuntimeStruct`, `RuntimeCell`, `RuntimeClassInstance`, `RuntimeFunction`, etc. incref every child value they accept at construction.
- **`bindField` / `bindElement`** — class methods on `RuntimeStruct`, `RuntimeClassInstance`, `RuntimeCell` for in-place mutation. Used by `setRTValueField` (handle-class branch and value-class non-shared branch) and `storeIntoCell*` (every direct write to `cell.data[i]` and every auto-grow append routes through `setCellElement` / `appendEmptyCellSlot`).
- **`callSuperConstructor`** ([`runtime/runtimeDispatch.ts`](../../../src/numbl-core/runtime/runtimeDispatch.ts)) — class-to-class inheritance threads `bindField`; the built-in-superclass case decrefs the old `_builtinData` and increfs the new one.

## Transients

A `RefScope` ([`runtime/refcount.ts`](../../../src/numbl-core/runtime/refcount.ts)) is established per top-level statement via `Runtime.withScope(fn)`. Every fresh container's auto-adopt routes into the active scope (`rc 0 → 1`, `_scopeHolds 0 → 1`). On scope drain at end of statement, every adopted value is decref'd:

- Values bound to a slot during the statement got an extra incref from the slot, so they survive drain at slot count.
- Unbound transients drop to `rc = 0` and self-destruct (decreffing children).

`execStmt` ([`interpreter/interpreterExec.ts`](../../../src/numbl-core/interpreter/interpreterExec.ts)) wraps the statement body in `withScope`. The JIT executors ([`executors/jit/`](../../../src/numbl-core/executors/jit)) cross the boundary through a value adapter ([`valueAdapter.ts`](../../../src/numbl-core/executors/jit/valueAdapter.ts)): owned inputs are cloned on the way in (MATLAB pass-by-value), and the spec returns a freshly-owned buffer that numbl takes ownership of on the way out — so a JIT'd call neither leaks into nor double-frees the caller's pooled slots.

`callUserFunction` ([`interpreter/interpreterFunctions.ts`](../../../src/numbl-core/interpreter/interpreterFunctions.ts)) adopts every output value into the **caller's** scope before running `fnEnv.clearLocals()` — outputs are otherwise held only by the callee's slot binding which is about to be decref'd to 0. The same adoption happens in `evalAnonFunc`'s synthetic-fn body so closure invocations don't drop their results.

## Copy-on-write on indexed stores

The lvalue chain walker `evalLValueBase` ([`interpreter/interpreterExec.ts`](../../../src/numbl-core/interpreter/interpreterExec.ts)) descends top-down through `Member` / `Index` / `IndexCell` levels. At each container level, if `isShared(parent)` it allocates a fresh copy via `cowCopy` ([`runtime/cow.ts`](../../../src/numbl-core/runtime/cow.ts)) and calls `writeLValueBase` to rebind it in the grandparent. After the walk, the chain from env root to the leaf's parent is uniquely owned.

The leaf-level mutation then runs an `isShared` check of its own:

- `storeIntoTensor` / `storeIntoCell` ([`runtime/indexing.ts`](../../../src/numbl-core/runtime/indexing.ts)) clone the tensor / cell wrapper if `isShared`, then mutate.
- `setRTValueField` ([`runtime/struct-access.ts`](../../../src/numbl-core/runtime/struct-access.ts)) mutates the struct/value-class instance in place via `bindField` when not shared, otherwise builds a new instance with copied fields.

Handle-class instances bypass COW unconditionally — they have reference semantics by design.

## Walk-throughs

```
c = 2 + 2;
```

1. Fresh tensor `T` from `2+2`. Refcounted ctor adopts: `T.rc=1`, `T.scopeHolds=1`.
2. `env.set("c", T)`: incref → `T.rc=2`, `T.scopeHolds=1`; old `c` undefined, no decref.
3. End of statement, scope drains: decref → `T.rc=1`, `T.scopeHolds=0` (just `env.c`).

```
b = a;        % a is already in env at rc=1
a(1) = 99;
```

1. After `b = a`: `T.rc=2` (env.a, env.b), `scopeHolds=0`. `isShared(T)` is `2 > 1` → true.
2. `a(1) = 99` calls `storeIntoTensor`. `isShared(T)` → COW: `T'` allocated, scope adopts → `T'.rc=1`, `T'.scopeHolds=1`.
3. `env.set("a", T')` increfs T' (rc=2, scopeHolds=1) and decrefs T (rc=2 → 1).
4. Mutate T'(0) = 99.
5. Scope drains. `T'.rc=1` (env.a). `T.rc=1` (env.b). ✓

```
s = t; s.a.b(1) = 99;        % s and t share a struct
```

1. `S.rc=2` (env.s, env.t).
2. `evalLValueBase` walks `Member("b") of Member("a") of Ident("s")`:
   - At `Ident("s")`: returns `S`.
   - At `Member("a")`: `isShared(S)` → COW `S` to `S'`, `writeLValueBase(Ident("s"), S')`. `env.s = S'`. Get `M = S'.fields["a"]`. `M.rc=2` (S, S' both hold).
   - At `Member("b")`: `isShared(M)` → COW `M` to `M'`, `writeLValueBase(Member("a") of Ident("s"), M')`. The intermediate `setMemberReturn(S', "a", M')` is now non-shared (`S'.rc=1`) so it `bindField`s in place. Get `T = M'.fields["b"]`. `T.rc=2`.
3. `storeIntoTensor(T, …)`: `isShared(T)` → COW to `T'`. Mutate.
4. `writeLValueBase` walks back up: `setMemberReturn(M', "b", T')` mutates `M'` in place; `setMemberReturn(S', "a", M')` is a no-op (same value); `env.set("s", S')` is a no-op.
5. Final: `env.t = S` with original chain; `env.s = S'` with `M'` with `T'` (modified). ✓

## Runtime flags

- **`rt.strictRefcount`** (default true) — when on, `decref` on a zero-count throws. Refcount-driven COW relies on accurate counts to make correct mutate-vs-copy decisions; an underflow indicates a missed incref or a double decref somewhere in the bookkeeping, so we want it loud rather than silently masked. Set to false only as a temporary debug aid while migrating new mutation paths.

## Buffer-sharing constraint

A `Float64Array` buffer can have **only one wrapper owner**. Operations that previously returned a wrapper aliasing the source's buffer (zero-copy `reshape`, `squeeze`) now copy the data. The refcount-driven COW relies on the wrapper-level rc faithfully reflecting buffer sharing; two wrappers pointing at one buffer would let a "rc=1, mutate in place" decision corrupt the other wrapper's view.
