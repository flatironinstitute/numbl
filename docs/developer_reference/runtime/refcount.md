# Refcount and Pool Reclamation

The runtime tracks reference counts on every container kind so that `Float64Array` buffers are released back to the per-runtime memory pool when their owning wrapper is no longer reachable. Strict ownership is enforced by an API on the container classes; the surrounding interpreter, JIT, and store paths thread the runtime through so every binding mutation goes through the API.

This system is for **buffer reclamation only**. Copy-on-write decisions still go through the sweep in [`runtime/aliasing.ts`](../../../src/numbl-core/runtime/aliasing.ts); refcount being slightly off is a leak or, with `memPool` on, a premature buffer release — never silent data corruption from missed COW.

## `Refcounted`

Defined in [`runtime/refcount.ts`](../../../src/numbl-core/runtime/refcount.ts). Every container class (`RuntimeTensor`, `RuntimeStruct`, `RuntimeCell`, …) extends `Refcounted`:

- `_rc: number` — starts at 0. A "newborn" wrapper is owned by no slot.
- `incref()` — bumps `_rc`.
- `decref(rt)` — decrements; if the count hits 0, runs `_destroy(rt)`.
- `_destroy(rt)` — kind-specific. Tensors and sparse matrices release their `Float64Array` buffers via `rt.pool.release(...)` (gated by `rt.memPool`). Containers with child refs decref each child.

The free helpers `incref(v)` and `decref(rt, v)` are noops on primitives (`number | boolean | string`).

A constructor finalizer auto-adopts the new wrapper into the active runtime's `currentScope` (see [Transients](#transients) below). Tests that build `RuntimeValue`s without an active runtime are unaffected (`getCurrentRuntime()` returns null and adoption is skipped).

## Slot ownership

Every "slot" — env binding, struct field, cell element, function capture, dictionary entry, persistent / global — increfs the value when it takes ownership and decrefs the previous occupant when it's overwritten.

- **`Environment.set` / `setLocal` / `clearLocals` / `delete`** ([`interpreter/types.ts`](../../../src/numbl-core/interpreter/types.ts)) — incref new before decref old, so self-rebind is safe.
- **`Environment.snapshot()`** — increfs every value copied into the new flat env so anonymous-function closures survive the source frame's exit.
- **`Runtime.setPersistent`** ([`runtime/runtime.ts`](../../../src/numbl-core/runtime/runtime.ts)) — same pattern for persistent variables.
- **Container constructors** ([`runtime/types.ts`](../../../src/numbl-core/runtime/types.ts)) — `RuntimeStruct`, `RuntimeCell`, `RuntimeClassInstance`, `RuntimeFunction`, etc. incref every child value they accept at construction.
- **`bindField` / `bindElement`** — class methods on `RuntimeStruct`, `RuntimeClassInstance`, `RuntimeCell` for in-place mutation. Used by `setRTValueField` (handle-class branch) and `storeIntoCell*` (every direct write to `cell.data[i]` and every auto-grow append routes through `setCellElement` / `appendEmptyCellSlot`).
- **`callSuperConstructor`** ([`runtime/runtimeDispatch.ts`](../../../src/numbl-core/runtime/runtimeDispatch.ts)) — class-to-class inheritance threads `bindField`; the built-in-superclass case decrefs the old `_builtinData` and increfs the new one.

`rt._envStack` (caller envs walked by the alias sweep across function boundaries) is a _borrowed_ reference — no incref. The slot-owning env still holds the count.

## Transients

A `RefScope` ([`runtime/refcount.ts`](../../../src/numbl-core/runtime/refcount.ts)) is established per top-level statement via `Runtime.withScope(fn)`. Every fresh container's auto-adopt routes into the active scope (`rc 0 → 1`). On scope drain at end of statement, every adopted value is decref'd:

- Values bound to a slot during the statement got an extra incref from the slot, so they survive drain at slot count.
- Unbound transients drop to `rc = 0` and self-destruct, releasing their buffers.

`execStmt` ([`interpreter/interpreterExec.ts`](../../../src/numbl-core/interpreter/interpreterExec.ts)) wraps the statement body in `withScope`. The JIT synthetic-fn runner ([`executors/jsJit/shared.ts`](../../../src/numbl-core/executors/jsJit/shared.ts) `runSyntheticFnAgainstEnv`) wraps the JIT-compiled function call so intermediates created by JIT'd loop bodies are subject to scope drain.

`callUserFunction` ([`interpreter/interpreterFunctions.ts`](../../../src/numbl-core/interpreter/interpreterFunctions.ts)) adopts every output value into the **caller's** scope before running `fnEnv.clearLocals()` — outputs are otherwise held only by the callee's slot binding which is about to be decref'd to 0.

## Walk-throughs

```
c = 2 + 2;
```

1. Fresh tensor `T` from `2+2`. Refcounted ctor adopts: `T.rc=1` (scope owns).
2. `env.set("c", T)`: incref → `T.rc=2`; old `c` undefined, no decref.
3. End of statement, scope drains: decref → `T.rc=1` (just `env.c`).

```
b = a;        % a is already in env at rc=1
```

1. `evalExpr(a)` returns env's value; no adopt (env reads aren't fresh).
2. `env.set("b", a_val)`: incref → `a_val.rc=2`.
3. Drain: nothing in scope, nothing changes.

```
2 + 2;        % unbound expression
```

1. `T = 2+2` adopted: `T.rc=1`.
2. `env.set("ans", T)`: incref → `T.rc=2`; old `ans` decref.
3. Drain: decref → `T.rc=1` (held by `env.ans`).

```
[~, x] = f();
```

1. Inside `f()`, outputs are read from `fnEnv`. Before `clearLocals`, each output is adopted into the **caller's** scope: `out_i.rc++`.
2. `fnEnv.clearLocals()` decrefs each binding: `out_i.rc--`.
3. Caller's `MultiAssign` binds `x` (incref). The `~` lvalue does nothing.
4. Caller's drain: every output decref'd. `x`'s tensor lands at `rc=1` (env.x); the `~` tensor lands at `rc=0` and is released.

## Runtime flags

- **`rt.memPool`** (default true) — when on, `RuntimeTensor._destroy` and `RuntimeSparseMatrix._destroy` release `data`/`imag`/`pr`/`pi` buffers to `rt.pool` for reuse. The pool poisons released buffers with `NaN`, so any use-after-free shows up loudly in test output rather than as a silent zero-fill match. Disable with the CLI's `--no-mem-pool` flag (or via the worker's `set_mem_pool` message in the browser IDE) to isolate bugs that may stem from buffer recycling: with the flag off, every allocation is a fresh `new Float64Array` and no buffer is ever released.
- **`rt.strictRefcount`** (default false) — when on, `decref` on a zero-count throws. Off by default because chained-lvalue assignments (e.g. `T.x(1).y = 10`) currently produce harmless underflows during scope drain. Flipping it on requires a cleanup pass over the back-write chain in `setMemberReturn` / indexed-store callbacks.

## Buffer-sharing constraint

A `Float64Array` buffer can have **only one wrapper owner**. Operations that previously returned a wrapper aliasing the source's buffer (zero-copy `reshape`, `squeeze`) now copy the data. The sweep-based COW system tolerates buffer aliasing during read paths but no construction path should produce two `RuntimeTensor`s pointing at the same `Float64Array`.
