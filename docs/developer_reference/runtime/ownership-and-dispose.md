# Buffer Ownership & Dispose Semantics

> Source of truth for who owns a tensor's data buffer at every seam in
> the interpreter. Any change to dispose / pool / clone must conform.
> If a code path can't be expressed under these rules, the rules are
> wrong — fix the doc, then fix the code.

## 1. Goal

Every dense float buffer (the `Float64Array` / `Float32Array` inside a
`RuntimeTensor`) should be **disposed exactly once** during the lifetime
of the run. "Disposed" means handed to the allocator pool via
`disposeFloat64` / `disposeFloatX`. Disposed buffers come back out of
the pool in subsequent `alloc*` / `zeroed*` / `copy*` calls.

## 2. Core invariant

> At any moment, every live buffer has **exactly one owner** — a single
> path through the runtime state graph (env binding, struct field, cell
> entry, output array, this.ans, persistent storage, `$g`) that has the
> right to dispose it.

Consequences:

- Two distinct paths must never reference the same buffer ("aliasing").
- A _borrow_ — a temporary, non-owning reference held during expression
  evaluation — is fine as long as it ends before the owner releases.
- Disposing the owner releases the buffer; any remaining borrow is a
  use-after-dispose bug. NaN-poison (in dispose) and double-dispose
  detection (in pool re-pool) catch these immediately.

Closures are the one structural exception — see §6.

## 3. Producers vs borrows

Each AST expression result is either **owned** (a fresh value the caller
takes ownership of) or **borrowed** (a non-owning reference into some
existing owner).

| Expression                | Result is                         | Notes                                                                                                                                                                                                                                                         |
| ------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TensorLit` (`[1 2 3]`)   | owned                             | Allocates fresh buffer(s).                                                                                                                                                                                                                                    |
| `Range` (`1:n`)           | owned                             | Fresh tensor.                                                                                                                                                                                                                                                 |
| `Binary` / `Unary`        | owned                             | Binops produce a fresh result tensor.                                                                                                                                                                                                                         |
| `FuncCall` / `MethodCall` | owned (intended)                  | Caller treats return as owned. **Caveat**: builtins like `uplus(x)` currently return their input verbatim, breaking the contract. Either fix offending builtins to clone, or treat `FuncCall` results as borrowed at the binding seam until that audit lands. |
| `Index` of a tensor       | owned                             | Slicing copies into a fresh tensor.                                                                                                                                                                                                                           |
| `Cell` literal (`{a, b}`) | owned wrapper, _borrowed_ entries | The cell wrapper is fresh. Each entry currently aliases what `evalExpr` returned — see §8 "Known gaps".                                                                                                                                                       |
| `Ident` (`x`)             | borrowed                          | Returns the env binding's RuntimeValue directly.                                                                                                                                                                                                              |
| `Member` (`s.x`)          | borrowed                          | Returns the struct/class field directly.                                                                                                                                                                                                                      |
| `IndexCell` (`c{i}`)      | borrowed                          | Returns the cell entry directly.                                                                                                                                                                                                                              |
| `MemberDynamic`           | borrowed                          | Same as `Member`.                                                                                                                                                                                                                                             |
| Numbers, strings, complex | n/a                               | No buffer.                                                                                                                                                                                                                                                    |

The expression-level rule is purely structural — it depends on the AST
node, not on runtime types. That makes it cheap to consult during
codegen / interpretation.

## 4. Binding seams (where ownership transfers)

A _seam_ is any place a value crosses from one owner to another. Each
seam decides whether to **move** (transfer ownership) or **clone** (mint
a fresh buffer for the new owner).

### 4.1 `Assign` (`x = expr`)

- If `expr` is **owned**: move the value into `x`'s binding. The
  expression result has no other holder; `x` is now its sole owner.
- If `expr` is **borrowed**: deep-clone into `x`'s binding. The original
  owner (env, struct, cell) keeps its buffer; `x` gets its own.
- The previous binding for `x` (if any) becomes garbage and is disposed.
  See §5 for why this isn't always safe under the current
  `setMemberReturn`-based mutators.

### 4.2 `AssignLValue` (`obj.x = expr`, `c{i} = expr`, `a(i) = expr`)

Same rule as `Assign` for the rhs. The destination container (struct,
cell, indexed tensor) takes ownership of the (possibly-cloned) rhs.

### 4.3 Function call entry (caller → callee)

- Each argument expression is evaluated by the caller. Per §3 it is
  either owned (the caller would have disposed it if not consumed) or
  borrowed (still owned by some env binding).
- At the call boundary, the callee's parameter binding takes ownership
  of its own buffer:
  - Owned arg → **move** into the parameter binding (no clone).
  - Borrowed arg → **deep-clone** into the parameter binding.
- Today's implementation deep-clones unconditionally (see §8). The doc
  is the target.

### 4.4 Function call exit (callee → caller)

- Output bindings hold owned values inside the callee. At exit:
  - The callee transfers ownership of each declared-output binding to
    the caller's `outputs[]` array.
  - Remaining locals are disposed by the function-exit dispose pass
    (see §6 for the keep-set rules).
- The caller then handles the returned value at its own seam (Assign,
  ExprStmt, FuncCall arg, etc.) per §4.1–§4.3.

### 4.5 ExprStmt (`expr;`)

The expression result becomes the new `ans` binding. Ownership transfer
follows the same owned-vs-borrowed rule as Assign. The previous `ans`
binding (env-level and `interpreter.ans`) is the prior owner being
released.

## 5. Container mutation (struct field set, cell store, indexed tensor write)

`setRTValueField` (struct/value-class field set) and the corresponding
cell/index store paths construct a _new_ container that shares the
unchanged entries with the previous container by reference (shallow
`new Map(base.fields)` / cell entries reused). The new container is
written back into `x`'s binding via `env.set("x", new_container)`.

This means **the old container is never the unique owner of its
unchanged entries** — the new container references those same buffers.
Disposing the old container would corrupt the new one.

Therefore, the binding-overwrite path in `env.set` must **not**
auto-dispose old values when the old value is a container that may have
been rebuilt this way. The simplest safe rule:

> `env.set` does not dispose the old binding. Disposal of the old
> binding is the responsibility of whoever last _moved_ a value into
> the binding (Assign with an owned rhs, function-exit dispose pass,
> `clear`).

Equivalently: container mutation produces a new container that _takes
over_ the binding without the old container being disposed. The buffers
are still owned, just via a different container instance. The previous
container instance is GC'd when its last reference (the rebuilt
container's shared map) drops.

### 5.1 Future tightening

If we later rewrite `setRTValueField` to deep-clone the unchanged
fields, the old container becomes the unique owner of its old field
values, and dispose-on-overwrite at `env.set` becomes safe (and
profitable for tight loops). Until that change lands, **do not** add
auto-dispose at `env.set`. NaN-poison + double-dispose detection will
flag it immediately if anyone tries.

## 6. Closure caveat (`envCaptured`)

Anonymous functions (`@(x) x + y`) and `@nestedFn` handles take a
_snapshot_ of the enclosing env. The snapshot's vars Map is independent
of the live env, but the values inside are direct references — the
snapshot becomes a co-owner of every wrapper it captured.

`Environment.snapshot()` and `Environment.markChainForNestedHandle()`
mark every visited env as `envCaptured = true`. Any dispose path
operating on a captured env must **either**:

- skip dispose entirely (just clear the map; let the snapshot's
  references keep the values alive for GC), **or**
- prove the specific value being disposed isn't reachable from any
  closure snapshot (not currently practical to check).

The function-exit dispose pass and the `clear` builtin both honor this
flag — no buffer recycling for captured envs.

## 7. The dispose / pool API

Layered. From bottom up:

1. **`disposeFloat64(buf)` / `disposeFloatX(buf)`** in
   [`runtime/alloc.ts`](../../../src/numbl-core/runtime/alloc.ts).
   Pre-conditions:
   - `buf` is **owned** by the caller (no other live reference).
   - `buf` was returned by an `alloc*` / `zeroed*` / `copy*` call (or
     equivalent — typed-array literals that go straight to the buffer
     never round-trip through the pool).
   - `buf` has not been disposed since its last allocation.
     Behavior:
   - Increments dispose tally; NaN-poisons the buffer; pushes onto
     length-keyed bucket subject to per-bucket / total caps.
   - Throws `DoubleDisposeError` if `buf` is already in the pool.

2. **`disposeValue(v)`** in
   [`runtime/utils.ts`](../../../src/numbl-core/runtime/utils.ts).
   Recursive walk of a `RuntimeValue`, calling `disposeFloatX` /
   `disposeFloat64` on every reachable buffer. Pre-condition: `v` and
   every value reachable from it is **uniquely owned by this caller**.
   Skips handle classes, function handles, graphics/dummy handles,
   primitives.

3. **Container-level dispose helpers** (e.g.
   `Environment.disposeAllLocals`, `Environment.disposeLocalsExcept`).
   Apply `disposeValue` across multiple bindings. Honor `envCaptured`.

4. **Public language seams** that produce dispose calls:
   - Function-exit (`callUserFunction` happy + catch paths).
   - `clear` / `clear all` / `clear x`.
   - (Future) Assign with owned rhs — dispose old binding.
   - (Future) Transient-arg dispose at FuncCall site.

## 8. Known gaps vs the doc

The implementation does not yet match every rule in this doc. Tracked
gaps:

| Gap                                                                                                      | Impact                                                | Notes                                                                                                                                            |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `catAlongDim` allocates 1-element wrapper tensors per scalar (`[1 2 3]` → 3 length-1 + 1 length-3 alloc) | tensor-literal cost ~Nx + 1 instead of 1              | the wrappers ARE disposed (gap closed for accounting), but the redundant alloc remains; could be eliminated by a scalar-fast-path in catAlongDim |
| Cell literal entries aliased to evaluator output                                                         | `{a, b}` shares with outer bindings if a/b were ident | currently saved by Assign-clone of the cell after construction                                                                                   |
| `setRTValueField` shallow-copies unchanged fields                                                        | blocks dispose-on-overwrite at env.set                | tighten by deep-clone-on-rebuild later                                                                                                           |

Closed gaps (kept here as historical context — see git log for the
specific commits):

- **Owned `FuncCall` args not disposed at call site** — closed by
  threading `evalArgsTracked` through `evalFuncCall`, plus defensive
  clones in pass-through builtins (`uplus`, `deal`, `struct`,
  `squeeze`, `assignin`).
- **`for` loop's iteration value not disposed at loop end** — closed by
  disposing `iterVal` (when owned) after the For case finishes, plus
  disposing the previous iter-var binding when iterating columns of a
  matrix.
- **`Unary` owned operand intermediates leak** — closed by mirroring
  the `evalBinary` dispose pattern in `evalUnary`.
- **`AssignLValue` always-clones the rhs even when owned** — closed by
  applying the same owned-vs-borrowed decision as `Assign`, plus
  disposing the rhs after a tensor-base index store consumes it.
- **`growTensor2D` leaks the old base buffer when extending past the
  current shape** — closed by disposing the old base in the
  `assignLValue → Index` path; the scalar-promo temp inside `indexStore`
  is also disposed.

When closing a gap, update this table; when opening a new behavior,
extend §3–§7 first, then implement.

## 8a. Copy-on-write plan (in progress)

Today, every borrowed seam (Assign with Ident rhs, function-arg entry,
struct/cell field set, closure snapshot) eager-deep-clones the tensor
buffer. Safe but wasteful — the chunkie repro2 allocates ~78,000 buffers
in a script that touches ~30 distinct tensors. The pool recycles, but
bytes still flow through `copyFloatX` for every clone.

The target model is **buffer-level reference counting**:

> A `RuntimeTensor` wrapper holds a shared `refcount: { count: number }`
> cell. The cell is shared across every wrapper that aliases the same
> `data` / `imag` typed-arrays; `shape` and `_isLogical` stay
> wrapper-local. Borrowed seams produce a fresh wrapper (cheap JS
> object) that copies the data/imag/refcount references and bumps the
> cell. Mutation seams (indexed write, in-place binop output, setter
> body) check `count > 1` and clone-then-mutate when it would corrupt a
> co-owner. `disposeValue` decrements the cell; the buffer goes to the
> pool only on the transition to zero.

Why a boxed cell instead of a per-wrapper count: it lets multiple
wrappers (different shapes — reshape views, transpose marker,
complex/real flag toggles) share one buffer with one count. Slicing
still always allocates (MATLAB semantics: `b = a(2:5)` is a copy), but
reshape and other shape-only transforms become "alloc a wrapper, bump
the cell" instead of "copy 1056 doubles".

### Migration phases

1. **Plumbing (landed).** `RuntimeTensor.refcount?: { count: number }`
   (absent means count=1). `release(t)` decrements; pools the buffer at
   zero. `disposeValue` tensor branch routes through `release`. The
   COW gate inside `storeIntoTensor` clones-then-mutates when the cell
   has `count > 1`. Dormant until Phase 2 starts producing shares — so
   far every wrapper has either no cell or count=1, and every release
   pools immediately, exactly like before.

2. **Convert read seams to share (on hold).** Make `cloneTensor` (the
   tensor branch of `deepCloneValue`) share the buffer + bump the cell
   instead of allocating a fresh one. This single change activates COW
   at every existing borrowed seam at once: function-arg entry, Assign
   / AssignLValue borrowed rhs, `setRTValueField` unchanged-field
   reuse, closure snapshot. The COW gate at `storeIntoTensor` protects
   interpreter mutations.

   **Blocker:** the JIT mutates tensor buffers directly via inline
   `data[i] = val` codegen (`set1r_h` / `set2r_h` / `set3r_h` in
   `jitCodegen.ts`); it doesn't consult the refcount cell. Activating
   Phase 2 globally without updating the JIT corrupts caller views
   anywhere a JIT-compiled function mutates one of its tensor args.
   Other secondary places to audit: any path that reads from an env
   binding and returns the wrapper (e.g. `evalin`) — under sharing,
   the returned wrapper must be a `cloneTensor` (own cell entry) or
   the callee's `release` would decrement the caller's view.

3. **Teach the JIT to honor refcounts.** Two viable approaches:
   - At JIT entry, walk tensor params (top level + container fields)
     and ensure-exclusive each one — break any share by copying the
     buffer. One-shot per call; no per-mutation overhead. Equivalent
     to today's per-call deep-copy cost, so JIT-targeted callsites are
     no worse off than pre-COW. Ship Phase 2 + this in the same commit.
   - Or: insert an inline COW check at each JIT mutation site
     (`if (cell.count > 1) data = cloneBuffer(data, cell)`). Per-write
     overhead but lets the JIT amortize when the share is genuinely
     shared.

   The first is simpler and recovers Phase 2's win for interpreter-
   only paths immediately; the second is a follow-up optimization.

4. **Retire `envCaptured` stickiness.** Once the snapshot path shares,
   the captured wrappers are protected by their refcounts. The
   function-exit dispose pass and `clear` no longer need the
   `envCaptured` / `capturedNames` skip — release lets refcount math
   sort it out. Simplifies `Environment` and the binding-overwrite
   paths in `interpreterExec.ts`.

### Invariants

- The `refcount` cell on a tensor is the count of distinct _owning_
  paths through the runtime state graph that hold any wrapper aliasing
  that buffer. Borrows held during expression evaluation do not count.
- Every `cloneTensor` has exactly one matching `release` later.
  Missing release leaks; extra release double-disposes the buffer
  once the cell hits zero.
- A mutation seam observing `count > 1` MUST clone the buffer (and
  allocate a fresh cell) before writing. No exceptions — that is the
  COW invariant.
- A wrapper handed across a seam is either shared-and-bumped (the
  borrowed case) or moved (the owned case). It is never "borrowed
  without bumping" across a function-call boundary; closures and
  structs all hold owning paths.
- `disposeValue` on a non-tensor (cell, struct, class instance) still
  recurses; the leaves are tensors that go through `release`.

## 9. Worked examples

### 9.1 `a = [1 2 3]`

1. `evalExpr(TensorLit)` → owned `T_lit` (length 3).
2. Assign: rhs is owned → **move** `T_lit` into `a`'s binding.
3. After Assign: `env.a = T_lit`. No clone, no dispose. One buffer for
   the literal, owned by `env.a`.
4. `clear a` → `disposeValue(T_lit)` → buffer pooled. Accounting balanced.

### 9.2 `b = a`

1. `evalExpr(Ident "a")` → borrowed reference to `env.a`'s tensor.
2. Assign: rhs is borrowed → deep-clone into `b`. Two buffers exist:
   `env.a`'s and `env.b`'s, each uniquely owned by its binding.
3. `clear a; clear b` disposes both.

### 9.3 `a = a + 1`

1. `evalExpr(Binary)` → owned fresh `T_sum`. The Binary case borrowed
   `env.a`'s buffer to read, then allocated `T_sum` to write.
2. Assign: rhs is owned → move `T_sum` into `a`. The previous
   `env.a` (the original input tensor) was the input owner; its only
   reference (the env binding) is being overwritten, so it should be
   disposed at this seam.
3. After Assign: `env.a = T_sum`. Old buffer pooled, new buffer owned.

### 9.4 `helper([1 2 3])`

1. `evalExpr(TensorLit)` → owned `T_lit`.
2. Function call entry: arg is owned → move `T_lit` into callee's `x`
   parameter binding.
3. Inside helper, normal evaluation. At fn exit, `x`'s binding is
   disposed (along with other locals not in the keep set).
4. No leaked buffer for the literal.

## 10. Test invariants

Tests should be able to assert:

- After a script ends with `clear all`, `getAllocStats()` satisfies
  `allocCount == disposeCount` for any program that does not exercise
  closures or globals/persistents.
- Running with closures, the program may legitimately leak buffers held
  by the snapshot — but never more than the closure references.
- Double-dispose detection never fires on conformant code (no
  `DoubleDisposeError` in any test run).
- NaN-poison never propagates into an `assert(isequal(...))` (no
  unexpected NaN failures).

A failure of any of these is either a runtime bug or a doc gap. Track
both in §8.
