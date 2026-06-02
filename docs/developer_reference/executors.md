# Executor Registry

Pluggable strategies for handling AST execution. The interpreter delegates each statement (or run of statements) — and each user-function call — to a registry of _executors_. Each executor implements one strategy (JS-JIT top-level/loop/call at `--opt 1`; C-JIT top-level/loop/call at `--opt 2`). The AST-walking interpreter is the dispatcher's hardcoded last-resort fallback — it is not a registered executor.

## Goals

- **Modularity** — each strategy is a self-contained executor with a uniform interface. Adding a strategy means adding one file, not threading a flag through several layers.
- **Composition** — an outer executor (e.g., JS-JIT loop) can delegate sub-statements to the registry rather than re-implementing them.
- **Runtime-driven dispatch** — selection happens at the moment of execution, with full live runtime info (shapes, exact values, etc.).
- **Mode-driven registration** — `--opt 0/1/2` selects which executors get registered. The browser bundle simply omits executors whose dependencies it can't satisfy (the C-JIT path needs `cc` + `koffi`, Node only).

## Concepts

- **Executor** — handles a piece of work. Typically one or more consecutive statements, or a user-function call. Implements the `Executor` interface below.
- **Registry** — holds executors, dispatches at runtime, owns per-executor caches.
- **Dispatch context** (`DispatchContext`) — passed to every executor call. Provides env access, the runtime, type-info queries, and runtime callbacks.
- **LoweredStmt** — the dispatcher's pre-propose lowering pass produces a discriminated union (`top-level` / `loop` / `call` / `synth`) carrying the lowered IR plus pre-computed feasibility flags (`hasReturn`, `hasIO`, ...). Executors filter on the kind in `propose()`.
- **CacheKey** — per-executor projection of the proposal data into a stable string. Drops volatile bits (e.g., exact scalar values) so unrelated runs of the same code reuse compiled artifacts.
- **Cost estimate** — three numbers the executor returns with each proposal: estimated compile time (ms, paid once on cache miss), per-call overhead (ns, paid every dispatch), and run time (ns, the actual work).
- **Bail-risk** — whether a specific proposal's compiled artifact may fail an invariant mid-execution. Per-proposal because a single executor can produce both bail-risky and bail-safe proposals depending on its inputs. The dispatcher refuses bail-risk proposals when the surrounding context has observable side effects that mustn't repeat.

## Executor interface

```ts
interface Executor<D = unknown, C = unknown> {
  readonly name: string;

  // Submit a bid to handle this stmt. Runs every dispatch — must be
  // cheap. Receives the `LoweredStmt` from the dispatcher's pre-propose
  // lowering pass; filter on `lowered.kind`. For lookahead, use
  // ctx.peekSibling(offset) or ctx.siblings.
  // Returns null to decline; on success, returns a Proposal.
  propose(lowered: LoweredStmt, ctx: DispatchContext): Proposal<D> | null;

  // Stable cache key projected from the proposal data.
  cacheKey(data: D): string;

  // Compile to a runnable artifact. Called only on cache miss. Cached
  // under (executor.name, owner, cacheKey) where owner is the head Stmt
  // (stmt-shape) or the FunctionDef (call-shape).
  compile(data: D, ctx: DispatchContext): C;

  // Execute. Stmt-shape success returns `{ ok: true }` and the
  // dispatcher advances by exactly one stmt. Call-shape success
  // returns `{ result }`. A bail signals the cache entry should be
  // invalidated and the next executor (or the interpreter) tried.
  run(compiled: C, data: D, ctx: DispatchContext): RunResult;
}

interface Proposal<D> {
  data: D; // opaque per-executor payload, flows through compile/run
  cost: CostEstimate;
  bailRisk: boolean;
}

interface CostEstimate {
  compileMs: number; // paid once per cache entry
  perCallNs: number; // dispatch overhead (marshaling, frame setup)
  runNs: number; // estimated work for the proposed input sizes
}

type RunResult =
  | { ok: true } // stmt-shape success (advances by one stmt)
  | { result: unknown } // call-shape success
  | { bail: BailReason; transient?: boolean };
```

## Dispatch flow

```
dispatch(siblings, i, ctx):
  lowered = tryLower(siblings, i, ctx)   // null → no specialized shape
  if lowered is null:
    interp.execStmt(siblings[i])         // hardcoded fallback
    return

  candidates = []
  for executor in registry:
    p = executor.propose(lowered, ctx)
    if p is null: continue
    if ctx.requireNoBail and p.bailRisk: continue
    candidates.push({ executor, data: p.data, cost: p.cost })

  // Pick best by perCallNs + runNs; backups in cost order.
  for c in [best, ...backups]:
    key = c.executor.cacheKey(c.data)
    compiled = cache.get(c.executor.name, head, key)
              ?? cache.set(..., c.executor.compile(c.data, ctx))
    result = c.executor.run(compiled, c.data, ctx)
    if result.bail:
      cache.invalidate(...)               // unless transient
      continue                            // try next candidate
    return                                // success (advances by one stmt)

  // All candidates bailed: AST interpreter, called directly.
  interp.execStmt(siblings[i])
```

`dispatchCall(fn, args, nargout, interp)` is parallel: lowering produces a `CallLoweredStmt`, executors filter on `lowered.kind === "call"`, and the result is the call's return value. When all decline or bail, `dispatchCall` returns `null` and the caller falls through to the interpreter's normal call path.

## Lowering

Lowering is shared across all executors. The dispatcher calls `tryLower(siblings, i, ctx)` once per stmt-dispatch (and `tryLowerCall(fn, args, nargout)` once per call-dispatch) and passes the result to every executor's `propose()`. Lowering produces an IR; it does NOT make codegen-feasibility decisions. "Can this be JIT'd?" lives in each executor's `propose`.

Today's specialized shapes:

- `top-level` — script body (top-level scope, first stmt). Whole script lowered as a synthetic FunctionDef. Lowered separately via `tryLowerTopLevel` and dispatched through `Registry.tryRunWholeScope` before the per-stmt loop runs, not through `tryLower`.
- `loop` — for/while loop stmt. Loop body lowered as a synthetic FunctionDef. A capture-free function-handle _input_ (defined before the loop, e.g. `f = @(t) …;` or `f = @name;`) would otherwise be an opaque `function_handle` the JIT can't type. Classification (`classifyLoop`) detects these and, instead of taking them as runtime inputs, emits them as `ConstHandle`s that the loop executors inline as in-scope `<name> = @…` assignments prepended to the synthetic body — reducing the boundary case to the supported in-scope handle case, identically on both backends. The inliner (`handleInline.ts`) reuses the handle's recorded defining AST + env (stored on `RuntimeFunction`) and only fires when capture-free and same-file. It also declines if any free body name (a name that was a function/undefined at definition, not a captured variable) collides with a name that will be in scope at the relocation site (the loop's inputs, assigned locals, or `for` variable) — otherwise re-lowering would re-resolve that name to the loop variable, silently turning `@(t) sq(t)` from a function call into an array index.
- `call` — user-function call. Lowered via `tryLowerCall` from `dispatchCall`.
- `synth` — a `Synth` AST stmt produced by a registered AST stmt-list transformer that collapses a contiguous run of stmts into a unit. The `tag` field discriminates among transformers. (No transformer is registered today; the shape remains in the union for future use.)

Stmts with no shape (e.g. a single Assign at non-top-level) skip the proposal loop and go straight to the hardcoded interpreter fallback — there are no executors that consume "raw" stmts.

Lowerings are cached in `LoweringCache`, keyed by (head Stmt or FunctionDef, classification cacheKey). The cache also tracks per-owner type-widening state so a callee invoked with shifting input types converges to a single specialization rather than thrashing.

## Composition

Two patterns, both first-class:

1. **AST stmt-list transformers (Synth)** — each executor handles exactly one stmt, but a registered AST transformer can rewrite a contiguous run of stmts into a single `Synth` stmt before dispatch. The matching `synth`-shape executor then handles that one rewritten stmt as a unit. (No transformer is registered today.) Lookahead at `propose()` time is also available via `ctx.peekSibling(offset)` and `ctx.siblings`.
2. **Sub-dispatch** — within `compile` or `run`, an executor calls back into the registry to handle sub-stmts. Sub-dispatch can recurse; the dispatcher guards against re-entering the same executor on the same stmt.

## Bailouts

The dispatcher supports bailing: a `{ bail }` from `run` invalidates the cache entry (unless `transient: true`) and falls through to the next candidate, and since the AST interpreter always matches and never bails, recovery is guaranteed.

In practice the current JIT executors (`src/numbl-core/jit`) **reject statically** rather than bailing mid-run: the compiler either lowers a spec cleanly or throws `UnsupportedConstruct` / `JitTypeError` at `compile()` time, which the executor catches and declines from. So once `compile()` returns an artifact it runs to completion, and the JIT executors report `bailRisk: false`. The `{ bail }` path is still used for hard runtime errors surfaced by the emitted code (re-routed to the interpreter), but there is no per-operation type-guard bailout.

### Bail-risk and side effects

`bailRisk: true` marks a proposal whose artifact might fail an invariant _after_ it has already emitted observable output (`disp`, `fprintf`, file writes, plot commands, …) — re-running on the interpreter would emit it twice. The dispatcher filters such proposals out of `requireNoBail` contexts. Because the current JIT declines statically, its executors set `bailRisk: false`; the field and the dispatcher filtering remain for executors that need them.

## Telemetry

Every successful `run()` call invokes `interp.onExecutorFired?.(executor.name, kind)`. For stmt-shape and whole-scope success the kind is `lowered.kind` (`top-level`, `loop`, `call`, `synth` — the variants of the `LoweredStmt` union); for call-shape success (via `dispatchCall`) the kind is hardcoded to `"call"`. Use this to track which optimizers fire in a session — a session-wide counter, a log entry per dispatch, etc. The hot path uses an `?.` undefined-check and pays nothing when no telemetry consumer is wired up.

## Mode-driven registration

`registerExecutorsForOpt(registry, opt)` in `executors/plugins.ts` is the single switch from an `--opt` level to a set of registered executors. Adding a new mode means extending that function — no other call-site changes.

| `--opt` | Executors registered                                                                                       |
| ------- | ---------------------------------------------------------------------------------------------------------- |
| 0       | (none — AST interpreter is the hardcoded fallback)                                                         |
| 1       | jit-top-level, jit-loop, jit-call (JS-JIT)                                                                 |
| 2       | cjit-top-level, cjit-loop, cjit-call (C-JIT) + the JS-JIT set as fallback (Node only — needs `cc`+`koffi`) |

At `--opt 2` both the C-JIT and JS-JIT executors are registered; they compete via the cost model. C-JIT proposes only where it can marshal the argument types and its lower per-call/run cost wins; everything else falls through to JS-JIT, then the interpreter.

The C-JIT native compile/load step (`cc` + `koffi` `dlopen`) lives behind a browser-safe stub: `executors/jit/compileC.ts` exposes `compileAndLoadC` whose implementation is `null` until a Node entry point calls `setCompileAndLoadCImpl(...)`. `cli.ts` does this at bootstrap via `registerNodeCompileC()` (from `executors/jit/compileC.node.ts`, which uses `node:fs`/`node:child_process`). The browser worker bundle never imports `compileC.node.ts`, so those Node modules stay out of the web build's graph; if `--opt 2` runs without the impl wired (or without `koffi`), the C-JIT executors decline and dispatch collapses to JS-JIT.

## Files

The executor _registry_ and the per-shape JIT/C-JIT _executors_ live under `executors/`; the JIT _compiler_ they call (lowering → IR → JS/C codegen, builtins, runtime snippets) is the self-contained, in-tree subsystem under [`src/numbl-core/jit`](jit/overview.md) — no external dependency.

```
executors/
  index.ts          public surface
  types.ts          Executor / Proposal / RunResult / CostEstimate
  registry.ts       Registry, dispatch, dispatchCall, makeRootContext
  context.ts        DispatchContext
  cache.ts          ExecutorCache (WeakMap with BAILED sentinel)
  lowering.ts       tryLower / tryLowerCall + LoweringCache
  classification.ts top-level / call classification (backend-independent)
  handleInline.ts   inlines capture-free function-handle inputs into loop bodies
  plugins.ts        registerExecutorsForOpt (the --opt → executors switch)
  jit/
    topLevelExecutor.ts  jit-top-level   (JS-JIT, opt 1)
    loopExecutor.ts      jit-loop        (JS-JIT, opt 1)
    callExecutor.ts      jit-call        (JS-JIT, opt 1)
    cJitTopLevelExecutor.ts cjit-top-level (C-JIT, opt 2)
    cJitLoopExecutor.ts     cjit-loop      (C-JIT, opt 2)
    cJitCallExecutor.ts     cjit-call      (C-JIT, opt 2)
    session.ts           per-context Workspace + Lowerer (the spec cache)
    typeAdapter.ts       numbl JitType → compiler Type (JS path)
    valueAdapter.ts      RuntimeValue ↔ emit-JS value shape
    typeAdapterC.ts      compiler Type → C decl (C path; `compilerTypeToCDecl`)
    valueAdapterC.ts     RuntimeValue ↔ C ABI marshaling (koffi)
    hostHelpers.ts       the `$h` object the emitted spec receives
    compileC.ts          browser-safe compile/load stub
    compileC.node.ts     cc-invoke + koffi dlopen (Node only)
```

The compiler invoked by these executors (`compileSpec` for JS, `compileSpecC` for C, plus `Workspace`/`Lowerer`) is imported from `src/numbl-core/jit/index.ts`; see [jit/overview.md](jit/overview.md).
