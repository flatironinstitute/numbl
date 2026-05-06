# Executor Registry

Pluggable strategies for handling AST execution. The interpreter delegates each statement (or run of statements) — and each user-function call — to a registry of _executors_. Each executor implements one strategy (JS-JIT top-level, JS-JIT loop, JS-JIT call, eventually C-JIT specializations, ...). The AST-walking interpreter is the dispatcher's hardcoded last-resort fallback — it is not a registered executor.

## Goals

- **Modularity** — each strategy is a self-contained executor with a uniform interface. Adding a strategy means adding one file, not threading a flag through several layers.
- **Composition** — an outer executor (e.g., JS-JIT loop) can delegate sub-statements to the registry rather than re-implementing them.
- **Runtime-driven dispatch** — selection happens at the moment of execution, with full live runtime info (shapes, exact values, etc.).
- **Mode-driven registration** — `--opt 0/1/e3` selects which executors get registered. The browser bundle simply omits executors whose dependencies it can't satisfy.

## Concepts

- **Executor** — handles a piece of work. Typically one or more consecutive statements, or a user-function call. Implements the `Executor` interface below.
- **Registry** — holds executors, dispatches at runtime, owns per-executor caches.
- **Dispatch context** (`DispatchContext`) — passed to every executor call. Provides env access, the runtime, type-info queries, and runtime callbacks.
- **LoweredStmt** — the dispatcher's pre-propose lowering pass produces a discriminated union (`top-level` / `loop` / `call`) carrying the lowered IR plus pre-computed feasibility flags (`hasReturn`, `hasIO`, `hasBailRisk`, ...). Executors filter on the kind in `propose()`.
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

  // Execute. Returns how many sibling stmts were consumed (stmt-shape)
  // or the call result (call-shape), or a bail signalling that the
  // cache entry should be invalidated and the next executor (or the
  // interpreter) tried.
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
  | { consumed: number } // stmt-shape success
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
    return result.consumed                // success

  // All candidates bailed: AST interpreter, called directly.
  interp.execStmt(siblings[i])
```

`dispatchCall(fn, args, nargout, interp)` is parallel: lowering produces a `CallLoweredStmt`, executors filter on `lowered.kind === "call"`, and the result is the call's return value. When all decline or bail, `dispatchCall` returns `null` and the caller falls through to the interpreter's normal call path.

## Lowering

Lowering is shared across all executors. The dispatcher calls `tryLower(siblings, i, ctx)` once per stmt-dispatch (and `tryLowerCall(fn, args, nargout)` once per call-dispatch) and passes the result to every executor's `propose()`. Lowering produces an IR; it does NOT make codegen-feasibility decisions. "Can this be JIT'd?" lives in each executor's `propose`.

Today's specialized shapes:

- `top-level` — script body (top-level scope, first stmt). Whole script lowered as a synthetic FunctionDef.
- `loop` — for/while loop stmt. Loop body lowered as a synthetic FunctionDef.
- `call` — user-function call. Lowered via `tryLowerCall` from `dispatchCall`.

Stmts with no shape (e.g. a single Assign at non-top-level) skip the proposal loop and go straight to the hardcoded interpreter fallback — there are no executors that consume "raw" stmts.

Lowerings are cached in `LoweringCache`, keyed by (head Stmt or FunctionDef, classification cacheKey). The cache also tracks per-owner type-widening state so a callee invoked with shifting input types converges to a single specialization rather than thrashing.

## Composition

Two patterns, both first-class:

1. **Window composition** — an executor's `run()` may consume more than one stmt. The executor reads ahead via `ctx.peekSibling(offset)` and reports `{ consumed: N }`.
2. **Sub-dispatch** — within `compile` or `run`, an executor calls back into the registry to handle sub-stmts. Sub-dispatch can recurse; the dispatcher guards against re-entering the same executor on the same stmt.

## Bailouts

A runtime invariant violation inside `run` — a complex value where the kernel was specialized for real, a tensor that grew, etc. — raises a Bail. The dispatcher invalidates the cache entry (unless `transient: true`) and tries the next candidate. Since the AST interpreter always matches and never bails, recovery is guaranteed.

### Bail-risk and side effects

A bail happens _during_ `run`, after the executor has already started producing output. If that output included observable side effects (`disp`, `fprintf`, file writes, plot commands, …), re-running on the interpreter would emit them a second time. To prevent this, `propose()` sets `bailRisk: true` when the proposed artifact may fail an invariant after starting to emit output. The lowering pipeline pre-computes `hasIO` and `hasBailRisk` flags on the LoweredStmt; executors typically reject the proposal when both are true.

## Telemetry

Every successful `run()` call invokes `interp.onExecutorFired?.(name, lowered.kind)`. Use this to track which optimizers fire in a session — a session-wide counter, a log entry per dispatch, etc. The hot path uses an `?.` undefined-check and pays nothing when no telemetry consumer is wired up.

## Mode-driven registration

`registerExecutorsForOpt(registry, opt)` in `executors/plugins.ts` is the single switch from an `--opt` level to a set of registered executors. Adding a new mode means extending that function — no other call-site changes.

| `--opt` | Executors registered                                                                     |
| ------- | ---------------------------------------------------------------------------------------- |
| 0       | (none — AST interpreter is the hardcoded fallback)                                       |
| 1       | js-jit-top-level, js-jit-loop, js-jit-call                                               |
| e3      | c-jit-loop, c-jit-fuse, c-jit-chain (Node only — see below; **excludes** the JS-JIT set) |

The C-JIT (e3) executors are wired in via `setCJitRegistrar` rather than imported directly by `plugins.ts`. A Node-only entry point (`cli.ts`, `lib.ts`) imports `executors/cJit/register.ts` for its side effect, which calls `setCJitRegistrar(...)`. The browser worker bundle never imports that file, so `cJit/compile.ts` (which uses `node:fs`/`node:os`/`node:child_process`) stays out of the web build's module graph. Passing `--opt e3` in a context where the registrar was never set throws.

## Files

```
executors/
  index.ts          public surface
  types.ts          Executor / Proposal / RunResult / CostEstimate
  registry.ts       Registry, dispatch, dispatchCall, makeRootContext
  context.ts        DispatchContext, DispatchScope
  cache.ts          ExecutorCache (WeakMap with BAILED sentinel)
  lowering.ts       tryLower / tryLowerCall + LoweringCache
  plugins.ts        registerExecutorsForOpt + setCJitRegistrar
  jsJit/
    topLevelExecutor.ts  js-jit-top-level
    loopExecutor.ts      js-jit-loop
    callExecutor.ts      js-jit-call
    jitTopLevel.ts       classify/lower/generate/run for top-level shape
    jitLoop.ts           classify/lower/generate/run for loop shape
    jitCall.ts           classify/lower/generate/run for call shape
    shared.ts            cross-shape helpers
    lower/               IR lowering (jitLower, jitLowerExpr/Stmt/Types,
                         jitBailSafety, blockAnalysis, scalarEmit)
    codegen/             JS codegen (jitCodegen, jsMultiReduction, ...)
    helpers/             $h runtime (jitHelpers, jitHelpersTensor, ...)
  cJit/
    register.ts          Node-only side-effect that wires the cJit
                         executors into plugins.ts
    loopExecutor.ts      c-jit-loop
    fuseExecutor.ts      c-jit-fuse
    chainExecutor.ts     c-jit-chain
    chainPass.ts         stmt-list pass that builds c-jit-chain Synth nodes
    compile.ts           cc-invoke + koffi load (Node only)
    codegen.ts           IR -> C source (loop shape)
    fuseCodegen.ts       IR -> C source (fuse shape)
    chainCodegen.ts      IR -> C source (chain shape)
    elemwiseCodegen.ts   shared elementwise C-expression emission
    elemwiseStructural.ts elementwise AST classifier shared by fuse/chain
    fuseAnalyze.ts       fuse classification (browser-safe)
    builtins.ts          builtin set permitted under c-jit
    whitelist.ts         feasibility filter
```
