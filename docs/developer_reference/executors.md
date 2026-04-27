# Executor Registry

Pluggable strategies for handling AST execution. The interpreter delegates each statement (or run of statements) to a registry of _executors_ — interpreter, JS-JIT, C-kernel, etc. The AST-walking interpreter is itself a registered executor: always available, lowest priority, the last-resort fallback. Anything not handled by another executor falls through to it transparently.

This subsystem replaces the scattered `tryE2Assign`, `tryJitFor`, `tryJitWhile`, `tryJitTopLevel`, `tryE2ScalarFn` hooks with one uniform dispatcher.

## Goals

- **Modularity** — each strategy is a self-contained executor with a uniform interface. Adding a strategy means adding one file, not threading a flag through several layers.
- **Testability** — executor correctness is exactly "matches the interpreter for the same input." A single mechanical contract, mechanically enforceable.
- **Composition** — an outer executor (e.g., loop JIT) can delegate sub-statements to the registry rather than re-implementing them.
- **Runtime-driven dispatch** — selection happens at the moment of execution, with full live runtime info (shapes, exact values, ref counts).
- **Mode-driven registration** — `--opt 0/1/e1/e2` selects which plugins get registered. The browser bundle simply omits plugins whose dependencies it can't satisfy (e.g., the C-kernel plugin). All registered executors are by definition available — there is no per-call platform check.

## Concepts

- **Executor** — handles a piece of work. Typically one or more consecutive statements. Implements the `Executor` interface below.
- **Plugin** — a TS module that registers one or more executors. Different `--opt` modes import different plugins; the browser bundle excludes plugins it can't support.
- **Registry** — holds executors, dispatches at runtime, owns per-executor caches.
- **Dispatch context** (`Ctx`) — passed to every executor call. Provides env access, the runtime, type-info queries, sub-dispatch into the registry, and runtime callbacks.
- **TypeInfo** — unified rich type information about a value or expression. Successor to `JitType`. Includes scalar kind, complex/logical, exact value if known, tensor shape if known, etc. Each executor reads what it cares about.
- **CacheKey** — per-executor projection of TypeInfo into a stable string. Drops volatile bits (e.g., exact scalar values; tensor shape if codegen is shape-agnostic) so unrelated runs of the same code reuse compiled artifacts.
- **Cost estimate** — three numbers the executor returns from its match: estimated compile time (ms, paid once on cache miss), per-call overhead (ns, paid every dispatch — koffi marshaling, frame setup), and run time (ns, the actual work). Estimates can be very rough at first; the dispatcher's policy can be refined as we learn what's accurate.
- **Bail-risk** — whether the executor's compiled artifact can fail an invariant mid-execution and need to be re-run by the interpreter. Executors that compile around runtime-type assumptions (JS-JIT) are bail-risk; the AST interpreter and pure C-kernel paths are not. The dispatcher refuses bail-risk executors when the surrounding context has observable side effects that mustn't repeat (see Bailouts).

## Executor interface

```ts
interface Executor<M = unknown, C = unknown> {
  name: string;

  // Whether this executor's compiled artifact can fail an invariant
  // mid-execution. Used by the dispatcher's eligibility filter when
  // the surrounding context has non-rerunnable side effects.
  bailRisk: boolean;

  // Try to match starting at siblings[i]. Runs every dispatch — must
  // be cheap. Returns null to decline. On match, returns a Match
  // (capturing how many siblings to consume, the input TypeInfo
  // snapshot, lowered IR / classification, anything compile() and
  // run() need) and a CostEstimate.
  match(
    siblings: Stmt[],
    i: number,
    ctx: Ctx
  ): { match: M; cost: CostEstimate } | null;

  // Stable cache key projected from the match.
  cacheKey(match: M): string;

  // Compile to a runnable artifact. Called only on cache miss. Cached
  // under the matched stmt's identity + cacheKey.
  compile(match: M, ctx: Ctx): C;

  // Execute. Returns how many siblings were consumed, or a Bail
  // signalling that the cache entry should be invalidated and the
  // next executor (or the interpreter) tried.
  run(compiled: C, match: M, ctx: Ctx): RunResult;
}

interface CostEstimate {
  compileMs: number; // paid once per cache entry
  perCallNs: number; // dispatch overhead (marshaling, frame setup)
  runNs: number; // estimated work for this match's input sizes
}

type RunResult = { consumed: number } | { bail: BailReason };
```

Cost numbers can be rough at first. The dispatcher's initial policy:

- Among matching executors, pick the lowest `perCallNs + runNs` (per-call work).
- Compile cost (`compileMs`) is amortized over the cache lifetime; on cache hit it doesn't matter, and on cache miss the policy treats it as paid up-front. We may add a "skip executors with high `compileMs` if this stmt is cold" heuristic later, but the first pass simply pays it.

Refining the policy doesn't require executor changes — they keep returning the same three numbers.

## Dispatch flow

```
execStmt(siblings, i, ctx):
  candidates = []
  for executor in registry:
    if ctx.requireNoBail and executor.bailRisk: continue
    m = executor.match(siblings, i, ctx)
    if m is null: continue
    candidates.push({ executor, match: m.match, cost: m.cost })

  // Pick by lowest per-call cost. The interpreter is always a
  // candidate (it matches every stmt with consumed=1) — and is the
  // last-resort because every other executor is faster on its
  // declared domain or it wouldn't be registered.
  candidates.sort by (cost.perCallNs + cost.runNs)

  for c in candidates:
    key = c.executor.cacheKey(c.match)
    compiled = cache.get(c.executor, siblings[i], key)
              ?? cache.set(..., c.executor.compile(c.match, ctx))
    result = c.executor.run(compiled, c.match, ctx)
    if result.bail:
      cache.invalidate(...)
      continue   // try next-best candidate (eventually the interpreter)
    advance i by result.consumed
    return
```

The AST interpreter executor always matches every statement, has `bailRisk: false`, and reports a high `runNs` — so it loses to specialized executors on their domain but always wins as a last resort. It cannot itself be filtered out by `requireNoBail`.

## Composition

Two patterns, both first-class:

1. **Window composition** — `match` consumes 1..N consecutive siblings. The chain-c-kernel executor reads ahead, classifies, and returns a Match capturing the run length. After `run` returns `{ consumed: 4 }`, the dispatcher advances four stmts.
2. **Sub-dispatch** — within `compile` or `run`, an executor calls `ctx.dispatch(subStmts)` to delegate sub-work to the registry. The JS-JIT loop executor uses this to handle stmts inside its loop body without re-implementing them. Sub-dispatch can recurse; the dispatcher guards against re-entering the same executor on the same stmt.

## TypeInfo

Computed lazily. When an executor's `match` asks for `ctx.typeOf(name)`, the dispatcher does the env lookup and inference once and memoizes for the rest of that dispatch.

The shape (sketched — full definition lives in the implementation):

```ts
type TypeInfo =
  | { kind: "number"; exact?: number; sign?: 1|0|-1; isInteger?: boolean; isFinite?: boolean }
  | { kind: "boolean"; value?: boolean }
  | { kind: "string"; value?: string }
  | { kind: "char"; value?: string }
  | { kind: "tensor"; shape?: ReadonlyArray<number>; elemType: ScalarType;
                       isComplex: boolean; isLogical: boolean }
  | { kind: "struct"; fields?: Record<string, TypeInfo> }
  | { kind: "cell"; length?: number; elem?: TypeInfo }
  | { kind: "function"; ... }
  | { kind: "unknown" };
```

The existing `JitType` becomes a strict subset; migration replaces uses of `JitType` with `TypeInfo` and populates the new fields opportunistically. `cacheKey` defaults drop `exact`, `value`, and tensor `shape`; executors that need any of those for codegen include them explicitly.

## Bailouts

A runtime invariant violation inside `run` — a complex value where the kernel was specialized for real, a tensor that grew, etc. — raises a `Bail`. The dispatcher invalidates the cache entry and tries the next candidate. Since the interpreter executor always matches and never bails, recovery is guaranteed.

This generalizes today's `JitBailToInterpreter` (JS-JIT) and `E2_BAILED` sentinel (e2).

### Bail-risk and side effects

A bail happens _during_ `run`, after the executor has already started producing output. If that output included observable side effects (`disp`, `fprintf`, file writes, plot commands, …), re-running on the interpreter would emit them a second time. To prevent this, the dispatcher carries a `requireNoBail` flag in `Ctx`:

- A top-level dispatch sets `requireNoBail = false`. Bail-risk executors are eligible.
- An executor whose compiled artifact contains observable side effects must declare its match with `requireNoBailInChildren: true`. When the dispatcher recurses into its child stmts (sub-dispatch), it sets `requireNoBail = true` on the child Ctx — only no-bail executors (the interpreter and pure C-kernel paths) are eligible there.
- `bailRisk` is also a property of an executor's emitted code at use time. JS-JIT executors that emit `disp` calls inside their compiled body are themselves `requireNoBailInChildren: true` for any nested compilation.

This generalizes today's `irHasIO` + `irHasBailRisk` check ([jit/index.ts:97-103](../../src/numbl-core/jit/index.ts#L97-L103)) — a function with both is rejected for JIT — into a uniform mechanism.

## Testing contract

The interpreter executor is the source of truth. Every other executor's correctness is exactly:

```
runWithRegistry(stmts, env, registry) ≡ runWithInterpreterOnly(stmts, env)
```

The harness `executorEquivalence(executor, corpus)` runs each `(stmts, env)` case through registry-with-only-interpreter and again through registry-with-executor-included, and asserts env-after equivalence. New executors add tests by adding `(stmts, env)` cases that exercise their match conditions — no hand-rolled expected outputs. The interpreter answers the question.

## Migration

Existing `--opt` flags become plugin-registration choices. The CLI parses `--opt` and calls the matching `register*Plugin(registry)` functions; from then on, dispatch is mode-agnostic.

| `--opt` | Plugins registered                                                |
| ------- | ----------------------------------------------------------------- |
| 0       | interpreter                                                       |
| 1       | interpreter, js-jit-fn, js-jit-loop, js-jit-toplevel              |
| e1      | interpreter, js-jit-\* + c-kernel-chain-inner, c-kernel-scalar-fn |
| e2      | interpreter, chain-c-kernel, loop-c-kernel, scalar-fn-c-kernel    |

The browser bundle never imports the C-kernel plugins; nothing else needs to know. CLI behavior is unchanged.

The port runs in increments. Each step is shippable:

1. Land the registry, the executor interface, the `TypeInfo` object, and the AST interpreter executor. No behavior change — existing try-hooks stay; the registry is wired but only the interpreter is registered.
2. Port `tryE2Assign` (chain + reduction) into a registered executor.
3. Port `tryE2Loop` (loop C kernel).
4. Port the JS-JIT loop, function, and top-level executors.
5. Port `tryE2ScalarFn` and the e1 paths.
6. Remove the old try-hooks from `interpreterExec.ts`.

## Files

(populated as the implementation lands)
