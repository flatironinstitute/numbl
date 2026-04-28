/**
 * Executor registry — core types.
 *
 * See docs/developer_reference/executors.md for the design overview.
 *
 * The interpreter delegates each statement (or run of statements) to a
 * registry of Executors. Each executor implements one strategy
 * (interpreter, JS-JIT, C-kernel, ...). Selection happens at runtime
 * based on cost estimates returned from `match`.
 */

import type { Stmt } from "../parser/types.js";
import type { ControlSignal, FunctionDef } from "../interpreter/types.js";
import type { DispatchContext } from "./context.js";

/** Estimated cost of using this executor for the matched work. Numbers
 *  can be very rough at first; the dispatcher's policy is refined
 *  separately from executors. */
export interface CostEstimate {
  /** One-time compile cost on cache miss. */
  compileMs: number;
  /** Per-call dispatch overhead (marshaling, frame setup, ...). */
  perCallNs: number;
  /** Estimated work done by the compiled artifact for this match's
   *  input sizes. */
  runNs: number;
}

/** Reason a `run` invocation could not complete. The dispatcher
 *  invalidates the cache entry and tries the next-best candidate. */
export interface BailReason {
  message: string;
  cause?: unknown;
}

export type RunResult =
  | { consumed: number; signal?: ControlSignal | null }
  | {
      bail: BailReason;
      /** When true, the bail is not cached: future dispatches re-enter
       *  the executor as if cache had never been touched. Use for
       *  shim/wrapper executors whose internal classify logic must
       *  re-run on every call (e.g., because the wrapped layer caches
       *  on its own keying scheme). Default false. */
      transient?: boolean;
    };

export interface MatchResult<M> {
  /** Opaque per-executor data; passed to compile() and run(). */
  match: M;
  cost: CostEstimate;
  /** When true, the compiled artifact emits observable side effects
   *  (`disp`, `fprintf`, file writes, ...) that mustn't repeat. The
   *  dispatcher sets `requireNoBail = true` on any sub-dispatch the
   *  executor performs while running. */
  requireNoBailInChildren?: boolean;
}

export interface Executor<M = unknown, C = unknown> {
  /** Stable identifier for logging, cache keys, and test selection. */
  readonly name: string;

  /** Whether this executor's compiled artifact can fail an invariant
   *  mid-execution (and thus need to be re-run by a fallback). The
   *  dispatcher filters bail-risk executors out of contexts marked
   *  `requireNoBail`. */
  readonly bailRisk: boolean;

  /** Runs every dispatch — must be cheap. Receives just the current
   *  stmt; for executors that need to look across multiple stmts
   *  (chain fusion, whole-script JIT), use `ctx.peekSibling(offset)`
   *  or `ctx.remainingSiblings()`. Returns null to decline; on
   *  success, returns the match data plus a cost estimate. */
  match(stmt: Stmt, ctx: DispatchContext): MatchResult<M> | null;

  /** Stable cache key projected from the match. Drops volatile bits
   *  (e.g., exact scalar values; tensor shape if codegen is shape-
   *  agnostic) so unrelated runs of the same code reuse compiled
   *  artifacts. */
  cacheKey(match: M): string;

  /** Compile to a runnable artifact. Called only on cache miss.
   *  Cached under (executor, headStmt, cacheKey). */
  compile(match: M, ctx: DispatchContext): C;

  /** Execute. Returns the number of consumed sibling stmts on success,
   *  or a Bail signalling the cache entry should be invalidated and
   *  the next-best candidate tried. */
  run(compiled: C, match: M, ctx: DispatchContext): RunResult;
}

// ── Function-call dispatch ──────────────────────────────────────────────
//
// Parallel to the stmt dispatch path. Function-call executors fire from
// `callUserFunction` (during expression evaluation), not from the
// interpreter's stmt loop. They handle JIT-compiled user-function calls
// (`tryJitCall`, `tryE2ScalarFn`).
//
// Call executors take the Interpreter directly rather than a
// DispatchContext. The ctx machinery (typeCache, reentrancy guard) is
// stmt-dispatch-specific; call executors only need env access via
// `interp`. Skipping ctx avoids a per-call Map+Set allocation, which
// dominates on user-function-call-heavy workloads (chunkie helmholtz).

export type CallRunResult =
  | { result: unknown }
  | {
      bail: BailReason;
      /** Same semantics as RunResult.transient. */
      transient?: boolean;
    };

export interface CallMatchResult<M> {
  /** Opaque per-executor data; passed to runCall(). */
  match: M;
  cost: CostEstimate;
}

// Forward declare the Interpreter type without creating a hard import
// cycle (types.ts is imported widely).
type Interpreter = import("../interpreter/interpreter.js").Interpreter;

/** Call executors don't have the registry's compile/cache layer.
 *  Function-call dispatch is per-call hot — a registry-level cache
 *  would add WeakMap+Map lookups per call without benefit, since the
 *  wrapped layers (`tryJitCall`, `tryE2ScalarFn`) already cache
 *  internally per (FunctionDef, argType-signature). Call executors
 *  match-and-run; any compile state is the executor's own
 *  responsibility. */
export interface CallExecutor<M = unknown> {
  readonly name: string;
  readonly bailRisk: boolean;

  /** Runs on every user-function call — must be cheap. Returns null
   *  to decline. */
  matchCall(
    fn: FunctionDef,
    args: unknown[],
    nargout: number,
    interp: Interpreter
  ): CallMatchResult<M> | null;

  /** Run. Returns the function's result (may be a value or an array
   *  of values for multi-output) on success, or a Bail. */
  runCall(
    match: M,
    fn: FunctionDef,
    args: unknown[],
    nargout: number,
    interp: Interpreter
  ): CallRunResult;
}
