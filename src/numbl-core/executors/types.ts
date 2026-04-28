/**
 * Executor registry — core types.
 *
 * See docs/developer_reference/executors.md for the design overview.
 *
 * The interpreter delegates each statement (or run of statements) — and
 * each user-function call — to a registry of Executors. Each executor
 * implements one strategy (interpreter, JS-JIT, C-kernel, ...). On each
 * dispatch, every executor may submit a Proposal; the dispatcher picks
 * the lowest-cost. Stmt-shape and call-shape work share the same
 * Executor interface — they're discriminated via the lowered statement
 * passed to propose().
 */

import type { DispatchContext } from "./context.js";
import type { LoweredStmt } from "./lowering.js";

/** Estimated cost of using this executor for the proposed work.
 *  Numbers can be very rough at first; the dispatcher's policy is
 *  refined separately from executors. */
export interface CostEstimate {
  /** One-time compile cost on cache miss. */
  compileMs: number;
  /** Per-call dispatch overhead (marshaling, frame setup, ...). */
  perCallNs: number;
  /** Estimated work done by the compiled artifact for the proposed
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
  /** Stmt-shape success — claims `consumed` consecutive sibling stmts
   *  starting at the current head. Registered executors don't produce
   *  control signals (break/continue/return); only the hardcoded
   *  interpreter fallback in `Registry.dispatch` does. */
  | { consumed: number }
  /** Call-shape success — used by executors that handle a
   *  CallLoweredStmt. The dispatcher's call entry point
   *  (`dispatchCall`) returns this `result` to the caller. */
  | { result: unknown }
  | {
      bail: BailReason;
      /** When true, the bail is not cached: future dispatches re-enter
       *  the executor as if cache had never been touched. Use for
       *  shim/wrapper executors whose internal classify logic must
       *  re-run on every call (e.g., because the wrapped layer caches
       *  on its own keying scheme). Default false. */
      transient?: boolean;
    };

/** An executor's bid to handle the current dispatch. The dispatcher
 *  picks the lowest-cost proposal; the executor's own `data` flows
 *  through to compile() and run() unchanged. */
export interface Proposal<D> {
  /** Opaque per-executor data; passed to compile() and run(). */
  data: D;
  cost: CostEstimate;
  /** Whether this specific proposal's compiled artifact may fail an
   *  invariant mid-execution (and thus need re-running by a fallback).
   *  The dispatcher filters bail-risk proposals out of contexts marked
   *  `requireNoBail`. Per-proposal because a single executor may
   *  produce both bail-risky and bail-safe proposals depending on the
   *  inputs it sees. */
  bailRisk: boolean;
}

export interface Executor<D = unknown, C = unknown> {
  /** Stable identifier for logging, cache keys, and test selection. */
  readonly name: string;

  /** Submit a bid to handle this stmt. Runs on every dispatch — must
   *  be cheap.
   *
   *  Receives the lowered statement produced by the dispatcher's
   *  pre-propose lowering pass. The `kind` field discriminates: a
   *  specialized shape (e.g. `"top-level"`) carries a lowered IR
   *  plus pre-computed feasibility flags; the fallback `"stmt"` kind
   *  carries the raw AST stmt for executors that classify from the
   *  AST directly.
   *
   *  Codegen-feasibility decisions (display mode, IO+bail-risk, etc.)
   *  belong here — the lowering pipeline produces an IR; the
   *  executor decides whether to commit. For lookahead across
   *  multiple stmts, use `ctx.peekSibling(offset)` or `ctx.siblings`.
   *
   *  Returns null to decline. */
  propose(lowered: LoweredStmt, ctx: DispatchContext): Proposal<D> | null;

  /** Stable cache key projected from the proposal data. Drops
   *  volatile bits (e.g., exact scalar values; tensor shape if
   *  codegen is shape-agnostic) so unrelated runs of the same code
   *  reuse compiled artifacts. */
  cacheKey(data: D): string;

  /** Compile to a runnable artifact. Called only on cache miss.
   *  Cached under (executor, headStmt, cacheKey). */
  compile(data: D, ctx: DispatchContext): C;

  /** Execute. Returns the number of consumed sibling stmts on success,
   *  or a Bail signalling the cache entry should be invalidated and
   *  the next-best candidate tried. */
  run(compiled: C, data: D, ctx: DispatchContext): RunResult;
}

// Function-call dispatch is unified with stmt-level dispatch: a call
// becomes a `CallLoweredStmt` (kind: "call") that the same Executor
// interface consumes. The dispatcher exposes a separate entry point
// (`Registry.dispatchCall`) for the (fn, args, nargout) input shape,
// but it iterates the same `executors` array — a regular Executor
// filters on `lowered.kind === "call"` and returns `{ result }` from
// run() on success.
