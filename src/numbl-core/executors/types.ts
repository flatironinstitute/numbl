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
import type { ControlSignal } from "../interpreter/types.js";
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
  | { bail: BailReason };

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

  /** Runs every dispatch — must be cheap. Returns null to decline.
   *  On success, returns the match data plus a cost estimate. */
  match(
    siblings: readonly Stmt[],
    i: number,
    ctx: DispatchContext
  ): MatchResult<M> | null;

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
