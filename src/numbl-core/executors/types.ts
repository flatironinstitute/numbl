/**
 * Executor registry — core types.
 *
 * See docs/developer_reference/executors.md for the design overview.
 *
 * The interpreter delegates whole-script bodies, loops, and user-function
 * calls to a registry of Executors. Each executor handles one lowered shape
 * (`top-level`, `loop`, or `call`). The registry tries the executors
 * registered for a shape in order; the first that proposes and runs
 * without bailing wins, and the AST interpreter handles the rest.
 */

import type { Interpreter } from "../interpreter/interpreter.js";
import type { LoweredStmt } from "./lowering.js";

export type ShapeKind = LoweredStmt["kind"];
export type LoweredOf<K extends ShapeKind> = Extract<LoweredStmt, { kind: K }>;

/** Outcome of `run`. `{ result }` on success (the call's return value
 *  for the `call` shape; `undefined` otherwise). `null` is a bail: the
 *  registry discards any captured output, caches the bail, and tries
 *  the next executor (or the interpreter). */
export type RunResult = { result: unknown } | null;

export interface Executor<
  K extends ShapeKind = ShapeKind,
  D = unknown,
  C = unknown,
> {
  /** Stable identifier for diagnostics and the compile cache. */
  readonly name: string;
  /** The lowered shape this executor handles. */
  readonly kind: K;

  /** Decide whether to take this unit. Runs on every dispatch, so it
   *  must be cheap. Returns per-executor data passed to `compile` and
   *  `run`, or null to decline. */
  propose(lowered: LoweredOf<K>, interp: Interpreter): D | null;

  /** Stable cache key projected from the proposal data. Drops volatile
   *  bits (e.g. exact scalar values) so repeated runs reuse artifacts. */
  cacheKey(data: D): string;

  /** Compile to a runnable artifact. Called only on cache miss; cached
   *  under (executor, owner, cacheKey). */
  compile(data: D, interp: Interpreter): C;

  run(compiled: C, data: D, interp: Interpreter): RunResult;
}
