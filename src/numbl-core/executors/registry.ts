/**
 * Executor registry and dispatch.
 *
 * The dispatcher is the single place that decides which executor
 * handles each statement (or run of statements). Plugins register
 * executors at startup; the dispatcher selects among them at runtime
 * based on cost estimates, with the AST interpreter as the always-
 * matching last-resort fallback.
 */

import type { Stmt } from "../parser/types.js";
import type { ControlSignal, FunctionDef } from "../interpreter/types.js";
import type { CallExecutor, Executor } from "./types.js";
import { DispatchContext } from "./context.js";
import { ExecutorCache } from "./cache.js";

interface RegisteredExecutor {
  readonly executor: Executor;
}

interface RegisteredCallExecutor {
  readonly executor: CallExecutor;
}

interface Candidate {
  readonly executor: Executor;
  readonly match: unknown;
  readonly total: number;
  readonly requireNoBailInChildren: boolean;
}

interface CallCandidate {
  readonly executor: CallExecutor;
  readonly match: unknown;
  readonly total: number;
}

export interface DispatchResult {
  /** Number of sibling stmts consumed (>= 1). */
  consumed: number;
  /** Control signal from interpreter execution (break/continue/return),
   *  if any. */
  signal: ControlSignal | null;
}

/** Result from `dispatchCall`. `null` means no executor handled the
 *  call — the caller should fall through to its own
 *  interpreter-execution path. */
export type CallDispatchResult = { result: unknown } | null;

export class Registry {
  private readonly executors: RegisteredExecutor[] = [];
  private readonly callExecutors: RegisteredCallExecutor[] = [];
  private cache = new ExecutorCache();

  register(executor: Executor): void {
    if (this.executors.some(r => r.executor.name === executor.name)) {
      throw new Error(`Executor already registered: ${executor.name}`);
    }
    this.executors.push({ executor });
  }

  /** Register a function-call executor. Function-call executors fire
   *  from `Interpreter.callUserFunction`, parallel to stmt-level
   *  executors. */
  registerCall(executor: CallExecutor): void {
    if (this.callExecutors.some(r => r.executor.name === executor.name)) {
      throw new Error(
        `Function-call executor already registered: ${executor.name}`
      );
    }
    this.callExecutors.push({ executor });
  }

  /** Number of registered stmt executors. Mainly for tests. */
  get size(): number {
    return this.executors.length;
  }

  /** Number of registered function-call executors. */
  get callSize(): number {
    return this.callExecutors.length;
  }

  /** Drop all cached compiled artifacts. Called from
   *  `Interpreter.clearAllCaches()` after addpath/rmpath etc. */
  clearCache(): void {
    // ExecutorCache uses a WeakMap, so we just throw away the whole
    // instance and start fresh.
    this.cache = new ExecutorCache();
  }

  /**
   * Dispatch one statement (or run of statements) starting at
   * `siblings[i]`. Returns the number of stmts consumed and any
   * control signal raised by the underlying execution.
   *
   * Hot path. Two optimizations matter for the chunkie-helmholtz-
   * scale workloads:
   *
   *   1. The AST interpreter is the always-applicable last-resort
   *      fallback. It is *not* a registered executor — the dispatcher
   *      hardcodes a direct call to `ctx.interp.execStmt(stmt)` when
   *      no specialized executor matches or all of them bail. This
   *      avoids per-dispatch match/Candidate/cache/result allocations
   *      for the most common path (where only the interpreter would
   *      fit anyway).
   *
   *   2. The reentrancy guard (`pushActive`/`popActive` /
   *      `isActive`) is short-circuited when the active set is
   *      empty — no sub-dispatch in flight means the bookkeeping is
   *      pure overhead.
   */
  dispatch(
    siblings: readonly Stmt[],
    i: number,
    ctx: DispatchContext
  ): DispatchResult {
    // Set position info on ctx so peekSibling / remainingSiblings /
    // isFirstInScope work for executors that consult them. Most
    // executors only see the single stmt and don't care.
    ctx._setPosition(siblings, i);

    // Single linear pass: pick the lowest-cost match among
    // *specialized* executors. Backup list (allocated lazily) holds
    // the rest, used only if the best bails.
    let bestExec: Executor | null = null;
    let bestMatch: unknown = null;
    let bestTotal = Infinity;
    let backups: Candidate[] | null = null;
    const stmt = siblings[i];
    const requireNoBail = ctx.requireNoBail;
    const checkActive = ctx.hasActive;
    const executors = this.executors;
    for (let k = 0; k < executors.length; k++) {
      const executor = executors[k].executor;
      if (requireNoBail && executor.bailRisk) continue;
      if (checkActive && ctx.isActive(executor.name, stmt)) continue;
      const m = executor.match(stmt, ctx);
      if (!m) continue;
      const total = m.cost.perCallNs + m.cost.runNs;
      if (total < bestTotal) {
        if (bestExec) {
          (backups ??= []).push({
            executor: bestExec,
            match: bestMatch,
            total: bestTotal,
            requireNoBailInChildren: false,
          });
        }
        bestExec = executor;
        bestMatch = m.match;
        bestTotal = total;
      } else {
        (backups ??= []).push({
          executor,
          match: m.match,
          total,
          requireNoBailInChildren: !!m.requireNoBailInChildren,
        });
      }
    }

    // Try best (if any), then backups in cost order. Each may bail.
    if (bestExec) {
      const r = this.runCandidate(siblings, i, ctx, bestExec, bestMatch);
      if (r) return r;
      if (backups) {
        backups.sort((a, b) => a.total - b.total);
        for (let k = 0; k < backups.length; k++) {
          const c = backups[k];
          const r2 = this.runCandidate(siblings, i, ctx, c.executor, c.match);
          if (r2) return r2;
        }
      }
    }

    // Fallback: AST interpreter, called directly. Bypasses the
    // executor protocol because the interpreter has no compiled
    // artifact, no cache benefit, and never bails — and this path
    // runs on every "uninteresting" stmt.
    const signal = ctx.interp.execStmt(stmt);
    return { consumed: 1, signal };
  }

  private runCandidate(
    siblings: readonly Stmt[],
    i: number,
    ctx: DispatchContext,
    executor: Executor,
    match: unknown
  ): DispatchResult | null {
    const stmt = siblings[i];
    const key = executor.cacheKey(match);

    let compiled = this.cache.get(executor.name, stmt, key);
    if (this.cache.isBailed(compiled)) return null;
    if (compiled === undefined) {
      compiled = executor.compile(match, ctx);
      this.cache.set(executor.name, stmt, key, compiled);
    }

    let result;
    if (ctx.hasActive) {
      ctx.pushActive(executor.name, stmt);
      try {
        result = executor.run(compiled, match, ctx);
      } finally {
        ctx.popActive(executor.name, stmt);
      }
    } else {
      result = executor.run(compiled, match, ctx);
    }

    if ("bail" in result) {
      if (!result.transient) {
        this.cache.markBailed(executor.name, stmt, key);
      }
      return null;
    }
    return { consumed: result.consumed, signal: result.signal ?? null };
  }

  /**
   * Dispatch a user-function call. Parallel to `dispatch` but for the
   * function-call entry point (`Interpreter.callUserFunction`). Returns
   * `{ result }` if a function-call executor handled the call, or
   * `null` to signal "no executor matched / all bailed; fall through
   * to the interpreter's normal call path."
   *
   * Takes the Interpreter directly instead of a DispatchContext — call
   * executors don't use the typeCache/active machinery, so skipping
   * the per-call Map+Set allocation matters on call-heavy workloads.
   *
   * Unlike stmt dispatch, there is no hardcoded fallback here — the
   * caller (callUserFunction) knows how to interpret-execute a
   * function and will do so when this returns null.
   */
  dispatchCall(
    fn: FunctionDef,
    args: unknown[],
    nargout: number,
    interp: import("../interpreter/interpreter.js").Interpreter
  ): CallDispatchResult {
    if (this.callExecutors.length === 0) return null;

    let bestExec: CallExecutor | null = null;
    let bestMatch: unknown = null;
    let bestTotal = Infinity;
    let backups: CallCandidate[] | null = null;
    const callExecs = this.callExecutors;
    for (let k = 0; k < callExecs.length; k++) {
      const executor = callExecs[k].executor;
      const m = executor.matchCall(fn, args, nargout, interp);
      if (!m) continue;
      const total = m.cost.perCallNs + m.cost.runNs;
      if (total < bestTotal) {
        if (bestExec) {
          (backups ??= []).push({
            executor: bestExec,
            match: bestMatch,
            total: bestTotal,
          });
        }
        bestExec = executor;
        bestMatch = m.match;
        bestTotal = total;
      } else {
        (backups ??= []).push({ executor, match: m.match, total });
      }
    }

    if (!bestExec) return null;

    const r = bestExec.runCall(bestMatch, fn, args, nargout, interp);
    if (!("bail" in r)) return { result: r.result };
    if (backups) {
      backups.sort((a, b) => a.total - b.total);
      for (let k = 0; k < backups.length; k++) {
        const c = backups[k];
        const r2 = c.executor.runCall(c.match, fn, args, nargout, interp);
        if (!("bail" in r2)) return { result: r2.result };
      }
    }
    return null;
  }
}

/** Build a fresh dispatch context. `scope` defaults to `"nested"` —
 *  pass `"top-level"` only from `Interpreter.run()`'s script-body
 *  loop, where whole-script executors (e.g., JS-JIT top-level) are
 *  eligible. */
export function makeRootContext(
  interp: import("../interpreter/interpreter.js").Interpreter,
  registry: Registry,
  scope: import("./context.js").DispatchScope = "nested"
): DispatchContext {
  return new DispatchContext(interp, registry, false, undefined, scope);
}
