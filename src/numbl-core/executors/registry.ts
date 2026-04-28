/**
 * Executor registry and dispatch.
 *
 * The dispatcher is the single place that decides which executor
 * handles each statement (or run of statements) — and each user-
 * function call. Plugins register executors at startup; the dispatcher
 * selects among them at runtime based on cost estimates, with the AST
 * interpreter as the always-matching last-resort fallback for stmts.
 *
 * Stmt-shape and call-shape work share the same Executor interface:
 * the dispatcher's lowering pass produces a discriminated `LoweredStmt`
 * (kind: "top-level" / "loop" / "call" / "stmt"), and each executor's
 * `propose()` filters on that kind. There are two entry points
 * (`dispatch` for stmts, `dispatchCall` for calls) because the inputs
 * and outputs differ, but both iterate the same `executors` array.
 */

import type { Stmt } from "../parser/types.js";
import type { ControlSignal, FunctionDef } from "../interpreter/types.js";
import type { Executor, RunResult } from "./types.js";
import { DispatchContext } from "./context.js";
import { ExecutorCache } from "./cache.js";
import { LoweringCache, tryLower, tryLowerCall } from "./lowering.js";

interface Candidate {
  readonly executor: Executor;
  readonly data: unknown;
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
  private readonly executors: Executor[] = [];
  private cache = new ExecutorCache();
  private loweringCache = new LoweringCache();
  /** Pre-allocated context reused for `dispatchCall`. Avoids the
   *  per-call Map+Set allocation cost that previously motivated
   *  bypassing ctx entirely on call-heavy workloads. Safe to reuse
   *  because dispatchCall isn't reentrant within itself: each call's
   *  propose/compile/run touches the context synchronously, and any
   *  nested call dispatch goes through `Interpreter.callUserFunction`
   *  which re-enters dispatchCall and re-resets the context. */
  private callCtx: DispatchContext | null = null;

  register(executor: Executor): void {
    if (this.executors.some(e => e.name === executor.name)) {
      throw new Error(`Executor already registered: ${executor.name}`);
    }
    this.executors.push(executor);
  }

  /** Number of registered executors. Mainly for tests. */
  get size(): number {
    return this.executors.length;
  }

  /** Drop all cached compiled artifacts. Called from
   *  `Interpreter.clearAllCaches()` after addpath/rmpath etc. */
  clearCache(): void {
    // ExecutorCache and LoweringCache use WeakMaps, so we just throw
    // away the whole instance and start fresh.
    this.cache = new ExecutorCache();
    this.loweringCache = new LoweringCache();
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

    // Pre-propose lowering pass. Returns the lowered IR for stmts
    // that match a specialized shape (top-level, loop, ...), or null
    // for stmts with no shape — those skip the proposal loop entirely
    // and go straight to the hardcoded interpreter fallback.
    const stmt = siblings[i];
    const lowered = tryLower(siblings, i, ctx, this.loweringCache);
    if (lowered === null) {
      const signal = ctx.interp.execStmt(stmt);
      return { consumed: 1, signal };
    }

    // Single linear pass: collect proposals, keep the lowest-cost one
    // as best; the rest go in a lazily-allocated backup list (only
    // used if the best bails).
    let bestExec: Executor | null = null;
    let bestData: unknown = null;
    let bestTotal = Infinity;
    let backups: Candidate[] | null = null;
    const requireNoBail = ctx.requireNoBail;
    const checkActive = ctx.hasActive;
    const executors = this.executors;
    for (let k = 0; k < executors.length; k++) {
      const executor = executors[k];
      if (checkActive && ctx.isActive(executor.name, stmt)) continue;
      const p = executor.propose(lowered, ctx);
      if (!p) continue;
      // bailRisk is per-proposal: an executor can produce both
      // bail-risky and bail-safe proposals depending on its input.
      if (requireNoBail && p.bailRisk) continue;
      const total = p.cost.perCallNs + p.cost.runNs;
      if (total < bestTotal) {
        if (bestExec) {
          (backups ??= []).push({
            executor: bestExec,
            data: bestData,
            total: bestTotal,
          });
        }
        bestExec = executor;
        bestData = p.data;
        bestTotal = total;
      } else {
        (backups ??= []).push({ executor, data: p.data, total });
      }
    }

    // Try best (if any), then backups in cost order. Each may bail.
    if (bestExec) {
      const r = this.runStmtCandidate(stmt, ctx, bestExec, bestData, lowered);
      if (r) return r;
      if (backups) {
        backups.sort((a, b) => a.total - b.total);
        for (let k = 0; k < backups.length; k++) {
          const c = backups[k];
          const r2 = this.runStmtCandidate(
            stmt,
            ctx,
            c.executor,
            c.data,
            lowered
          );
          if (r2) return r2;
        }
      }
    }

    // Fallback: AST interpreter, called directly. Reached when every
    // proposal bailed.
    const signal = ctx.interp.execStmt(stmt);
    return { consumed: 1, signal };
  }

  private runStmtCandidate(
    stmt: Stmt,
    ctx: DispatchContext,
    executor: Executor,
    data: unknown,
    lowered: import("./lowering.js").LoweredStmt
  ): DispatchResult | null {
    const result = this.executeCandidate(ctx, executor, stmt, data, stmt);
    if (!result) return null;
    if ("consumed" in result) {
      ctx.interp.onExecutorFired?.(executor.name, lowered.kind);
      return { consumed: result.consumed, signal: null };
    }
    throw new Error(
      `Stmt-shape executor ${executor.name} returned an invalid RunResult`
    );
  }

  /** Cache lookup + compile-on-miss + run + bail handling. Returns
   *  the executor's success RunResult, or null when the candidate
   *  bailed (cache may have been marked BAILED). When `guardKey` is
   *  non-null, the reentrancy guard pushes/pops around `run()` —
   *  passed by the stmt path; the call path passes null because
   *  `dispatchCall` isn't reentrant within itself. */
  private executeCandidate(
    ctx: DispatchContext,
    executor: Executor,
    owner: object,
    data: unknown,
    guardKey: Stmt | null
  ): RunResult | null {
    const key = executor.cacheKey(data);

    let compiled = this.cache.get(executor.name, owner, key);
    if (this.cache.isBailed(compiled)) return null;
    if (compiled === undefined) {
      compiled = executor.compile(data, ctx);
      this.cache.set(executor.name, owner, key, compiled);
    }

    let result: RunResult;
    if (guardKey !== null && ctx.hasActive) {
      ctx.pushActive(executor.name, guardKey);
      try {
        result = executor.run(compiled, data, ctx);
      } finally {
        ctx.popActive(executor.name, guardKey);
      }
    } else {
      result = executor.run(compiled, data, ctx);
    }

    if ("bail" in result) {
      if (!result.transient) {
        this.cache.markBailed(executor.name, owner, key);
      }
      return null;
    }
    return result;
  }

  /**
   * Dispatch a user-function call. Iterates the same `executors`
   * array as stmt dispatch — call-shape executors filter on
   * `lowered.kind === "call"`. Returns `{ result }` when one of them
   * handles the call, or `null` so the caller (callUserFunction)
   * falls through to the AST interpreter.
   *
   * Reuses a pre-allocated DispatchContext (`callCtx`) to avoid the
   * per-call Map+Set allocation that previously motivated bypassing
   * ctx for call dispatch entirely.
   */
  dispatchCall(
    fn: FunctionDef,
    args: unknown[],
    nargout: number,
    interp: import("../interpreter/interpreter.js").Interpreter
  ): CallDispatchResult {
    const lowered = tryLowerCall(fn, args, nargout, interp, this.loweringCache);
    if (!lowered) return null;

    const ctx = this.getCallCtx(interp);

    let bestExec: Executor | null = null;
    let bestData: unknown = null;
    let bestTotal = Infinity;
    let backups: Candidate[] | null = null;
    const executors = this.executors;
    for (let k = 0; k < executors.length; k++) {
      const executor = executors[k];
      const p = executor.propose(lowered, ctx);
      if (!p) continue;
      const total = p.cost.perCallNs + p.cost.runNs;
      if (total < bestTotal) {
        if (bestExec) {
          (backups ??= []).push({
            executor: bestExec,
            data: bestData,
            total: bestTotal,
          });
        }
        bestExec = executor;
        bestData = p.data;
        bestTotal = total;
      } else {
        (backups ??= []).push({ executor, data: p.data, total });
      }
    }

    if (!bestExec) return null;

    const r = this.runCallCandidate(fn, ctx, bestExec, bestData);
    if (r) return r;
    if (backups) {
      backups.sort((a, b) => a.total - b.total);
      for (let k = 0; k < backups.length; k++) {
        const c = backups[k];
        const r2 = this.runCallCandidate(fn, ctx, c.executor, c.data);
        if (r2) return r2;
      }
    }
    return null;
  }

  private runCallCandidate(
    fn: FunctionDef,
    ctx: DispatchContext,
    executor: Executor,
    data: unknown
  ): CallDispatchResult {
    const result = this.executeCandidate(ctx, executor, fn, data, null);
    if (!result) return null;
    if ("result" in result) {
      ctx.interp.onExecutorFired?.(executor.name, "call");
      return { result: result.result };
    }
    throw new Error(
      `Call-shape executor ${executor.name} returned an invalid RunResult`
    );
  }

  private getCallCtx(
    interp: import("../interpreter/interpreter.js").Interpreter
  ): DispatchContext {
    let ctx = this.callCtx;
    if (!ctx) {
      ctx = new DispatchContext(interp, this, false, undefined, "nested");
      this.callCtx = ctx;
    } else {
      ctx.resetForNextDispatch();
    }
    return ctx;
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
