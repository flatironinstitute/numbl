/**
 * Executor registry and dispatch.
 *
 * Three entry points, three concerns:
 *   - `dispatch(siblings, i, ctx)` — per-stmt dispatch. Each executor
 *     handles exactly one stmt (no consumed-N).
 *   - `dispatchCall(fn, args, nargout, interp)` — user-function call.
 *   - `tryRunWholeScope(siblings, interp)` — whole-script attempt,
 *     called once before the per-stmt loop runs. Top-level executors
 *     register here separately from per-stmt ones.
 *
 * Plugins register executors at startup; the dispatcher selects among
 * them at runtime based on cost estimates. The AST interpreter is the
 * always-matching last-resort fallback for per-stmt dispatch and isn't
 * a registered executor.
 */

import type { Stmt } from "../parser/types.js";
import type { ControlSignal, FunctionDef } from "../interpreter/types.js";
import type { Executor, RunResult } from "./types.js";
import { DispatchContext } from "./context.js";
import { ExecutorCache } from "./cache.js";
import {
  LoweringCache,
  tryLower,
  tryLowerCall,
  tryLowerTopLevel,
} from "./lowering.js";

interface Candidate {
  readonly executor: Executor;
  readonly data: unknown;
  readonly total: number;
}

/** AST stmt-list transformer. Receives a stmt list and returns a
 *  list of the same semantics, possibly with `Synth` stmts inserted
 *  where the transformer can collapse a contiguous run of stmts into
 *  a unit handled by a specialized executor. The transform is
 *  shallow — recursive descent into For/While/If bodies happens
 *  lazily on the next call to `transformStmts`, cached separately. */
export type StmtTransformer = (stmts: readonly Stmt[]) => Stmt[];

export interface DispatchResult {
  /** Control signal from interpreter execution (break/continue/return),
   *  if any. */
  signal: ControlSignal | null;
}

/** Result from `dispatchCall`. `null` means no executor handled the
 *  call — the caller should fall through to its own
 *  interpreter-execution path. */
export type CallDispatchResult = { result: unknown } | null;

/** Result from `tryRunWholeScope`. `null` means no whole-scope
 *  executor matched (or all bailed) — the caller should fall through
 *  to the per-stmt dispatch loop. */
export type WholeScopeResult = { signal: ControlSignal | null } | null;

export class Registry {
  /** Per-stmt executors (loop, call, fuse). */
  private readonly executors: Executor[] = [];
  /** Whole-scope executors (top-level). Iterated only by
   *  `tryRunWholeScope`, separate from per-stmt dispatch. */
  private readonly wholeScopeExecutors: Executor[] = [];
  /** AST stmt-list transformers. Run lazily on each stmt list before
   *  the per-stmt loop walks it; result cached per input list. */
  private readonly transformers: StmtTransformer[] = [];
  /** Memoize transformed lists by input list identity. WeakMap so
   *  entries vanish when the AST is dropped. The transformed list
   *  is also stored mapped to itself so a second call with the
   *  result hits the cache (idempotency). */
  private transformCache = new WeakMap<readonly Stmt[], Stmt[]>();
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

  /** Register an executor that handles a whole stmt list as a unit
   *  (e.g. JS-JIT top-level). Iterated only by `tryRunWholeScope`
   *  and never seen by per-stmt dispatch. */
  registerWholeScope(executor: Executor): void {
    if (this.wholeScopeExecutors.some(e => e.name === executor.name)) {
      throw new Error(
        `Whole-scope executor already registered: ${executor.name}`
      );
    }
    this.wholeScopeExecutors.push(executor);
  }

  /** Register an AST stmt-list transformer. Transformers run on every
   *  stmt list the interpreter is about to walk (script body, function
   *  body, loop body, if branch, ...) before the per-stmt loop kicks
   *  in. They typically wrap contiguous runs of stmts in `Synth`
   *  nodes that a matching executor will recognize. Adding new
   *  transformers is additive — they compose in registration order. */
  registerStmtTransformer(fn: StmtTransformer): void {
    this.transformers.push(fn);
    // Different transformer set ⇒ stale cached results.
    this.transformCache = new WeakMap();
  }

  /** Transform a stmt list, applying all registered transformers in
   *  registration order. Cached per input-list identity (WeakMap).
   *  When no transformers are registered, returns the input unchanged
   *  without populating the cache (saves a Map lookup per dispatch). */
  transformStmts(stmts: readonly Stmt[]): Stmt[] {
    if (this.transformers.length === 0) return stmts as Stmt[];
    const cached = this.transformCache.get(stmts);
    if (cached) return cached;
    let result: Stmt[] = stmts as Stmt[];
    for (const t of this.transformers) {
      result = t(result);
    }
    this.transformCache.set(stmts, result);
    // Idempotency: walking the result list later (e.g. via a different
    // entry point) should not re-run the transformers.
    if (result !== (stmts as Stmt[])) this.transformCache.set(result, result);
    return result;
  }

  /** Number of registered per-stmt executors. Mainly for tests. */
  get size(): number {
    return this.executors.length;
  }

  /** Drop all cached compiled artifacts. Called from
   *  `Interpreter.clearAllCaches()` after addpath/rmpath etc. */
  clearCache(): void {
    // ExecutorCache, LoweringCache, and transformCache use WeakMaps,
    // so we just throw away the whole instance and start fresh.
    this.cache = new ExecutorCache();
    this.loweringCache = new LoweringCache();
    this.transformCache = new WeakMap();
  }

  /**
   * Dispatch one statement at `siblings[i]`. Each executor handles
   * exactly one stmt — the dispatcher always advances by one on
   * success.
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
    // Set position info on ctx so peekSibling / siblings work for
    // executors that consult them. Most executors only see the
    // single stmt and don't care.
    ctx._setPosition(siblings, i);

    // Pre-propose lowering pass. Returns the lowered IR for stmts
    // that match a specialized shape (loop, fuse, ...), or null for
    // stmts with no shape — those skip the proposal loop entirely
    // and go straight to the hardcoded interpreter fallback.
    const stmt = siblings[i];
    const lowered = tryLower(siblings, i, ctx.interp, this.loweringCache);
    if (lowered === null) {
      const signal = ctx.interp.execStmt(stmt);
      return { signal };
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
    return { signal };
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
    if ("ok" in result) {
      ctx.interp.onExecutorFired?.(executor.name, lowered.kind);
      return { signal: null };
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
    const lowered = tryLowerCall(fn, args, nargout, this.loweringCache);
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
      ctx = new DispatchContext(interp, this, false, undefined);
      this.callCtx = ctx;
    } else {
      ctx.resetForNextDispatch();
    }
    return ctx;
  }

  /**
   * Whole-scope dispatch: try to handle the entire stmt list as a
   * single unit (e.g. JS-JIT top-level). Called by the Interpreter
   * once before the per-stmt loop runs. Returns:
   *   - `{ signal }` when a whole-scope executor handled the body.
   *     The caller should skip the per-stmt loop entirely.
   *   - `null` when no whole-scope executor matched (or all bailed).
   *     The caller falls through to per-stmt dispatch.
   */
  tryRunWholeScope(
    siblings: readonly Stmt[],
    interp: import("../interpreter/interpreter.js").Interpreter
  ): WholeScopeResult {
    if (this.wholeScopeExecutors.length === 0) return null;

    const lowered = tryLowerTopLevel(interp, siblings, this.loweringCache);
    if (!lowered) return null;

    const ctx = new DispatchContext(interp, this, false, undefined);
    ctx._setPosition(siblings, 0);

    let bestExec: Executor | null = null;
    let bestData: unknown = null;
    let bestTotal = Infinity;
    let backups: Candidate[] | null = null;
    const requireNoBail = ctx.requireNoBail;
    const executors = this.wholeScopeExecutors;
    for (let k = 0; k < executors.length; k++) {
      const executor = executors[k];
      const p = executor.propose(lowered, ctx);
      if (!p) continue;
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

    if (!bestExec) return null;

    // Whole-scope artifacts are cached against the head Stmt as
    // owner — same scheme stmt-shape executors use.
    const owner = siblings[0];
    const r = this.executeCandidate(ctx, bestExec, owner, bestData, null);
    if (r) return this.finishWholeScope(interp, bestExec, lowered, r);
    if (backups) {
      backups.sort((a, b) => a.total - b.total);
      for (let k = 0; k < backups.length; k++) {
        const c = backups[k];
        const r2 = this.executeCandidate(ctx, c.executor, owner, c.data, null);
        if (r2) return this.finishWholeScope(interp, c.executor, lowered, r2);
      }
    }
    return null;
  }

  private finishWholeScope(
    interp: import("../interpreter/interpreter.js").Interpreter,
    executor: Executor,
    lowered: import("./lowering.js").LoweredStmt,
    result: RunResult
  ): WholeScopeResult {
    if ("ok" in result) {
      interp.onExecutorFired?.(executor.name, lowered.kind);
      return { signal: null };
    }
    throw new Error(
      `Whole-scope executor ${executor.name} returned an invalid RunResult`
    );
  }
}

/** Build a fresh dispatch context. */
export function makeRootContext(
  interp: import("../interpreter/interpreter.js").Interpreter,
  registry: Registry
): DispatchContext {
  return new DispatchContext(interp, registry, false, undefined);
}
