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
import type { ControlSignal } from "../interpreter/types.js";
import type { Executor } from "./types.js";
import { DispatchContext } from "./context.js";
import { ExecutorCache } from "./cache.js";

interface RegisteredExecutor {
  readonly executor: Executor;
}

interface Candidate {
  readonly executor: Executor;
  readonly match: unknown;
  readonly cost: { readonly perCallNs: number; readonly runNs: number };
  readonly requireNoBailInChildren: boolean;
}

export interface DispatchResult {
  /** Number of sibling stmts consumed (>= 1). */
  consumed: number;
  /** Control signal from interpreter execution (break/continue/return),
   *  if any. */
  signal: ControlSignal | null;
}

export class Registry {
  private readonly executors: RegisteredExecutor[] = [];
  private cache = new ExecutorCache();

  register(executor: Executor): void {
    if (this.executors.some(r => r.executor.name === executor.name)) {
      throw new Error(`Executor already registered: ${executor.name}`);
    }
    this.executors.push({ executor });
  }

  /** Number of registered executors. Mainly for tests. */
  get size(): number {
    return this.executors.length;
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
   */
  dispatch(
    siblings: readonly Stmt[],
    i: number,
    ctx: DispatchContext
  ): DispatchResult {
    // Phase 1: collect candidates by asking each eligible executor
    // for a match.
    const candidates: Candidate[] = [];
    for (const { executor } of this.executors) {
      if (ctx.requireNoBail && executor.bailRisk) continue;
      if (ctx.isActive(executor.name, siblings[i])) continue;
      const m = executor.match(siblings, i, ctx);
      if (!m) continue;
      candidates.push({
        executor,
        match: m.match,
        cost: m.cost,
        requireNoBailInChildren: !!m.requireNoBailInChildren,
      });
    }

    // No candidate should ever be empty: the interpreter executor
    // always matches and is never bail-risk.
    if (candidates.length === 0) {
      throw new Error(
        "Executor registry: no candidate for stmt — is the interpreter " +
          "executor registered?"
      );
    }

    // Phase 2: sort by per-call cost (compileMs not in the comparison
    // for now — see the design doc).
    candidates.sort(
      (a, b) =>
        a.cost.perCallNs + a.cost.runNs - (b.cost.perCallNs + b.cost.runNs)
    );

    // Phase 3: try candidates in order. On bail, invalidate and try
    // the next one.
    for (const c of candidates) {
      const result = this.runCandidate(siblings, i, ctx, c);
      if (result) return result;
    }

    throw new Error(
      "Executor registry: every candidate bailed — interpreter executor " +
        "should never bail."
    );
  }

  private runCandidate(
    siblings: readonly Stmt[],
    i: number,
    ctx: DispatchContext,
    c: Candidate
  ): DispatchResult | null {
    const stmt = siblings[i];
    const key = c.executor.cacheKey(c.match);

    let compiled = this.cache.get(c.executor.name, stmt, key);
    if (this.cache.isBailed(compiled)) return null;
    if (compiled === undefined) {
      compiled = c.executor.compile(c.match, ctx);
      this.cache.set(c.executor.name, stmt, key, compiled);
    }

    ctx.pushActive(c.executor.name, stmt);
    let result;
    try {
      result = c.executor.run(compiled, c.match, ctx);
    } finally {
      ctx.popActive(c.executor.name, stmt);
    }

    if ("bail" in result) {
      if (!result.transient) {
        this.cache.markBailed(c.executor.name, stmt, key);
      }
      return null;
    }
    return { consumed: result.consumed, signal: result.signal ?? null };
  }
}

/** Build a fresh dispatch context for top-level interpreter use. */
export function makeRootContext(
  interp: import("../interpreter/interpreter.js").Interpreter,
  registry: Registry
): DispatchContext {
  return new DispatchContext(interp, registry, false);
}
