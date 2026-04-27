/**
 * Per-executor compilation cache.
 *
 * Keyed by (executor, headStmt, cacheKey). The headStmt is the AST
 * node identity at `siblings[i]` where the match started, so cache
 * entries are scoped to the specific source location and can be
 * collected when the AST is dropped (WeakMap).
 *
 * Each registered executor gets its own slot in the cache; a single
 * Stmt can host entries from multiple executors with different
 * specializations.
 */

import type { Stmt } from "../parser/types.js";

const BAILED = Symbol("EXECUTOR_BAILED");
type Bailed = typeof BAILED;

type ExecutorSlot = Map<string, unknown | Bailed>;

export class ExecutorCache {
  /** stmt → executorName → cacheKey → compiled artifact (or BAILED) */
  private readonly slots = new WeakMap<Stmt, Map<string, ExecutorSlot>>();

  get(executorName: string, stmt: Stmt, key: string): unknown | undefined {
    const perStmt = this.slots.get(stmt);
    if (!perStmt) return undefined;
    const perExec = perStmt.get(executorName);
    if (!perExec) return undefined;
    const v = perExec.get(key);
    if (v === BAILED) return BAILED;
    return v;
  }

  set(executorName: string, stmt: Stmt, key: string, compiled: unknown): void {
    let perStmt = this.slots.get(stmt);
    if (!perStmt) {
      perStmt = new Map();
      this.slots.set(stmt, perStmt);
    }
    let perExec = perStmt.get(executorName);
    if (!perExec) {
      perExec = new Map();
      perStmt.set(executorName, perExec);
    }
    perExec.set(key, compiled);
  }

  markBailed(executorName: string, stmt: Stmt, key: string): void {
    this.set(executorName, stmt, key, BAILED);
  }

  isBailed(value: unknown): value is Bailed {
    return value === BAILED;
  }
}
