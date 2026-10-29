/**
 * Per-executor compilation cache.
 *
 * Keyed by (executor, owner, cacheKey). The `owner` is the object the
 * cache entry's lifetime is tied to:
 *   - For stmt dispatch: the head Stmt at `siblings[i]`.
 *   - For function-call dispatch: the FunctionDef.
 *
 * WeakMap scoping ensures entries are reclaimed when the owner is
 * garbage-collected (e.g., when an AST is dropped after addpath).
 *
 * Each registered executor gets its own slot in the cache; a single
 * owner can host entries from multiple executors with different
 * specializations.
 */

const BAILED = Symbol("EXECUTOR_BAILED");

/** See `ExecutorCache.keepsBailing`. */
export const MAX_RUNTIME_BAILS = 2;
type Bailed = typeof BAILED;

type ExecutorSlot = Map<string, unknown | Bailed>;

export class ExecutorCache {
  /** owner → executorName → cacheKey → compiled artifact (or BAILED) */
  private readonly slots = new WeakMap<object, Map<string, ExecutorSlot>>();

  get(executorName: string, owner: object, key: string): unknown | undefined {
    const perOwner = this.slots.get(owner);
    if (!perOwner) return undefined;
    const perExec = perOwner.get(executorName);
    if (!perExec) return undefined;
    const v = perExec.get(key);
    if (v === BAILED) return BAILED;
    return v;
  }

  set(
    executorName: string,
    owner: object,
    key: string,
    compiled: unknown
  ): void {
    let perOwner = this.slots.get(owner);
    if (!perOwner) {
      perOwner = new Map();
      this.slots.set(owner, perOwner);
    }
    let perExec = perOwner.get(executorName);
    if (!perExec) {
      perExec = new Map();
      perOwner.set(executorName, perExec);
    }
    perExec.set(key, compiled);
  }

  markBailed(executorName: string, owner: object, key: string): void {
    this.set(executorName, owner, key, BAILED);
  }

  /** owner → executorName → runs that bailed at run time, or -1 once a
   *  run has succeeded. */
  private readonly runtimeBails = new WeakMap<object, Map<string, number>>();

  /** Record how a run of a compiled unit ended: a success, or a bail at
   *  run time (not a compile-time decline). */
  noteRun(executorName: string, owner: object, bailed: boolean): void {
    let perOwner = this.runtimeBails.get(owner);
    if (!perOwner) {
      perOwner = new Map();
      this.runtimeBails.set(owner, perOwner);
    }
    const n = perOwner.get(executorName) ?? 0;
    if (n < 0) return;
    perOwner.set(executorName, bailed ? n + 1 : -1);
  }

  /** Whether `owner` has bailed at run time under `executorName` with
   *  `MAX_RUNTIME_BAILS` specializations and never run to completion: a
   *  unit that keeps bailing (an array that grows in a loop, say) would
   *  bail with a new specialization too, so the executor stops compiling
   *  new ones for it. */
  keepsBailing(executorName: string, owner: object): boolean {
    const n = this.runtimeBails.get(owner)?.get(executorName) ?? 0;
    return n >= MAX_RUNTIME_BAILS;
  }

  isBailed(value: unknown): value is Bailed {
    return value === BAILED;
  }
}
