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

  isBailed(value: unknown): value is Bailed {
    return value === BAILED;
  }
}
