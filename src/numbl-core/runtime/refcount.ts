/**
 * Reference counting for RuntimeValue containers.
 *
 * Every container kind extends `Refcounted`. The count starts at 0
 * (newly-constructed value, not yet bound to anything). It's incremented
 * when a slot (env binding, struct field, cell element, ...) takes
 * ownership, and decremented when the slot releases ownership. When the
 * count reaches 0, `_destroy` runs and decrefs every child value.
 *
 * Kept in place for the upcoming COW overhaul, which will use the count
 * to decide whether a write needs to copy. No buffer-pool reclamation
 * happens here — V8 GC reclaims wrappers and their backing buffers
 * automatically once they're unreachable.
 *
 * Strict mode (`rt.strictRefcount`) makes a decref of an already-zero
 * count throw.
 */

// ── Active runtime stack ─────────────────────────────────────────────────
//
// `Refcounted`'s constructor consults the active runtime so freshly built
// values can adopt into the current per-statement transient scope. Each
// `executeCode` invocation pushes its rt; nested invocations stack.

const runtimeStack: RefcountRuntime[] = [];

export function pushCurrentRuntime(rt: RefcountRuntime): void {
  runtimeStack.push(rt);
}

export function popCurrentRuntime(rt: RefcountRuntime): void {
  const top = runtimeStack[runtimeStack.length - 1];
  if (top !== rt) {
    throw new Error(
      "refcount: popCurrentRuntime mismatch — push/pop misnested"
    );
  }
  runtimeStack.pop();
}

export function getCurrentRuntime(): RefcountRuntime | null {
  const top = runtimeStack[runtimeStack.length - 1];
  return top ?? null;
}

/** Minimal runtime surface used by the refcount API. The real `Runtime`
 *  class satisfies this; using an interface here avoids a circular
 *  import between refcount.ts ↔ runtime.ts ↔ types.ts. */
export interface RefcountRuntime {
  /** When true, decref of a zero count throws. */
  strictRefcount?: boolean;
  /** Per-statement transient scope. Constructors auto-adopt fresh
   *  values here so an unbound expression result stays alive until end
   *  of statement. Null when no statement is in flight. */
  currentScope?: RefScope | null;
}

/** Base class for every refcounted runtime value. */
export abstract class Refcounted {
  /** Public so type guards can look for a `kind` discriminator without
   *  walking the prototype. The string is set by every subclass. */
  abstract readonly kind: string;

  /** Reference count. Starts at 0; incremented when bound to a slot. */
  _rc: number = 0;

  constructor() {
    // Auto-adopt into the active runtime's transient scope (if any).
    // The scope owns one ref (rc=0 → 1) so the value survives until the
    // current statement finishes. Statements that bind the value to a
    // slot add their own refs; statements that don't get the value
    // released on scope drain.
    //
    // In production the current runtime is always set during execution.
    // Test harnesses that build values without an active runtime simply
    // skip adoption — the rc stays at 0 and JS GC collects the wrapper.
    const rt = getCurrentRuntime();
    if (rt && rt.currentScope) rt.currentScope.adopt(this);
  }

  incref(): void {
    this._rc++;
  }

  decref(rt: RefcountRuntime): void {
    if (this._rc <= 0) {
      if (rt.strictRefcount) {
        throw new Error(`refcount underflow on ${this.kind} (rc=${this._rc})`);
      }
      // Lax mode: silently ignore, plumbing can be incomplete during migration.
      return;
    }
    this._rc--;
    if (this._rc === 0) {
      this._destroy(rt);
    }
  }

  /** Subclasses release child refs. Idempotent must NOT be assumed —
   *  callers guarantee this fires exactly once. Default is a no-op for
   *  kinds that have no child refs (RuntimeChar, RuntimeComplexNumber,
   *  RuntimeDummyHandle, etc.). */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  protected _destroy(_rt: RefcountRuntime): void {
    // no-op default
  }
}

/** True if the value is a container that participates in refcounting.
 *  Primitives (number, boolean, string) return false. */
export function isRefcountable(v: unknown): v is Refcounted {
  return v instanceof Refcounted;
}

/** Increment refcount on a value if it's a container; primitives are noop. */
export function incref(v: unknown): void {
  if (v instanceof Refcounted) v.incref();
}

/** Decrement refcount on a value if it's a container; primitives are noop. */
export function decref(rt: RefcountRuntime, v: unknown): void {
  if (v instanceof Refcounted) v.decref(rt);
}

/** Per-statement transients harness. Fresh values produced by constructors
 *  and operators are adopted here at rc=1; on `drain`, every member is
 *  decref'd. Anything bound to a slot during the statement gets an extra
 *  incref from the slot, so it survives drain at slot count. */
export class RefScope {
  private members: Refcounted[] = [];

  /** Take ownership of a fresh value: incref and remember for drain. */
  adopt(v: unknown): void {
    if (v instanceof Refcounted) {
      v.incref();
      this.members.push(v);
    }
  }

  /** Release every adopted value. Called in `withScope`'s finally. */
  drain(rt: RefcountRuntime): void {
    for (const v of this.members) {
      v.decref(rt);
    }
    this.members.length = 0;
  }

  /** Number of currently-adopted values (for assertions/tests). */
  size(): number {
    return this.members.length;
  }
}
