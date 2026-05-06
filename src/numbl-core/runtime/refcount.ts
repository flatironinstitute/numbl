/**
 * Refcount-driven memory-pool reclamation for RuntimeValue containers.
 *
 * Every container kind extends `Refcounted`. The count starts at 0
 * (newly-constructed value, not yet bound to anything). It's incremented
 * when a slot (env binding, struct field, cell element, ...) takes
 * ownership, and decremented when the slot releases ownership. When the
 * count reaches 0, `_destroy` runs: every child value is decref'd and
 * any owned buffers are returned to the pool.
 *
 * Strict mode (`rt.strictRefcount`) makes a decref of an already-zero
 * count throw. The memory pool is gated by `rt.memPool` — when off,
 * `_destroy` skips the buffer release so every allocation is a fresh
 * `new Float64Array` and no buffer is ever recycled. Useful for
 * isolating bugs that may be caused by buffer recycling.
 */

import { getCurrentRuntime } from "./memoryPool.js";

/** Minimal runtime surface used by the refcount API. The real `Runtime`
 *  class satisfies this, but using an interface here avoids a circular
 *  import between refcount.ts ↔ runtime.ts ↔ types.ts. */
export interface RefcountRuntime {
  pool: {
    acquire(size: number): Float64Array;
    acquireFrom(src: number[] | Float64Array): Float64Array;
    release(buf: Float64Array): void;
  };
  /** When true, decref of a zero count throws. */
  strictRefcount?: boolean;
  /** When true, `RuntimeTensor._destroy` etc. release buffers back to
   *  the pool for reuse. When false, buffers are dropped on the floor
   *  (JS GC reclaims the wrapper; the underlying `Float64Array` is
   *  collected with it) and every allocation is a fresh
   *  `new Float64Array`. */
  memPool?: boolean;
  /** Optional transients harness. Constructors push freshly-built values
   *  here so an unbound expression result is decref'd at end of statement. */
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
    // Auto-adopt into the active runtime's transient scope (if any). The
    // scope owns one ref (rc=0 → 1) so the value survives until the
    // current statement finishes. Statements that bind the value to a
    // slot (env, struct field, cell element, ...) add their own refs;
    // statements that don't get the value released on scope drain.
    //
    // In production code the current runtime is always set during
    // execution. Test harnesses that build values without an active
    // runtime simply skip adoption — the rc stays at 0 and JS GC
    // collects the wrapper on its own.
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

  /** Subclasses release child refs and any owned buffers. Idempotent
   *  must NOT be assumed — callers guarantee this fires exactly once.
   *  Default is a no-op for kinds that own no buffers and have no child
   *  refs (RuntimeChar, RuntimeComplexNumber, RuntimeDummyHandle, etc.). */
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
 *  incref from the slot, so it survives drain at slot count.
 *
 *  The harness is wired up in phase 5 of the refcount rollout. Until
 *  then, `RTV.*` factories may construct without adopting. */
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
