/**
 * Float64Array memory pool — alloc-tracking only, no sweep.
 *
 * Every Float64Array handed out by `allocFloat64Array` is recorded in
 * `liveSet` and counters are bumped. The pool exposes a `release(buf)`
 * method that moves a buffer to `freePool` (one bucket per size, unbounded);
 * subsequent `acquire(N)` calls pop from the bucket instead of allocating
 * fresh. Nothing currently calls `release` automatically — sweep-based
 * reclamation is intentionally absent in this iteration.
 *
 * Per-runtime: each Runtime owns its own pool. A module-level stack tracks
 * the *currently active* runtime so `allocFloat64Array` can find it
 * without threading rt through every callsite.
 */

// ── Runtime stack ────────────────────────────────────────────────────────

interface PoolHolder {
  pool: MemoryPool;
}

const runtimeStack: PoolHolder[] = [];

export function pushCurrentRuntime(rt: PoolHolder): void {
  runtimeStack.push(rt);
}

export function popCurrentRuntime(rt: PoolHolder): void {
  const top = runtimeStack[runtimeStack.length - 1];
  if (top !== rt) {
    throw new Error(
      "memoryPool: popCurrentRuntime mismatch — push/pop misnested"
    );
  }
  runtimeStack.pop();
}

export function getCurrentRuntime(): PoolHolder | null {
  const top = runtimeStack[runtimeStack.length - 1];
  return top ?? null;
}

// ── Stats ────────────────────────────────────────────────────────────────

export interface MemoryPoolStats {
  // counts
  attemptedAllocs: number;
  actualAllocs: number;
  cacheHits: number;
  releases: number;
  // bytes (length * 8)
  attemptedBytes: number;
  actualAllocBytes: number;
  cacheHitBytes: number;
  releaseBytes: number;
  // current state
  liveSetSize: number;
  freePoolBufferCount: number;
  freePoolBytes: number;
  freePoolBuckets: Array<{ size: number; count: number }>;
}

// ── MemoryPool ───────────────────────────────────────────────────────────

export class MemoryPool {
  private liveSet = new Set<Float64Array>();
  private freePool = new Map<number, Float64Array[]>();

  // Counters (never reset within an execution).
  private _attemptedAllocs = 0;
  private _actualAllocs = 0;
  private _cacheHits = 0;
  private _releases = 0;
  private _attemptedBytes = 0;
  private _actualAllocBytes = 0;
  private _cacheHitBytes = 0;
  private _releaseBytes = 0;
  private _freePoolBufferCount = 0;
  private _freePoolBytes = 0;

  /** Allocate (or reuse) a zero-filled Float64Array of length `size`. */
  acquire(size: number): Float64Array {
    const bytes = 8 * size;
    this._attemptedAllocs++;
    this._attemptedBytes += bytes;

    const bucket = this.freePool.get(size);
    let buf: Float64Array;
    if (bucket && bucket.length > 0) {
      buf = bucket.pop()!;
      buf.fill(0);
      this._cacheHits++;
      this._cacheHitBytes += bytes;
      this._freePoolBufferCount--;
      this._freePoolBytes -= bytes;
    } else {
      buf = new Float64Array(size);
      this._actualAllocs++;
      this._actualAllocBytes += bytes;
    }
    this.liveSet.add(buf);
    return buf;
  }

  /** Allocate a Float64Array containing `src`'s values (copies). */
  acquireFrom(src: number[] | Float64Array): Float64Array {
    const n = src.length;
    const bytes = 8 * n;
    this._attemptedAllocs++;
    this._attemptedBytes += bytes;

    const bucket = this.freePool.get(n);
    let buf: Float64Array;
    if (bucket && bucket.length > 0) {
      buf = bucket.pop()!;
      buf.set(src);
      this._cacheHits++;
      this._cacheHitBytes += bytes;
      this._freePoolBufferCount--;
      this._freePoolBytes -= bytes;
    } else {
      buf = new Float64Array(src);
      this._actualAllocs++;
      this._actualAllocBytes += bytes;
    }
    this.liveSet.add(buf);
    return buf;
  }

  /**
   * Return a buffer to the free pool. Caller asserts no live wrapper still
   * references this buffer; misuse silently corrupts data. Currently
   * unused — exposed so a future reclamation strategy (refcount, GC-based,
   * region-based) can plug in without touching alloc-side code.
   */
  release(buf: Float64Array): void {
    if (!this.liveSet.has(buf)) return;
    this.liveSet.delete(buf);
    let bucket = this.freePool.get(buf.length);
    if (!bucket) {
      bucket = [];
      this.freePool.set(buf.length, bucket);
    }
    bucket.push(buf);
    const bytes = 8 * buf.length;
    this._releases++;
    this._releaseBytes += bytes;
    this._freePoolBufferCount++;
    this._freePoolBytes += bytes;
  }

  /** Snapshot the pool's stats into a plain JSON-serializable object. */
  getStats(): MemoryPoolStats {
    const buckets: Array<{ size: number; count: number }> = [];
    for (const [size, arr] of this.freePool) {
      if (arr.length > 0) buckets.push({ size, count: arr.length });
    }
    buckets.sort((a, b) => a.size - b.size);

    return {
      attemptedAllocs: this._attemptedAllocs,
      actualAllocs: this._actualAllocs,
      cacheHits: this._cacheHits,
      releases: this._releases,
      attemptedBytes: this._attemptedBytes,
      actualAllocBytes: this._actualAllocBytes,
      cacheHitBytes: this._cacheHitBytes,
      releaseBytes: this._releaseBytes,
      liveSetSize: this.liveSet.size,
      freePoolBufferCount: this._freePoolBufferCount,
      freePoolBytes: this._freePoolBytes,
      freePoolBuckets: buckets,
    };
  }
}
