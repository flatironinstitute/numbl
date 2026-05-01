/**
 * Tensor buffer pool: per-length free lists for Float64Array / Float32Array.
 *
 * Tensors dominate heap usage and are typically allocated and discarded in
 * shape-stable patterns (the same length appears repeatedly). The pool lets
 * us hand the same backing buffer back to the next allocation of the
 * matching length instead of paying for `new Float64Array(n)` (zero-fill) or
 * `Buffer.allocUnsafe(...)` (Node slab) every time.
 *
 * PR 2 wires the pool through `uninitFloat64` / `uninitFloatX` for
 * acquisition; release is opt-in (only test code calls `pool.release(buf)`
 * for now). PR 3 will insert release calls at scope/assignment seams so the
 * pool actually fills under real workloads.
 *
 * Use-after-release safety: `NUMBL_DEBUG_POOL=true` enables sentinel mode,
 * which fills released buffers with NaN. A subsequent stale read by a caller
 * that retained a reference past the release will surface as NaN propagation
 * rather than silently reusing the new owner's bytes.
 */

import { FloatXArray } from "./types.js";
import type { RuntimeTensor } from "./types.js";

const debugFromEnv =
  import.meta.env?.NUMBL_DEBUG_POOL === "true" ||
  (typeof process !== "undefined" && process.env?.NUMBL_DEBUG_POOL === "true")
    ? true
    : false;

const useFloat32 = (FloatXArray as unknown) === Float32Array;

export interface BufferPoolOptions {
  /** Global cap on bytes held in the pool. Defaults to 1 GB. */
  maxBytes?: number;
  /** Per-length cap on the number of buffers retained. Defaults to 256. */
  maxPerBucket?: number;
  /** Scribble released buffers with NaN to make use-after-release visible. */
  debug?: boolean;
}

const DEFAULT_MAX_BYTES = 1024 * 1024 * 1024; // 1 GB
const DEFAULT_MAX_PER_BUCKET = 256;

const hasBuffer = typeof Buffer !== "undefined";

/** Telemetry snapshot of a BufferPool. Exposed via ProfileData so the
 *  --profile output can show acquire / release totals and the pool's
 *  effectiveness — divergence between `acquireBytes` and `releaseBytes`
 *  signals a leak. */
export interface BufferPoolStats {
  /** Total acquireF64 + acquireF32 calls (excluding zero-length). */
  acquireCount: number;
  /** Total bytes returned by acquire calls. */
  acquireBytes: number;
  /** Subset of acquireCount that came from the pool's free list (vs. a
   *  fresh allocation). `acquireHits / acquireCount` is the reuse rate. */
  acquireHits: number;
  /** Total release() calls that ran (including those that were discarded
   *  due to caps). */
  releaseCount: number;
  /** Total bytes that were attempted to be released. */
  releaseBytes: number;
  /** Bytes currently sitting in the pool's buckets (not yet handed back
   *  to a caller). */
  currentBytes: number;
}

export class BufferPool {
  private f64: Map<number, Float64Array[]> = new Map();
  private f32: Map<number, Float32Array[]> = new Map();
  private bytesInPool = 0;
  readonly maxBytes: number;
  readonly maxPerBucket: number;
  readonly debug: boolean;

  // ── Telemetry counters ───────────────────────────────────────────────
  // Always tracked (negligible cost); surfaced via stats() / --profile.
  private acquireCount = 0;
  private acquireBytes = 0;
  private acquireHits = 0;
  private releaseCount = 0;
  private releaseBytes = 0;

  constructor(opts?: BufferPoolOptions) {
    this.maxBytes = opts?.maxBytes ?? DEFAULT_MAX_BYTES;
    this.maxPerBucket = opts?.maxPerBucket ?? DEFAULT_MAX_PER_BUCKET;
    this.debug = opts?.debug ?? debugFromEnv;
  }

  acquireF64(n: number): Float64Array {
    if (n === 0) return new Float64Array(0);
    this.acquireCount++;
    this.acquireBytes += n * 8;
    const bucket = this.f64.get(n);
    if (bucket && bucket.length > 0) {
      const buf = bucket.pop()!;
      this.bytesInPool -= n * 8;
      this.acquireHits++;
      return buf;
    }
    return allocF64(n);
  }

  acquireF32(n: number): Float32Array {
    if (n === 0) return new Float32Array(0);
    this.acquireCount++;
    this.acquireBytes += n * 4;
    const bucket = this.f32.get(n);
    if (bucket && bucket.length > 0) {
      const buf = bucket.pop()!;
      this.bytesInPool -= n * 4;
      this.acquireHits++;
      return buf;
    }
    return allocF32(n);
  }

  /** Return a buffer to the pool. Caller must not retain the reference.
   *  Returns true if pooled, false if discarded (caps hit, zero length, or
   *  unrecognized type). */
  release(buf: Float64Array | Float32Array): boolean {
    const n = buf.length;
    if (n === 0) return false;
    const bytes = buf.BYTES_PER_ELEMENT * n;
    this.releaseCount++;
    this.releaseBytes += bytes;
    let bucketMap: Map<number, Float64Array[] | Float32Array[]>;
    if (buf instanceof Float64Array) {
      bucketMap = this.f64;
    } else if (buf instanceof Float32Array) {
      bucketMap = this.f32;
    } else {
      return false;
    }
    let bucket = bucketMap.get(n);
    if (!bucket) {
      bucket = [];
      bucketMap.set(n, bucket as Float64Array[] & Float32Array[]);
    }
    if (bucket.length >= this.maxPerBucket) return false;
    if (this.bytesInPool + bytes > this.maxBytes) return false;
    if (this.debug) buf.fill(NaN);
    (bucket as (Float64Array | Float32Array)[]).push(buf);
    this.bytesInPool += bytes;
    return true;
  }

  /** Total bytes currently held in the pool. */
  totalBytes(): number {
    return this.bytesInPool;
  }

  /** Number of buffers retained for a given length and kind. */
  bucketSize(kind: "f64" | "f32", n: number): number {
    const m = kind === "f64" ? this.f64 : this.f32;
    return m.get(n)?.length ?? 0;
  }

  /** Drop all retained buffers. Does not reset telemetry counters. */
  clear(): void {
    this.f64.clear();
    this.f32.clear();
    this.bytesInPool = 0;
  }

  /** Snapshot of acquire/release telemetry. */
  stats(): BufferPoolStats {
    return {
      acquireCount: this.acquireCount,
      acquireBytes: this.acquireBytes,
      acquireHits: this.acquireHits,
      releaseCount: this.releaseCount,
      releaseBytes: this.releaseBytes,
      currentBytes: this.bytesInPool,
    };
  }
}

function allocF64(n: number): Float64Array {
  if (hasBuffer) {
    const b = Buffer.allocUnsafe(n * 8);
    return new Float64Array(b.buffer as ArrayBuffer, b.byteOffset, n);
  }
  return new Float64Array(n);
}

function allocF32(n: number): Float32Array {
  if (hasBuffer) {
    const b = Buffer.allocUnsafe(n * 4);
    return new Float32Array(b.buffer as ArrayBuffer, b.byteOffset, n);
  }
  return new Float32Array(n);
}

/** The active pool that `uninitFloat64` / `uninitFloatX` route through.
 *  Defaults to a process-wide pool; `Runtime` replaces it on construction so
 *  serially-active runtimes don't share buffers. */
let _activePool = new BufferPool();

export function getActivePool(): BufferPool {
  return _activePool;
}

export function setActivePool(p: BufferPool): void {
  _activePool = p;
}

/** Acquire from the active pool, dispatching to the FloatX kind in use. */
export function acquireFloatX(n: number): InstanceType<typeof FloatXArray> {
  return (
    useFloat32 ? _activePool.acquireF32(n) : _activePool.acquireF64(n)
  ) as InstanceType<typeof FloatXArray>;
}

// ── Tensor release helpers ─────────────────────────────────────────────
//
// PR 3 wiring: every binding seam (Environment overwrites, function exit,
// JIT Assign overwrites and epilogues) calls one of these to decrement the
// shared `_refs.c` and pool the buffer when the count hits zero.

/** Decrement a tensor's shared refcount; if the buffer becomes unaliased,
 *  return its data (and imag, if present) to the active pool. Throws if
 *  the count would go negative — that indicates a double-release bug in
 *  the runtime and is never legal. */
export function releaseTensor(t: RuntimeTensor): void {
  const next = --t._refs.c;
  if (next === 0) {
    _activePool.release(t.data);
    if (t.imag) _activePool.release(t.imag);
  } else if (next < 0) {
    throw new Error(
      `Tensor refcount underflow: _refs.c went to ${next}. ` +
        `This indicates a double-release in the runtime — a slot dropped ` +
        `an alias that was never registered (or was released twice).`
    );
  }
}

/** Release `v` only if it is a RuntimeTensor. Cheap no-op for everything
 *  else — usable as a blanket cleanup at scope exits where the slot type
 *  is not statically known. */
export function releaseIfTensor(v: unknown): void {
  if (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeTensor).kind === "tensor"
  ) {
    releaseTensor(v as RuntimeTensor);
  }
}

/** Bump the shared refcount for a tensor wrapper. Used when a value gains
 *  a new aliasing slot (function output, persistent store, return-value
 *  bump before clearLocals) so that the matching release at the other end
 *  doesn't pool a still-aliased buffer. No-op for non-tensors. */
export function retainIfTensor(v: unknown): void {
  if (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeTensor).kind === "tensor"
  ) {
    (v as RuntimeTensor)._refs.c++;
  }
}

/** Assign-with-release: release `prev` (if a tensor and not the same
 *  wrapper as `next`) and return `next`. Used by JIT codegen so the dest-
 *  hint reuse path stays intact: when the RHS op writes into prev's buffer
 *  in place, it returns the same wrapper, prev === next, and no release
 *  fires. */
export function assignReleasing<T>(prev: unknown, next: T): T {
  if (prev !== next && prev !== undefined) releaseIfTensor(prev);
  return next;
}
