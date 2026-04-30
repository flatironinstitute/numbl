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

const debugFromEnv =
  import.meta.env?.NUMBL_DEBUG_POOL === "true" ? true : false;

const useFloat32 = (FloatXArray as unknown) === Float32Array;

export interface BufferPoolOptions {
  /** Global cap on bytes held in the pool. Defaults to 256 MB. */
  maxBytes?: number;
  /** Per-length cap on the number of buffers retained. Defaults to 16. */
  maxPerBucket?: number;
  /** Scribble released buffers with NaN to make use-after-release visible. */
  debug?: boolean;
}

const DEFAULT_MAX_BYTES = 256 * 1024 * 1024;
const DEFAULT_MAX_PER_BUCKET = 16;

const hasBuffer = typeof Buffer !== "undefined";

export class BufferPool {
  private f64: Map<number, Float64Array[]> = new Map();
  private f32: Map<number, Float32Array[]> = new Map();
  private bytesInPool = 0;
  readonly maxBytes: number;
  readonly maxPerBucket: number;
  readonly debug: boolean;

  constructor(opts?: BufferPoolOptions) {
    this.maxBytes = opts?.maxBytes ?? DEFAULT_MAX_BYTES;
    this.maxPerBucket = opts?.maxPerBucket ?? DEFAULT_MAX_PER_BUCKET;
    this.debug = opts?.debug ?? debugFromEnv;
  }

  acquireF64(n: number): Float64Array {
    if (n === 0) return new Float64Array(0);
    const bucket = this.f64.get(n);
    if (bucket && bucket.length > 0) {
      const buf = bucket.pop()!;
      this.bytesInPool -= n * 8;
      return buf;
    }
    return allocF64(n);
  }

  acquireF32(n: number): Float32Array {
    if (n === 0) return new Float32Array(0);
    const bucket = this.f32.get(n);
    if (bucket && bucket.length > 0) {
      const buf = bucket.pop()!;
      this.bytesInPool -= n * 4;
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

  /** Drop all retained buffers. */
  clear(): void {
    this.f64.clear();
    this.f32.clear();
    this.bytesInPool = 0;
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
