import { describe, it, expect } from "vitest";
import {
  BufferPool,
  getActivePool,
  setActivePool,
} from "../numbl-core/runtime/bufferPool.js";

describe("BufferPool", () => {
  it("acquire from empty pool allocates a fresh buffer of the requested length", () => {
    const pool = new BufferPool();
    const buf = pool.acquireF64(100);
    expect(buf).toBeInstanceOf(Float64Array);
    expect(buf.length).toBe(100);
    expect(pool.bucketSize("f64", 100)).toBe(0);
    expect(pool.totalBytes()).toBe(0);
  });

  it("release + acquire of the same length round-trips the same buffer", () => {
    const pool = new BufferPool();
    const buf = pool.acquireF64(64);
    buf[0] = 42;
    expect(pool.release(buf)).toBe(true);
    expect(pool.bucketSize("f64", 64)).toBe(1);
    const reacquired = pool.acquireF64(64);
    expect(reacquired).toBe(buf);
    expect(pool.bucketSize("f64", 64)).toBe(0);
  });

  it("acquire of a different length does not reuse a pooled buffer", () => {
    const pool = new BufferPool();
    pool.release(pool.acquireF64(32));
    const other = pool.acquireF64(33);
    expect(other.length).toBe(33);
    expect(pool.bucketSize("f64", 32)).toBe(1);
  });

  it("Float32 and Float64 buckets are independent", () => {
    const pool = new BufferPool();
    const f64 = pool.acquireF64(16);
    const f32 = pool.acquireF32(16);
    pool.release(f64);
    pool.release(f32);
    expect(pool.bucketSize("f64", 16)).toBe(1);
    expect(pool.bucketSize("f32", 16)).toBe(1);
    expect(pool.totalBytes()).toBe(16 * 8 + 16 * 4);
  });

  it("zero-length acquire/release is a no-op for the pool", () => {
    const pool = new BufferPool();
    const buf = pool.acquireF64(0);
    expect(buf.length).toBe(0);
    expect(pool.release(buf)).toBe(false);
    expect(pool.totalBytes()).toBe(0);
  });

  it("release respects maxPerBucket — extra buffers are discarded", () => {
    const pool = new BufferPool({ maxPerBucket: 2 });
    // Pre-allocate three distinct buffers so each release sees a fresh
    // buffer (otherwise an inline acquireF64 between releases would pop the
    // just-released buffer back out and the cap would never be reached).
    const a = pool.acquireF64(8);
    const b = pool.acquireF64(8);
    const c = pool.acquireF64(8);
    expect(pool.release(a)).toBe(true);
    expect(pool.release(b)).toBe(true);
    expect(pool.release(c)).toBe(false);
    expect(pool.bucketSize("f64", 8)).toBe(2);
  });

  it("release respects maxBytes — large buffer is discarded once cap hit", () => {
    // Cap = 1 KB. First 100-element f64 (800 bytes) fits; second one would
    // push past the cap and is discarded.
    const pool = new BufferPool({ maxBytes: 1024 });
    const a = pool.acquireF64(100);
    const b = pool.acquireF64(100);
    expect(pool.release(a)).toBe(true);
    expect(pool.release(b)).toBe(false);
    expect(pool.bucketSize("f64", 100)).toBe(1);
    expect(pool.totalBytes()).toBe(800);
  });

  it("totalBytes tracks acquire and release across both kinds", () => {
    const pool = new BufferPool();
    const a = pool.acquireF64(10);
    const b = pool.acquireF32(20);
    expect(pool.totalBytes()).toBe(0);
    pool.release(a);
    expect(pool.totalBytes()).toBe(10 * 8);
    pool.release(b);
    expect(pool.totalBytes()).toBe(10 * 8 + 20 * 4);
    const a2 = pool.acquireF64(10);
    expect(a2).toBe(a);
    expect(pool.totalBytes()).toBe(20 * 4);
  });

  it("clear drops all retained buffers", () => {
    const pool = new BufferPool();
    pool.release(pool.acquireF64(4));
    pool.release(pool.acquireF32(8));
    expect(pool.totalBytes()).toBeGreaterThan(0);
    pool.clear();
    expect(pool.totalBytes()).toBe(0);
    expect(pool.bucketSize("f64", 4)).toBe(0);
    expect(pool.bucketSize("f32", 8)).toBe(0);
  });

  it("debug mode scribbles released buffers with NaN", () => {
    const pool = new BufferPool({ debug: true });
    const buf = pool.acquireF64(4);
    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    buf[3] = 4;
    pool.release(buf);
    // After release, every element is NaN — a use-after-release read sees
    // NaN propagation rather than silently aliasing the next owner's bytes.
    for (let i = 0; i < 4; i++) expect(buf[i]).toBeNaN();
    // Re-acquiring returns the same (still scribbled) buffer; the contract
    // is that the caller writes every element before reading it.
    const reacquired = pool.acquireF64(4);
    expect(reacquired).toBe(buf);
    for (let i = 0; i < 4; i++) expect(reacquired[i]).toBeNaN();
  });

  it("non-debug mode preserves released contents (uninit semantics)", () => {
    const pool = new BufferPool({ debug: false });
    const buf = pool.acquireF64(4);
    buf[0] = 7;
    pool.release(buf);
    expect(buf[0]).toBe(7);
  });

  it("two pools are isolated — buffers released into A do not appear in B", () => {
    const a = new BufferPool();
    const b = new BufferPool();
    a.release(a.acquireF64(16));
    expect(a.bucketSize("f64", 16)).toBe(1);
    expect(b.bucketSize("f64", 16)).toBe(0);
    const fromB = b.acquireF64(16);
    expect(a.bucketSize("f64", 16)).toBe(1); // unchanged
    expect(fromB.length).toBe(16);
  });

  it("setActivePool / getActivePool swap the module-level default", () => {
    const original = getActivePool();
    const fresh = new BufferPool();
    setActivePool(fresh);
    expect(getActivePool()).toBe(fresh);
    setActivePool(original);
    expect(getActivePool()).toBe(original);
  });
});
