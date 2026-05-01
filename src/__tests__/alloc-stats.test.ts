import { describe, it, expect, beforeEach } from "vitest";
import {
  allocFloat64,
  allocFloatX,
  zeroedFloat64,
  zeroedFloatX,
  copyFloat64,
  copyFloatX,
  disposeFloat64,
  disposeFloatX,
  getAllocStats,
  resetAllocStats,
  clearPool,
} from "../numbl-core/runtime/alloc.js";
import { FloatXArray } from "../numbl-core/runtime/types.js";

const FLOATX_BYTES = (FloatXArray as unknown) === Float32Array ? 4 : 8;

const ZERO_STATS = {
  allocCount: 0,
  allocBytes: 0,
  freshAllocBytes: 0,
  disposeCount: 0,
  disposeBytes: 0,
  poolHits: 0,
  poolMisses: 0,
  poolBuffersHeld: 0,
  poolBytesHeld: 0,
};

describe("alloc-stats", () => {
  beforeEach(() => {
    clearPool();
    resetAllocStats();
  });

  it("counts each non-zero allocation once", () => {
    allocFloat64(10);
    allocFloat64(5);
    expect(getAllocStats()).toMatchObject({
      allocCount: 2,
      allocBytes: 10 * 8 + 5 * 8,
    });
  });

  it("ignores zero-length allocations and disposals", () => {
    allocFloat64(0);
    allocFloatX(0);
    zeroedFloat64(0);
    zeroedFloatX(0);
    disposeFloat64(new Float64Array(0));
    disposeFloatX(new FloatXArray(0));
    expect(getAllocStats()).toEqual(ZERO_STATS);
  });

  it("tallies allocFloat64 (uninitialized) bytes correctly", () => {
    const buf = allocFloat64(7);
    expect(buf).toBeInstanceOf(Float64Array);
    expect(buf.length).toBe(7);
    expect(getAllocStats()).toMatchObject({ allocCount: 1, allocBytes: 56 });
  });

  it("tallies zeroedFloat64 and returns zero-filled buffer", () => {
    const buf = zeroedFloat64(4);
    expect(buf).toBeInstanceOf(Float64Array);
    expect(Array.from(buf)).toEqual([0, 0, 0, 0]);
    expect(getAllocStats()).toMatchObject({ allocCount: 1, allocBytes: 32 });
  });

  it("tallies copyFloat64 from a number array", () => {
    const buf = copyFloat64([1, 2, 3]);
    expect(Array.from(buf)).toEqual([1, 2, 3]);
    expect(getAllocStats()).toMatchObject({ allocCount: 1, allocBytes: 24 });
  });

  it("tallies copyFloat64 from a typed array", () => {
    const src = new Float32Array([4, 5]);
    const buf = copyFloat64(src);
    expect(Array.from(buf)).toEqual([4, 5]);
    expect(getAllocStats()).toMatchObject({ allocCount: 1, allocBytes: 16 });
  });

  it("tallies allocFloatX with the active precision", () => {
    allocFloatX(3);
    expect(getAllocStats()).toMatchObject({
      allocCount: 1,
      allocBytes: 3 * FLOATX_BYTES,
    });
  });

  it("tallies zeroedFloatX with the active precision", () => {
    const buf = zeroedFloatX(2);
    expect(Array.from(buf)).toEqual([0, 0]);
    expect(getAllocStats()).toMatchObject({
      allocCount: 1,
      allocBytes: 2 * FLOATX_BYTES,
    });
  });

  it("tallies copyFloatX from an array", () => {
    copyFloatX([1, 2, 3, 4]);
    expect(getAllocStats()).toMatchObject({
      allocCount: 1,
      allocBytes: 4 * FLOATX_BYTES,
    });
  });

  it("resetAllocStats zeroes the alloc/dispose/hit counters", () => {
    allocFloat64(100);
    expect(getAllocStats().allocCount).toBe(1);
    resetAllocStats();
    expect(getAllocStats()).toMatchObject({
      allocCount: 0,
      allocBytes: 0,
      disposeCount: 0,
      disposeBytes: 0,
      poolHits: 0,
      poolMisses: 0,
    });
  });

  it("multiple paths share the same counter", () => {
    allocFloat64(1);
    zeroedFloat64(2);
    copyFloat64([0, 0, 0]);
    allocFloatX(4);
    zeroedFloatX(5);
    copyFloatX([0, 0, 0, 0, 0, 0]);
    const s = getAllocStats();
    expect(s.allocCount).toBe(6);
    expect(s.allocBytes).toBe(
      8 + 16 + 24 + 4 * FLOATX_BYTES + 5 * FLOATX_BYTES + 6 * FLOATX_BYTES
    );
  });
});

describe("alloc-stats: pool + dispose", () => {
  beforeEach(() => {
    clearPool();
    resetAllocStats();
  });

  it("dispose tallies bytes and pushes onto the pool", () => {
    const buf = allocFloat64(8);
    disposeFloat64(buf);
    const s = getAllocStats();
    expect(s.disposeCount).toBe(1);
    expect(s.disposeBytes).toBe(64);
    expect(s.poolBuffersHeld).toBe(1);
    expect(s.poolBytesHeld).toBe(64);
  });

  it("alloc draws from the pool when a matching length is held", () => {
    const a = allocFloat64(16);
    disposeFloat64(a);
    expect(getAllocStats().poolBuffersHeld).toBe(1);
    const b = allocFloat64(16);
    expect(b).toBe(a); // same buffer instance returned
    const s = getAllocStats();
    expect(s.poolHits).toBe(1);
    expect(s.poolMisses).toBe(1); // only the first alloc missed
    expect(s.poolBuffersHeld).toBe(0);
  });

  it("zeroedFloat64 zero-fills a recycled buffer", () => {
    const a = allocFloat64(4);
    a[0] = 7;
    a[1] = 8;
    a[2] = 9;
    a[3] = 10;
    disposeFloat64(a);
    const b = zeroedFloat64(4);
    expect(b).toBe(a);
    expect(Array.from(b)).toEqual([0, 0, 0, 0]);
    expect(getAllocStats().poolHits).toBe(1);
  });

  it("copyFloat64 reuses a pooled buffer", () => {
    const a = allocFloat64(3);
    disposeFloat64(a);
    const b = copyFloat64([4, 5, 6]);
    expect(b).toBe(a);
    expect(Array.from(b)).toEqual([4, 5, 6]);
  });

  it("disposing a different length does not satisfy alloc of another length", () => {
    const a = allocFloat64(10);
    disposeFloat64(a);
    const b = allocFloat64(5);
    expect(b).not.toBe(a);
    const s = getAllocStats();
    expect(s.poolHits).toBe(0);
    expect(s.poolMisses).toBe(2);
    expect(s.poolBuffersHeld).toBe(1); // the size-10 buffer still held
  });

  it("disposeFloatX tallies bytes for the active precision", () => {
    const buf = allocFloatX(5);
    disposeFloatX(buf);
    const s = getAllocStats();
    expect(s.disposeCount).toBe(1);
    expect(s.disposeBytes).toBe(5 * FLOATX_BYTES);
    expect(s.poolBuffersHeld).toBe(1);
    expect(s.poolBytesHeld).toBe(5 * FLOATX_BYTES);
  });

  it("allocFloatX recycles buffers", () => {
    const a = allocFloatX(7);
    disposeFloatX(a);
    const b = allocFloatX(7);
    expect(b).toBe(a);
    expect(getAllocStats().poolHits).toBe(1);
  });

  it("clearPool drops held buffers and resets buffers-held counters", () => {
    const a = allocFloat64(4);
    const b = allocFloat64(4);
    disposeFloat64(a);
    disposeFloat64(b);
    expect(getAllocStats().poolBuffersHeld).toBe(2);
    clearPool();
    const s = getAllocStats();
    expect(s.poolBuffersHeld).toBe(0);
    expect(s.poolBytesHeld).toBe(0);
    // alloc/dispose counters are NOT cleared by clearPool
    expect(s.allocCount).toBe(2);
    expect(s.disposeCount).toBe(2);
  });
});
