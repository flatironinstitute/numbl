import { describe, it, expect, beforeEach } from "vitest";
import {
  allocFloat64,
  allocFloatX,
  zeroedFloat64,
  zeroedFloatX,
  copyFloat64,
  copyFloatX,
  getAllocStats,
  resetAllocStats,
} from "../numbl-core/runtime/alloc.js";
import { FloatXArray } from "../numbl-core/runtime/types.js";

const FLOATX_BYTES = (FloatXArray as unknown) === Float32Array ? 4 : 8;

describe("alloc-stats", () => {
  beforeEach(() => {
    resetAllocStats();
  });

  it("counts each non-zero allocation once", () => {
    allocFloat64(10);
    allocFloat64(5);
    expect(getAllocStats()).toEqual({ count: 2, bytes: 10 * 8 + 5 * 8 });
  });

  it("ignores zero-length allocations", () => {
    allocFloat64(0);
    allocFloatX(0);
    zeroedFloat64(0);
    zeroedFloatX(0);
    expect(getAllocStats()).toEqual({ count: 0, bytes: 0 });
  });

  it("tallies allocFloat64 (uninitialized) bytes correctly", () => {
    const buf = allocFloat64(7);
    expect(buf).toBeInstanceOf(Float64Array);
    expect(buf.length).toBe(7);
    expect(getAllocStats()).toEqual({ count: 1, bytes: 56 });
  });

  it("tallies zeroedFloat64 and returns zero-filled buffer", () => {
    const buf = zeroedFloat64(4);
    expect(buf).toBeInstanceOf(Float64Array);
    expect(Array.from(buf)).toEqual([0, 0, 0, 0]);
    expect(getAllocStats()).toEqual({ count: 1, bytes: 32 });
  });

  it("tallies copyFloat64 from a number array", () => {
    const buf = copyFloat64([1, 2, 3]);
    expect(Array.from(buf)).toEqual([1, 2, 3]);
    expect(getAllocStats()).toEqual({ count: 1, bytes: 24 });
  });

  it("tallies copyFloat64 from a typed array", () => {
    const src = new Float32Array([4, 5]);
    const buf = copyFloat64(src);
    expect(Array.from(buf)).toEqual([4, 5]);
    expect(getAllocStats()).toEqual({ count: 1, bytes: 16 });
  });

  it("tallies allocFloatX with the active precision", () => {
    allocFloatX(3);
    expect(getAllocStats()).toEqual({ count: 1, bytes: 3 * FLOATX_BYTES });
  });

  it("tallies zeroedFloatX with the active precision", () => {
    const buf = zeroedFloatX(2);
    expect(Array.from(buf)).toEqual([0, 0]);
    expect(getAllocStats()).toEqual({ count: 1, bytes: 2 * FLOATX_BYTES });
  });

  it("tallies copyFloatX from an array", () => {
    copyFloatX([1, 2, 3, 4]);
    expect(getAllocStats()).toEqual({ count: 1, bytes: 4 * FLOATX_BYTES });
  });

  it("resetAllocStats zeroes the counters", () => {
    allocFloat64(100);
    expect(getAllocStats().count).toBe(1);
    resetAllocStats();
    expect(getAllocStats()).toEqual({ count: 0, bytes: 0 });
  });

  it("multiple paths share the same counter", () => {
    allocFloat64(1);
    zeroedFloat64(2);
    copyFloat64([0, 0, 0]);
    allocFloatX(4);
    zeroedFloatX(5);
    copyFloatX([0, 0, 0, 0, 0, 0]);
    const s = getAllocStats();
    expect(s.count).toBe(6);
    expect(s.bytes).toBe(
      8 + 16 + 24 + 4 * FLOATX_BYTES + 5 * FLOATX_BYTES + 6 * FLOATX_BYTES
    );
  });
});
