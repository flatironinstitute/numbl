import { describe, it, expect } from "vitest";
import {
  numel,
  colMajorIndex,
  ind2sub,
  sub2ind,
  tensorSize2D,
} from "../numbl-core/runtime/utils.js";
import { RTV } from "../numbl-core/runtime/constructors.js";
import { valuesAreEqual } from "../numbl-core/runtime/compare.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeTensor,
  isRuntimeCell,
  FloatXArray,
} from "../numbl-core/runtime/types.js";

describe("numel", () => {
  it("returns 1 for scalar shape", () => {
    expect(numel([1, 1])).toBe(1);
  });

  it("computes product of dimensions", () => {
    expect(numel([2, 3])).toBe(6);
    expect(numel([2, 3, 4])).toBe(24);
  });

  it("returns 0 for empty dimension", () => {
    expect(numel([0, 5])).toBe(0);
  });

  it("handles 1D shape", () => {
    expect(numel([5])).toBe(5);
  });
});

describe("colMajorIndex", () => {
  it("computes column-major index for 2D", () => {
    // In a 3x4 matrix, element at row=1, col=2 is at index 2*3+1 = 7
    expect(colMajorIndex(1, 2, 3)).toBe(7);
  });

  it("first element is at index 0", () => {
    expect(colMajorIndex(0, 0, 5)).toBe(0);
  });
});

describe("ind2sub", () => {
  it("converts linear index to subscripts", () => {
    // 3x4 matrix, linear index 7 -> row 1, col 2
    expect(ind2sub([3, 4], 7)).toEqual([1, 2]);
  });

  it("handles index 0", () => {
    expect(ind2sub([3, 4], 0)).toEqual([0, 0]);
  });

  it("handles 3D shape", () => {
    // 2x3x4: index 13 -> (1, 0, 2): 13 = 1 + 0*2 + 2*6
    expect(ind2sub([2, 3, 4], 13)).toEqual([1, 0, 2]);
  });
});

describe("sub2ind", () => {
  it("converts subscripts to linear index", () => {
    expect(sub2ind([3, 4], [1, 2])).toBe(7);
  });

  it("handles index [0,0]", () => {
    expect(sub2ind([3, 4], [0, 0])).toBe(0);
  });

  it("is inverse of ind2sub", () => {
    const shape = [3, 4, 5];
    for (const idx of [0, 1, 10, 59]) {
      expect(sub2ind(shape, ind2sub(shape, idx))).toBe(idx);
    }
  });
});

describe("tensorSize2D", () => {
  it("returns [1,1] for empty shape", () => {
    const t = {
      kind: "tensor" as const,
      data: new FloatXArray(1),
      shape: [],
      _rc: 1,
    };
    expect(tensorSize2D(t)).toEqual([1, 1]);
  });

  it("returns [1,n] for 1D shape", () => {
    const t = {
      kind: "tensor" as const,
      data: new FloatXArray(3),
      shape: [3],
      _rc: 1,
    };
    expect(tensorSize2D(t)).toEqual([1, 3]);
  });

  it("returns shape for 2D", () => {
    const t = {
      kind: "tensor" as const,
      data: new FloatXArray(6),
      shape: [2, 3],
      _rc: 1,
    };
    expect(tensorSize2D(t)).toEqual([2, 3]);
  });
});

describe("type guards", () => {
  it("isRuntimeNumber", () => {
    expect(isRuntimeNumber(42)).toBe(true);
    expect(isRuntimeNumber(0)).toBe(true);
    expect(isRuntimeNumber("hello")).toBe(false);
    expect(isRuntimeNumber(true)).toBe(false);
  });

  it("isRuntimeLogical", () => {
    expect(isRuntimeLogical(true)).toBe(true);
    expect(isRuntimeLogical(false)).toBe(true);
    expect(isRuntimeLogical(1)).toBe(false);
  });

  it("isRuntimeString", () => {
    expect(isRuntimeString("hello")).toBe(true);
    expect(isRuntimeString("")).toBe(true);
    expect(isRuntimeString(42)).toBe(false);
  });

  it("isRuntimeTensor", () => {
    const t = RTV.row([1, 2, 3]);
    expect(isRuntimeTensor(t)).toBe(true);
    expect(isRuntimeTensor(42)).toBe(false);
  });

  it("isRuntimeCell", () => {
    const c = { kind: "cell" as const, data: [1, "hi"], shape: [1, 2], _rc: 1 };
    expect(isRuntimeCell(c)).toBe(true);
    expect(isRuntimeCell(42)).toBe(false);
  });
});

describe("valuesAreEqual", () => {
  it("compares numbers", () => {
    expect(valuesAreEqual(1, 1)).toBe(true);
    expect(valuesAreEqual(1, 2)).toBe(false);
  });

  it("compares booleans", () => {
    expect(valuesAreEqual(true, true)).toBe(true);
    expect(valuesAreEqual(true, false)).toBe(false);
  });

  it("compares strings", () => {
    expect(valuesAreEqual("abc", "abc")).toBe(true);
    expect(valuesAreEqual("abc", "def")).toBe(false);
  });

  it("returns false for different types", () => {
    expect(valuesAreEqual(1, true)).toBe(false);
    expect(valuesAreEqual(1, "1")).toBe(false);
  });

  it("compares tensors by data", () => {
    const a = RTV.row([1, 2, 3]);
    const b = RTV.row([1, 2, 3]);
    const c = RTV.row([1, 2, 4]);
    expect(valuesAreEqual(a, b)).toBe(true);
    expect(valuesAreEqual(a, c)).toBe(false);
  });

  it("compares complex numbers", () => {
    const a = { kind: "complex_number" as const, re: 1, im: 2 };
    const b = { kind: "complex_number" as const, re: 1, im: 2 };
    const c = { kind: "complex_number" as const, re: 1, im: 3 };
    expect(valuesAreEqual(a, b)).toBe(true);
    expect(valuesAreEqual(a, c)).toBe(false);
  });
});

describe("RTV constructors", () => {
  it("creates a number", () => {
    expect(RTV.num(42)).toBe(42);
  });

  it("creates a row vector", () => {
    const r = RTV.row([1, 2, 3]);
    expect(isRuntimeTensor(r)).toBe(true);
    expect(r.shape).toEqual([1, 3]);
    expect(r.data[0]).toBe(1);
    expect(r.data[2]).toBe(3);
  });

  it("creates a column vector", () => {
    const c = RTV.col([4, 5, 6]);
    expect(c.shape).toEqual([3, 1]);
    expect(c.data[1]).toBe(5);
  });

  it("creates a tensor with given shape", () => {
    const t = RTV.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.data.length).toBe(6);
  });
});
