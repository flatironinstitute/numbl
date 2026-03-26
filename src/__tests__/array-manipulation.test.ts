import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeTensor } from "../numbl-core/runtime/types.js";

function num(code: string, varName: string): number {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (typeof v !== "number") throw new Error(`${varName} is not a number`);
  return v;
}

function tensorData(code: string, varName: string): number[] {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (!isRuntimeTensor(v)) throw new Error(`${varName} is not a tensor`);
  return Array.from(v.data);
}

function tensorShape(code: string, varName: string): number[] {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (!isRuntimeTensor(v)) throw new Error(`${varName} is not a tensor`);
  return v.shape;
}

// ── flip operations ────────────────────────────────────────────────────

describe("fliplr", () => {
  it("reverses a row vector", () => {
    expect(tensorData("v = fliplr([1,2,3,4]);", "v")).toEqual([4, 3, 2, 1]);
  });

  it("reverses columns of a matrix", () => {
    // [1 2; 3 4] -> [2 1; 4 3]
    const data = tensorData("A = [1,2;3,4]; B = fliplr(A);", "B");
    // column-major: [2,4,1,3]
    expect(data[0]).toBe(2);
    expect(data[1]).toBe(4);
    expect(data[2]).toBe(1);
    expect(data[3]).toBe(3);
  });

  it("scalar is unchanged", () => {
    expect(num("x = fliplr(5);", "x")).toBe(5);
  });
});

describe("flipud", () => {
  it("reverses a column vector", () => {
    const data = tensorData("v = flipud([1;2;3]);", "v");
    expect(data).toEqual([3, 2, 1]);
  });

  it("reverses rows of a matrix", () => {
    // [1 2; 3 4] -> [3 4; 1 2]
    const data = tensorData("A = [1,2;3,4]; B = flipud(A);", "B");
    // column-major: [3,1,4,2]
    expect(data[0]).toBe(3);
    expect(data[1]).toBe(1);
    expect(data[2]).toBe(4);
    expect(data[3]).toBe(2);
  });

  it("scalar is unchanged", () => {
    expect(num("x = flipud(5);", "x")).toBe(5);
  });
});

describe("flip", () => {
  it("flip along dim 1 reverses rows", () => {
    const data = tensorData("A = [1,2;3,4]; B = flip(A, 1);", "B");
    expect(data[0]).toBe(3);
    expect(data[1]).toBe(1);
  });

  it("flip along dim 2 reverses cols", () => {
    const data = tensorData("A = [1,2;3,4]; B = flip(A, 2);", "B");
    expect(data[0]).toBe(2);
    expect(data[2]).toBe(1);
  });

  it("flip of row vector with default dim", () => {
    expect(tensorData("v = flip([1,2,3]);", "v")).toEqual([3, 2, 1]);
  });
});

// ── rot90 ──────────────────────────────────────────────────────────────

describe("rot90", () => {
  it("rotates 2x2 matrix 90 CCW", () => {
    // [1 2; 3 4] -> [2 4; 1 3]
    const shape = tensorShape("A = [1,2;3,4]; B = rot90(A);", "B");
    expect(shape).toEqual([2, 2]);
  });

  it("rotation 4 times returns original", () => {
    const data = tensorData("A = [1,2;3,4]; B = rot90(A, 4);", "B");
    expect(data).toEqual([1, 3, 2, 4]); // column-major: same as original
  });

  it("scalar is unchanged", () => {
    expect(num("x = rot90(5);", "x")).toBe(5);
  });
});

// ── repmat ────────────────────────────────────────────────────────────

describe("repmat", () => {
  it("repeats scalar into matrix", () => {
    const shape = tensorShape("A = repmat(1, 2, 3);", "A");
    expect(shape).toEqual([2, 3]);
  });

  it("tiles a vector horizontally", () => {
    const data = tensorData("v = repmat([1,2], 1, 3);", "v");
    expect(data).toEqual([1, 2, 1, 2, 1, 2]);
  });

  it("tiles a vector vertically", () => {
    const shape = tensorShape("v = repmat([1,2], 3, 1);", "v");
    expect(shape).toEqual([3, 2]);
  });

  it("repmat with scalar n tiles n x n", () => {
    const shape = tensorShape("A = repmat([1,2], 2);", "A");
    expect(shape).toEqual([2, 4]);
  });
});

// ── repelem ───────────────────────────────────────────────────────────

describe("repelem", () => {
  it("repeats each element n times", () => {
    const data = tensorData("v = repelem([1,2,3], 2);", "v");
    expect(data).toEqual([1, 1, 2, 2, 3, 3]);
  });

  it("repelem of scalar", () => {
    const shape = tensorShape("v = repelem(5, 3);", "v");
    expect(shape).toEqual([1, 3]);
  });

  it("repelem with r,c args", () => {
    const shape = tensorShape("A = repelem([1,2;3,4], 2, 3);", "A");
    expect(shape).toEqual([4, 6]);
  });
});

// ── circshift ─────────────────────────────────────────────────────────

describe("circshift", () => {
  it("shifts row vector by 1", () => {
    const data = tensorData("v = circshift([1,2,3,4], 1);", "v");
    expect(data).toEqual([4, 1, 2, 3]);
  });

  it("shifts row vector by -1", () => {
    const data = tensorData("v = circshift([1,2,3,4], -1);", "v");
    expect(data).toEqual([2, 3, 4, 1]);
  });

  it("shifts column vector", () => {
    const data = tensorData("v = circshift([1;2;3], 1);", "v");
    expect(data).toEqual([3, 1, 2]);
  });
});

// ── squeeze ───────────────────────────────────────────────────────────

describe("squeeze", () => {
  it("squeeze 2D matrix is no-op", () => {
    const shape = tensorShape("A = [1,2;3,4]; B = squeeze(A);", "B");
    expect(shape).toEqual([2, 2]);
  });

  it("squeeze scalar is no-op", () => {
    expect(num("x = squeeze(5);", "x")).toBe(5);
  });
});

// ── cat ───────────────────────────────────────────────────────────────

describe("cat", () => {
  it("cat(1, A, B) concatenates vertically", () => {
    const shape = tensorShape("C = cat(1, [1,2], [3,4]);", "C");
    expect(shape).toEqual([2, 2]);
  });

  it("cat(2, A, B) concatenates horizontally", () => {
    const shape = tensorShape("C = cat(2, [1,2], [3,4]);", "C");
    expect(shape).toEqual([1, 4]);
  });

  it("cat(2, ...) matches bracket horzcat", () => {
    const data = tensorData("C = cat(2, [1,2], [3,4]);", "C");
    expect(data).toEqual([1, 2, 3, 4]);
  });
});

// ── horzcat / vertcat builtins ─────────────────────────────────────────

describe("horzcat / vertcat builtins", () => {
  it("horzcat function concatenates horizontally", () => {
    const shape = tensorShape("C = horzcat([1,2], [3,4,5]);", "C");
    expect(shape).toEqual([1, 5]);
  });

  it("vertcat function concatenates vertically", () => {
    const shape = tensorShape("C = vertcat([1,2], [3,4]);", "C");
    expect(shape).toEqual([2, 2]);
  });
});

// ── permute ───────────────────────────────────────────────────────────

describe("permute", () => {
  it("permute [2,1] transposes 2D matrix", () => {
    const shape = tensorShape("A = [1,2,3;4,5,6]; B = permute(A, [2,1]);", "B");
    expect(shape).toEqual([3, 2]);
  });

  it("permute identity [1,2] is no-op", () => {
    const data = tensorData("A = [1,2;3,4]; B = permute(A, [1,2]);", "B");
    expect(data).toEqual([1, 3, 2, 4]);
  });
});

// ── sub2ind / ind2sub builtins ─────────────────────────────────────────

describe("sub2ind builtin", () => {
  it("converts subscripts to linear index", () => {
    // [3,4] matrix, row=2, col=3 (1-based) -> 3*3+2-1 = 8? No:
    // col-major: index = (col-1)*nrows + row = (3-1)*3 + 2 = 8 (1-based)
    expect(num("i = sub2ind([3,4], 2, 3);", "i")).toBe(8);
  });

  it("first element is 1", () => {
    expect(num("i = sub2ind([3,4], 1, 1);", "i")).toBe(1);
  });
});

describe("ind2sub builtin", () => {
  it("converts linear index to subscripts", () => {
    const result = executeCode("[r, c] = ind2sub([3,4], 8);");
    expect(result.variableValues["r"]).toBe(2);
    expect(result.variableValues["c"]).toBe(3);
  });
});

// ── meshgrid ─────────────────────────────────────────────────────────

describe("meshgrid", () => {
  it("meshgrid produces correct shapes", () => {
    const result = executeCode("[X, Y] = meshgrid(1:3, 1:2);");
    const X = result.variableValues["X"];
    const Y = result.variableValues["Y"];
    expect(isRuntimeTensor(X)).toBe(true);
    expect(isRuntimeTensor(Y)).toBe(true);
    if (isRuntimeTensor(X)) expect(X.shape).toEqual([2, 3]);
    if (isRuntimeTensor(Y)) expect(Y.shape).toEqual([2, 3]);
  });
});

// ── ndgrid ────────────────────────────────────────────────────────────

describe("ndgrid", () => {
  it("ndgrid produces correct shapes", () => {
    const result = executeCode("[X, Y] = ndgrid(1:3, 1:4);");
    const X = result.variableValues["X"];
    const Y = result.variableValues["Y"];
    expect(isRuntimeTensor(X)).toBe(true);
    expect(isRuntimeTensor(Y)).toBe(true);
    if (isRuntimeTensor(X)) expect(X.shape).toEqual([3, 4]);
    if (isRuntimeTensor(Y)) expect(Y.shape).toEqual([3, 4]);
  });
});

// ── diag with offset ──────────────────────────────────────────────────

describe("diag with offset", () => {
  it("diag(v, 1) creates superdiagonal matrix", () => {
    const shape = tensorShape("A = diag([1,2], 1);", "A");
    expect(shape).toEqual([3, 3]);
  });

  it("diag(v, -1) creates subdiagonal matrix", () => {
    const shape = tensorShape("A = diag([1,2], -1);", "A");
    expect(shape).toEqual([3, 3]);
  });

  it("diag extracts superdiagonal", () => {
    const data = tensorData("A = [1,2,3;4,5,6;7,8,9]; d = diag(A, 1);", "d");
    expect(data[0]).toBe(2);
    expect(data[1]).toBe(6);
  });
});

// ── reshape with size vector ───────────────────────────────────────────

describe("reshape with size vector", () => {
  it("reshape(A, [m, n]) using vector form", () => {
    const shape = tensorShape("A = reshape(1:6, [2, 3]);", "A");
    expect(shape).toEqual([2, 3]);
  });
});

// ── colon range ───────────────────────────────────────────────────────

describe("colon range", () => {
  it("1:5 creates row vector", () => {
    const data = tensorData("v = 1:5;", "v");
    expect(data).toEqual([1, 2, 3, 4, 5]);
  });

  it("1:2:9 creates step vector", () => {
    const data = tensorData("v = 1:2:9;", "v");
    expect(data).toEqual([1, 3, 5, 7, 9]);
  });

  it("5:-1:1 creates descending vector", () => {
    const data = tensorData("v = 5:-1:1;", "v");
    expect(data).toEqual([5, 4, 3, 2, 1]);
  });
});

// ── sort ──────────────────────────────────────────────────────────────

describe("sort", () => {
  it("sort ascending by default", () => {
    const data = tensorData("v = sort([3,1,4,1,5]);", "v");
    expect(data).toEqual([1, 1, 3, 4, 5]);
  });

  it("sort descending", () => {
    const data = tensorData("v = sort([3,1,4,1,5], 'descend');", "v");
    expect(data).toEqual([5, 4, 3, 1, 1]);
  });

  it("sort with indices", () => {
    const result = executeCode("[v, idx] = sort([3,1,2]);");
    const v = result.variableValues["v"];
    const idx = result.variableValues["idx"];
    expect(isRuntimeTensor(v)).toBe(true);
    expect(isRuntimeTensor(idx)).toBe(true);
    if (isRuntimeTensor(v)) expect(Array.from(v.data)).toEqual([1, 2, 3]);
    if (isRuntimeTensor(idx)) expect(Array.from(idx.data)).toEqual([2, 3, 1]);
  });
});

// ── unique ────────────────────────────────────────────────────────────

describe("unique", () => {
  it("unique removes duplicates", () => {
    const data = tensorData("v = unique([3,1,2,1,3]);", "v");
    expect(data).toEqual([1, 2, 3]);
  });
});

// ── cumsum ────────────────────────────────────────────────────────────

describe("cumsum", () => {
  it("cumsum of vector", () => {
    const data = tensorData("v = cumsum([1,2,3,4]);", "v");
    expect(data).toEqual([1, 3, 6, 10]);
  });
});

// ── cumprod ───────────────────────────────────────────────────────────

describe("cumprod", () => {
  it("cumprod of vector", () => {
    const data = tensorData("v = cumprod([1,2,3,4]);", "v");
    expect(data).toEqual([1, 2, 6, 24]);
  });
});

// ── logical indexing ──────────────────────────────────────────────────

describe("logical indexing", () => {
  it("extracts elements where condition is true", () => {
    const data = tensorData("v = [1,2,3,4,5]; w = v(v > 3);", "w");
    expect(data).toEqual([4, 5]);
  });

  it("assigns to elements where condition is true", () => {
    const data = tensorData("v = [1,2,3,4,5]; v(v > 3) = 0;", "v");
    expect(data).toEqual([1, 2, 3, 0, 0]);
  });
});

// ── end indexing ──────────────────────────────────────────────────────

describe("end indexing", () => {
  it("v(end) returns last element", () => {
    expect(num("v = [1,2,3,4,5]; x = v(end);", "x")).toBe(5);
  });

  it("v(end-1) returns second to last", () => {
    expect(num("v = [1,2,3,4,5]; x = v(end-1);", "x")).toBe(4);
  });

  it("v(2:end) returns tail", () => {
    const data = tensorData("v = [1,2,3,4,5]; w = v(2:end);", "w");
    expect(data).toEqual([2, 3, 4, 5]);
  });
});

// ── setdiff / intersect / union ───────────────────────────────────────

describe("set operations", () => {
  it("intersect finds common elements", () => {
    const data = tensorData("v = intersect([1,2,3], [2,3,4]);", "v");
    expect(data).toEqual([2, 3]);
  });

  it("union combines unique elements", () => {
    const data = tensorData("v = union([1,2,3], [2,3,4]);", "v");
    expect(data).toEqual([1, 2, 3, 4]);
  });

  it("setdiff finds elements in first but not second", () => {
    const data = tensorData("v = setdiff([1,2,3,4], [2,4]);", "v");
    expect(data).toEqual([1, 3]);
  });
});

// ── ismember ─────────────────────────────────────────────────────────

describe("ismember", () => {
  it("returns logical vector for membership", () => {
    const result = executeCode("tf = ismember([1,2,3,4], [2,4]);");
    const tf = result.variableValues["tf"];
    expect(isRuntimeTensor(tf)).toBe(true);
    if (isRuntimeTensor(tf)) {
      expect(Array.from(tf.data)).toEqual([0, 1, 0, 1]);
    }
  });
});

// ── sparse awareness (linspace edge cases) ────────────────────────────

describe("linspace edge cases", () => {
  it("linspace with n=1 returns start", () => {
    const result = executeCode("v = linspace(3, 7, 1);");
    const v = result.variableValues["v"];
    if (typeof v === "number") {
      expect(v).toBe(7);
    } else if (isRuntimeTensor(v)) {
      expect(v.data[0]).toBe(7);
    } else {
      throw new Error("v is not a number or tensor");
    }
  });

  it("linspace with n=2 returns endpoints", () => {
    const data = tensorData("v = linspace(0, 10, 2);", "v");
    expect(data[0]).toBeCloseTo(0);
    expect(data[1]).toBeCloseTo(10);
  });
});

// ── max/min with matrix and dim ───────────────────────────────────────

describe("max/min along dimension", () => {
  it("max along dim 1 (columnwise max)", () => {
    const data = tensorData("A = [1,4;3,2]; m = max(A, [], 1);", "m");
    expect(data[0]).toBe(3);
    expect(data[1]).toBe(4);
  });

  it("min along dim 2 (rowwise min)", () => {
    const data = tensorData("A = [1,4;3,2]; m = min(A, [], 2);", "m");
    expect(data[0]).toBe(1);
    expect(data[1]).toBe(2);
  });

  it("max with indices", () => {
    const result = executeCode("[m, i] = max([3,1,4,1,5]);");
    expect(result.variableValues["m"]).toBe(5);
    expect(result.variableValues["i"]).toBe(5);
  });

  it("min with indices", () => {
    const result = executeCode("[m, i] = min([3,1,4,1,5]);");
    expect(result.variableValues["m"]).toBe(1);
    expect(result.variableValues["i"]).toBe(2);
  });
});

// ── tril / triu ───────────────────────────────────────────────────────

describe("tril / triu", () => {
  it("tril extracts lower triangular", () => {
    const data = tensorData("A = [1,2;3,4]; L = tril(A);", "L");
    // Column-major: [1,3,0,4]
    expect(data[0]).toBe(1);
    expect(data[1]).toBe(3);
    expect(data[2]).toBe(0);
    expect(data[3]).toBe(4);
  });

  it("triu extracts upper triangular", () => {
    const data = tensorData("A = [1,2;3,4]; U = triu(A);", "U");
    // Column-major: [1,0,2,4]
    expect(data[0]).toBe(1);
    expect(data[1]).toBe(0);
    expect(data[2]).toBe(2);
    expect(data[3]).toBe(4);
  });
});

// ── kron ─────────────────────────────────────────────────────────────

describe("kron", () => {
  it("kron product", () => {
    const shape = tensorShape("C = kron([1,0;0,1], [1,2;3,4]);", "C");
    expect(shape).toEqual([4, 4]);
  });
});

// ── cellfun / arrayfun ────────────────────────────────────────────────

describe("arrayfun", () => {
  it("arrayfun applies function to each element", () => {
    const data = tensorData("v = arrayfun(@(x) x^2, [1,2,3,4]);", "v");
    expect(data).toEqual([1, 4, 9, 16]);
  });
});

// ── cell operations ───────────────────────────────────────────────────

describe("cell operations", () => {
  it("creates cell array with curly braces", () => {
    const result = executeCode("c = {1, 'hello', [1,2,3]};");
    const c = result.variableValues["c"];
    expect(c).toBeDefined();
    expect((c as { kind: string }).kind).toBe("cell");
  });

  it("cell indexing with curly braces", () => {
    const result = executeCode("c = {10, 20, 30}; x = c{2};");
    expect(result.variableValues["x"]).toBe(20);
  });

  it("assigns to cell element", () => {
    const result = executeCode("c = {1, 2, 3}; c{2} = 99;");
    const c = result.variableValues["c"];
    expect(c).toBeDefined();
  });
});
