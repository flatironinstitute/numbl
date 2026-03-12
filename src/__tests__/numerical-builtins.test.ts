import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeTensor } from "../numbl-core/runtime/types.js";

function num(code: string, varName: string): number {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (typeof v !== "number") throw new Error(`${varName} is not a number`);
  return v;
}

function bool(code: string, varName: string): boolean {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (typeof v !== "boolean") throw new Error(`${varName} is not a boolean`);
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

// ── conv ──────────────────────────────────────────────────────────────

describe("conv", () => {
  it("full convolution of [1,1] and [1,1] is [1,2,1]", () => {
    const data = tensorData("v = conv([1,1], [1,1]);", "v");
    expect(data).toEqual([1, 2, 1]);
  });

  it("conv of polynomial coefficients", () => {
    // [1,2] * [1,3] = [1,5,6]: x^2 + 5x + 6 = (x+2)(x+3)
    const data = tensorData("v = conv([1,2], [1,3]);", "v");
    expect(data[0]).toBe(1);
    expect(data[1]).toBe(5);
    expect(data[2]).toBe(6);
  });

  it("conv with same shape", () => {
    const shape = tensorShape("v = conv([1,2,3], [1,1], 'same');", "v");
    expect(shape[1]).toBe(3);
  });

  it("conv with valid shape", () => {
    const data = tensorData("v = conv([1,2,3,4], [1,1], 'valid');", "v");
    expect(data).toEqual([3, 5, 7]);
  });
});

// ── polyval ───────────────────────────────────────────────────────────

describe("polyval", () => {
  it("evaluates polynomial at scalar", () => {
    // p(x) = x^2 - 1 = [1, 0, -1], at x=3: 9-1=8
    expect(num("y = polyval([1, 0, -1], 3);", "y")).toBe(8);
  });

  it("evaluates polynomial at x=0", () => {
    // [1, 2, 3] at x=0: 3
    expect(num("y = polyval([1, 2, 3], 0);", "y")).toBe(3);
  });

  it("evaluates polynomial on vector", () => {
    // p(x) = x at x=[1,2,3]: [1,2,3]
    const data = tensorData("v = polyval([1, 0], [1,2,3]);", "v");
    expect(data).toEqual([1, 2, 3]);
  });
});

// ── polyfit ───────────────────────────────────────────────────────────

describe("polyfit", () => {
  it("fits linear polynomial to linear data", () => {
    // y = 2x + 1 at x = [0,1,2]: coefficients [2, 1]
    const data = tensorData("p = polyfit([0,1,2], [1,3,5], 1);", "p");
    expect(data[0]).toBeCloseTo(2);
    expect(data[1]).toBeCloseTo(1);
  });

  it("fits quadratic polynomial", () => {
    // y = x^2: at x=[0,1,2], y=[0,1,4]
    const data = tensorData("p = polyfit([0,1,2], [0,1,4], 2);", "p");
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(0, 5);
    expect(data[2]).toBeCloseTo(0, 5);
  });
});

// ── trapz ─────────────────────────────────────────────────────────────

describe("trapz", () => {
  it("trapezoidal integration of constant 1 over [0,1] = 1", () => {
    // y=[1,1] over x=[0,1]: area = 1
    expect(num("a = trapz([0,1], [1,1]);", "a")).toBeCloseTo(1);
  });

  it("trapz of y=[0,1] over [0,1] = 0.5", () => {
    expect(num("a = trapz([0,1], [0,1]);", "a")).toBeCloseTo(0.5);
  });

  it("trapz with unit spacing", () => {
    // y=[0,1,2,3,4], unit spacing: sum of trapezoids = 8
    expect(num("a = trapz([0,1,2,3,4]);", "a")).toBeCloseTo(8);
  });

  it("trapz of single point = 0", () => {
    expect(num("a = trapz([5]);", "a")).toBe(0);
  });
});

// ── cumtrapz ──────────────────────────────────────────────────────────

describe("cumtrapz", () => {
  it("cumtrapz of [0,1,2,3] with unit spacing", () => {
    const data = tensorData("v = cumtrapz([0,1,2,3]);", "v");
    expect(data[0]).toBeCloseTo(0);
    expect(data[1]).toBeCloseTo(0.5);
    expect(data[2]).toBeCloseTo(2);
    expect(data[3]).toBeCloseTo(4.5);
  });

  it("cumtrapz with x vector", () => {
    const data = tensorData("v = cumtrapz([0,1,2], [0,2,4]);", "v");
    expect(data[0]).toBeCloseTo(0);
    expect(data[2]).toBeCloseTo(4);
  });
});

// ── gradient ─────────────────────────────────────────────────────────

describe("gradient", () => {
  it("gradient of linear function is constant", () => {
    // [0,1,2,3,4] -> [1,1,1,1,1]
    const data = tensorData("v = gradient([0,1,2,3,4]);", "v");
    for (const d of data) expect(d).toBeCloseTo(1);
  });

  it("gradient with custom spacing", () => {
    // [0,2,4] with h=2 -> [1,1,1]
    const data = tensorData("v = gradient([0,2,4], 2);", "v");
    for (const d of data) expect(d).toBeCloseTo(1);
  });
});

// ── accumarray ────────────────────────────────────────────────────────

describe("accumarray", () => {
  it("sums values by group", () => {
    // subs=[1,2,1,2], vals=[1,2,3,4]: result=[4,6]
    const data = tensorData("v = accumarray([1;2;1;2], [1;2;3;4]);", "v");
    expect(data[0]).toBe(4);
    expect(data[1]).toBe(6);
  });

  it("accumarray with scalar value", () => {
    // counts: how many fall in each bin
    const data = tensorData("v = accumarray([1;2;1;3;2], 1);", "v");
    expect(data[0]).toBe(2);
    expect(data[1]).toBe(2);
    expect(data[2]).toBe(1);
  });
});

// ── interp1 ──────────────────────────────────────────────────────────

describe("interp1", () => {
  it("linear interpolation at midpoint", () => {
    // x=[0,1,2], y=[0,1,2]: interp at 0.5 -> 0.5
    expect(num("y = interp1([0,1,2], [0,1,2], 0.5);", "y")).toBeCloseTo(0.5);
  });

  it("interp1 returns NaN for out-of-range by default", () => {
    const result = executeCode("y = interp1([0,1], [0,1], 2);");
    const v = result.variableValues["y"];
    expect(isNaN(v as number)).toBe(true);
  });

  it("interp1 extrapolates with extrap option", () => {
    expect(
      num("y = interp1([0,1,2], [0,1,2], 3, 'linear', 'extrap');", "y")
    ).toBeCloseTo(3);
  });

  it("interp1 on vector of query points", () => {
    const data = tensorData("v = interp1([0,1,2], [0,2,4], [0.5, 1.5]);", "v");
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(3);
  });
});

// ── bitwise operations ────────────────────────────────────────────────

describe("bitand", () => {
  it("bitand of 5 and 3 is 1", () => {
    // 5 = 101, 3 = 011, AND = 001 = 1
    expect(num("x = bitand(5, 3);", "x")).toBe(1);
  });

  it("bitand of 12 and 10 is 8", () => {
    // 12=1100, 10=1010, AND=1000=8
    expect(num("x = bitand(12, 10);", "x")).toBe(8);
  });
});

describe("bitor", () => {
  it("bitor of 5 and 3 is 7", () => {
    // 5=101, 3=011, OR=111=7
    expect(num("x = bitor(5, 3);", "x")).toBe(7);
  });
});

describe("bitxor", () => {
  it("bitxor of 5 and 3 is 6", () => {
    // 5=101, 3=011, XOR=110=6
    expect(num("x = bitxor(5, 3);", "x")).toBe(6);
  });

  it("bitxor of identical values is 0", () => {
    expect(num("x = bitxor(7, 7);", "x")).toBe(0);
  });
});

describe("bitshift", () => {
  it("left shift by 1 doubles", () => {
    expect(num("x = bitshift(4, 1);", "x")).toBe(8);
  });

  it("right shift by 1 halves", () => {
    expect(num("x = bitshift(8, -1);", "x")).toBe(4);
  });

  it("bitshift on vector", () => {
    const data = tensorData("v = bitshift([1,2,4], 2);", "v");
    expect(data).toEqual([4, 8, 16]);
  });
});

// ── cov ───────────────────────────────────────────────────────────────

describe("cov", () => {
  it("cov of a vector is its variance", () => {
    // cov([1,3]) = var([1,3]) = 2
    expect(num("x = cov([1,3]);", "x")).toBeCloseTo(2);
  });

  it("cov of scalar is 0", () => {
    expect(num("x = cov(5);", "x")).toBe(0);
  });

  it("cov of two vectors returns 2x2 matrix", () => {
    const shape = tensorShape("C = cov([1,2,3], [4,5,6]);", "C");
    expect(shape).toEqual([2, 2]);
  });
});

// ── corrcoef ──────────────────────────────────────────────────────────

describe("corrcoef", () => {
  it("corrcoef of a single vector is 1", () => {
    expect(num("x = corrcoef([1,2,3,4]);", "x")).toBeCloseTo(1);
  });

  it("corrcoef of two perfectly correlated vectors has diagonal 1", () => {
    const data = tensorData("C = corrcoef([1,2,3], [2,4,6]);", "C");
    // diagonal elements = 1
    expect(data[0]).toBeCloseTo(1); // C(1,1) - col major: [C11,C21,C12,C22]
    expect(data[3]).toBeCloseTo(1); // C(2,2)
    // off-diagonal = 1 (perfect correlation)
    expect(data[1]).toBeCloseTo(1);
  });
});

// ── roots / poly ─────────────────────────────────────────────────────

describe("roots", () => {
  it("roots of linear polynomial [1, -2] is 2", () => {
    // roots([1,-2]) returns [2] as a 1x1 tensor or scalar
    const result = executeCode("r = roots([1, -2]);");
    const r = result.variableValues["r"];
    const val = isRuntimeTensor(r) ? r.data[0] : (r as number);
    expect(val).toBeCloseTo(2);
  });

  it("roots of quadratic [1, -3, 2] are 1 and 2", () => {
    const data = tensorData("r = roots([1, -3, 2]);", "r");
    const sorted = [...data].sort((a, b) => a - b);
    expect(sorted[0]).toBeCloseTo(1);
    expect(sorted[1]).toBeCloseTo(2);
  });
});

describe("poly", () => {
  it("poly([1,2]) produces polynomial with those roots", () => {
    // poly([1,2]) = (x-1)(x-2) = x^2 - 3x + 2 = [1,-3,2]
    const data = tensorData("p = poly([1,2]);", "p");
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(-3);
    expect(data[2]).toBeCloseTo(2);
  });

  it("poly(scalar) produces [1, -scalar]", () => {
    const data = tensorData("p = poly(3);", "p");
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(-3);
  });
});

// ── deconv ───────────────────────────────────────────────────────────

describe("deconv", () => {
  it("deconv is inverse of conv for exact division", () => {
    // conv([1,2], [1,3]) = [1,5,6]
    // deconv([1,5,6], [1,3]) = [1,2]
    const result = executeCode("[q, r] = deconv([1,5,6], [1,3]);");
    const q = result.variableValues["q"];
    expect(isRuntimeTensor(q)).toBe(true);
    if (isRuntimeTensor(q)) {
      expect(q.data[0]).toBeCloseTo(1);
      expect(q.data[1]).toBeCloseTo(2);
    }
  });
});

// ── operator-name builtins ────────────────────────────────────────────

describe("operator-name builtins", () => {
  it("plus(a, b) adds", () => {
    expect(num("x = plus(3, 4);", "x")).toBe(7);
  });

  it("minus(a, b) subtracts", () => {
    expect(num("x = minus(10, 3);", "x")).toBe(7);
  });

  it("times(a, b) element-wise multiplies", () => {
    const data = tensorData("v = times([1,2,3], [4,5,6]);", "v");
    expect(data).toEqual([4, 10, 18]);
  });

  it("rdivide(a, b) element-wise divides", () => {
    expect(num("x = rdivide(10, 2);", "x")).toBe(5);
  });

  it("mtimes matrix multiply", () => {
    const result = executeCode("C = mtimes([1,2;3,4], [1;1]);");
    const C = result.variableValues["C"];
    expect(isRuntimeTensor(C)).toBe(true);
    if (isRuntimeTensor(C)) {
      expect(C.data[0]).toBe(3);
      expect(C.data[1]).toBe(7);
    }
  });

  it("eq element-wise equal", () => {
    const result = executeCode("v = eq([1,2,3], [1,0,3]);");
    const v = result.variableValues["v"];
    expect(isRuntimeTensor(v)).toBe(true);
    if (isRuntimeTensor(v)) {
      expect(Array.from(v.data)).toEqual([1, 0, 1]);
    }
  });

  it("ne element-wise not-equal", () => {
    expect(bool("x = ne(1, 2);", "x")).toBe(true);
    expect(bool("x = ne(1, 1);", "x")).toBe(false);
  });

  it("lt less-than", () => {
    expect(bool("x = lt(1, 2);", "x")).toBe(true);
  });

  it("gt greater-than", () => {
    expect(bool("x = gt(2, 1);", "x")).toBe(true);
  });

  it("le less-or-equal", () => {
    expect(bool("x = le(1, 1);", "x")).toBe(true);
  });

  it("ge greater-or-equal", () => {
    expect(bool("x = ge(2, 1);", "x")).toBe(true);
  });

  it("power(2, 8) = 256", () => {
    expect(num("x = power(2, 8);", "x")).toBe(256);
  });

  it("uminus negates", () => {
    expect(num("x = uminus(5);", "x")).toBe(-5);
  });

  it("uplus is identity", () => {
    expect(num("x = uplus(5);", "x")).toBe(5);
  });
});
