import { describe, it, expect } from "vitest";
import { displayValue } from "../numbl-core/runtime/display.js";

describe("displayValue numeric formatting", () => {
  it("preserves the exponent for tiny values (regression: was 1.0000e-1)", () => {
    expect(displayValue(1e-100)).toBe("1e-100");
  });

  it("preserves the exponent for large values whose exponent ends in 0", () => {
    // 1e10 falls into the integer fast path
    expect(displayValue(1e10)).toBe("10000000000");
    expect(displayValue(1e20)).toBe("1e+20");
    expect(displayValue(1e-20)).toBe("1e-20");
  });

  it("strips trailing zeros from the mantissa but not the exponent", () => {
    expect(displayValue(1.5e-100)).toBe("1.5e-100");
  });

  it("still strips trailing zeros for non-exponential reals", () => {
    expect(displayValue(0.5)).toBe("0.5");
    expect(displayValue(3.14)).toBe("3.14");
  });
});
