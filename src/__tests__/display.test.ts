import { describe, it, expect } from "vitest";
import { displayValue } from "../numbl-core/runtime/display.js";
import { RTV } from "../numbl-core/runtime/constructors.js";
import { makeMException } from "../numbl-core/runtime/mexception.js";

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

describe("MException display", () => {
  it("lists the properties as MATLAB does", () => {
    const stack = RTV.structArray(
      ["file", "name", "line"],
      [RTV.struct({ file: RTV.char("f.m"), name: RTV.char("f"), line: 3 })]
    );
    const me = makeMException("my:id", "bad value 3", stack);
    expect(displayValue(me)).toBe(
      [
        "  MException with properties:",
        "",
        "    identifier: 'my:id'",
        "       message: 'bad value 3'",
        "         cause: {}",
        "         stack: [1×1 struct]",
        "    Correction: []",
        "",
      ].join("\n")
    );
  });
});
