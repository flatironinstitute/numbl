import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

// A complex value whose imaginary part is entirely zero is real in value (the
// JIT routinely produces such tensors when it cannot prove realness at compile
// time). jsonencode must treat it as real — matching isreal — rather than
// rejecting on the imaginary lane's mere presence. Genuinely complex data is
// still unsupported.
function jenc(expr: string): string {
  const r = executeCode(`r = jsonencode(${expr});`);
  return (r.variableValues.r as { value: string }).value;
}

describe("jsonencode zero-imaginary handling", () => {
  it("encodes a zero-imag complex scalar as real", () => {
    expect(jenc("complex(1,0)")).toBe("1");
  });

  it("encodes a zero-imag complex vector as real", () => {
    expect(jenc("complex([1 2 3],[0 0 0])")).toBe("[1,2,3]");
  });

  it("encodes a struct field with a zero-imag complex value as real", () => {
    expect(jenc("struct('v', complex([1 2],[0 0]))")).toBe('{"v":[1,2]}');
  });

  it("still rejects genuinely complex values", () => {
    expect(() => executeCode("r = jsonencode(complex(1,2));")).toThrow(
      /complex/
    );
  });
});
