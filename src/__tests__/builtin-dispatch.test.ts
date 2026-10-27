import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

const count = (code: string, type: string, opt: "0" | "1" = "0") =>
  executeCode(code, { optimization: opt }).plotInstructions.filter(
    i => i.type === type
  ).length;

describe("builtin dispatch", () => {
  it("builtin('plot', ...) draws once", () => {
    expect(count("builtin('plot', 1:3);", "plot")).toBe(1);
    expect(count("builtin('surf', magic(3));", "surf")).toBe(1);
    expect(count("builtin('pcolor', magic(3));", "pcolor")).toBe(1);
  });

  it("plot calls from compiled code return the same handles", () => {
    const code =
      "function r = f()\nh = pcolor(magic(3));\nr = isgraphics(h);\nend\n";
    for (const opt of ["0", "1"] as const) {
      const res = executeCode("r = f();", { optimization: opt }, [
        { name: "f.m", source: code },
      ]);
      expect(res.variableValues["r"]).toBe(true);
    }
  });
});
