import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

describe("subplot", () => {
  it("subplot(m,n,p) emits set_subplot with those values", () => {
    const r = executeCode("subplot(2,3,4);");
    const s = r.plotInstructions.find(i => i.type === "set_subplot");
    expect(s).toMatchObject({ rows: 2, cols: 3, index: 4 });
  });

  it("subplot(mnp) three-digit shorthand maps to subplot(m,n,p)", () => {
    const r = executeCode("subplot(131);");
    const s = r.plotInstructions.find(i => i.type === "set_subplot");
    expect(s).toMatchObject({ rows: 1, cols: 3, index: 1 });
  });

  it("subplot(234) maps to rows=2, cols=3, index=4", () => {
    const r = executeCode("subplot(234);");
    const s = r.plotInstructions.find(i => i.type === "set_subplot");
    expect(s).toMatchObject({ rows: 2, cols: 3, index: 4 });
  });

  it("routes plots into three separate axes via the shorthand", () => {
    const code = [
      "[X,Y] = meshgrid(linspace(-1,1,5));",
      "subplot(131), surf(X, Y, X)",
      "subplot(132), surf(X, Y, Y)",
      "subplot(133), surf(X, Y, X + Y)",
    ].join("\n");
    const subs = executeCode(code).plotInstructions.filter(
      i => i.type === "set_subplot"
    );
    expect(subs).toHaveLength(3);
    expect(subs.map(s => (s as { index: number }).index)).toEqual([1, 2, 3]);
  });

  it("ignores an out-of-range three-digit value", () => {
    // index 0 is invalid (subplot(130) -> index 0): emit nothing.
    const r = executeCode("subplot(130);");
    expect(
      r.plotInstructions.find(i => i.type === "set_subplot")
    ).toBeUndefined();
  });
});
