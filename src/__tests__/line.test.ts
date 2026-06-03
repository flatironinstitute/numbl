import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import type { PlotTrace, Plot3Trace } from "../graphics/types.js";

type LineInstr = { type: "line"; traces: PlotTrace[] };
type Line3Instr = { type: "line3"; traces: Plot3Trace[] };

function lineInstrs(code: string): LineInstr[] {
  return executeCode(code).plotInstructions.filter(
    i => i.type === "line"
  ) as LineInstr[];
}

describe("line", () => {
  it("line(x, y) emits a single 2-D line trace", () => {
    const instrs = lineInstrs("x = 1:5; y = x.^2; line(x, y);");
    expect(instrs).toHaveLength(1);
    expect(instrs[0].traces).toHaveLength(1);
    expect(instrs[0].traces[0].x).toEqual([1, 2, 3, 4, 5]);
    expect(instrs[0].traces[0].y).toEqual([1, 4, 9, 16, 25]);
  });

  it("matrix Y produces one line per column", () => {
    const instrs = lineInstrs("x = (1:4)'; y = [x, 2*x]; line(x, y);");
    expect(instrs).toHaveLength(1);
    expect(instrs[0].traces).toHaveLength(2);
    expect(instrs[0].traces[0].y).toEqual([1, 2, 3, 4]);
    expect(instrs[0].traces[1].y).toEqual([2, 4, 6, 8]);
  });

  it("line(x, y, z) emits a 3-D line trace", () => {
    const result = executeCode("line([0 1 2], [0 1 0], [0 1 2]);");
    const instrs = result.plotInstructions.filter(
      i => i.type === "line3"
    ) as Line3Instr[];
    expect(instrs).toHaveLength(1);
    expect(instrs[0].traces[0].z).toEqual([0, 1, 2]);
  });

  it("line with no arguments draws (0,0) to (1,1)", () => {
    const instrs = lineInstrs("line;");
    expect(instrs).toHaveLength(1);
    expect(instrs[0].traces[0].x).toEqual([0, 1]);
    expect(instrs[0].traces[0].y).toEqual([0, 1]);
  });

  it("Name-Value pairs set appearance", () => {
    const instrs = lineInstrs(
      "line([1 9], [2 12], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 3);"
    );
    const t = instrs[0].traces[0];
    expect(t.color).toEqual([1, 0, 0]);
    expect(t.lineStyle).toBe("--");
    expect(t.lineWidth).toBe(3);
  });

  it("low-level form line('XData',x,'YData',y) draws a black line", () => {
    const instrs = lineInstrs("line('XData', [0 1 2], 'YData', [3 4 5]);");
    const t = instrs[0].traces[0];
    expect(t.x).toEqual([0, 1, 2]);
    expect(t.y).toEqual([3, 4, 5]);
    expect(t.color).toEqual([0, 0, 0]);
  });

  it("low-level form with ZData makes a black 3-D line", () => {
    const result = executeCode(
      "line('XData', [0 1], 'YData', [0 1], 'ZData', [0 2]);"
    );
    const instrs = result.plotInstructions.filter(
      i => i.type === "line3"
    ) as Line3Instr[];
    expect(instrs[0].traces[0].z).toEqual([0, 2]);
    expect(instrs[0].traces[0].color).toEqual([0, 0, 0]);
  });

  it("line adds to the current axes without resetting hold", () => {
    // Two line() calls with no hold should both render (line ignores hold,
    // unlike plot which would replace).
    const instrs = lineInstrs("line([0 1],[0 1]); line([0 1],[1 0]);");
    expect(instrs).toHaveLength(2);
  });

  it("pl = line(...) returns a handle whose properties can be set", () => {
    const result = executeCode(
      "pl = line([3 2], [15 12]); pl.Color = 'green'; c = pl.Color;"
    );
    const c = result.variableValues["c"] as { data: Float64Array };
    expect(Array.from(c.data)).toEqual([0, 1, 0]);
  });
});
