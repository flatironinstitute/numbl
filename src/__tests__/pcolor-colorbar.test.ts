import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

describe("pcolor", () => {
  it("pcolor(C) emits a pcolor instruction with correct shape", () => {
    const result = executeCode("pcolor(magic(5));");
    const pcolorInstrs = result.plotInstructions.filter(
      i => i.type === "pcolor"
    );
    expect(pcolorInstrs).toHaveLength(1);
    const trace = (
      pcolorInstrs[0] as { trace: { rows: number; cols: number; c: number[] } }
    ).trace;
    expect(trace.rows).toBe(5);
    expect(trace.cols).toBe(5);
    expect(trace.c).toHaveLength(25);
  });

  it("pcolor(X, Y, C) accepts explicit X/Y matrices", () => {
    const result = executeCode(
      "[X,Y]=meshgrid(1:4,1:3); Z = X.*Y; pcolor(X,Y,Z);"
    );
    const pcolorInstrs = result.plotInstructions.filter(
      i => i.type === "pcolor"
    );
    expect(pcolorInstrs).toHaveLength(1);
    const trace = (
      pcolorInstrs[0] as {
        trace: {
          rows: number;
          cols: number;
          x: number[];
          y: number[];
          c: number[];
        };
      }
    ).trace;
    expect(trace.rows).toBe(3);
    expect(trace.cols).toBe(4);
    expect(trace.x).toHaveLength(12);
    expect(trace.y).toHaveLength(12);
    expect(trace.c).toHaveLength(12);
    // x[i,j] should equal X(i,j) (column 0 is the first column of meshgrid output)
    // X = [1 2 3 4; 1 2 3 4; 1 2 3 4]; column-major: [1,1,1, 2,2,2, 3,3,3, 4,4,4]
    expect(trace.x[0]).toBe(1);
    expect(trace.x[3]).toBe(2);
    expect(trace.y[0]).toBe(1);
    expect(trace.y[1]).toBe(2);
  });

  it("rejects pcolor with no arguments", () => {
    expect(() => executeCode("pcolor();")).toThrow();
  });
});

describe("colorbar", () => {
  it("colorbar with no args emits set_colorbar value=on", () => {
    const result = executeCode("colorbar;");
    const cb = result.plotInstructions.find(i => i.type === "set_colorbar");
    expect(cb).toBeDefined();
    expect((cb as { value: string }).value).toBe("on");
  });

  it("colorbar('off') emits set_colorbar value=off", () => {
    const result = executeCode("colorbar('off');");
    const cb = result.plotInstructions.find(i => i.type === "set_colorbar");
    expect(cb).toBeDefined();
    expect((cb as { value: string }).value).toBe("off");
  });

  it("colorbar('northoutside') emits the location", () => {
    const result = executeCode("colorbar('northoutside');");
    const cb = result.plotInstructions.find(i => i.type === "set_colorbar");
    expect(cb).toBeDefined();
    expect((cb as { value: string; location?: string }).location).toBe(
      "northoutside"
    );
  });

  it("colorbar with name-value pairs is accepted (silently)", () => {
    const result = executeCode("colorbar('Direction','reverse');");
    const cb = result.plotInstructions.find(i => i.type === "set_colorbar");
    expect(cb).toBeDefined();
    expect((cb as { value: string }).value).toBe("on");
  });
});
