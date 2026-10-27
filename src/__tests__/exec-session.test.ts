import { describe, it, expect } from "vitest";
import { ExecSession } from "../numbl-core/execSession.js";

describe("ExecSession", () => {
  it("carries variables, hold state and figures across calls", () => {
    const s = new ExecSession();
    s.execute("x = 3; figure; hold on;");
    const r = s.execute("y = x + 1; h = figure;");
    expect(r.variableValues.y).toBe(4);
    expect(r.variableValues.h).toBe(2);
    expect(s.holdState).toBe(true);
    expect(s.figureHandles).toEqual({ current: 2, max: 2 });
  });

  it("leaves the state unchanged when a call fails", () => {
    const s = new ExecSession();
    s.execute("x = 1;");
    expect(() => s.execute("x = 2; error('boom');")).toThrow();
    expect(s.execute("y = x;").variableValues.y).toBe(1);
  });

  it("clearWorkspace forgets variables", () => {
    const s = new ExecSession();
    s.execute("x = 1;");
    s.clearWorkspace();
    expect(s.variableValues).toEqual({});
    expect(() => s.execute("y = x;")).toThrow();
  });
});
