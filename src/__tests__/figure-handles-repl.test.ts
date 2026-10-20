import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

describe("figure handles across REPL commands", () => {
  it("a bare figure in a later command opens a new figure", () => {
    const first = executeCode("figure; plot(1:3);");
    expect(first.figureHandles).toEqual({ current: 1, max: 1 });
    const second = executeCode("h = figure;", {
      initialFigureHandles: first.figureHandles,
    });
    expect(second.variableValues.h).toBe(2);
    expect(second.figureHandles).toEqual({ current: 2, max: 2 });
  });
});
