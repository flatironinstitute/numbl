import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeTensor } from "../numbl-core/runtime/types.js";

function tensor(
  code: string,
  varName: string
): { data: number[]; shape: number[] } {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (!isRuntimeTensor(v)) throw new Error(`${varName} is not a tensor`);
  return { data: Array.from(v.data), shape: v.shape };
}

function num(code: string, varName: string): number {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (typeof v !== "number") throw new Error(`${varName} is not a number`);
  return v;
}

describe("convhull", () => {
  it("2-D hull is a closed counter-clockwise loop of the corner indices", () => {
    // 2x2 square plus an interior point (index 5) that must be excluded.
    const k = tensor("k = convhull([0 0; 2 0; 2 2; 0 2; 1 1]);", "k");
    // Column vector, closed (first == last).
    expect(k.shape[1]).toBe(1);
    expect(k.data[0]).toBe(k.data[k.data.length - 1]);
    // The four corners (1..4), not the interior point (5).
    const unique = [...new Set(k.data)].sort((a, b) => a - b);
    expect(unique).toEqual([1, 2, 3, 4]);
    // Counter-clockwise: signed area of the loop is positive.
    let signed = 0;
    const pts = [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
    ];
    for (let i = 0; i + 1 < k.data.length; i++) {
      const a = pts[k.data[i] - 1];
      const b = pts[k.data[i + 1] - 1];
      signed += a[0] * b[1] - b[0] * a[1];
    }
    expect(signed).toBeGreaterThan(0);
  });

  it("2-D area matches MATLAB's documented example (1.75)", () => {
    const av = num(
      "[k,av] = convhull([0 0; 1 1; 1.5 0.5; 1.5 -0.5; 1.25 0.3; 1 0; 1.25 -0.3; 1 -1]);",
      "av"
    );
    expect(av).toBeCloseTo(1.75, 10);
  });

  it("3-D hull returns a triangle matrix and correct volume", () => {
    // 2x2x2 cube, volume 8, surface = 12 triangles.
    const code =
      "[k,v] = convhull([0 0 0; 2 0 0; 2 2 0; 0 2 0; 0 0 2; 2 0 2; 2 2 2; 0 2 2]);";
    const k = tensor(code, "k");
    expect(k.shape).toEqual([12, 3]);
    expect(num(code, "v")).toBeCloseTo(8, 10);
  });

  it("accepts coordinate-vector forms convhull(x,y) and convhull(x,y,z)", () => {
    expect(tensor("k = convhull([0;2;2;0],[0;0;2;2]);", "k").shape[1]).toBe(1);
    expect(
      tensor(
        "k = convhull([0;2;2;0;0;2;2;0],[0;0;2;2;0;0;2;2],[0;0;0;0;2;2;2;2]);",
        "k"
      ).shape
    ).toEqual([12, 3]);
  });

  it("accepts the 'Simplify' name-value pair", () => {
    const av = num(
      "[k,av] = convhull([0 0; 2 0; 2 2; 0 2], 'Simplify', true);",
      "av"
    );
    expect(av).toBeCloseTo(4, 10);
  });
});

describe("convhulln", () => {
  it("2-D returns an edge matrix and area", () => {
    const code = "[k,a] = convhulln([0 0; 2 0; 2 2; 0 2; 1 1]);";
    const k = tensor(code, "k");
    expect(k.shape).toEqual([4, 2]); // 4 edges, 2 indices each
    expect(num(code, "a")).toBeCloseTo(4, 10);
  });

  it("3-D returns a triangle matrix and volume", () => {
    const code =
      "[k,v] = convhulln([0 0 0; 2 0 0; 2 2 0; 0 2 0; 0 0 2; 2 0 2; 2 2 2; 0 2 2]);";
    expect(tensor(code, "k").shape).toEqual([12, 3]);
    expect(num(code, "v")).toBeCloseTo(8, 10);
  });

  it("rejects dimensions above 3", () => {
    expect(() =>
      executeCode(
        "k = convhulln([0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]);"
      )
    ).toThrow(/supported/i);
  });
});
