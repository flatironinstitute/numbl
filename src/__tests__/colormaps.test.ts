import { describe, it, expect } from "vitest";
import {
  colormapLookup,
  colormapTable,
  COLORMAP_NAMES,
  PARULA_ANCHORS,
} from "../numbl-core/plotting/colormaps.js";
import { executeCode } from "../numbl-core/executeCode.js";

describe("colormapLookup", () => {
  it("defaults to the parula table used by the parula builtin", () => {
    expect(colormapLookup(0)).toEqual(PARULA_ANCHORS[0]);
    expect(colormapLookup(1)).toEqual(
      PARULA_ANCHORS[PARULA_ANCHORS.length - 1]
    );
    expect(colormapLookup(NaN)).toEqual(PARULA_ANCHORS[0]);
    expect(colormapLookup(0.5)).toEqual(PARULA_ANCHORS[4]);
  });

  it("honors names and custom data", () => {
    expect(colormapLookup(0, { name: "redblue" })).toEqual([0, 0, 1]);
    expect(colormapLookup(0.5, { name: "jet" })).toEqual([0.5, 1, 0.5]);
    const data = [
      [0, 0, 0],
      [1, 1, 1],
    ];
    expect(colormapLookup(0.25, { name: "jet", data })).toEqual([
      0.25, 0.25, 0.25,
    ]);
  });
});

describe("colormapTable", () => {
  it("returns m rows for every named colormap", () => {
    for (const name of COLORMAP_NAMES) {
      expect(colormapTable(name, 7)).toHaveLength(7);
      expect(colormapTable(name, 0)).toEqual([]);
    }
    expect(colormapTable("nosuchmap", 4)).toBeUndefined();
  });

  it("follows MATLAB's formulas", () => {
    // Values from MATLAB R2026b.
    expect(colormapTable("jet", 4)).toEqual([
      [0, 0, 1],
      [0, 1, 1],
      [1, 1, 0],
      [1, 0, 0],
    ]);
    expect(
      colormapTable("hot", 5)?.map(r => r.map(v => +v.toFixed(4)))
    ).toEqual([
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 0.3333],
      [1, 1, 0.6667],
      [1, 1, 1],
    ]);
    expect(colormapTable("gray", 3)).toEqual([
      [0, 0, 0],
      [0.5, 0.5, 0.5],
      [1, 1, 1],
    ]);
  });

  it("is sent with colormap('name') so the viewer draws it", () => {
    const r = executeCode("imagesc(magic(3)); colormap hot", {
      implicitCwdPath: null,
    });
    const cm = r.plotInstructions.find(i => i.type === "set_colormap");
    expect(cm).toMatchObject({ type: "set_colormap", name: "hot" });
    const data = (cm as { data?: number[][] }).data;
    expect(data).toHaveLength(256);
    expect(data?.[255]).toEqual([1, 1, 1]);
  });
});
