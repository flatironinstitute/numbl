import { describe, it, expect } from "vitest";
import {
  colormapLookup,
  PARULA_ANCHORS,
} from "../numbl-core/plotting/colormaps.js";

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
