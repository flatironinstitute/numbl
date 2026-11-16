import { describe, it, expect } from "vitest";
import {
  applyView,
  colorbarFraction,
  plotFractions,
  shiftRange,
  valueAt,
  zoomRange,
  type PlotLayout,
} from "../graphics/plotInteraction.js";
import type { AxesState } from "../numbl-core/plotting/figuresReducer.js";

const close = (a: number[], b: number[]) =>
  a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 10));

const layout = (over: Partial<PlotLayout> = {}): PlotLayout => ({
  plot: { left: 100, top: 50, width: 200, height: 100 },
  x: [0, 10],
  y: [0, 1],
  logX: false,
  logY: false,
  yReversed: false,
  ...over,
});

describe("plot interaction", () => {
  it("zooms about a point, keeping it fixed", () => {
    close(zoomRange([0, 10], 2, 0.5), [1, 6]);
    close(zoomRange([0, 10], 5, 2), [-5, 15]);
  });

  it("zooms and shifts log ranges in log space", () => {
    close(zoomRange([1, 1e4], 100, 0.5, true), [10, 1000]);
    close(shiftRange([1, 100], 0.5), [50.5, 149.5]);
    close(shiftRange([1, 100], 0.5, true), [10, 1000]);
  });

  it("maps canvas points to axis fractions", () => {
    const L = layout();
    expect(plotFractions(L, 100, 150)).toEqual({ fx: 0, fy: 0 });
    expect(plotFractions(L, 300, 50)).toEqual({ fx: 1, fy: 1 });
    const R = layout({ yReversed: true });
    expect(plotFractions(R, 100, 50).fy).toBe(0);
    expect(valueAt([0, 10], 0.25)).toBe(2.5);
  });

  it("maps colorbar points to fractions (low limit at 0)", () => {
    const v = { left: 0, top: 0, width: 16, height: 100, vertical: true };
    expect(colorbarFraction(v, 5, 100)).toBe(0);
    expect(colorbarFraction(v, 5, 25)).toBe(0.75);
    const h = { left: 0, top: 0, width: 200, height: 16, vertical: false };
    expect(colorbarFraction(h, 50, 5)).toBe(0.25);
  });

  it("applies a view override over the axes limits", () => {
    const axes = { xlim: [0, 1], caxis: [0, 2] } as unknown as AxesState;
    expect(applyView(axes, null)).toBe(axes);
    const v = applyView(axes, { ylim: [3, 4], caxis: [1, 5] });
    expect(v.xlim).toEqual([0, 1]);
    expect(v.ylim).toEqual([3, 4]);
    expect(v.caxis).toEqual([1, 5]);
  });
});
