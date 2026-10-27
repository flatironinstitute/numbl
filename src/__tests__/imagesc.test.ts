import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import {
  figuresReducer,
  initialFiguresState,
  type AxesState,
  type FiguresState,
} from "../numbl-core/plotting/figuresReducer.js";
import { computeAxisLimits } from "../numbl-core/plotting/axisLimits.js";
import {
  imagescColorLimits,
  imagescExtent,
} from "../numbl-core/plotting/imagescExtent.js";

function renderedAxes(code: string): AxesState {
  const result = executeCode(code);
  let state: FiguresState = initialFiguresState;
  for (const instr of result.plotInstructions) {
    state = figuresReducer(state, instr);
  }
  const fig = state.figs[state.currentHandle];
  return fig.axes[fig.currentAxesIndex];
}

describe("imagesc", () => {
  it("treats XData/YData as pixel centers, as MATLAB does", () => {
    expect(
      imagescExtent(renderedAxes("imagesc(magic(4));").imagescTrace!)
    ).toEqual({
      dx: 1,
      dy: 1,
      x: [0.5, 4.5],
      y: [0.5, 4.5],
    });
    // A single row is one unit tall rather than empty.
    const row = imagescExtent(renderedAxes("imagesc([1 2 3]);").imagescTrace!);
    expect(row.y).toEqual([0.5, 1.5]);
    // Decreasing XData flips the pixel step but not the extent.
    const flipped = imagescExtent(
      renderedAxes("imagesc([20 10], [5 8], rand(4, 3));").imagescTrace!
    );
    expect(flipped.dx).toBe(-5);
    expect(flipped.x).toEqual([7.5, 22.5]);
    expect(flipped.y).toEqual([4.5, 8.5]);
  });

  it("gives tight axis limits at the pixel edges", () => {
    expect(computeAxisLimits(renderedAxes("imagesc(magic(4));"))).toEqual([
      0.5, 4.5, 0.5, 4.5,
    ]);
  });

  it("maps colors over caxis/clim when set, else the data range", () => {
    const limits = (code: string) => {
      const ax = renderedAxes(code);
      return imagescColorLimits(ax.imagescTrace!, ax.caxis);
    };
    expect(limits("imagesc([1 NaN; 3 4]);")).toEqual([1, 4]);
    // A singular point must not set the color scale once caxis is given.
    const spike = "Z = peaks(20); Z(10,10) = 1e6; imagesc(Z);";
    expect(limits(spike + " caxis([-5 5]);")).toEqual([-5, 5]);
    expect(limits(spike + " clim([-5 5]);")).toEqual([-5, 5]);
    expect(limits("imagesc(peaks(20), [-5 5]);")).toEqual([-5, 5]);
    expect(limits("M = magic(3); imagesc(1:3, 1:2, M(1:2,:), [2 6]);")).toEqual(
      [2, 6]
    );
    expect(() => executeCode("imagesc(magic(3), [5 1]);")).toThrow(/clims/);
  });

  it("reverses the y axis (axis ij) unless hold is on", () => {
    expect(renderedAxes("imagesc(magic(3));").yDir).toBe("reverse");
    expect(
      renderedAxes("plot(1:3); hold on; imagesc(magic(3));").yDir
    ).not.toBe("reverse");
  });
});
