import { describe, it, expect } from "vitest";
import { exportFigureHdf5 } from "../graphics/exportFigureHdf5.js";
import { importFigureHdf5 } from "../graphics/importFigureHdf5.js";
import type { FigureState } from "../graphics/figuresReducer.js";

function sampleFigure(): FigureState {
  return {
    currentAxesIndex: 1,
    sgtitle: "Round Trip",
    subplotGrid: { rows: 1, cols: 1 },
    axes: {
      1: {
        holdOn: true,
        title: "Axes 1",
        xlabel: "t",
        ylabel: "v",
        legend: ["A"],
        gridOn: true,
        xlim: [0, null],
        view: { az: 30, el: 45 },
        colormapData: [
          [0, 0, 0],
          [1, 1, 1],
        ],
        traces: [
          {
            x: [1, 2, 3, 4],
            y: [10, 20, NaN, 40],
            color: [1, 0, 0],
            lineWidth: 2,
            marker: "o",
          },
        ],
        plot3Traces: [],
        surfTraces: [
          {
            x: [0, 0, 1, 1, 2, 2],
            y: [0, 1, 0, 1, 0, 1],
            z: [1, 2, 3, 4, 5, 6], // column-major, rows=2 cols=3
            rows: 2,
            cols: 3,
            faceColor: "interp",
          },
        ],
        pcolorTraces: [],
        contourTraces: [],
        barTraces: [],
        barhTraces: [],
        bar3Traces: [],
        bar3hTraces: [],
        errorBarTraces: [],
        boxTraces: [],
        quiverTraces: [],
        quiver3Traces: [],
        areaTraces: [],
        areaBaseValue: 0,
        patchTraces: [
          {
            vertices: [
              [0, 0],
              [1, 0],
              [1, 1],
              [0, 1],
              [2, 2],
            ],
            faces: [
              [0, 1, 2],
              [0, 2, 3, 4],
            ],
            faceColor: [0, 0, 1],
            faceAlpha: 0.5,
          },
        ],
      },
    },
  } as unknown as FigureState;
}

describe("figure HDF5 round-trip", () => {
  it("export then import reproduces the figure data", async () => {
    const bytes = await exportFigureHdf5(sampleFigure());
    const fig = await importFigureHdf5(bytes);

    expect(fig.sgtitle).toBe("Round Trip");
    expect(fig.currentAxesIndex).toBe(1);
    expect(fig.subplotGrid).toEqual({ rows: 1, cols: 1 });

    const ax = fig.axes[1];
    expect(ax.title).toBe("Axes 1");
    expect(ax.xlabel).toBe("t");
    expect(ax.legend).toEqual(["A"]);
    expect(ax.holdOn).toBe(true);
    expect(ax.gridOn).toBe(true);
    expect(ax.xlim).toEqual([0, null]);
    expect(ax.view).toEqual({ az: 30, el: 45 });
    expect(ax.colormapData).toEqual([
      [0, 0, 0],
      [1, 1, 1],
    ]);

    // line trace: NaN preserved, color/style restored
    expect(ax.traces).toHaveLength(1);
    const line = ax.traces[0];
    expect(line.x).toEqual([1, 2, 3, 4]);
    expect(line.y[2]).toBeNaN();
    expect([line.y[0], line.y[1], line.y[3]]).toEqual([10, 20, 40]);
    expect(line.color).toEqual([1, 0, 0]);
    expect(line.lineWidth).toBe(2);
    expect(line.marker).toBe("o");

    // surf: z restored to original column-major flat
    expect(ax.surfTraces).toHaveLength(1);
    const surf = ax.surfTraces[0];
    expect(surf.rows).toBe(2);
    expect(surf.cols).toBe(3);
    expect(surf.z).toEqual([1, 2, 3, 4, 5, 6]);
    expect(surf.faceColor).toBe("interp");

    // patch: ragged faces restored, vertices, alpha
    expect(ax.patchTraces).toHaveLength(1);
    const patch = ax.patchTraces[0];
    expect(patch.faces).toEqual([
      [0, 1, 2],
      [0, 2, 3, 4],
    ]);
    expect(patch.vertices).toEqual([
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [2, 2],
    ]);
    expect(patch.faceColor).toEqual([0, 0, 1]);
    expect(patch.faceAlpha).toBe(0.5);
  });
});
