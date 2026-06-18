import { describe, it, expect } from "vitest";
import h5wasm from "h5wasm";
import { exportFigureHdf5 } from "../graphics/exportFigureHdf5.js";
import { FIGURE_HDF5_VERSION } from "../graphics/figureHdf5Schema.js";
import type { FigureState } from "../graphics/figuresReducer.js";

/** Minimal shape of an h5wasm read node (group or dataset). */
interface H5ReadNode {
  attrs: Record<string, { value: unknown }>;
  value: Iterable<number>;
  shape: number[];
}

function sampleFigure(): FigureState {
  return {
    currentAxesIndex: 1,
    sgtitle: "My Figure",
    axes: {
      1: {
        holdOn: false,
        title: "Axes 1",
        xlabel: "t",
        legend: ["line A"],
        xlim: [0, null], // partial → [0, NaN]
        view: { az: 30, el: 45 },
        traces: [
          {
            x: [1, 2, 3, 4],
            y: [10, 20, NaN, 40],
            color: [1, 0, 0],
            lineWidth: 2,
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
            ], // ragged → padded with -1
            faceColor: [0, 0, 1],
          },
        ],
      },
    },
  } as unknown as FigureState;
}

describe("exportFigureHdf5", () => {
  it("writes a valid HDF5 file with the numbl figure layout", async () => {
    const bytes = await exportFigureHdf5(sampleFigure());

    // HDF5 signature: \x89 H D F \r \n \x1a \n
    expect(Array.from(bytes.slice(0, 8))).toEqual([
      0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a,
    ]);

    const { FS } = await h5wasm.ready;
    const fname = `/test_${Math.random().toString(36).slice(2)}.h5`;
    FS.writeFile(fname, bytes);
    const f = new h5wasm.File(fname, "r");
    try {
      const get = (p: string) => f.get(p) as unknown as H5ReadNode;
      const num = (v: unknown) => (typeof v === "bigint" ? Number(v) : v);
      const root = get("/");
      expect(num(root.attrs.numbl_figure_version.value)).toBe(
        FIGURE_HDF5_VERSION
      );
      expect(root.attrs.sgtitle.value).toBe("My Figure");

      const ax = get("axes/1");
      expect(ax.attrs.title.value).toBe("Axes 1");
      expect(ax.attrs.legend.value).toEqual(["line A"]);
      // partial xlim: null bound encoded as NaN
      const xlim = Array.from(ax.attrs.xlim.value as number[]);
      expect(xlim[0]).toBe(0);
      expect(Number.isNaN(xlim[1])).toBe(true);

      // trace 0 = plot: NaN preserved natively in the y dataset
      const y = get("axes/1/traces/0/y");
      expect(
        Array.from(y.value).map(v => (Number.isNaN(v) ? "nan" : v))
      ).toEqual([10, 20, "nan", 40]);

      // trace 1 = patch: ragged faces padded with -1, 5x2 vertices
      const faces = get("axes/1/traces/1/faces");
      expect(faces.shape).toEqual([2, 4]);
      expect(Array.from(faces.value)).toEqual([0, 1, 2, -1, 0, 2, 3, 4]);
      expect(get("axes/1/traces/1/vertices").shape).toEqual([5, 2]);

      // trace 2 = surf: column-major z transposed to row-major [rows, cols]
      const z = get("axes/1/traces/2/z");
      expect(z.shape).toEqual([2, 3]);
      expect(Array.from(z.value)).toEqual([1, 3, 5, 2, 4, 6]);
    } finally {
      f.close();
      FS.unlink(fname);
    }
  });
});
