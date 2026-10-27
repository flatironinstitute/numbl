import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import {
  figuresReducer,
  initialFiguresState,
  type FiguresState,
} from "../numbl-core/plotting/figuresReducer.js";
import type { AxesState } from "../numbl-core/plotting/figuresReducer.js";
import { computeAxisLimits } from "../numbl-core/plotting/axisLimits.js";

/** Reduce all emitted plot instructions (as the viewer does) and return the
 *  current axes — the rendered state after the script runs. */
function renderedAxes(code: string): AxesState {
  const result = executeCode(code);
  let state: FiguresState = initialFiguresState;
  for (const instr of result.plotInstructions) {
    state = figuresReducer(state, instr);
  }
  const fig = state.figs[state.currentHandle];
  return fig.axes[fig.currentAxesIndex];
}

describe("trimesh", () => {
  it("2-D form builds a patch with faces and 2-D vertices", () => {
    // Unit square split along the diagonal into two triangles.
    const axes = renderedAxes("trimesh([1 2 3; 1 3 4], [0;1;1;0], [0;0;1;1]);");
    expect(axes.patchTraces).toHaveLength(1);
    const p = axes.patchTraces[0];
    expect(p.faces).toEqual([
      [0, 1, 2],
      [0, 2, 3],
    ]);
    expect(p.vertices).toEqual([
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ]);
    expect(p.is3D).toBe(false);
    // Documented defaults: light-gray faces, black edges.
    expect(p.faceColor).toEqual([0.85, 0.85, 0.85]);
    expect(p.edgeColor).toEqual([0, 0, 0]);
  });

  it("3-D form carries z-coordinates on each vertex (is3D)", () => {
    const axes = renderedAxes(
      "trimesh([1 2 3; 1 3 4], [0;1;1;0], [0;0;1;1], [0;0;2;2]);"
    );
    const p = axes.patchTraces[0];
    expect(p.is3D).toBe(true);
    expect(p.vertices).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 2],
      [0, 1, 2],
    ]);
  });

  it("honors FaceColor/EdgeColor name-value pairs (DistMesh idiom)", () => {
    const axes = renderedAxes(
      "trimesh([1 2 3], [0;1;0], [0;0;1], 0*[0;1;0], 'facecolor', [0.8 0.9 1], 'edgecolor', 'k');"
    );
    const p = axes.patchTraces[0];
    expect(p.faceColor).toEqual([0.8, 0.9, 1]);
    expect(p.edgeColor).toEqual([0, 0, 0]);
  });

  it("works with meshgrid/delaunay matrix inputs (peaks-style mesh)", () => {
    const axes = renderedAxes(
      "[x,y]=meshgrid(1:5,1:5); z=peaks(5); T=delaunay(x,y); trimesh(T,x,y,z);"
    );
    const p = axes.patchTraces[0];
    expect(p.is3D).toBe(true);
    expect(p.vertices).toHaveLength(25);
    expect(p.faces.length).toBeGreaterThan(0);
    // z is not constant, so the vertices span a real z range.
    const zs = p.vertices.map(v => v[2]);
    expect(Math.max(...zs) - Math.min(...zs)).toBeGreaterThan(0.1);
  });

  it("axis() reflects a flat patch's extent (DistMesh simpplot pattern)", () => {
    // A flat mesh spanning [-1,1]x[-1,1]. simpplot does `ax=axis;
    // axis(ax*1.001)`, so axis() must cover the patch — not default to
    // [0 1 0 1], which previously cropped the view to one quadrant.
    const axes = renderedAxes(
      "trimesh([1 2 3; 1 3 4], [-1;1;1;-1], [-1;-1;1;1], 0*[-1;1;1;-1]);"
    );
    const lim = computeAxisLimits(axes);
    expect(lim).toHaveLength(4); // flat patch -> 2-D limits
    expect(lim[0]).toBeLessThanOrEqual(-1);
    expect(lim[1]).toBeGreaterThanOrEqual(1);
    expect(lim[2]).toBeLessThanOrEqual(-1);
    expect(lim[3]).toBeGreaterThanOrEqual(1);
  });

  it("redraws replace the previous mesh when hold is off (no accumulation)", () => {
    // trimesh is a high-level function (calls newplot), so re-drawing with hold
    // off replaces the axes contents — an animation/iteration loop must not
    // accumulate overlaid meshes (the DistMesh simpplot bug).
    const axes = renderedAxes(
      "trimesh([1 2 3], [0;1;0], [0;0;1], [0;0;0]);" +
        "trimesh([1 2 3], [0;2;0], [0;0;2], [0;0;0]);"
    );
    expect(axes.patchTraces).toHaveLength(1);
    // Only the second mesh survives.
    expect(axes.patchTraces[0].vertices).toEqual([
      [0, 0, 0],
      [2, 0, 0],
      [0, 2, 0],
    ]);
  });

  it("accumulates meshes across calls when hold is on", () => {
    const axes = renderedAxes(
      "hold on;" +
        "trimesh([1 2 3], [0;1;0], [0;0;1], [0;0;0]);" +
        "trimesh([1 2 3], [0;2;0], [0;0;2], [0;0;0]);"
    );
    expect(axes.patchTraces).toHaveLength(2);
  });

  it("a hold-off redraw also clears other trace types (newplot)", () => {
    // plot() then trimesh() with hold off: the trimesh replaces the line.
    const axes = renderedAxes(
      "plot([0 1],[0 1]); trimesh([1 2 3], [0;1;0], [0;0;1], [0;0;0]);"
    );
    expect(axes.patchTraces).toHaveLength(1);
    expect(axes.traces).toHaveLength(0);
  });

  it("axis() returns a 6-vector for a z-varying 3-D mesh", () => {
    const axes = renderedAxes(
      "trimesh([1 2 3; 1 3 4], [0;1;1;0], [0;0;1;1], [0;0;5;5]);"
    );
    const lim = computeAxisLimits(axes);
    expect(lim).toHaveLength(6);
    expect(lim[4]).toBeLessThanOrEqual(0);
    expect(lim[5]).toBeGreaterThanOrEqual(5);
  });
});

describe("trisurf", () => {
  it("defaults to flat faces colored by per-vertex z", () => {
    const axes = renderedAxes(
      "trisurf([1 2 3; 1 3 4], [0;1;1;0], [0;0;1;1], [0;1;2;3]);"
    );
    const p = axes.patchTraces[0];
    expect(p.is3D).toBe(true);
    expect(p.faceColor).toBe("flat");
    expect(p.edgeColor).toEqual([0, 0, 0]);
    expect(p.faceVertexCData).toEqual([0, 1, 2, 3]);
  });

  it("uses c for color data and passes Name-Value pairs through", () => {
    const axes = renderedAxes(
      "trisurf([1 2 3; 1 3 4], [0;1;1;0], [0;0;1;1], [0;1;2;3], [5;6;7;8], " +
        "'EdgeColor', 'black', 'EdgeAlpha', 0.1, 'AmbientStrength', 0.6);"
    );
    const p = axes.patchTraces[0];
    expect(p.faceVertexCData).toEqual([5, 6, 7, 8]);
    expect(p.edgeAlpha).toBe(0.1);
  });

  it("replaces the previous plot when hold is off (newplot)", () => {
    const axes = renderedAxes(
      "trisurf([1 2 3], [0;1;0], [0;0;1], [0;0;1]);" +
        "trisurf([1 2 3], [0;2;0], [0;0;2], [0;0;1]);"
    );
    expect(axes.patchTraces).toHaveLength(1);
  });
});
