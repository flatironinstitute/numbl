import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import {
  figuresReducer,
  initialFiguresState,
  type FiguresState,
} from "../graphics/figuresReducer.js";
import type { AxesState } from "../graphics/figuresReducer.js";

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
});
