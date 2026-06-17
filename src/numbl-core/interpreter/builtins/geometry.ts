/**
 * Interpreter IBuiltins for computational geometry functions: delaunay.
 *
 * Backed by the `delaunay-triangulate` npm package (Delaunay triangulation
 * in arbitrary dimension via lift-to-paraboloid + incremental convex hull).
 */

import type { RuntimeValue } from "../../runtime/types.js";
import { isRuntimeTensor } from "../../runtime/types.js";
import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import type { JitType } from "../../jitTypes.js";
import { defineBuiltin } from "./types.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import delaunayTriangulate from "delaunay-triangulate";

/** Extract a flat list of numbers from a scalar or tensor argument. */
function toFlatArray(v: RuntimeValue, name: string): number[] {
  if (typeof v === "number") return [v];
  if (typeof v === "boolean") return [v ? 1 : 0];
  if (isRuntimeTensor(v)) {
    if (v.imag) throw new RuntimeError(`${name}: inputs must be real`);
    return Array.from(v.data);
  }
  throw new RuntimeError(`${name}: inputs must be numeric`);
}

// ── delaunay ───────────────────────────────────────────────────────────────

defineBuiltin({
  name: "delaunay",
  help: {
    signatures: [
      "DT = delaunay(P)",
      "DT = delaunay(x,y)",
      "DT = delaunay(x,y,z)",
    ],
    description:
      "Delaunay triangulation of a set of points. With a single matrix P " +
      "(N-by-2 or N-by-3), or coordinate vectors x,y (2-D) or x,y,z (3-D). " +
      "Returns a matrix where each row holds the 1-based point indices of one " +
      "triangle (2-D) or tetrahedron (3-D).",
  },
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1) return null;
        if (argTypes.length < 1 || argTypes.length > 3) return null;
        const numeric = (t: JitType) =>
          t.kind === "number" || t.kind === "boolean" || t.kind === "tensor";
        if (!argTypes.every(numeric)) return null;
        return [{ kind: "tensor", isComplex: false }];
      },
      apply: args => {
        let points: number[][];

        if (args.length === 1) {
          // delaunay(P): P is N-by-d, d in {2,3}
          const P = args[0];
          if (!isRuntimeTensor(P))
            throw new RuntimeError(
              "delaunay: P must be an N-by-2 or N-by-3 matrix"
            );
          if (P.imag) throw new RuntimeError("delaunay: inputs must be real");
          const [n, d] = tensorSize2D(P);
          if (d !== 2 && d !== 3)
            throw new RuntimeError("delaunay: P must have 2 or 3 columns");
          points = new Array(n);
          for (let i = 0; i < n; i++) {
            const pt = new Array(d);
            for (let j = 0; j < d; j++) pt[j] = P.data[j * n + i]; // column-major
            points[i] = pt;
          }
        } else {
          // delaunay(x,y) or delaunay(x,y,z): coordinate vectors
          const coords = args.map(a => toFlatArray(a, "delaunay"));
          const n = coords[0].length;
          if (coords.some(c => c.length !== n))
            throw new RuntimeError(
              "delaunay: coordinate vectors must have the same length"
            );
          points = new Array(n);
          for (let i = 0; i < n; i++) points[i] = coords.map(c => c[i]);
        }

        const dim = args.length === 1 ? (points[0]?.length ?? 2) : args.length;
        if (points.length < dim + 1)
          throw new RuntimeError(
            `delaunay: need at least ${dim + 1} points for a ${dim}-D triangulation`
          );

        const cells = delaunayTriangulate(points);
        const numCells = cells.length;
        const cols = dim + 1;
        const out = allocFloat64Array(numCells * cols);
        // Store column-major, converting 0-based indices to 1-based.
        for (let i = 0; i < numCells; i++) {
          const cell = cells[i];
          for (let j = 0; j < cols; j++) out[j * numCells + i] = cell[j] + 1;
        }
        return RTV.tensor(out, [numCells, cols]);
      },
    },
  ],
});
