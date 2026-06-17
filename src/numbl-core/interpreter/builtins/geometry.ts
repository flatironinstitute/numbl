/**
 * Interpreter IBuiltins for computational geometry functions: delaunay, delaunayn.
 *
 * Backed by the `delaunay-triangulate` npm package (Delaunay triangulation
 * in arbitrary dimension via lift-to-paraboloid + incremental convex hull).
 */

import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
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

/** Convert an m-by-d (column-major) tensor into an array of m d-dimensional points. */
function matrixToPoints(P: RuntimeTensor, name: string): number[][] {
  if (P.imag) throw new RuntimeError(`${name}: inputs must be real`);
  const [m, d] = tensorSize2D(P);
  const points: number[][] = new Array(m);
  for (let i = 0; i < m; i++) {
    const pt = new Array(d);
    for (let j = 0; j < d; j++) pt[j] = P.data[j * m + i]; // column-major
    points[i] = pt;
  }
  return points;
}

/** Triangulate points and pack the resulting simplices into a numt-by-(dim+1)
 *  tensor of 1-based vertex indices. */
function triangulateToTensor(points: number[][], dim: number): RuntimeTensor {
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
          const [, d] = tensorSize2D(P);
          if (d !== 2 && d !== 3)
            throw new RuntimeError("delaunay: P must have 2 or 3 columns");
          points = matrixToPoints(P, "delaunay");
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

        return triangulateToTensor(points, dim);
      },
    },
  ],
});

// ── delaunayn ────────────────────────────────────────────────────────────────
//
// DIMENSION LIMITATION (n <= 3 only)
// ----------------------------------
// MATLAB's delaunayn supports arbitrary n, but our `delaunay-triangulate`
// backend only produces CORRECT results for n = 2 and n = 3. We deliberately
// reject n >= 4 rather than return wrong triangulations.
//
// Root cause: delaunay-triangulate computes the n-D Delaunay triangulation as
// the lower convex hull of the points lifted onto a paraboloid in (n+1)
// dimensions. That hull (via `incremental-convex-hull`) relies on the
// `robust-orientation` predicate, which must evaluate the orientation of
// (n+2) points in (n+1)-D space.
//
// `robust-orientation` only implements exact predicates up to 5 points
// (NUM_EXPAND = 5, with hardcoded orientation_3 / _4 / _5 routines). For more
// than 5 points its `orientation(k)` dispatch silently falls back to the
// 5-point routine `orientation_5`, which ignores the extra arguments and
// returns a meaningless sign. n-D Delaunay needs (n+2) points, so:
//     n + 2 <= 5   =>   n <= 3
// is the largest dimension that is actually correct. Empirically n=2 and n=3
// match MATLAB exactly, while n>=4 yields wrong or empty results (e.g. only 3
// of the expected 11 simplices for 8 points in 4-D, with some input vertices
// missing entirely).
//
// To lift this cap in the future we would need robust orientation/insphere
// predicates valid in arbitrary dimension (Shewchuk-style adaptive precision),
// which neither robust-orientation nor mourner's robust-predicates provide.

const DELAUNAYN_MAX_DIM = 3;

defineBuiltin({
  name: "delaunayn",
  help: {
    signatures: ["T = delaunayn(X)", "T = delaunayn(X,opts)"],
    description:
      "N-D Delaunay triangulation. X is an m-by-n matrix of m points in " +
      "n-dimensional space. Returns a numt-by-(n+1) matrix where each row " +
      "holds the 1-based point indices of one simplex. The opts argument " +
      "(Qhull options) is accepted for MATLAB compatibility but ignored. " +
      "Note: only n = 2 and n = 3 are supported (see source comments).",
  },
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1) return null;
        // X is required; opts (any type) is optional and ignored.
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        const x = argTypes[0];
        if (x.kind !== "tensor" && x.kind !== "number" && x.kind !== "boolean")
          return null;
        return [{ kind: "tensor", isComplex: false }];
      },
      apply: args => {
        const X = args[0];
        if (!isRuntimeTensor(X))
          throw new RuntimeError("delaunayn: X must be an m-by-n matrix");
        const [m, n] = tensorSize2D(X);
        if (n < 2)
          throw new RuntimeError("delaunayn: X must have at least 2 columns");
        if (n > DELAUNAYN_MAX_DIM)
          // See the dimension-limitation note above: the backend's robust
          // orientation predicate is only correct up to 4-D space (n <= 3).
          throw new RuntimeError(
            `delaunayn: only 2-D and 3-D triangulations are supported (got ${n}-D); ` +
              `higher dimensions are not reliable with the current backend`
          );
        if (m < n + 1)
          throw new RuntimeError(
            `delaunayn: need at least ${n + 1} points for a ${n}-D triangulation`
          );
        const points = matrixToPoints(X, "delaunayn");
        return triangulateToTensor(points, n);
      },
    },
  ],
});
