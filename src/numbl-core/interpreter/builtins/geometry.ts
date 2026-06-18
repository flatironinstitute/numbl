/**
 * Interpreter IBuiltins for computational geometry functions: delaunay,
 * delaunayn, inpolygon.
 *
 * delaunay/delaunayn are backed by the qhull WASM module, installed as the
 * Delaunay backend at startup (see geometry-bridge.ts and the qhull-node /
 * qhull-browser loaders). The backend must be installed before these builtins
 * run; otherwise they throw.
 */

import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
import { isRuntimeTensor } from "../../runtime/types.js";
import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import type { JitType } from "../../jitTypes.js";
import { defineBuiltin } from "./types.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { getDelaunayBackend } from "../../native/geometry-bridge.js";

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

/** Absolute volume of a dim-simplex (1/d! * |det| of its edge-vector matrix),
 *  used to detect degenerate (flat, zero-volume) simplices. */
function simplexVolume(
  points: number[][],
  cell: number[],
  dim: number
): number {
  const M: number[][] = [];
  const p0 = points[cell[0]];
  for (let r = 0; r < dim; r++) {
    const pr = points[cell[r + 1]];
    const row = new Array<number>(dim);
    for (let c = 0; c < dim; c++) row[c] = pr[c] - p0[c];
    M.push(row);
  }
  // Determinant via Gaussian elimination with partial pivoting.
  let det = 1;
  for (let col = 0; col < dim; col++) {
    let piv = col;
    for (let r = col + 1; r < dim; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (M[piv][col] === 0) return 0;
    if (piv !== col) {
      [M[piv], M[col]] = [M[col], M[piv]];
      det = -det;
    }
    det *= M[col][col];
    for (let r = col + 1; r < dim; r++) {
      const f = M[r][col] / M[col][col];
      for (let c = col; c < dim; c++) M[r][c] -= f * M[col][c];
    }
  }
  let fact = 1;
  for (let k = 2; k <= dim; k++) fact *= k;
  return Math.abs(det) / fact;
}

/** Triangulate points and pack the resulting simplices into a numt-by-(dim+1)
 *  tensor of 1-based vertex indices. */
function triangulateToTensor(points: number[][], dim: number): RuntimeTensor {
  // The qhull WASM backend (exact and robust to cospherical/coplanar input
  // such as a regular grid) is installed at startup. It must be present.
  const backend = getDelaunayBackend();
  if (!backend)
    throw new RuntimeError(
      "delaunay/delaunayn: triangulation backend not initialized. " +
        "In Node call loadQhullNodeBackend() (the CLI and library do this " +
        "automatically); in the browser worker it loads on startup."
    );
  const raw = backend(points, dim);
  // Discard degenerate (zero-volume) simplices. Triangulating cospherical /
  // coplanar facets (e.g. a regular grid) emits flat slivers — qhull's "Qt"
  // does this, and the JS fallback does too — which corrupt downstream use
  // (e.g. DistMesh edge forces). MATLAB's delaunayn likewise discards them.
  // The threshold is tiny relative to the data scale so only numerically-zero
  // volumes are removed, never legitimately thin simplices.
  let scale = 0;
  for (let c = 0; c < dim; c++) {
    let lo = Infinity,
      hi = -Infinity;
    for (const p of points) {
      if (p[c] < lo) lo = p[c];
      if (p[c] > hi) hi = p[c];
    }
    scale = Math.max(scale, hi - lo);
  }
  const volTol = scale > 0 ? Math.pow(scale, dim) * 1e-10 : 0;
  const cells =
    volTol > 0
      ? raw.filter(cell => simplexVolume(points, cell, dim) > volTol)
      : raw;
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
// MATLAB's delaunayn supports arbitrary n. The qhull backend does too, but we
// have only validated n = 2 and n = 3 against MATLAB (and the WASM wrapper is
// exercised only in those dimensions), so we conservatively reject n >= 4 for
// now rather than return unvalidated triangulations. Raising this cap is a
// matter of validating qhull's higher-dimensional output.

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
          // See the dimension-limitation note above: only n <= 3 is validated.
          throw new RuntimeError(
            `delaunayn: only 2-D and 3-D triangulations are supported (got ${n}-D); ` +
              `higher dimensions are not yet validated`
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

// ── inpolygon ────────────────────────────────────────────────────────────

/** A query point's value and shape (shape is null for a scalar input). */
function getQueryValues(
  v: RuntimeValue,
  name: string
): { data: number[]; shape: number[] | null } {
  if (typeof v === "number") return { data: [v], shape: null };
  if (typeof v === "boolean") return { data: [v ? 1 : 0], shape: null };
  if (isRuntimeTensor(v)) {
    if (v.imag) throw new RuntimeError(`${name}: inputs must be real`);
    return { data: Array.from(v.data), shape: [...v.shape] };
  }
  throw new RuntimeError(`${name}: inputs must be numeric`);
}

/** Split NaN-separated vertices into closed loops (each a list of [x, y]). */
function buildPolygonLoops(xv: number[], yv: number[]): number[][][] {
  const loops: number[][][] = [];
  let cur: number[][] = [];
  for (let i = 0; i < xv.length; i++) {
    const x = xv[i];
    const y = yv[i];
    if (x !== x || y !== y) {
      // NaN separates distinct loops (multiply connected / disjoint).
      if (cur.length > 0) {
        loops.push(cur);
        cur = [];
      }
    } else {
      cur.push([x, y]);
    }
  }
  if (cur.length > 0) loops.push(cur);
  return loops;
}

/** Shortest distance from (px,py) to segment (x1,y1)-(x2,y2). */
function distToSegment(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(px - x1, py - y1);
  let t = ((px - x1) * dx + (py - y1) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

/** Classify a query point against the polygon loops. `inside` uses the
 *  nonzero winding-number rule — this is why MATLAB requires multiply
 *  connected polygons to orient their outer/inner loops oppositely (so a
 *  hole's winding cancels to 0). `on` flags points within `tol` of any edge.
 *  The `in` output (inside-or-on) is `inside || on`. */
function classifyPoint(
  px: number,
  py: number,
  loops: number[][][],
  tol: number
): { in: boolean; on: boolean } {
  let wn = 0;
  let on = false;
  for (const loop of loops) {
    const m = loop.length;
    if (m === 1) {
      if (Math.hypot(px - loop[0][0], py - loop[0][1]) <= tol) on = true;
      continue;
    }
    // Iterate directed edges loop[b] -> loop[a] (b is the previous vertex, so
    // edges run in forward order with the closing edge handled by the wrap).
    for (let a = 0, b = m - 1; a < m; b = a++) {
      const x0 = loop[b][0];
      const y0 = loop[b][1];
      const x1 = loop[a][0];
      const y1 = loop[a][1];
      if (!on && distToSegment(px, py, x1, y1, x0, y0) <= tol) on = true;
      // Winding number: count signed crossings of the upward/downward edges.
      const isLeft = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0);
      if (y0 <= py) {
        if (y1 > py && isLeft > 0) wn++;
      } else if (y1 <= py && isLeft < 0) {
        wn--;
      }
    }
  }
  return { in: wn !== 0 || on, on };
}

defineBuiltin({
  name: "inpolygon",
  help: {
    signatures: [
      "in = inpolygon(xq,yq,xv,yv)",
      "[in,on] = inpolygon(xq,yq,xv,yv)",
    ],
    description:
      "Determine whether query points (xq,yq) are inside or on the edge of " +
      "the polygon with vertices (xv,yv). Returns logical arrays the same " +
      "size as xq: `in` is true inside or on the edge, `on` is true only on " +
      "the edge. Use NaN to separate vertices of multiply connected or " +
      "disjoint polygons.",
  },
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 4) return null;
        const out: JitType = {
          kind: "tensor",
          isComplex: false,
          isLogical: true,
          shape: undefined,
        };
        return nargout > 1 ? [out, out] : [out];
      },
      apply: (args, nargout) => {
        if (args.length !== 4)
          throw new RuntimeError("inpolygon requires 4 arguments");
        const xq = getQueryValues(args[0], "inpolygon");
        const yq = getQueryValues(args[1], "inpolygon");
        if (xq.data.length !== yq.data.length)
          throw new RuntimeError("inpolygon: xq and yq must be the same size");
        const xv = toFlatArray(args[2], "inpolygon");
        const yv = toFlatArray(args[3], "inpolygon");
        if (xv.length !== yv.length)
          throw new RuntimeError("inpolygon: xv and yv must be the same size");

        const loops = buildPolygonLoops(xv, yv);

        // Tolerance for the on-edge test, scaled to the polygon's magnitude
        // (mirrors MATLAB, where `on` is a near-edge test, not exact equality).
        let scale = 0;
        for (let i = 0; i < xv.length; i++) {
          if (Number.isFinite(xv[i])) scale = Math.max(scale, Math.abs(xv[i]));
          if (Number.isFinite(yv[i])) scale = Math.max(scale, Math.abs(yv[i]));
        }
        const tol = Math.max(scale, 1) * Math.sqrt(Number.EPSILON);

        const n = xq.data.length;
        const inData = allocFloat64Array(n);
        const onData = nargout > 1 ? allocFloat64Array(n) : undefined;
        for (let i = 0; i < n; i++) {
          const r = classifyPoint(xq.data[i], yq.data[i], loops, tol);
          inData[i] = r.in ? 1 : 0;
          if (onData) onData[i] = r.on ? 1 : 0;
        }

        const makeOut = (data: Float64Array): RuntimeValue => {
          if (xq.shape === null) return RTV.logical(data[0] !== 0);
          const t = RTV.tensor(data, xq.shape) as RuntimeTensor;
          t._isLogical = true;
          return t;
        };

        const inResult = makeOut(inData);
        if (nargout > 1) return [inResult, makeOut(onData!)];
        return inResult;
      },
    },
  ],
});
