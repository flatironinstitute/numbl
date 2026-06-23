/**
 * Interpreter IBuiltins for computational geometry functions: delaunay,
 * delaunayn, convhull, convhulln, inpolygon.
 *
 * delaunay/delaunayn/convhull/convhulln are backed by the qhull WASM module,
 * installed at startup (see geometry-bridge.ts and the qhull-node /
 * qhull-browser loaders). The backend must be installed before these builtins
 * run; otherwise they throw.
 */

import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
import { isRuntimeTensor, isRuntimeChar } from "../../runtime/types.js";
import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import type { JitType } from "../../jitTypes.js";
import { defineBuiltin } from "./types.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import {
  getDelaunayBackend,
  getConvexHullBackend,
} from "../../native/geometry-bridge.js";

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
function triangulateToTensor(
  points: number[][],
  dim: number,
  orientCCW = false
): RuntimeTensor {
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
  // MATLAB's delaunay returns 2-D triangles with a consistent CCW winding
  // (positive signed area). Swap the last two vertices of any clockwise
  // triangle so downstream code (boundary tracing, signed areas) can rely on
  // it. Tetrahedra (3-D) carry no analogous MATLAB guarantee.
  if (orientCCW && dim === 2) {
    for (const cell of cells) {
      const [a, b, c] = cell;
      const signedArea2 =
        (points[b][0] - points[a][0]) * (points[c][1] - points[a][1]) -
        (points[c][0] - points[a][0]) * (points[b][1] - points[a][1]);
      if (signedArea2 < 0) {
        cell[1] = c;
        cell[2] = b;
      }
    }
  }
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

        return triangulateToTensor(points, dim, true);
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

// ── convhull / convhulln ─────────────────────────────────────────────────
//
// Backed by the qhull convex-hull backend (same WASM module as delaunay).
// qhull returns the MINIMAL hull — collinear (2-D) / coplanar (3-D) points
// that do not contribute to the hull are dropped. This matches MATLAB's
// `Simplify=true`. The `Simplify` name-value pair (convhull) and `opts`
// argument (convhulln) are accepted for MATLAB compatibility but do not change
// the result; the hull geometry, area, and volume are exact in either case.

const CONVHULLN_MAX_DIM = 3;

/** Run the convex-hull backend, returning simplicial facets as 0-based indices. */
function convexHullFacets(
  points: number[][],
  dim: number,
  name: string
): number[][] {
  const backend = getConvexHullBackend();
  if (!backend)
    throw new RuntimeError(
      `${name}: convex-hull backend not initialized. In Node call ` +
        `loadQhullNodeBackend() (the CLI and library do this automatically); ` +
        `in the browser worker it loads on startup.`
    );
  return backend(points, dim);
}

/** A point strictly interior to the convex hull: the mean of its vertices. */
function hullInteriorPoint(
  points: number[][],
  facets: number[][],
  dim: number
): number[] {
  const used = new Set<number>();
  for (const f of facets) for (const idx of f) used.add(idx);
  const c = new Array<number>(dim).fill(0);
  for (const idx of used) {
    const p = points[idx];
    for (let k = 0; k < dim; k++) c[k] += p[k];
  }
  const m = used.size || 1;
  for (let k = 0; k < dim; k++) c[k] /= m;
  return c;
}

/** Area (dim=2) or volume (dim=3) of the hull, via a fan decomposition from an
 *  interior point. Each boundary simplex + the interior point forms a
 *  non-overlapping piece tiling the convex hull, so summing |measure| is exact
 *  regardless of facet orientation. */
function hullMeasure(
  points: number[][],
  facets: number[][],
  dim: number
): number {
  const c = hullInteriorPoint(points, facets, dim);
  let total = 0;
  if (dim === 2) {
    for (const f of facets) {
      const a = points[f[0]];
      const b = points[f[1]];
      const ax = a[0] - c[0],
        ay = a[1] - c[1];
      const bx = b[0] - c[0],
        by = b[1] - c[1];
      total += Math.abs(ax * by - ay * bx) / 2;
    }
  } else {
    for (const f of facets) {
      const a = points[f[0]];
      const b = points[f[1]];
      const d = points[f[2]];
      const ax = a[0] - c[0],
        ay = a[1] - c[1],
        az = a[2] - c[2];
      const bx = b[0] - c[0],
        by = b[1] - c[1],
        bz = b[2] - c[2];
      const dx = d[0] - c[0],
        dy = d[1] - c[1],
        dz = d[2] - c[2];
      const det =
        ax * (by * dz - bz * dy) -
        ay * (bx * dz - bz * dx) +
        az * (bx * dy - by * dx);
      total += Math.abs(det) / 6;
    }
  }
  return total;
}

/** Pack facets (0-based) into a numFacets-by-cols tensor of 1-based indices. */
function facetsToTensor(facets: number[][], cols: number): RuntimeTensor {
  const n = facets.length;
  const out = allocFloat64Array(n * cols);
  for (let i = 0; i < n; i++) {
    const f = facets[i];
    for (let j = 0; j < cols; j++) out[j * n + i] = f[j] + 1; // column-major
  }
  return RTV.tensor(out, [n, cols]);
}

/** Order the 2-D hull edges into a single counter-clockwise boundary loop of
 *  1-based point indices, closed (last == first), as MATLAB's convhull returns. */
function orderHull2D(facets: number[][], points: number[][]): RuntimeTensor {
  // Build adjacency: each hull vertex of a convex polygon has exactly 2 edges.
  const adj = new Map<number, number[]>();
  for (const [a, b] of facets) {
    (adj.get(a) ?? adj.set(a, []).get(a)!).push(b);
    (adj.get(b) ?? adj.set(b, []).get(b)!).push(a);
  }
  const start = facets[0][0];
  const loop: number[] = [start];
  let prev = -1;
  let cur = start;
  for (let guard = 0; guard <= facets.length; guard++) {
    const nbrs = adj.get(cur)!;
    const next = nbrs[0] === prev ? nbrs[1] : nbrs[0];
    if (next === undefined || next === start) break;
    loop.push(next);
    prev = cur;
    cur = next;
  }
  // Orient counter-clockwise (positive signed area).
  let signed = 0;
  for (let i = 0; i < loop.length; i++) {
    const p = points[loop[i]];
    const q = points[loop[(i + 1) % loop.length]];
    signed += p[0] * q[1] - q[0] * p[1];
  }
  if (signed < 0) loop.reverse();
  loop.push(loop[0]); // close the polygon

  const out = allocFloat64Array(loop.length);
  for (let i = 0; i < loop.length; i++) out[i] = loop[i] + 1; // 1-based
  return RTV.tensor(out, [loop.length, 1]); // column vector
}

defineBuiltin({
  name: "convhull",
  help: {
    signatures: [
      "k = convhull(P)",
      "k = convhull(x,y)",
      "k = convhull(x,y,z)",
      "k = convhull(___,'Simplify',tf)",
      "[k,av] = convhull(___)",
    ],
    description:
      "Convex hull of a set of 2-D or 3-D points. With a single matrix P " +
      "(N-by-2 or N-by-3), or coordinate vectors x,y (2-D) or x,y,z (3-D). " +
      "For 2-D input, k is a column vector of 1-based point indices tracing " +
      "the hull boundary counter-clockwise (closed: k(1)==k(end)). For 3-D " +
      "input, k is a numFacets-by-3 matrix of triangle vertex indices. The " +
      "second output av is the area (2-D) or volume (3-D). The 'Simplify' " +
      "name-value pair is accepted for compatibility; the hull is always " +
      "returned in minimal (simplified) form.",
  },
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 2) return null;
        if (argTypes.length < 1 || argTypes.length > 5) return null;
        const ok = (t: JitType) =>
          t.kind === "number" ||
          t.kind === "boolean" ||
          t.kind === "tensor" ||
          t.kind === "string" ||
          t.kind === "char";
        if (!argTypes.every(ok)) return null;
        const k: JitType = { kind: "tensor", isComplex: false };
        return nargout > 1 ? [k, { kind: "number" }] : [k];
      },
      apply: (args, nargout) => {
        // Strip a trailing 'Simplify', tf name-value pair (accepted, ignored).
        let coordArgs = args;
        const tail = args[args.length - 2];
        const isSimplifyName = (v: RuntimeValue) =>
          (typeof v === "string" && v.toLowerCase() === "simplify") ||
          (isRuntimeChar(v) && v.value.toLowerCase() === "simplify");
        if (args.length >= 3 && isSimplifyName(tail)) {
          coordArgs = args.slice(0, args.length - 2);
        }

        let points: number[][];
        let dim: number;
        if (coordArgs.length === 1) {
          const P = coordArgs[0];
          if (!isRuntimeTensor(P))
            throw new RuntimeError(
              "convhull: P must be an N-by-2 or N-by-3 matrix"
            );
          const [, d] = tensorSize2D(P);
          if (d !== 2 && d !== 3)
            throw new RuntimeError("convhull: P must have 2 or 3 columns");
          points = matrixToPoints(P, "convhull");
          dim = d;
        } else if (coordArgs.length === 2 || coordArgs.length === 3) {
          const coords = coordArgs.map(a => toFlatArray(a, "convhull"));
          const n = coords[0].length;
          if (coords.some(c => c.length !== n))
            throw new RuntimeError(
              "convhull: coordinate vectors must have the same length"
            );
          points = new Array(n);
          for (let i = 0; i < n; i++) points[i] = coords.map(c => c[i]);
          dim = coordArgs.length;
        } else {
          throw new RuntimeError("convhull: invalid arguments");
        }

        if (points.length < dim + 1)
          throw new RuntimeError(
            `convhull: need at least ${dim + 1} points for a ${dim}-D hull`
          );

        const facets = convexHullFacets(points, dim, "convhull");
        const k =
          dim === 2 ? orderHull2D(facets, points) : facetsToTensor(facets, 3);
        if (nargout > 1) return [k, hullMeasure(points, facets, dim)];
        return k;
      },
    },
  ],
});

defineBuiltin({
  name: "convhulln",
  help: {
    signatures: [
      "k = convhulln(P)",
      "k = convhulln(P,opts)",
      "[k,vol] = convhulln(___)",
    ],
    description:
      "N-D convex hull. P is an m-by-n matrix of m points in n-dimensional " +
      "space. Returns a numFacets-by-n matrix where each row holds the 1-based " +
      "point indices of one simplicial facet (edges for n=2, triangles for " +
      "n=3). The second output is the area (n=2) or volume (n=3). The opts " +
      "argument (Qhull options) is accepted for MATLAB compatibility but " +
      "ignored. Note: only n = 2 and n = 3 are supported (see source comments).",
  },
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 2) return null;
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        const x = argTypes[0];
        if (x.kind !== "tensor" && x.kind !== "number" && x.kind !== "boolean")
          return null;
        const k: JitType = { kind: "tensor", isComplex: false };
        return nargout > 1 ? [k, { kind: "number" }] : [k];
      },
      apply: (args, nargout) => {
        const P = args[0];
        if (!isRuntimeTensor(P))
          throw new RuntimeError("convhulln: P must be an m-by-n matrix");
        const [m, n] = tensorSize2D(P);
        if (n < 2)
          throw new RuntimeError("convhulln: P must have at least 2 columns");
        if (n > CONVHULLN_MAX_DIM)
          // Only n <= 3 is validated (parity with delaunayn).
          throw new RuntimeError(
            `convhulln: only 2-D and 3-D hulls are supported (got ${n}-D); ` +
              `higher dimensions are not yet validated`
          );
        if (m < n + 1)
          throw new RuntimeError(
            `convhulln: need at least ${n + 1} points for a ${n}-D hull`
          );
        const points = matrixToPoints(P, "convhulln");
        const facets = convexHullFacets(points, n, "convhulln");
        const k = facetsToTensor(facets, n);
        if (nargout > 1) return [k, hullMeasure(points, facets, n)];
        return k;
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
