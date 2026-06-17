/**
 * Marching cubes over a regular scalar grid, plus the MATLAB `isosurface`
 * argument parsing. Backs the `isosurface` builtin.
 *
 * The 256-case lookup tables are vendored in `marching-cubes-tables.ts`
 * (MIT, mikolalysenko/isosurface — Paul Bourke's classic table). The loop
 * here adds the two things that library lacks for our use:
 *   - vertices interpolated from arbitrary X/Y/Z coordinate arrays (so it is
 *     correct for ndgrid, meshgrid, vector, and implicit grids), and
 *   - cross-cell vertex sharing (keyed by grid-edge identity), which MATLAB
 *     produces by default and which downstream code (e.g. triangle adjacency)
 *     relies on.
 */

import type { RuntimeValue, RuntimeTensor } from "../runtime/types.js";
import {
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
} from "../runtime/types.js";
import { toNumber, toString } from "../runtime/convert.js";
import { RuntimeError } from "../runtime/error.js";
import {
  edgeTable,
  triTable,
  cubeVerts,
  edgeIndex,
} from "./marching-cubes-tables.js";

export interface IsoMesh {
  /** Physical vertex coordinates, one `[x, y, z]` per vertex. */
  vertices: number[][];
  /** Triangles as 0-based vertex-index triples. */
  faces: number[][];
  /** Per-vertex scalar color data (present only when a colors array is given). */
  colors?: number[];
}

type ScalarFn = (i: number, j: number, k: number) => number;
type CoordFn = (i: number, j: number, k: number) => [number, number, number];

/**
 * Extract the isosurface of `getVal` at `iso` over an `m×n×p` grid.
 * `getCoord` maps a grid corner to its physical position; `getColor` (if
 * given) supplies a scalar field interpolated onto the surface vertices.
 */
export function marchingCubes(
  dims: [number, number, number],
  getVal: ScalarFn,
  getCoord: CoordFn,
  iso: number,
  opts?: { share?: boolean; getColor?: ScalarFn }
): IsoMesh {
  const [m, n, p] = dims;
  const share = opts?.share !== false;
  const getColor = opts?.getColor;
  const vertices: number[][] = [];
  const faces: number[][] = [];
  const colors: number[] | undefined = getColor ? [] : undefined;
  // Shared-vertex cache keyed by the unordered pair of corner linear indices.
  const vmap = share ? new Map<string, number>() : null;
  const cornerVal = new Array<number>(8);
  const edges = new Array<number>(12);

  const lin = (i: number, j: number, k: number) => i + m * j + m * n * k;

  const makeVertex = (
    ia: number,
    ja: number,
    ka: number,
    ib: number,
    jb: number,
    kb: number,
    va: number,
    vb: number
  ): number => {
    const a = va - iso;
    const d = a - (vb - iso);
    const t = Math.abs(d) > 1e-12 ? a / d : 0;
    const pa = getCoord(ia, ja, ka);
    const pb = getCoord(ib, jb, kb);
    const idx = vertices.length;
    vertices.push([
      pa[0] + t * (pb[0] - pa[0]),
      pa[1] + t * (pb[1] - pa[1]),
      pa[2] + t * (pb[2] - pa[2]),
    ]);
    if (colors && getColor) {
      const ca = getColor(ia, ja, ka);
      const cb = getColor(ib, jb, kb);
      colors.push(ca + t * (cb - ca));
    }
    return idx;
  };

  for (let k = 0; k < p - 1; k++) {
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < m - 1; i++) {
        let cubeIndex = 0;
        for (let c = 0; c < 8; c++) {
          const o = cubeVerts[c];
          const v = getVal(i + o[0], j + o[1], k + o[2]);
          cornerVal[c] = v;
          if (v > iso) cubeIndex |= 1 << c;
        }
        const em = edgeTable[cubeIndex];
        if (em === 0) continue;
        for (let e = 0; e < 12; e++) {
          if ((em & (1 << e)) === 0) continue;
          const ea = edgeIndex[e][0];
          const eb = edgeIndex[e][1];
          const oa = cubeVerts[ea];
          const ob = cubeVerts[eb];
          const ia = i + oa[0];
          const ja = j + oa[1];
          const ka = k + oa[2];
          const ib = i + ob[0];
          const jb = j + ob[1];
          const kb = k + ob[2];
          if (vmap) {
            const ga = lin(ia, ja, ka);
            const gb = lin(ib, jb, kb);
            const key = ga < gb ? ga + "_" + gb : gb + "_" + ga;
            let vid = vmap.get(key);
            if (vid === undefined) {
              vid = makeVertex(
                ia,
                ja,
                ka,
                ib,
                jb,
                kb,
                cornerVal[ea],
                cornerVal[eb]
              );
              vmap.set(key, vid);
            }
            edges[e] = vid;
          } else {
            edges[e] = makeVertex(
              ia,
              ja,
              ka,
              ib,
              jb,
              kb,
              cornerVal[ea],
              cornerVal[eb]
            );
          }
        }
        const tri = triTable[cubeIndex];
        for (let f = 0; f + 2 < tri.length; f += 3) {
          faces.push([edges[tri[f]], edges[tri[f + 1]], edges[tri[f + 2]]]);
        }
      }
    }
  }
  return { vertices, faces, colors };
}

// ── MATLAB isosurface() argument parsing ───────────────────────────────────

function isNumericArg(v: RuntimeValue): boolean {
  return typeof v === "number" || typeof v === "boolean" || isRuntimeTensor(v);
}

/** 3-D dimensions of a volume tensor (pages default to 1 for a 2-D array). */
function dims3(v: RuntimeTensor): [number, number, number] {
  const s = v.shape;
  return [s[0] ?? 1, s[1] ?? 1, s[2] ?? 1];
}

/** Build a per-corner scalar accessor over a tensor the same size as V. */
function tensorAccessor(t: RuntimeTensor, m: number, n: number): ScalarFn {
  return (i, j, k) => t.data[i + m * j + m * n * k];
}

/**
 * Parse `isosurface(...)` arguments and compute the mesh.
 *
 * Forms (by numeric-argument count, after stripping 'verbose'/'noshare'):
 *   1: isosurface(V)               4: isosurface(X,Y,Z,V)
 *   2: isosurface(V,iso)           5: isosurface(X,Y,Z,V,iso)
 *   3: isosurface(V,iso,colors)    6: isosurface(X,Y,Z,V,iso,colors)
 *
 * When the isovalue is omitted it is chosen automatically (see autoIsovalue) —
 * an approximation that does not match MATLAB's exact histogram heuristic.
 */
export function isosurfaceFromArgs(args: RuntimeValue[]): IsoMesh {
  let share = true;
  const nums: RuntimeValue[] = [];
  for (const a of args) {
    if (isRuntimeChar(a) || isRuntimeString(a)) {
      const s = toString(a).toLowerCase();
      if (s === "noshare") share = false;
      // 'verbose' is accepted and ignored.
    } else if (isNumericArg(a)) {
      nums.push(a);
    }
  }

  let X: RuntimeValue | undefined;
  let Y: RuntimeValue | undefined;
  let Z: RuntimeValue | undefined;
  let V: RuntimeValue;
  let isoArg: RuntimeValue | undefined;
  let colorsArg: RuntimeValue | undefined;

  switch (nums.length) {
    case 1:
      V = nums[0];
      break;
    case 2:
      V = nums[0];
      isoArg = nums[1];
      break;
    case 3:
      V = nums[0];
      isoArg = nums[1];
      colorsArg = nums[2];
      break;
    case 4:
      [X, Y, Z, V] = nums;
      break;
    case 5:
      [X, Y, Z, V, isoArg] = nums;
      break;
    case 6:
      [X, Y, Z, V, isoArg, colorsArg] = nums;
      break;
    default:
      throw new RuntimeError("isosurface: invalid number of arguments");
  }

  if (!isRuntimeTensor(V))
    throw new RuntimeError("isosurface: V must be a 3-D array");
  const [m, n, p] = dims3(V);
  const getVal = tensorAccessor(V, m, n);

  // Coordinate accessor: explicit X/Y/Z (3-D arrays or vectors), or the
  // implicit 1-based index grid.
  let getCoord: CoordFn;
  if (X && Y && Z) {
    const axisAccessor = (c: RuntimeValue, axis: 0 | 1 | 2): ScalarFn => {
      if (isRuntimeTensor(c) && c.data.length === m * n * p) {
        return tensorAccessor(c, m, n);
      }
      // Vector form (meshgrid convention): X varies along columns (j), Y along
      // rows (i), Z along pages (k).
      const arr = isRuntimeTensor(c) ? Array.from(c.data) : [toNumber(c)];
      const pick =
        axis === 0
          ? (j: number) => arr[j]
          : axis === 1
            ? (i: number) => arr[i]
            : (k: number) => arr[k];
      return (i, j, k) =>
        axis === 0 ? pick(j) : axis === 1 ? pick(i) : pick(k);
    };
    const gx = axisAccessor(X, 0);
    const gy = axisAccessor(Y, 1);
    const gz = axisAccessor(Z, 2);
    getCoord = (i, j, k) => [gx(i, j, k), gy(i, j, k), gz(i, j, k)];
  } else {
    getCoord = (i, j, k) => [j + 1, i + 1, k + 1];
  }

  const iso = isoArg !== undefined ? toNumber(isoArg) : autoIsovalue(V);

  let getColor: ScalarFn | undefined;
  if (colorsArg !== undefined && isRuntimeTensor(colorsArg)) {
    if (colorsArg.data.length !== m * n * p)
      throw new RuntimeError("isosurface: colors must be the same size as V");
    getColor = tensorAccessor(colorsArg, m, n);
  }

  return marchingCubes([m, n, p], getVal, getCoord, iso, { share, getColor });
}

/** Approximate auto-isovalue (midpoint of the finite data range). Does NOT
 *  match MATLAB's histogram-based heuristic. */
function autoIsovalue(V: RuntimeTensor): number {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < V.data.length; i++) {
    const v = V.data[i];
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo)) return 0;
  return (lo + hi) / 2;
}
