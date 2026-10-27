/**
 * Argument parsers for patch-based plots: patch, fill, trimesh, and trisurf.
 */

import {
  type RuntimeValue,
  type RuntimeStruct,
  isRuntimeTensor,
  isRuntimeStruct,
} from "../runtime/types.js";
import { toNumber } from "../runtime/convert.js";
import type { PatchTrace } from "./types.js";
import {
  getMatrixInfo,
  getStringValue,
  isAxesHandleArg,
  isNumericArg,
  isStringArg,
  resolveColor,
  toNumberArray,
} from "./parseArgsCommon.js";

// ── patch argument parser ────────────────────────────────────────────────

type PatchColor = [number, number, number] | "flat" | "interp" | "none";

/** Resolve a patch FaceColor/EdgeColor value: the keywords 'flat'/'interp'/
 *  'none' pass through, color names/RGB triplets resolve to an [r,g,b] triplet,
 *  and 'auto' (or anything unrecognized) returns undefined (use the default). */
function resolvePatchColor(v: RuntimeValue): PatchColor | undefined {
  if (isStringArg(v)) {
    const s = getStringValue(v as RuntimeValue)
      .toLowerCase()
      .replace(/^["']|["']$/g, "");
    if (s === "flat" || s === "interp" || s === "none") return s;
    if (s === "auto") return undefined;
    return resolveColor(s);
  }
  return resolveColor(v);
}

/** Convert a numeric color-data value into per-entry scalars or RGB triplets.
 *  An N×3 matrix is read as one RGB triplet per row; anything else flattens to
 *  scalars (column-major), each later mapped through the colormap. */
function toCData(v: RuntimeValue): (number | [number, number, number])[] {
  if (isRuntimeTensor(v) && v.shape.length >= 2 && v.shape[1] === 3) {
    const rows = v.shape[0];
    const out: [number, number, number][] = [];
    for (let i = 0; i < rows; i++) {
      out.push([v.data[i], v.data[rows + i], v.data[2 * rows + i]]);
    }
    return out;
  }
  return toNumberArray(v);
}

const PATCH_NAME_VALUE_KEYS = new Set([
  "facecolor",
  "edgecolor",
  "facealpha",
  "edgealpha",
  "linewidth",
  "linestyle",
  "marker",
  "markerfacecolor",
  "facevertexcdata",
  "cdata",
  "xdata",
  "ydata",
  "zdata",
  "faces",
  "vertices",
]);

/** Build vertices + faces from X/Y[/Z] coordinate matrices. A vector is one
 *  polygon; an m×n matrix is n polygons of m vertices each (column-major). */
function patchVerticesFromXYZ(
  xv: RuntimeValue,
  yv: RuntimeValue,
  zv: RuntimeValue | undefined
): { vertices: number[][]; faces: number[][]; rows: number; cols: number } {
  const xi = getMatrixInfo(xv);
  const yi = getMatrixInfo(yv);
  const zi = zv ? getMatrixInfo(zv) : undefined;
  // A vector of any orientation (row or column) is a single polygon with all
  // its elements as vertices; only a true matrix (both dims > 1) is split into
  // one polygon per column.
  let rows = xi.rows;
  let cols = xi.cols;
  if (rows === 1 || cols === 1) {
    rows = xi.data.length;
    cols = 1;
  }
  const vertices: number[][] = [];
  const faces: number[][] = [];
  for (let j = 0; j < cols; j++) {
    const face: number[] = [];
    for (let i = 0; i < rows; i++) {
      const k = j * rows + i;
      const vx = xi.data[k];
      const vy = yi.data[Math.min(k, yi.data.length - 1)];
      const v = zi
        ? [vx, vy, zi.data[Math.min(k, zi.data.length - 1)]]
        : [vx, vy];
      face.push(vertices.length);
      vertices.push(v);
    }
    faces.push(face);
  }
  return { vertices, faces, rows, cols };
}

/** Apply a color C (from `patch(X,Y,C)` / `patch(X,Y,Z,C)`) to the trace,
 *  choosing single-color / flat / interp per MATLAB's orientation rules. */
function applyPatchCData(
  trace: PatchTrace,
  c: RuntimeValue,
  nFaces: number,
  nVerts: number
): void {
  if (isStringArg(c)) {
    const col = resolvePatchColor(c);
    if (col) trace.faceColor = col;
    return;
  }
  const info = getMatrixInfo(c);
  const n = info.data.length;
  if (n === 1) {
    // Scalar: one flat color for all faces.
    trace.faceColor = "flat";
    trace.faceVertexCData = new Array(nFaces).fill(info.data[0]);
  } else if (info.rows === 1 && info.cols === 3) {
    // 1×3 row → a single RGB triplet for all faces.
    trace.faceColor = [info.data[0], info.data[1], info.data[2]];
  } else if (n === nVerts && nVerts !== nFaces) {
    // One value per vertex → interpolated face color.
    trace.faceColor = "interp";
    trace.faceVertexCData = toCData(c);
  } else {
    // Otherwise treat as one value per face → flat.
    trace.faceColor = "flat";
    trace.faceVertexCData = toCData(c);
  }
}

/** Apply patch Name-Value pairs starting at args[pos]. */
function applyPatchNameValues(
  trace: PatchTrace,
  args: RuntimeValue[],
  pos: number
): void {
  for (let i = pos; i + 1 < args.length; i += 2) {
    if (!isStringArg(args[i])) break;
    const key = getStringValue(args[i] as RuntimeValue).toLowerCase();
    const val = args[i + 1];
    applyPatchProp(trace, key, val);
  }
}

/** Apply a single patch property (lower-cased key) to the trace. Shared by the
 *  Name-Value, struct, and `set`/handle paths. */
export function applyPatchProp(
  trace: PatchTrace,
  key: string,
  val: RuntimeValue
): void {
  switch (key) {
    case "facecolor": {
      const c = resolvePatchColor(val);
      if (c) trace.faceColor = c;
      break;
    }
    case "edgecolor": {
      const c = resolvePatchColor(val);
      if (c) trace.edgeColor = c;
      break;
    }
    case "facealpha":
      trace.faceAlpha = toNumber(val);
      break;
    case "edgealpha":
      trace.edgeAlpha = toNumber(val);
      break;
    case "linewidth":
      trace.lineWidth = toNumber(val);
      break;
    case "linestyle":
      trace.lineStyle = getStringValue(val as RuntimeValue);
      break;
    case "marker":
      trace.marker = getStringValue(val as RuntimeValue);
      break;
    case "markerfacecolor": {
      const s = isStringArg(val)
        ? getStringValue(val as RuntimeValue).toLowerCase()
        : "";
      if (s === "flat" || s === "none") trace.markerFaceColor = s;
      else {
        const c = resolveColor(val);
        if (c) trace.markerFaceColor = c;
      }
      break;
    }
    case "facevertexcdata":
    case "cdata":
      trace.faceVertexCData = toCData(val);
      break;
  }
}

/** Build a patch trace from explicit Faces/Vertices. F may be a row vector
 *  (one face) or a matrix (one row per face); NaN entries pad ragged faces and
 *  are dropped. Face indices are 1-based in MATLAB → stored 0-based. */
function patchFromFacesVertices(
  fv: RuntimeValue,
  vv: RuntimeValue
): { vertices: number[][]; faces: number[][]; is3D: boolean } {
  const vInfo = getMatrixInfo(vv);
  const nv = vInfo.rows;
  const ncol = vInfo.cols;
  const is3D = ncol >= 3;
  const vertices: number[][] = [];
  for (let i = 0; i < nv; i++) {
    const x = vInfo.data[i];
    const y = vInfo.data[nv + i];
    vertices.push(is3D ? [x, y, vInfo.data[2 * nv + i]] : [x, y]);
  }
  const fInfo = getMatrixInfo(fv);
  const faces: number[][] = [];
  for (let r = 0; r < fInfo.rows; r++) {
    const face: number[] = [];
    for (let c = 0; c < fInfo.cols; c++) {
      const idx = fInfo.data[c * fInfo.rows + r];
      if (Number.isFinite(idx)) face.push(idx - 1);
    }
    if (face.length > 0) faces.push(face);
  }
  return { vertices, faces, is3D };
}

/** Build a patch trace from a struct S (patch(S)): field names are patch
 *  property names, field values the corresponding values. */
function parsePatchStruct(s: RuntimeStruct): PatchTrace {
  const get = (name: string): RuntimeValue | undefined => {
    for (const [k, v] of s.fields) {
      if (k.toLowerCase() === name) return v;
    }
    return undefined;
  };
  const trace: PatchTrace = { vertices: [], faces: [] };
  const faces = get("faces");
  const verts = get("vertices");
  const xd = get("xdata");
  const yd = get("ydata");
  if (faces && verts) {
    const built = patchFromFacesVertices(faces, verts);
    trace.vertices = built.vertices;
    trace.faces = built.faces;
    trace.is3D = built.is3D;
  } else if (xd && yd) {
    const zd = get("zdata");
    const built = patchVerticesFromXYZ(xd, yd, zd);
    trace.vertices = built.vertices;
    trace.faces = built.faces;
    trace.is3D = !!zd;
  }
  for (const [k, v] of s.fields) {
    const key = k.toLowerCase();
    if (
      key === "faces" ||
      key === "vertices" ||
      key === "xdata" ||
      key === "ydata" ||
      key === "zdata"
    ) {
      continue;
    }
    applyPatchProp(trace, key, v);
  }
  return trace;
}

/**
 * Parse patch() arguments into a canonical PatchTrace.
 *
 * Supported forms:
 *   patch(X, Y, C)                              — 2-D polygons, color C
 *   patch(X, Y, Z, C)                           — 3-D polygons
 *   patch('XData', X, 'YData', Y[, 'ZData', Z]) — coords as name-value
 *   patch('Faces', F, 'Vertices', V)            — faces/vertices model
 *   patch(S)                                    — struct of patch properties
 *   patch(___, Name, Value)                     — patch properties
 *   patch(ax, ___)                              — leading axes handle (ignored)
 */
export function parsePatchArgs(args: RuntimeValue[]): PatchTrace {
  let a = args;
  if (a.length > 0 && isAxesHandleArg(a[0])) a = a.slice(1);
  if (a.length === 0) throw new Error("patch requires input arguments");

  // patch(S)
  if (a.length === 1 && isRuntimeStruct(a[0])) {
    return finalizePatch(parsePatchStruct(a[0]));
  }

  const trace: PatchTrace = { vertices: [], faces: [] };

  if (isStringArg(a[0])) {
    // Name-value form: collect coords from XData/YData/ZData or Faces/Vertices.
    const opts: Record<string, RuntimeValue> = {};
    for (let i = 0; i + 1 < a.length; i += 2) {
      if (!isStringArg(a[i])) break;
      opts[getStringValue(a[i] as RuntimeValue).toLowerCase()] = a[i + 1];
    }
    if (opts.faces && opts.vertices) {
      const built = patchFromFacesVertices(opts.faces, opts.vertices);
      trace.vertices = built.vertices;
      trace.faces = built.faces;
      trace.is3D = built.is3D;
    } else if (opts.xdata && opts.ydata) {
      const built = patchVerticesFromXYZ(opts.xdata, opts.ydata, opts.zdata);
      trace.vertices = built.vertices;
      trace.faces = built.faces;
      trace.is3D = !!opts.zdata;
    }
    for (const [key, val] of Object.entries(opts)) {
      if (
        key === "faces" ||
        key === "vertices" ||
        key === "xdata" ||
        key === "ydata" ||
        key === "zdata"
      ) {
        continue;
      }
      applyPatchProp(trace, key, val);
    }
    return finalizePatch(trace);
  }

  // Positional X, Y [, Z], C form.
  let nc = 0;
  for (let i = 0; i < a.length; i++) {
    if (isNumericArg(a[i])) nc++;
    else break;
  }
  if (nc < 2) throw new Error("patch: expected X and Y coordinates");

  // Decide whether a Z coordinate is present.
  //  - ≥4 numeric → X,Y,Z,C
  //  - exactly 3 numeric followed by a color string → X,Y,Z,colorString
  //  - otherwise → X,Y[,C]
  let has3D = false;
  if (nc >= 4) has3D = true;
  else if (nc === 3 && a[3] !== undefined && isStringArg(a[3])) {
    const s = getStringValue(a[3] as RuntimeValue).toLowerCase();
    if (!PATCH_NAME_VALUE_KEYS.has(s) && resolvePatchColor(a[3])) has3D = true;
  }

  let pos: number;
  let cArg: RuntimeValue | undefined;
  const built = has3D
    ? patchVerticesFromXYZ(a[0], a[1], a[2])
    : patchVerticesFromXYZ(a[0], a[1], undefined);
  trace.vertices = built.vertices;
  trace.faces = built.faces;
  trace.is3D = has3D;
  if (has3D) {
    cArg = a[3];
    pos = 4;
  } else {
    // C may be a 3rd numeric/color arg, unless it's a Name-Value key string.
    if (
      a.length > 2 &&
      !(
        isStringArg(a[2]) &&
        PATCH_NAME_VALUE_KEYS.has(
          getStringValue(a[2] as RuntimeValue).toLowerCase()
        )
      )
    ) {
      cArg = a[2];
      pos = 3;
    } else {
      pos = 2;
    }
  }
  if (cArg !== undefined) {
    applyPatchCData(trace, cArg, built.faces.length, built.vertices.length);
  }
  applyPatchNameValues(trace, a, pos);
  return finalizePatch(trace);
}

/** Fill in default colors a patch needs to render. */
function finalizePatch(trace: PatchTrace): PatchTrace {
  if (trace.faceColor === undefined) {
    trace.faceColor = trace.faceVertexCData ? "flat" : [0, 0.447, 0.741];
  }
  if (trace.edgeColor === undefined) trace.edgeColor = [0, 0, 0];
  return trace;
}

/**
 * Parse fill() arguments into one patch trace per (X,Y,C) group. fill is the
 * 2-D coordinate form of patch repeated over groups, so it reuses the patch
 * vertex/color/property helpers — the only fill-specific logic is splitting
 * the argument list into X,Y,C triplets.
 *
 * Supported forms:
 *   fill(X, Y, C)
 *   fill(X1, Y1, C1, ..., Xn, Yn, Cn)
 *   fill(___, Name, Value)
 *   fill(ax, ___)
 */
export function parseFillArgs(args: RuntimeValue[]): PatchTrace[] {
  let a = args;
  if (a.length > 0 && isAxesHandleArg(a[0])) a = a.slice(1);

  const traces: PatchTrace[] = [];
  let pos = 0;
  // Consume X,Y,C triplets until we hit a Name-Value key (or non-numeric X).
  while (
    pos + 1 < a.length &&
    isNumericArg(a[pos]) &&
    isNumericArg(a[pos + 1])
  ) {
    const built = patchVerticesFromXYZ(a[pos], a[pos + 1], undefined);
    const trace: PatchTrace = {
      vertices: built.vertices,
      faces: built.faces,
      is3D: false,
    };
    const cArg = a[pos + 2];
    const cIsNameValue =
      cArg !== undefined &&
      isStringArg(cArg) &&
      PATCH_NAME_VALUE_KEYS.has(
        getStringValue(cArg as RuntimeValue).toLowerCase()
      );
    if (cArg !== undefined && !cIsNameValue) {
      applyPatchCData(trace, cArg, built.faces.length, built.vertices.length);
      pos += 3;
    } else {
      // No color given for this group — leave the default; Name-Value follows.
      pos += 2;
    }
    traces.push(trace);
  }
  // Remaining args are Name-Value pairs applied to every patch.
  for (const trace of traces) applyPatchNameValues(trace, a, pos);
  return traces.map(finalizePatch);
}

/** Build 0-based triangle faces from a connectivity matrix T (1-based vertex
 *  indices, one triangle per row). Non-finite entries are dropped so ragged
 *  rows are tolerated. */
function facesFromConnectivity(fv: RuntimeValue): number[][] {
  const fInfo = getMatrixInfo(fv);
  const faces: number[][] = [];
  for (let r = 0; r < fInfo.rows; r++) {
    const face: number[] = [];
    for (let c = 0; c < fInfo.cols; c++) {
      const idx = fInfo.data[c * fInfo.rows + r];
      if (Number.isFinite(idx)) face.push(idx - 1);
    }
    if (face.length > 0) faces.push(face);
  }
  return faces;
}

/** trimesh color defaults, per the MATLAB docs: light-gray faces, black
 *  edges. Explicit Name-Value colors set earlier are preserved. */
function finalizeTriMesh(trace: PatchTrace): PatchTrace {
  if (trace.faceColor === undefined) trace.faceColor = [0.85, 0.85, 0.85];
  if (trace.edgeColor === undefined) trace.edgeColor = [0, 0, 0];
  return trace;
}

/**
 * Parse trimesh()/trisurf() arguments into a PatchTrace (colors not yet
 * defaulted).
 *
 * Both draw a triangular mesh and return a patch object: the connectivity
 * matrix T supplies the faces (one triangle per row, 1-based vertex indices)
 * and the x/y[/z] vectors supply the vertices.
 *
 * Supported forms:
 *   trimesh(T, x, y)              — 2-D triangular mesh (trimesh only)
 *   trimesh(T, x, y, z)           — 3-D triangular mesh
 *   trimesh(T, x, y, z, c)        — c is per-vertex color data
 *   trimesh(TO)                   — triangulation object (struct exposing
 *                                   ConnectivityList + Points)
 *   trimesh(___, Name, Value)     — patch properties
 *   trimesh(ax, ___)              — leading axes handle (ignored)
 */
function parseTriArgs(args: RuntimeValue[], fname: string): PatchTrace {
  let a = args;
  if (a.length > 0 && isAxesHandleArg(a[0])) a = a.slice(1);
  if (a.length === 0) throw new Error(`${fname} requires input arguments`);

  // trimesh(TO): triangulation / delaunayTriangulation object, modeled as a
  // struct with ConnectivityList and Points fields.
  if (isRuntimeStruct(a[0])) {
    const s = a[0] as RuntimeStruct;
    const get = (name: string): RuntimeValue | undefined => {
      for (const [k, v] of s.fields) if (k.toLowerCase() === name) return v;
      return undefined;
    };
    const conn = get("connectivitylist");
    const pts = get("points");
    if (!conn || !pts) {
      throw new Error(
        `${fname}: triangulation object must have ConnectivityList and Points fields`
      );
    }
    const pInfo = getMatrixInfo(pts);
    const np = pInfo.rows;
    const is3D = pInfo.cols >= 3;
    const vertices: number[][] = [];
    for (let i = 0; i < np; i++) {
      const x = pInfo.data[i];
      const y = pInfo.data[np + i];
      vertices.push(is3D ? [x, y, pInfo.data[2 * np + i]] : [x, y]);
    }
    const trace: PatchTrace = {
      vertices,
      faces: facesFromConnectivity(conn),
      is3D,
    };
    applyPatchNameValues(trace, a, 1);
    return trace;
  }

  if (a.length < 3) throw new Error(`${fname}: expected T, x and y`);

  // Positional T, x, y [, z [, c]] followed by Name-Value pairs.
  let pos = 3;
  let zv: RuntimeValue | undefined;
  let cv: RuntimeValue | undefined;
  if (a.length > 3 && isNumericArg(a[3])) {
    zv = a[3];
    pos = 4;
    if (a.length > 4 && isNumericArg(a[4])) {
      cv = a[4];
      pos = 5;
    }
  }

  const xi = getMatrixInfo(a[1]);
  const yi = getMatrixInfo(a[2]);
  const zi = zv ? getMatrixInfo(zv) : undefined;
  const nv = xi.data.length;
  const vertices: number[][] = [];
  for (let i = 0; i < nv; i++) {
    const x = xi.data[i];
    const y = yi.data[Math.min(i, yi.data.length - 1)];
    if (zi) vertices.push([x, y, zi.data[Math.min(i, zi.data.length - 1)]]);
    else vertices.push([x, y]);
  }

  const trace: PatchTrace = {
    vertices,
    faces: facesFromConnectivity(a[0]),
    is3D: !!zi,
  };
  if (cv) trace.faceVertexCData = toCData(cv);
  applyPatchNameValues(trace, a, pos);
  return trace;
}

export function parseTriMeshArgs(args: RuntimeValue[]): PatchTrace {
  return finalizeTriMesh(parseTriArgs(args, "trimesh"));
}

/** Parse trisurf() arguments. Same forms as trimesh, but (matching MATLAB)
 *  the faces default to 'flat' coloring by the per-vertex z values (or c). */
export function parseTriSurfArgs(args: RuntimeValue[]): PatchTrace {
  const trace = parseTriArgs(args, "trisurf");
  if (!trace.is3D) throw new Error("trisurf: expected T, x, y and z");
  if (trace.faceVertexCData === undefined) {
    trace.faceVertexCData = trace.vertices.map(v => v[2]);
  }
  if (trace.faceColor === undefined) trace.faceColor = "flat";
  if (trace.edgeColor === undefined) trace.edgeColor = [0, 0, 0];
  return trace;
}
