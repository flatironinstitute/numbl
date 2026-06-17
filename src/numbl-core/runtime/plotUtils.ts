/**
 * Utilities for parsing plot() arguments at runtime.
 *
 * Handles: plot(Y), plot(X,Y), plot(X1,Y1,...,Xn,Yn),
 * plot(X1,Y1,LineSpec1,...), and Name-Value pairs like 'Color','r','LineWidth',2.
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  type RuntimeStruct,
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeStruct,
} from "./types.js";
import { toNumber, toString } from "./convert.js";

// Re-export trace types from the shared graphics types module.
export type {
  PlotTrace,
  Plot3Trace,
  PatchTrace,
  SurfTrace,
  ImagescTrace,
  PcolorTrace,
  ContourTrace,
  BarTrace,
  Bar3Trace,
  ErrorBarTrace,
  BoxTrace,
  PieTrace,
  HeatmapTrace,
  QuiverTrace,
  Quiver3Trace,
} from "../../graphics/types.js";

import type {
  PlotTrace,
  Plot3Trace,
  PatchTrace,
  SurfTrace,
  ImagescTrace,
  PcolorTrace,
  ContourTrace,
  BarTrace,
  Bar3Trace,
  ErrorBarTrace,
  BoxTrace,
  PieTrace,
  HeatmapTrace,
  QuiverTrace,
  Quiver3Trace,
} from "../../graphics/types.js";
import { allocFloat64Array } from "./alloc.js";

// ── Color mapping ───────────────────────────────────────────────────────

const COLOR_SHORT: Record<string, [number, number, number]> = {
  r: [1, 0, 0],
  g: [0, 1, 0],
  b: [0, 0, 1],
  c: [0, 1, 1],
  m: [1, 0, 1],
  y: [1, 1, 0],
  k: [0, 0, 0],
  w: [1, 1, 1],
};

const COLOR_NAMES: Record<string, [number, number, number]> = {
  red: [1, 0, 0],
  green: [0, 1, 0],
  blue: [0, 0, 1],
  cyan: [0, 1, 1],
  magenta: [1, 0, 1],
  yellow: [1, 1, 0],
  black: [0, 0, 0],
  white: [1, 1, 1],
};

export function resolveColor(
  v: RuntimeValue | string
): [number, number, number] | undefined {
  if (typeof v === "string") {
    const lower = v.toLowerCase();
    if (COLOR_SHORT[lower]) return COLOR_SHORT[lower];
    if (COLOR_NAMES[lower]) return COLOR_NAMES[lower];
    return undefined;
  }
  if (isRuntimeString(v) || isRuntimeChar(v)) {
    return resolveColor(isRuntimeString(v) ? v : v.value);
  }
  if (isRuntimeTensor(v) && v.data.length === 3) {
    return [v.data[0], v.data[1], v.data[2]];
  }
  return undefined;
}

// ── LineSpec parser ─────────────────────────────────────────────────────

const COLOR_CHARS = new Set(["r", "g", "b", "c", "m", "y", "k", "w"]);
const MARKER_CHARS = new Set([
  "o",
  "+",
  "*",
  ".",
  "x",
  "_",
  "|",
  "s",
  "d",
  "^",
  "v",
  "<",
  ">",
  "p",
  "h",
]);

export interface ParsedLineSpec {
  color?: string;
  lineStyle?: string;
  marker?: string;
}

function parseLineSpec(s: string): ParsedLineSpec | null {
  const result: ParsedLineSpec = {};
  let pos = 0;

  while (pos < s.length) {
    // Try two-char line styles first
    if (pos + 1 < s.length) {
      const two = s.slice(pos, pos + 2);
      if (two === "--" || two === "-.") {
        if (result.lineStyle !== undefined) return null;
        result.lineStyle = two;
        pos += 2;
        continue;
      }
    }

    const ch = s[pos];

    // Single-char line style: '-' or ':'
    if (ch === "-" || ch === ":") {
      if (result.lineStyle !== undefined) return null;
      result.lineStyle = ch;
      pos++;
      continue;
    }

    // Color
    if (COLOR_CHARS.has(ch)) {
      if (result.color !== undefined) return null;
      result.color = ch;
      pos++;
      continue;
    }

    // Marker
    if (MARKER_CHARS.has(ch)) {
      // '.' could be ambiguous but since we checked '-.' above, standalone '.' is a marker
      if (result.marker !== undefined) return null;
      result.marker = ch;
      pos++;
      continue;
    }

    // Unrecognized character → not a LineSpec
    return null;
  }

  // Must have matched at least one thing
  if (
    result.color === undefined &&
    result.lineStyle === undefined &&
    result.marker === undefined
  )
    return null;

  return result;
}

// ── Name-Value key detection ────────────────────────────────────────────

const NAME_VALUE_KEYS = new Set([
  "color",
  "linestyle",
  "linewidth",
  "marker",
  "markersize",
  "markeredgecolor",
  "markerfacecolor",
  "markerindices",
]);

function isNameValueKey(v: RuntimeValue): string | null {
  if (!isRuntimeString(v) && !isRuntimeChar(v)) return null;
  const lower = isRuntimeString(v)
    ? v.toLocaleLowerCase()
    : v.value.toLowerCase();
  if (NAME_VALUE_KEYS.has(lower)) return lower;
  return null;
}

// ── Numeric data helpers ────────────────────────────────────────────────

function isNumericArg(v: unknown): boolean {
  if (typeof v === "number") return true;
  if (typeof v === "boolean") return true;
  if (v && typeof v === "object" && "kind" in v) {
    const mv = v as RuntimeValue;
    return isRuntimeNumber(mv) || isRuntimeTensor(mv) || isRuntimeLogical(mv);
  }
  return false;
}

function isStringArg(v: unknown): boolean {
  if (typeof v === "string") return true;
  if (v && typeof v === "object" && "kind" in v) {
    const mv = v as RuntimeValue;
    return isRuntimeString(mv) || isRuntimeChar(mv);
  }
  return false;
}

function getStringValue(v: RuntimeValue): string {
  if (isRuntimeString(v)) return v;
  if (isRuntimeChar(v)) return v.value;
  return toString(v);
}

/** Convert runtime value to a flat number array */
function toNumberArray(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  return [0];
}

/** Extract column j from a column-major tensor with shape [m, n] */
function tensorColumn(tensor: RuntimeTensor, col: number): number[] {
  const m = tensor.shape[0];
  const offset = col * m;
  const result = new Array(m);
  for (let i = 0; i < m; i++) {
    result[i] = tensor.data[offset + i];
  }
  return result;
}

/**
 * Get the number of columns of a numeric value for matrix splitting.
 * Only returns > 1 for true matrices (nRows > 1 && nCols > 1).
 * Row vectors (1×N) and column vectors (N×1) return 1.
 */
function numColumns(v: RuntimeValue): number {
  if (
    isRuntimeTensor(v) &&
    v.shape.length >= 2 &&
    v.shape[0] > 1 &&
    v.shape[1] > 1
  )
    return v.shape[1];
  return 1;
}

/** Generate 1-based index array [1, 2, ..., n] */
function oneBasedIndices(n: number): number[] {
  const arr = new Array(n);
  for (let i = 0; i < n; i++) arr[i] = i + 1;
  return arr;
}

/**
 * Expand a numeric value into column-wise traces.
 * For a vector (row or column), returns one array.
 * For an m×n matrix (m>1, n>1), returns n arrays (one per column).
 */
function expandColumns(v: RuntimeValue): number[][] {
  if (
    isRuntimeTensor(v) &&
    v.shape.length >= 2 &&
    v.shape[0] > 1 &&
    v.shape[1] > 1
  ) {
    const cols: number[][] = [];
    for (let j = 0; j < v.shape[1]; j++) {
      cols.push(tensorColumn(v, j));
    }
    return cols;
  }
  // Scalar or vector — single column
  return [toNumberArray(v)];
}

// ── Main argument parser ────────────────────────────────────────────────

export function parsePlotArgs(args: RuntimeValue[]): PlotTrace[] {
  const traces: PlotTrace[] = [];
  let pos = 0;

  // First pass: collect X-Y pairs with optional LineSpec
  while (pos < args.length) {
    // Check if we've hit Name-Value pairs
    if (isStringArg(args[pos]) && isNameValueKey(args[pos])) {
      break;
    }

    // Must have at least one numeric arg
    if (!isNumericArg(args[pos])) {
      // Unexpected string that's not a Name-Value key — try as LineSpec for previous traces
      if (isStringArg(args[pos]) && traces.length > 0) {
        const spec = parseLineSpec(getStringValue(args[pos]));
        if (spec) {
          // This shouldn't normally happen here, but handle gracefully
          pos++;
          continue;
        }
      }
      break;
    }

    const first = args[pos];
    pos++;

    // Check if next arg is numeric (making this an X-Y pair)
    // or if next arg is a string (making 'first' a Y-only value)
    let second: RuntimeValue | undefined = undefined;

    if (pos < args.length && isNumericArg(args[pos])) {
      // Could be X,Y pair OR this 'first' might be Y-only followed by another Y
      // Peek ahead: if after consuming second we see a LineSpec or Name-Value or end,
      // then first=X, second=Y. If we see another numeric, it's also X-Y pair pattern.
      second = args[pos];
      pos++;
    }

    let lineSpec: ParsedLineSpec | undefined;

    // Check for LineSpec after the pair/single
    if (
      pos < args.length &&
      isStringArg(args[pos]) &&
      !isNameValueKey(args[pos])
    ) {
      const spec = parseLineSpec(getStringValue(args[pos]));
      if (spec) {
        lineSpec = spec;
        pos++;
      }
    }

    // Build traces from this group
    if (second !== undefined) {
      // X-Y pair
      buildTracesFromXY(first, second, lineSpec, traces);
    } else {
      // Y-only
      buildTracesFromY(first, lineSpec, traces);
    }
  }

  // Second pass: apply Name-Value pairs to all traces
  while (pos < args.length) {
    const key = isNameValueKey(args[pos]);
    if (!key) break;
    pos++;
    if (pos >= args.length) break;
    const value = args[pos];
    pos++;
    applyNameValue(traces, key, value);
  }

  return traces;
}

function buildTracesFromY(
  yVal: RuntimeValue,
  lineSpec: ParsedLineSpec | undefined,
  traces: PlotTrace[]
): void {
  const yCols = expandColumns(yVal);

  for (const yData of yCols) {
    const xData = oneBasedIndices(yData.length);
    traces.push(makeTrace(xData, yData, lineSpec));
  }
}

function buildTracesFromXY(
  xVal: RuntimeValue,
  yVal: RuntimeValue,
  lineSpec: ParsedLineSpec | undefined,
  traces: PlotTrace[]
): void {
  const xCols = numColumns(xVal);
  const yCols = numColumns(yVal);

  if (xCols === 1 && yCols === 1) {
    // Simple vector-vector pair
    traces.push(makeTrace(toNumberArray(xVal), toNumberArray(yVal), lineSpec));
  } else if (xCols === 1 && yCols > 1) {
    // Vector X, matrix Y → shared X, one trace per Y column
    const xData = toNumberArray(xVal);
    const yColumns = expandColumns(yVal);
    for (const yData of yColumns) {
      traces.push(makeTrace(xData, yData, lineSpec));
    }
  } else if (xCols > 1 && yCols > 1 && xCols === yCols) {
    // Matrix X, matrix Y (same size) → pair columns
    const xColumns = expandColumns(xVal);
    const yColumns = expandColumns(yVal);
    for (let i = 0; i < xCols; i++) {
      traces.push(makeTrace(xColumns[i], yColumns[i], lineSpec));
    }
  } else {
    // Fallback: flatten both
    traces.push(makeTrace(toNumberArray(xVal), toNumberArray(yVal), lineSpec));
  }
}

function makeTrace(
  x: number[],
  y: number[],
  lineSpec: ParsedLineSpec | undefined
): PlotTrace {
  const trace: PlotTrace = { x, y };
  if (lineSpec) {
    if (lineSpec.color) {
      trace.color = COLOR_SHORT[lineSpec.color];
    }
    if (lineSpec.lineStyle) {
      trace.lineStyle = lineSpec.lineStyle;
    }
    if (lineSpec.marker) {
      trace.marker = lineSpec.marker;
      // If marker specified but no line style, show markers only
      if (!lineSpec.lineStyle) {
        trace.lineStyle = "none";
      }
    }
  }
  return trace;
}

// ── plot3 argument parser ────────────────────────────────────────────────

/**
 * Parse plot3() arguments.
 *
 * Supported forms:
 *   plot3(X, Y, Z)
 *   plot3(X, Y, Z, LineSpec)
 *   plot3(X1,Y1,Z1,...,Xn,Yn,Zn)
 *   plot3(X1,Y1,Z1,LineSpec1,...,Xn,Yn,Zn,LineSpecn)
 *   plot3(..., Name, Value)
 */
export function parsePlot3Args(args: RuntimeValue[]): Plot3Trace[] {
  const traces: Plot3Trace[] = [];
  let pos = 0;

  // First pass: collect X-Y-Z triplets with optional LineSpec
  while (pos < args.length) {
    // Check if we've hit Name-Value pairs
    if (isStringArg(args[pos]) && isNameValueKey(args[pos])) {
      break;
    }

    // Must have at least three numeric args for X, Y, Z
    if (!isNumericArg(args[pos])) break;
    if (pos + 1 >= args.length || !isNumericArg(args[pos + 1])) break;
    if (pos + 2 >= args.length || !isNumericArg(args[pos + 2])) break;

    const xVal = args[pos];
    const yVal = args[pos + 1];
    const zVal = args[pos + 2];
    pos += 3;

    let lineSpec: ParsedLineSpec | undefined;

    // Check for LineSpec after the triplet
    if (
      pos < args.length &&
      isStringArg(args[pos]) &&
      !isNameValueKey(args[pos])
    ) {
      const spec = parseLineSpec(getStringValue(args[pos]));
      if (spec) {
        lineSpec = spec;
        pos++;
      }
    }

    // Build traces from this group (handle matrix expansion)
    buildPlot3Traces(xVal, yVal, zVal, lineSpec, traces);
  }

  // Second pass: apply Name-Value pairs to all traces
  while (pos < args.length) {
    const key = isNameValueKey(args[pos]);
    if (!key) break;
    pos++;
    if (pos >= args.length) break;
    const value = args[pos];
    pos++;
    applyPlot3NameValue(traces, key, value);
  }

  return traces;
}

function buildPlot3Traces(
  xVal: RuntimeValue,
  yVal: RuntimeValue,
  zVal: RuntimeValue,
  lineSpec: ParsedLineSpec | undefined,
  traces: Plot3Trace[]
): void {
  const xCols = numColumns(xVal);
  const yCols = numColumns(yVal);
  const zCols = numColumns(zVal);
  const maxCols = Math.max(xCols, yCols, zCols);

  if (maxCols === 1) {
    // Simple vector-vector-vector
    traces.push(
      makePlot3Trace(
        toNumberArray(xVal),
        toNumberArray(yVal),
        toNumberArray(zVal),
        lineSpec
      )
    );
  } else {
    // At least one is a matrix — expand columns
    const xColumns = expandColumns(xVal);
    const yColumns = expandColumns(yVal);
    const zColumns = expandColumns(zVal);
    for (let i = 0; i < maxCols; i++) {
      traces.push(
        makePlot3Trace(
          xColumns[Math.min(i, xColumns.length - 1)],
          yColumns[Math.min(i, yColumns.length - 1)],
          zColumns[Math.min(i, zColumns.length - 1)],
          lineSpec
        )
      );
    }
  }
}

function makePlot3Trace(
  x: number[],
  y: number[],
  z: number[],
  lineSpec: ParsedLineSpec | undefined
): Plot3Trace {
  const trace: Plot3Trace = { x, y, z };
  if (lineSpec) {
    if (lineSpec.color) {
      trace.color = COLOR_SHORT[lineSpec.color];
    }
    if (lineSpec.lineStyle) {
      trace.lineStyle = lineSpec.lineStyle;
    }
    if (lineSpec.marker) {
      trace.marker = lineSpec.marker;
      if (!lineSpec.lineStyle) {
        trace.lineStyle = "none";
      }
    }
  }
  return trace;
}

function applyPlot3NameValue(
  traces: Plot3Trace[],
  key: string,
  value: RuntimeValue
): void {
  switch (key) {
    case "color": {
      const c = resolveColor(value);
      if (c) for (const t of traces) t.color = c;
      break;
    }
    case "linestyle": {
      const s = getStringValue(value);
      for (const t of traces) t.lineStyle = s;
      break;
    }
    case "linewidth": {
      const n = typeof value === "number" ? value : toNumber(value);
      for (const t of traces) t.lineWidth = n;
      break;
    }
    case "marker": {
      const s = getStringValue(value);
      for (const t of traces) t.marker = s === "none" ? undefined : s;
      break;
    }
    case "markersize": {
      const n = typeof value === "number" ? value : toNumber(value);
      for (const t of traces) t.markerSize = n;
      break;
    }
    case "markeredgecolor": {
      const c = resolveColor(value);
      if (c) for (const t of traces) t.markerEdgeColor = c;
      break;
    }
    case "markerfacecolor": {
      const c = resolveColor(value);
      if (c) for (const t of traces) t.markerFaceColor = c;
      break;
    }
    case "markerindices": {
      const arr = toNumberArray(value);
      for (const t of traces) t.markerIndices = arr;
      break;
    }
  }
}

// ── line argument parser ─────────────────────────────────────────────────

/** Result of parsing line() arguments: a set of 2-D or 3-D traces. */
export type ParsedLine =
  | { kind: "2d"; traces: PlotTrace[] }
  | { kind: "3d"; traces: Plot3Trace[] };

/** A leading argument that should be skipped as an axes handle. numbl has no
 *  real axes handles, so `line(ax, ___)` is accepted and the handle ignored. */
function isAxesHandleArg(v: RuntimeValue): boolean {
  return (
    !!v &&
    typeof v === "object" &&
    "kind" in v &&
    ((v as RuntimeValue & { kind: string }).kind === "graphics_handle" ||
      (v as RuntimeValue & { kind: string }).kind === "dummy_handle")
  );
}

/**
 * Parse line() arguments.
 *
 * Supported forms:
 *   line                                  — line from (0,0) to (1,1)
 *   line(x, y)                            — 2-D line(s)
 *   line(x, y, z)                         — 3-D line(s)
 *   line(___, Name, Value)               — Color/LineStyle/LineWidth/Marker/...
 *   line('XData', x, 'YData', y, ...)    — low-level form (black line)
 *   line(ax, ___)                        — leading axes handle (ignored)
 *
 * Like plot/plot3, vector inputs make a single line and matrix inputs make one
 * line per column.
 */
export function parseLineArgs(args: RuntimeValue[]): ParsedLine {
  // Drop a leading axes handle: line(ax, ___).
  let a = args;
  if (a.length > 0 && isAxesHandleArg(a[0])) a = a.slice(1);

  // line  →  a line from (0,0) to (1,1) with default properties.
  if (a.length === 0) {
    return { kind: "2d", traces: [{ x: [0, 1], y: [0, 1] }] };
  }

  // Low-level form: data supplied via Name-Value pairs. Resulting line is
  // black (per MATLAB), unless an explicit Color is also given.
  if (isStringArg(a[0])) {
    return parseLowLevelLine(a);
  }

  // High-level form: x, y[, z], then optional Name-Value pairs. Three or more
  // leading numeric arguments means a 3-D line.
  let numericCount = 0;
  for (let i = 0; i < a.length; i++) {
    if (isNumericArg(a[i])) numericCount++;
    else break;
  }
  if (numericCount >= 3) {
    return { kind: "3d", traces: parsePlot3Args(a) };
  }
  return { kind: "2d", traces: parsePlotArgs(a) };
}

/** Parse the low-level `line('XData',x,'YData',y,...)` form. */
function parseLowLevelLine(args: RuntimeValue[]): ParsedLine {
  let xData: number[] | undefined;
  let yData: number[] | undefined;
  let zData: number[] | undefined;
  const styleArgs: RuntimeValue[] = [];

  let i = 0;
  while (i + 1 < args.length) {
    if (!isStringArg(args[i])) break;
    const key = getStringValue(args[i] as RuntimeValue).toLowerCase();
    const val = args[i + 1];
    if (key === "xdata") {
      xData = toNumberArray(val);
    } else if (key === "ydata") {
      yData = toNumberArray(val);
    } else if (key === "zdata") {
      zData = toNumberArray(val);
    } else {
      // Styling pair (Color, LineStyle, LineWidth, Marker, ...). Defer to the
      // shared Name-Value application so color names/RGB triplets resolve.
      styleArgs.push(args[i], val);
    }
    i += 2;
  }

  // Omitted coordinates default to [0 1] (MATLAB's default line endpoints).
  if (!xData) xData = [0, 1];
  if (!yData) yData = [0, 1];

  if (zData) {
    const trace: Plot3Trace = {
      x: xData,
      y: yData,
      z: zData,
      color: [0, 0, 0],
    };
    for (let k = 0; k + 1 < styleArgs.length; k += 2) {
      const nv = isNameValueKey(styleArgs[k]);
      if (nv) applyPlot3NameValue([trace], nv, styleArgs[k + 1]);
    }
    return { kind: "3d", traces: [trace] };
  }
  const trace: PlotTrace = { x: xData, y: yData, color: [0, 0, 0] };
  for (let k = 0; k + 1 < styleArgs.length; k += 2) {
    const nv = isNameValueKey(styleArgs[k]);
    if (nv) applyNameValue([trace], nv, styleArgs[k + 1]);
  }
  return { kind: "2d", traces: [trace] };
}

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
 * Parse trimesh() arguments into a canonical PatchTrace.
 *
 * trimesh draws a triangular mesh and (per the MATLAB docs) returns a patch
 * object, so it maps onto the PatchTrace model: the connectivity matrix T
 * supplies the faces (one triangle per row, 1-based vertex indices) and the
 * x/y[/z] vectors supply the vertices.
 *
 * Supported forms:
 *   trimesh(T, x, y)              — 2-D triangular mesh
 *   trimesh(T, x, y, z)           — 3-D triangular mesh
 *   trimesh(T, x, y, z, c)        — c is per-vertex color data
 *   trimesh(TO)                   — triangulation object (struct exposing
 *                                   ConnectivityList + Points)
 *   trimesh(___, Name, Value)     — patch properties
 *   trimesh(ax, ___)              — leading axes handle (ignored)
 */
export function parseTriMeshArgs(args: RuntimeValue[]): PatchTrace {
  let a = args;
  if (a.length > 0 && isAxesHandleArg(a[0])) a = a.slice(1);
  if (a.length === 0) throw new Error("trimesh requires input arguments");

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
        "trimesh: triangulation object must have ConnectivityList and Points fields"
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
    return finalizeTriMesh(trace);
  }

  if (a.length < 3) throw new Error("trimesh: expected T, x and y");

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
  return finalizeTriMesh(trace);
}

// ── Surf Name-Value key detection ────────────────────────────────────────

const SURF_NAME_VALUE_KEYS = new Set(["edgecolor", "facecolor", "facealpha"]);

function isSurfNameValueKey(v: RuntimeValue): string | null {
  if (!isRuntimeString(v) && !isRuntimeChar(v)) return null;
  const lower = isRuntimeString(v)
    ? v.toLocaleLowerCase()
    : v.value.toLowerCase();
  if (SURF_NAME_VALUE_KEYS.has(lower)) return lower;
  return null;
}

// ── Surf argument parser ────────────────────────────────────────────────

/**
 * Parse surf() arguments.
 *
 * Supported forms:
 *   surf(Z)           — Z is m×n, X = 1:n, Y = 1:m
 *   surf(Z, C)        — Z is m×n, C is m×n color data
 *   surf(X, Y, Z)     — X, Y, Z are m×n matrices (or X is 1×n / Y is m×1)
 *   surf(X, Y, Z, C)  — with explicit color data
 *   surf(..., Name, Value) — name-value pairs
 */
export function parseSurfArgs(args: RuntimeValue[]): SurfTrace {
  let pos = 0;

  // Skip axes handle argument (not supported, but tolerate it)
  // For simplicity, we assume all numeric args are data args

  let xData: number[] | undefined;
  let yData: number[] | undefined;
  let zData: number[];
  let rows: number;
  let cols: number;
  let cData: number[] | undefined;

  // Count leading numeric args
  let numericCount = 0;
  for (let i = pos; i < args.length; i++) {
    if (isNumericArg(args[i])) numericCount++;
    else break;
  }

  if (numericCount === 1) {
    // surf(Z)
    const z = args[pos++];
    const info = getMatrixInfo(z);
    rows = info.rows;
    cols = info.cols;
    zData = info.data;
    // Generate meshgrid: X = 1:cols, Y = 1:rows
    const gen = generateMeshgrid(rows, cols);
    xData = gen.x;
    yData = gen.y;
  } else if (numericCount === 2) {
    // surf(Z, C)
    const z = args[pos++];
    const c = args[pos++];
    const info = getMatrixInfo(z);
    rows = info.rows;
    cols = info.cols;
    zData = info.data;
    cData = toNumberArray(c);
    const gen = generateMeshgrid(rows, cols);
    xData = gen.x;
    yData = gen.y;
  } else if (numericCount === 3) {
    // surf(X, Y, Z)
    const x = args[pos++];
    const y = args[pos++];
    const z = args[pos++];
    const zInfo = getMatrixInfo(z);
    rows = zInfo.rows;
    cols = zInfo.cols;
    zData = zInfo.data;
    const expanded = expandXY(x, y, rows, cols);
    xData = expanded.x;
    yData = expanded.y;
  } else if (numericCount >= 4) {
    // surf(X, Y, Z, C)
    const x = args[pos++];
    const y = args[pos++];
    const z = args[pos++];
    const c = args[pos++];
    const zInfo = getMatrixInfo(z);
    rows = zInfo.rows;
    cols = zInfo.cols;
    zData = zInfo.data;
    cData = toNumberArray(c);
    const expanded = expandXY(x, y, rows, cols);
    xData = expanded.x;
    yData = expanded.y;
  } else {
    throw new Error("surf requires at least one input argument");
  }

  const trace: SurfTrace = {
    x: xData!,
    y: yData!,
    z: zData,
    rows,
    cols,
  };
  if (cData) trace.c = cData;

  // Parse name-value pairs
  while (pos < args.length) {
    const key = isSurfNameValueKey(args[pos]);
    if (!key) break;
    pos++;
    if (pos >= args.length) break;
    const value = args[pos++];
    applySurfNameValue(trace, key, value);
  }

  return trace;
}

function getMatrixInfo(v: RuntimeValue): {
  data: number[];
  rows: number;
  cols: number;
} {
  if (isRuntimeTensor(v) && v.shape.length >= 2) {
    return {
      data: Array.from(v.data),
      rows: v.shape[0],
      cols: v.shape[1],
    };
  }
  if (isRuntimeNumber(v)) {
    return { data: [v], rows: 1, cols: 1 };
  }
  // Fallback: treat as column vector
  const arr = toNumberArray(v);
  return { data: arr, rows: arr.length, cols: 1 };
}

/** Generate meshgrid X and Y for a rows×cols grid: X = 1:cols, Y = 1:rows (column-major) */
function generateMeshgrid(
  rows: number,
  cols: number
): { x: number[]; y: number[] } {
  const n = rows * cols;
  const x = new Array(n);
  const y = new Array(n);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      const idx = j * rows + i; // column-major
      x[idx] = j + 1;
      y[idx] = i + 1;
    }
  }
  return { x, y };
}

/**
 * Expand X and Y to match the grid dimensions of Z.
 * Handles:
 *   - X and Y as full m×n matrices (pass through)
 *   - X as 1×n row vector, Y as m×1 column vector (meshgrid expansion)
 *   - X as n-element vector, Y as m-element vector (meshgrid expansion)
 */
function expandXY(
  xVal: RuntimeValue,
  yVal: RuntimeValue,
  rows: number,
  cols: number
): { x: number[]; y: number[] } {
  const xArr = toNumberArray(xVal);
  const yArr = toNumberArray(yVal);
  const n = rows * cols;

  // If X and Y are already full matrices, use them directly
  if (xArr.length === n && yArr.length === n) {
    return { x: xArr, y: yArr };
  }

  // Meshgrid expansion: X has cols elements, Y has rows elements
  const xVec = xArr.length === cols ? xArr : xArr.slice(0, cols);
  const yVec = yArr.length === rows ? yArr : yArr.slice(0, rows);
  const x = new Array(n);
  const y = new Array(n);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      const idx = j * rows + i; // column-major
      x[idx] = xVec[j];
      y[idx] = yVec[i];
    }
  }
  return { x, y };
}

// ── Pcolor argument parser ──────────────────────────────────────────────

const PCOLOR_NAME_VALUE_KEYS = new Set(["edgecolor", "facealpha"]);

function isPcolorNameValueKey(v: RuntimeValue): string | null {
  if (!isRuntimeString(v) && !isRuntimeChar(v)) return null;
  const lower = isRuntimeString(v)
    ? v.toLocaleLowerCase()
    : v.value.toLowerCase();
  if (PCOLOR_NAME_VALUE_KEYS.has(lower)) return lower;
  return null;
}

function applyPcolorNameValue(
  trace: PcolorTrace,
  key: string,
  value: RuntimeValue
): void {
  switch (key) {
    case "edgecolor": {
      const s = getStringValueIfString(value);
      if (s !== undefined && s.toLowerCase() === "none") {
        trace.edgeColor = "none";
        break;
      }
      const c = resolveColor(value);
      if (c) trace.edgeColor = c;
      break;
    }
    case "facealpha": {
      const n = typeof value === "number" ? value : toNumber(value);
      trace.faceAlpha = n;
      break;
    }
  }
}

/**
 * Parse pcolor() arguments.
 *
 * Supported forms:
 *   pcolor(C)         — C is m×n, X = 1:n, Y = 1:m
 *   pcolor(X, Y, C)   — X, Y, C are m×n matrices (or X is 1×n / Y is m×1)
 *   pcolor(..., Name, Value) — name-value pairs (EdgeColor, FaceAlpha)
 */
export function parsePcolorArgs(args: RuntimeValue[]): PcolorTrace {
  let pos = 0;

  let xData: number[];
  let yData: number[];
  let cData: number[];
  let rows: number;
  let cols: number;

  // Count leading numeric args
  let numericCount = 0;
  for (let i = pos; i < args.length; i++) {
    if (isNumericArg(args[i])) numericCount++;
    else break;
  }

  if (numericCount === 1) {
    // pcolor(C)
    const c = args[pos++];
    const info = getMatrixInfo(c);
    rows = info.rows;
    cols = info.cols;
    cData = info.data;
    const gen = generateMeshgrid(rows, cols);
    xData = gen.x;
    yData = gen.y;
  } else if (numericCount >= 3) {
    // pcolor(X, Y, C)
    const x = args[pos++];
    const y = args[pos++];
    const c = args[pos++];
    const cInfo = getMatrixInfo(c);
    rows = cInfo.rows;
    cols = cInfo.cols;
    cData = cInfo.data;
    const expanded = expandXY(x, y, rows, cols);
    xData = expanded.x;
    yData = expanded.y;
  } else {
    throw new Error("pcolor requires 1 or 3 numeric input arguments");
  }

  const trace: PcolorTrace = {
    x: xData,
    y: yData,
    c: cData,
    rows,
    cols,
  };

  // Parse name-value pairs
  while (pos < args.length) {
    const key = isPcolorNameValueKey(args[pos]);
    if (!key) break;
    pos++;
    if (pos >= args.length) break;
    const value = args[pos++];
    applyPcolorNameValue(trace, key, value);
  }

  return trace;
}

function applySurfNameValue(
  trace: SurfTrace,
  key: string,
  value: RuntimeValue
): void {
  switch (key) {
    case "edgecolor": {
      const s = getStringValueIfString(value);
      if (s !== undefined) {
        const lower = s.toLowerCase();
        if (lower === "none" || lower === "flat" || lower === "interp") {
          trace.edgeColor = lower;
          break;
        }
      }
      const c = resolveColor(value);
      if (c) trace.edgeColor = c;
      break;
    }
    case "facecolor": {
      const s = getStringValueIfString(value);
      if (s !== undefined) {
        const lower = s.toLowerCase();
        if (
          lower === "flat" ||
          lower === "interp" ||
          lower === "none" ||
          lower === "texturemap"
        ) {
          trace.faceColor = lower;
          break;
        }
      }
      const c = resolveColor(value);
      if (c) trace.faceColor = c;
      break;
    }
    case "facealpha": {
      const n = typeof value === "number" ? value : toNumber(value);
      trace.faceAlpha = n;
      break;
    }
  }
}

/** Get string value if the value is a string/char, otherwise undefined */
function getStringValueIfString(v: RuntimeValue): string | undefined {
  if (isRuntimeString(v)) return v;
  if (isRuntimeChar(v)) return v.value;
  return undefined;
}

// ── Scatter argument parser ──────────────────────────────────────────────

/**
 * Parse scatter() arguments.
 *
 * Supported forms:
 *   scatter(X, Y)
 *   scatter(X, Y, sz)
 *   scatter(X, Y, sz, c)
 *   scatter(X, Y, ..., 'filled')
 *   scatter(X, Y, ..., mkr)
 *   scatter(X, Y, ..., Name, Value)
 */
export function parseScatterArgs(args: RuntimeValue[]): PlotTrace[] {
  if (args.length < 2) throw new Error("scatter requires at least 2 arguments");

  const xData = toNumberArray(args[0]);
  const yData = toNumberArray(args[1]);

  const trace: PlotTrace = {
    x: xData,
    y: yData,
    marker: "o",
    lineStyle: "none",
  };

  let pos = 2;
  let filled = false;

  // Check for size argument (numeric, 3rd positional arg)
  if (pos < args.length && isNumericArg(args[pos])) {
    const sz = args[pos];
    // In MATLAB, sz is area in points^2. We convert to markerSize (diameter-like).
    if (isRuntimeNumber(sz)) {
      trace.markerSize = Math.sqrt(sz as number) * 0.5;
    } else if (isRuntimeTensor(sz)) {
      // If scalar size, use it; if vector, just use the first value for now
      const sArr = toNumberArray(sz);
      if (sArr.length === 1 || new Set(sArr).size === 1) {
        trace.markerSize = Math.sqrt(sArr[0]) * 0.5;
      }
    }
    pos++;
  }

  // Check for color argument (4th positional: string color spec or RGB matrix)
  if (pos < args.length && !isNameValueKey(args[pos])) {
    if (isStringArg(args[pos])) {
      const s = getStringValue(args[pos]);
      if (s === "filled") {
        filled = true;
        pos++;
      } else {
        // Could be a color short name or marker spec
        const c = resolveColor(s);
        if (c) {
          trace.color = c;
          pos++;
        } else {
          const spec = parseLineSpec(s);
          if (spec?.marker) {
            trace.marker = spec.marker;
            if (spec.color) trace.color = COLOR_SHORT[spec.color];
            pos++;
          }
        }
      }
    } else if (isNumericArg(args[pos])) {
      // RGB color matrix or color vector
      const c = resolveColor(args[pos]);
      if (c) {
        trace.color = c;
        pos++;
      }
    }
  }

  // Check for 'filled' or marker string in remaining positional args
  while (
    pos < args.length &&
    isStringArg(args[pos]) &&
    !isNameValueKey(args[pos])
  ) {
    const s = getStringValue(args[pos]);
    if (s === "filled") {
      filled = true;
      pos++;
    } else {
      const spec = parseLineSpec(s);
      if (spec?.marker) {
        trace.marker = spec.marker;
        if (spec.color) trace.color = COLOR_SHORT[spec.color];
        pos++;
      } else {
        break;
      }
    }
  }

  // Apply 'filled': set markerFaceColor to match edge color
  if (filled) {
    // Will be resolved at render time if color not set yet
    trace.markerFaceColor = trace.color || [0, 0, 1]; // default blue
  }

  // Name-Value pairs
  while (pos < args.length) {
    const key = isNameValueKey(args[pos]);
    if (!key) break;
    pos++;
    if (pos >= args.length) break;
    const value = args[pos];
    pos++;
    applyNameValue([trace], key, value);
  }

  return [trace];
}

function applyNameValue(
  traces: PlotTrace[],
  key: string,
  value: RuntimeValue
): void {
  switch (key) {
    case "color": {
      const c = resolveColor(value);
      if (c) for (const t of traces) t.color = c;
      break;
    }
    case "linestyle": {
      const s = getStringValue(value);
      for (const t of traces) t.lineStyle = s;
      break;
    }
    case "linewidth": {
      const n = typeof value === "number" ? value : toNumber(value);
      for (const t of traces) t.lineWidth = n;
      break;
    }
    case "marker": {
      const s = getStringValue(value);
      for (const t of traces) t.marker = s === "none" ? undefined : s;
      break;
    }
    case "markersize": {
      const n = typeof value === "number" ? value : toNumber(value);
      for (const t of traces) t.markerSize = n;
      break;
    }
    case "markeredgecolor": {
      const c = resolveColor(value);
      if (c) for (const t of traces) t.markerEdgeColor = c;
      break;
    }
    case "markerfacecolor": {
      const c = resolveColor(value);
      if (c) for (const t of traces) t.markerFaceColor = c;
      break;
    }
    case "markerindices": {
      const arr = toNumberArray(value);
      for (const t of traces) t.markerIndices = arr;
      break;
    }
  }
}

// ── Imagesc argument parser ──────────────────────────────────────────────

/**
 * Parse imagesc() arguments.
 *
 * Supported forms:
 *   imagesc(C)         — C is m×n matrix
 *   imagesc(x, y, C)   — x and y are vectors defining axis limits
 */
export function parseImagescArgs(args: RuntimeValue[]): ImagescTrace {
  if (args.length === 1) {
    const info = getMatrixInfo(args[0]);
    return {
      x: [1, info.cols],
      y: [1, info.rows],
      z: info.data,
      rows: info.rows,
      cols: info.cols,
    };
  }
  if (args.length >= 3) {
    const xArr = toNumberArray(args[0]);
    const yArr = toNumberArray(args[1]);
    const info = getMatrixInfo(args[2]);
    return {
      x: [xArr[0], xArr[xArr.length - 1]],
      y: [yArr[0], yArr[yArr.length - 1]],
      z: info.data,
      rows: info.rows,
      cols: info.cols,
    };
  }
  throw new Error("imagesc requires 1 or 3+ arguments");
}

// ── Contour argument parser ──────────────────────────────────────────────

/**
 * Parse contour()/contourf() arguments.
 *
 * Supported forms:
 *   contour(Z)
 *   contour(Z, N)
 *   contour(X, Y, Z)
 *   contour(X, Y, Z, N)
 *   contour(..., Name, Value)
 */
export function parseContourArgs(
  args: RuntimeValue[],
  filled: boolean
): ContourTrace {
  let pos = 0;

  // Count leading numeric args
  let numericCount = 0;
  for (let i = pos; i < args.length; i++) {
    if (isNumericArg(args[i])) numericCount++;
    else break;
  }

  let xData: number[];
  let yData: number[];
  let zData: number[];
  let rows: number;
  let cols: number;
  let nLevels = 10;
  let levels: number[] | undefined;

  // The optional level argument that follows Z may be a scalar N (number of
  // levels) or a vector of explicit levels — see contour(...,levels).
  const applyLevelArg = (arg: RuntimeValue): void => {
    const vals = toNumberArray(arg);
    if (vals.length <= 1) nLevels = vals.length === 1 ? vals[0] : nLevels;
    else levels = vals;
  };

  if (numericCount === 1) {
    // contour(Z)
    const info = getMatrixInfo(args[pos++]);
    rows = info.rows;
    cols = info.cols;
    zData = info.data;
    const gen = generateMeshgrid(rows, cols);
    xData = gen.x;
    yData = gen.y;
  } else if (numericCount === 2) {
    // contour(Z, levels)
    const info = getMatrixInfo(args[pos++]);
    rows = info.rows;
    cols = info.cols;
    zData = info.data;
    applyLevelArg(args[pos++]);
    const gen = generateMeshgrid(rows, cols);
    xData = gen.x;
    yData = gen.y;
  } else if (numericCount === 3) {
    // contour(X, Y, Z)
    const x = args[pos++];
    const y = args[pos++];
    const zInfo = getMatrixInfo(args[pos++]);
    rows = zInfo.rows;
    cols = zInfo.cols;
    zData = zInfo.data;
    const expanded = expandXY(x, y, rows, cols);
    xData = expanded.x;
    yData = expanded.y;
  } else if (numericCount >= 4) {
    // contour(X, Y, Z, levels)
    const x = args[pos++];
    const y = args[pos++];
    const zInfo = getMatrixInfo(args[pos++]);
    rows = zInfo.rows;
    cols = zInfo.cols;
    zData = zInfo.data;
    applyLevelArg(args[pos++]);
    const expanded = expandXY(x, y, rows, cols);
    xData = expanded.x;
    yData = expanded.y;
  } else {
    throw new Error("contour requires at least 1 argument");
  }

  // Trailing Name-Value pairs (e.g. 'LineWidth', 2 / 'LineColor', 'flat').
  let lineWidth: number | undefined;
  let lineStyle: string | undefined;
  let lineColor: number[] | string | undefined;
  for (; pos + 1 < args.length; pos += 2) {
    if (!isStringArg(args[pos])) break;
    const name = getStringValue(args[pos] as RuntimeValue).toLowerCase();
    const val = args[pos + 1];
    switch (name) {
      case "linewidth":
        lineWidth = toNumber(val);
        break;
      case "linestyle":
        lineStyle = getStringValue(val as RuntimeValue);
        break;
      case "linecolor":
      case "color":
        lineColor = isStringArg(val)
          ? getStringValue(val as RuntimeValue)
          : toNumberArray(val as RuntimeValue);
        break;
      case "levellist":
      case "levels": {
        const vals = toNumberArray(val as RuntimeValue);
        if (vals.length >= 1) levels = vals;
        break;
      }
      // Unknown options are ignored (numbl renders a subset of properties).
    }
  }

  if (levels) nLevels = levels.length;
  return {
    x: xData,
    y: yData,
    z: zData,
    rows,
    cols,
    nLevels,
    ...(levels ? { levels } : {}),
    ...(lineWidth !== undefined ? { lineWidth } : {}),
    ...(lineStyle !== undefined ? { lineStyle } : {}),
    ...(lineColor !== undefined ? { lineColor } : {}),
    filled,
  };
}

/** Line segments [x1,y1,x2,y2][] for a single marching-squares cell.
 *  Corners: 00=(i,j) 10=(i+1,j) 01=(i,j+1) 11=(i+1,j+1). */
function marchingSquaresCell(
  z00: number,
  z10: number,
  z01: number,
  z11: number,
  x00: number,
  y00: number,
  x10: number,
  y10: number,
  x01: number,
  y01: number,
  x11: number,
  y11: number,
  level: number
): [number, number, number, number][] {
  const code =
    (z00 >= level ? 1 : 0) |
    (z10 >= level ? 2 : 0) |
    (z01 >= level ? 4 : 0) |
    (z11 >= level ? 8 : 0);
  if (code === 0 || code === 15) return [];

  const lerp = (a: number, b: number, za: number, zb: number) =>
    a + ((level - za) / (zb - za || 1)) * (b - a);

  // Edge crossings: bottom (00-10), right (10-11), top (01-11), left (00-01).
  const bx = lerp(x00, x10, z00, z10),
    by = lerp(y00, y10, z00, z10);
  const rx = lerp(x10, x11, z10, z11),
    ry = lerp(y10, y11, z10, z11);
  const tx = lerp(x01, x11, z01, z11),
    ty = lerp(y01, y11, z01, z11);
  const lx = lerp(x00, x01, z00, z01),
    ly = lerp(y00, y01, z00, z01);

  const segs: [number, number, number, number][] = [];
  switch (code) {
    case 1:
    case 14:
      segs.push([bx, by, lx, ly]);
      break;
    case 2:
    case 13:
      segs.push([bx, by, rx, ry]);
      break;
    case 3:
    case 12:
      segs.push([lx, ly, rx, ry]);
      break;
    case 4:
    case 11:
      segs.push([lx, ly, tx, ty]);
      break;
    case 5:
    case 10:
      segs.push([bx, by, tx, ty]);
      break;
    case 6:
    case 9:
      segs.push([bx, by, lx, ly]);
      segs.push([tx, ty, rx, ry]);
      break;
    case 7:
    case 8:
      segs.push([tx, ty, rx, ry]);
      break;
  }
  return segs;
}

/** Result of computing contour geometry: the MATLAB contour matrix `C`
 *  (column-major `2 × n` data) plus the level list actually used. */
export interface ContourMatrix {
  /** Column-major `2 × n` data for a `RTV.tensor(data, [2, n])`. */
  data: number[];
  n: number;
  levelList: number[];
}

/** Compute the MATLAB contour matrix for a parsed contour trace via marching
 *  squares. Each marching-squares segment is emitted as its own two-vertex
 *  contour line (header `[level; 2]` followed by the two `[x; y]` vertices) —
 *  a valid contour matrix whose geometry matches MATLAB's connected polylines
 *  segment-for-segment. */
export function computeContourMatrix(trace: ContourTrace): ContourMatrix {
  const { x, y, z, rows, cols } = trace;
  const at = (a: number[], i: number, j: number) => a[j * rows + i];

  let zMin = Infinity;
  let zMax = -Infinity;
  for (const v of z) {
    if (Number.isFinite(v)) {
      if (v < zMin) zMin = v;
      if (v > zMax) zMax = v;
    }
  }

  let levelList: number[];
  if (trace.levels && trace.levels.length > 0) {
    levelList = trace.levels.filter(v => Number.isFinite(v));
  } else {
    const n = Math.max(1, Math.round(trace.nLevels));
    levelList = [];
    if (Number.isFinite(zMin) && Number.isFinite(zMax) && zMax > zMin) {
      // Interior levels, mirroring MATLAB's exclusion of the data extremes.
      const step = (zMax - zMin) / (n + 1);
      for (let k = 1; k <= n; k++) levelList.push(zMin + k * step);
    }
  }

  const data: number[] = [];
  for (const level of levelList) {
    for (let j = 0; j < cols - 1; j++) {
      for (let i = 0; i < rows - 1; i++) {
        const segs = marchingSquaresCell(
          at(z, i, j),
          at(z, i + 1, j),
          at(z, i, j + 1),
          at(z, i + 1, j + 1),
          at(x, i, j),
          at(y, i, j),
          at(x, i + 1, j),
          at(y, i + 1, j),
          at(x, i, j + 1),
          at(y, i, j + 1),
          at(x, i + 1, j + 1),
          at(y, i + 1, j + 1),
          level
        );
        for (const [x1, y1, x2, y2] of segs) {
          // header column, then the two vertex columns (column-major).
          data.push(level, 2, x1, y1, x2, y2);
        }
      }
    }
  }
  return { data, n: data.length / 2, levelList };
}

// ── Mesh argument parser ─────────────────────────────────────────────────

/**
 * Parse mesh() arguments — same as surf but with different default rendering.
 */
export function parseMeshArgs(args: RuntimeValue[]): SurfTrace {
  const trace = parseSurfArgs(args);
  if (!trace.edgeColor) trace.edgeColor = "flat";
  if (!trace.faceColor) trace.faceColor = "none";
  return trace;
}

// ── Bar argument parser ─────────────────────────────────────────────────

/**
 * Parse bar() arguments.
 *
 * Supported forms:
 *   bar(Y)             — Y values with x = 1:length(Y)
 *   bar(X, Y)          — explicit X positions
 *   bar(..., width)     — relative bar width (scalar 0–1)
 *   bar(..., color)     — single-char color spec
 */
export function parseBarArgs(args: RuntimeValue[]): BarTrace[] {
  let pos = 0;
  let xData: number[] | undefined;
  let yData: number[];
  let width = 0.8;
  let color: [number, number, number] | undefined;

  if (args.length === 0) throw new Error("bar requires at least 1 argument");

  // bar(Y) or bar(X, Y, ...)
  const first = args[pos++];

  if (pos < args.length && isNumericArg(args[pos])) {
    // bar(X, Y, ...)
    xData = toNumberArray(first);
    yData = toNumberArray(args[pos++]);
  } else {
    // bar(Y) — could also be bar(Y, width) or bar(Y, color)
    yData = toNumberArray(first);
  }

  // Check for optional width (scalar) or color (string)
  while (pos < args.length) {
    if (isNumericArg(args[pos])) {
      const v = args[pos];
      if (isRuntimeNumber(v)) {
        width = v as number;
        pos++;
      } else {
        break;
      }
    } else if (isStringArg(args[pos])) {
      const s = getStringValue(args[pos]);
      const c = resolveColor(s);
      if (c) {
        color = c;
        pos++;
      } else {
        break;
      }
    } else {
      break;
    }
  }

  if (!xData) {
    xData = oneBasedIndices(yData.length);
  }

  const trace: BarTrace = { x: xData, y: yData, width };
  if (color) trace.color = color;
  return [trace];
}

/**
 * Parse barh() arguments — same forms as bar() but rendered horizontally.
 *
 * Supported forms:
 *   barh(Y)             — Y values (bar lengths) with positions = 1:length(Y)
 *   barh(X, Y)          — explicit positions X, bar lengths Y
 *   barh(..., width)    — relative bar thickness (scalar 0–1)
 *   barh(..., color)    — single-char color spec
 */
export function parseBarHArgs(args: RuntimeValue[]): BarTrace[] {
  return parseBarArgs(args);
}

/**
 * Parse bar3() / bar3h() arguments.
 *
 * Supported forms:
 *   bar3(Z)             — Z values with y = 1:m, x = 1:n
 *   bar3(Y, Z)          — explicit Y positions
 *   bar3(..., width)    — relative bar width (scalar 0–1)
 *   bar3(..., color)    — single-char color spec
 */
export function parseBar3Args(args: RuntimeValue[]): Bar3Trace {
  let pos = 0;
  let yPositions: number[] | undefined;
  let width = 0.8;
  let color: [number, number, number] | undefined;

  if (args.length === 0) throw new Error("bar3 requires at least 1 argument");

  const first = args[pos++];

  let zData: number[];
  let rows: number;
  let cols: number;

  if (pos < args.length && isNumericArg(args[pos])) {
    // bar3(Y, Z, ...)
    yPositions = toNumberArray(first);
    const second = args[pos++];
    if (isRuntimeTensor(second)) {
      const info = getMatrixInfo(second);
      zData = Array.from(info.data);
      rows = info.rows;
      cols = info.cols;
    } else {
      zData = toNumberArray(second);
      rows = zData.length;
      cols = 1;
    }
  } else {
    // bar3(Z)
    if (isRuntimeTensor(first)) {
      const info = getMatrixInfo(first);
      zData = Array.from(info.data);
      rows = info.rows;
      cols = info.cols;
    } else {
      zData = toNumberArray(first);
      rows = zData.length;
      cols = 1;
    }
  }

  // Check for optional width (scalar) or color (string)
  while (pos < args.length) {
    if (isNumericArg(args[pos])) {
      const v = args[pos];
      if (isRuntimeNumber(v)) {
        width = v as number;
        pos++;
      } else {
        break;
      }
    } else if (isStringArg(args[pos])) {
      const s = getStringValue(args[pos]);
      const c = resolveColor(s);
      if (c) {
        color = c;
        pos++;
      } else {
        break;
      }
    } else {
      break;
    }
  }

  if (!yPositions) {
    yPositions = oneBasedIndices(rows);
  }

  // Build x, y, z arrays for each bar (column-major → per-bar)
  const xArr: number[] = [];
  const yArr: number[] = [];
  const zArr: number[] = [];
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      xArr.push(j + 1); // 1-based column index
      yArr.push(yPositions[i]);
      zArr.push(zData[j * rows + i]); // column-major
    }
  }

  const trace: Bar3Trace = { x: xArr, y: yArr, z: zArr, rows, cols, width };
  if (color) trace.color = color;
  return trace;
}

/**
 * Parse stairs() arguments — same forms as plot() but with staircase-transformed coordinates.
 *
 * Supported forms:
 *   stairs(Y)
 *   stairs(X, Y)
 *   stairs(..., LineSpec)
 *   stairs(..., Name, Value)
 */
export function parseStairsArgs(args: RuntimeValue[]): PlotTrace[] {
  const traces = parsePlotArgs(args);
  // Transform each trace's x/y into staircase coordinates
  for (const t of traces) {
    const n = t.x.length;
    if (n < 2) continue;
    const xb: number[] = [];
    const yb: number[] = [];
    for (let i = 0; i < n - 1; i++) {
      xb.push(t.x[i]);
      yb.push(t.y[i]);
      xb.push(t.x[i + 1]);
      yb.push(t.y[i]);
    }
    xb.push(t.x[n - 1]);
    yb.push(t.y[n - 1]);
    t.x = xb;
    t.y = yb;
  }
  return traces;
}

/**
 * Parse errorbar() arguments.
 *
 * Supported forms:
 *   errorbar(y, err)                    — symmetric vertical
 *   errorbar(x, y, err)                 — symmetric vertical with explicit x
 *   errorbar(x, y, neg, pos)            — asymmetric vertical
 *   errorbar(x, y, yneg, ypos, xneg, xpos) — both axes
 *   errorbar(..., LineSpec)
 */
export function parseErrorBarArgs(args: RuntimeValue[]): ErrorBarTrace[] {
  if (args.length < 2)
    throw new Error("errorbar requires at least 2 arguments");

  let pos = 0;
  let xData: number[] | undefined;
  let yData: number[];
  let yNeg: number[];
  let yPos: number[];
  let xNeg: number[] | undefined;
  let xPos: number[] | undefined;
  let color: [number, number, number] | undefined;
  let lineStyle: string | undefined;
  let marker: string | undefined;

  // Count leading numeric args
  let numericCount = 0;
  for (let i = 0; i < args.length; i++) {
    if (isNumericArg(args[i])) numericCount++;
    else break;
  }

  if (numericCount === 2) {
    // errorbar(y, err)
    yData = toNumberArray(args[pos++]);
    const err = toNumberArray(args[pos++]);
    yNeg = err;
    yPos = err;
  } else if (numericCount === 3) {
    // errorbar(x, y, err)
    xData = toNumberArray(args[pos++]);
    yData = toNumberArray(args[pos++]);
    const err = toNumberArray(args[pos++]);
    yNeg = err;
    yPos = err;
  } else if (numericCount === 4) {
    // errorbar(x, y, neg, pos)
    xData = toNumberArray(args[pos++]);
    yData = toNumberArray(args[pos++]);
    yNeg = toNumberArray(args[pos++]);
    yPos = toNumberArray(args[pos++]);
  } else if (numericCount >= 6) {
    // errorbar(x, y, yneg, ypos, xneg, xpos)
    xData = toNumberArray(args[pos++]);
    yData = toNumberArray(args[pos++]);
    yNeg = toNumberArray(args[pos++]);
    yPos = toNumberArray(args[pos++]);
    xNeg = toNumberArray(args[pos++]);
    xPos = toNumberArray(args[pos++]);
  } else {
    throw new Error("errorbar: invalid number of numeric arguments");
  }

  if (!xData) {
    xData = oneBasedIndices(yData.length);
  }

  // Check for optional LineSpec or color
  while (pos < args.length) {
    if (isStringArg(args[pos])) {
      const s = getStringValue(args[pos]);
      const spec = parseLineSpec(s);
      if (spec) {
        if (spec.color) {
          const c = resolveColor(spec.color);
          if (c) color = c;
        }
        if (spec.lineStyle) lineStyle = spec.lineStyle;
        if (spec.marker) marker = spec.marker;
        pos++;
      } else {
        const c = resolveColor(s);
        if (c) {
          color = c;
          pos++;
        } else {
          break;
        }
      }
    } else {
      break;
    }
  }

  const trace: ErrorBarTrace = { x: xData, y: yData, yNeg, yPos };
  if (xNeg) trace.xNeg = xNeg;
  if (xPos) trace.xPos = xPos;
  if (color) trace.color = color;
  if (lineStyle) trace.lineStyle = lineStyle;
  if (marker) trace.marker = marker;
  return [trace];
}

/**
 * Parse area() arguments.
 *
 * Supported forms:
 *   area(Y)              — fill under Y with x = 1:length(Y)
 *   area(X, Y)           — explicit X
 *   area(..., basevalue)  — scalar baseline (default 0)
 *   area(..., Name, Value)
 *
 * Returns { traces, baseValue }.
 */
export function parseAreaArgs(args: RuntimeValue[]): {
  traces: PlotTrace[];
  baseValue: number;
} {
  let baseValue = 0;

  // Check if the last numeric arg is a scalar (basevalue)
  // We need to peek: if last numeric arg is a scalar and there are ≥2 numeric args, it could be basevalue
  const argsCopy = [...args];
  const lastIdx = argsCopy.length - 1;
  if (
    lastIdx >= 1 &&
    isNumericArg(argsCopy[lastIdx]) &&
    isRuntimeNumber(argsCopy[lastIdx])
  ) {
    // If the previous arg is also numeric (vector/matrix), this scalar is basevalue
    if (lastIdx >= 1 && isNumericArg(argsCopy[lastIdx - 1])) {
      // Check it's truly a scalar, not a 1-element vector used as Y
      baseValue = toNumber(argsCopy[lastIdx]);
      argsCopy.pop();
    }
  }

  // Now parse remaining args as plot-style (handles Y, X/Y, matrix columns, etc.)
  const traces = parsePlotArgs(argsCopy);
  return { traces, baseValue };
}

// ── scatter3 argument parser ────────────────────────────────────────────

/**
 * Parse scatter3() arguments.
 *
 * Supported forms:
 *   scatter3(X, Y, Z)
 *   scatter3(X, Y, Z, S)
 *   scatter3(X, Y, Z, S, C)
 *   scatter3(..., 'filled')
 *   scatter3(..., markertype)
 */
export function parseScatter3Args(args: RuntimeValue[]): Plot3Trace[] {
  if (args.length < 3)
    throw new Error("scatter3 requires at least 3 arguments");

  const xData = toNumberArray(args[0]);
  const yData = toNumberArray(args[1]);
  const zData = toNumberArray(args[2]);

  const trace: Plot3Trace = {
    x: xData,
    y: yData,
    z: zData,
    marker: "o",
    lineStyle: "none",
  };

  let pos = 3;
  let filled = false;

  // Check for size argument (numeric, 4th positional arg)
  if (pos < args.length && isNumericArg(args[pos])) {
    const sz = args[pos];
    if (isRuntimeNumber(sz)) {
      trace.markerSize = Math.sqrt(sz as number) * 0.5;
    } else if (isRuntimeTensor(sz)) {
      const sArr = toNumberArray(sz);
      if (sArr.length === 1 || new Set(sArr).size === 1) {
        trace.markerSize = Math.sqrt(sArr[0]) * 0.5;
      }
    }
    pos++;
  }

  // Check for color argument (5th positional: string color spec or RGB tensor)
  if (pos < args.length && !isNameValueKey(args[pos])) {
    if (isStringArg(args[pos])) {
      const s = getStringValue(args[pos]);
      if (s === "filled") {
        filled = true;
        pos++;
      } else {
        const c = resolveColor(s);
        if (c) {
          trace.color = c;
          pos++;
        } else {
          const spec = parseLineSpec(s);
          if (spec?.marker) {
            trace.marker = spec.marker;
            if (spec.color) trace.color = COLOR_SHORT[spec.color];
            pos++;
          }
        }
      }
    } else if (isNumericArg(args[pos])) {
      const c = resolveColor(args[pos]);
      if (c) {
        trace.color = c;
        pos++;
      }
    }
  }

  // Check for 'filled' or marker string in remaining positional args
  while (
    pos < args.length &&
    isStringArg(args[pos]) &&
    !isNameValueKey(args[pos])
  ) {
    const s = getStringValue(args[pos]);
    if (s === "filled") {
      filled = true;
      pos++;
    } else {
      const spec = parseLineSpec(s);
      if (spec?.marker) {
        trace.marker = spec.marker;
        if (spec.color) trace.color = COLOR_SHORT[spec.color];
        pos++;
      } else {
        break;
      }
    }
  }

  if (filled) {
    trace.markerFaceColor = trace.color || [0, 0, 1];
  }

  // Name-Value pairs
  while (pos < args.length) {
    const key = isNameValueKey(args[pos]);
    if (!key) break;
    pos++;
    if (pos >= args.length) break;
    const value = args[pos];
    pos++;
    applyPlot3NameValue([trace], key, value);
  }

  return [trace];
}

// ── histogram argument parser ───────────────────────────────────────────

/**
 * Parse histogram() arguments.
 *
 * Supported forms:
 *   histogram(X)
 *   histogram(X, nbins)
 *   histogram(X, edges)
 */
export function parseHistogramArgs(args: RuntimeValue[]): BarTrace[] {
  if (args.length === 0)
    throw new Error("histogram requires at least 1 argument");

  let pos = 0;
  const xData = toNumberArray(args[pos++]);

  // Filter out non-finite values
  const data = xData.filter(v => isFinite(v));
  if (data.length === 0) return [];

  let edges: number[];

  if (pos < args.length && isNumericArg(args[pos])) {
    const arg = args[pos++];
    if (isRuntimeNumber(arg)) {
      // histogram(X, nbins)
      const nbins = arg as number;
      edges = computeUniformEdges(data, Math.max(1, Math.round(nbins)));
    } else {
      // histogram(X, edges)
      edges = toNumberArray(arg);
    }
  } else {
    // Auto bins using Sturges' rule
    const nbins = Math.max(1, Math.ceil(Math.log2(data.length) + 1));
    edges = computeUniformEdges(data, nbins);
  }

  // Compute bin counts
  const nBins = edges.length - 1;
  if (nBins <= 0) return [];
  const counts = new Array(nBins).fill(0);
  for (const v of data) {
    // Find bin (last bin is right-inclusive)
    let idx = -1;
    for (let i = 0; i < nBins; i++) {
      if (i < nBins - 1) {
        if (v >= edges[i] && v < edges[i + 1]) {
          idx = i;
          break;
        }
      } else {
        if (v >= edges[i] && v <= edges[i + 1]) {
          idx = i;
          break;
        }
      }
    }
    if (idx >= 0) counts[idx]++;
  }

  // Build bar trace with bin centers
  const xBins: number[] = [];
  const yBins: number[] = [];
  const width = edges[1] - edges[0];
  for (let i = 0; i < nBins; i++) {
    xBins.push((edges[i] + edges[i + 1]) / 2);
    yBins.push(counts[i]);
  }

  return [{ x: xBins, y: yBins, width }];
}

function computeUniformEdges(data: number[], nbins: number): number[] {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const step = range / nbins;
  const edges: number[] = [];
  for (let i = 0; i <= nbins; i++) {
    edges.push(min + i * step);
  }
  return edges;
}

// ── histogram2 argument parser ──────────────────────────────────────────

/**
 * Parse histogram2() arguments.
 *
 * Supported forms:
 *   histogram2(X, Y)
 *   histogram2(X, Y, nbins)        — nbins is scalar or [nx, ny]
 *   histogram2(X, Y, Xedges, Yedges)
 */
export function parseHistogram2Args(args: RuntimeValue[]): Bar3Trace {
  if (args.length < 2)
    throw new Error("histogram2 requires at least 2 arguments");

  let pos = 0;
  const xData = toNumberArray(args[pos++]).filter(v => isFinite(v));
  const yData = toNumberArray(args[pos++]).filter(v => isFinite(v));
  const n = Math.min(xData.length, yData.length);

  if (n === 0) {
    return { x: [], y: [], z: [], rows: 0, cols: 0, width: 0.8 };
  }

  let xEdges: number[];
  let yEdges: number[];

  if (pos < args.length && isNumericArg(args[pos])) {
    const arg1 = args[pos++];
    if (pos < args.length && isNumericArg(args[pos])) {
      // histogram2(X, Y, Xedges, Yedges)
      xEdges = toNumberArray(arg1);
      yEdges = toNumberArray(args[pos++]);
    } else if (isRuntimeNumber(arg1)) {
      // histogram2(X, Y, nbins) — scalar
      const nb = Math.max(1, Math.round(arg1 as number));
      xEdges = computeUniformEdges(xData.slice(0, n), nb);
      yEdges = computeUniformEdges(yData.slice(0, n), nb);
    } else {
      // histogram2(X, Y, [nx, ny])
      const nbArr = toNumberArray(arg1);
      const nx = Math.max(1, Math.round(nbArr[0]));
      const ny = nbArr.length > 1 ? Math.max(1, Math.round(nbArr[1])) : nx;
      xEdges = computeUniformEdges(xData.slice(0, n), nx);
      yEdges = computeUniformEdges(yData.slice(0, n), ny);
    }
  } else {
    // Auto bins
    const nb = Math.max(1, Math.ceil(Math.sqrt(n)));
    xEdges = computeUniformEdges(xData.slice(0, n), nb);
    yEdges = computeUniformEdges(yData.slice(0, n), nb);
  }

  const nxBins = xEdges.length - 1;
  const nyBins = yEdges.length - 1;
  if (nxBins <= 0 || nyBins <= 0) {
    return { x: [], y: [], z: [], rows: 0, cols: 0, width: 0.8 };
  }

  // Compute 2D bin counts
  const counts = new Array(nxBins * nyBins).fill(0);
  for (let k = 0; k < n; k++) {
    const xi = findBin(xData[k], xEdges);
    const yi = findBin(yData[k], yEdges);
    if (xi >= 0 && yi >= 0) {
      counts[xi * nyBins + yi]++; // column-major: col=xi, row=yi
    }
  }

  // Build Bar3Trace — rows = nyBins, cols = nxBins
  const xArr: number[] = [];
  const yArr: number[] = [];
  const zArr: number[] = [];
  for (let j = 0; j < nxBins; j++) {
    for (let i = 0; i < nyBins; i++) {
      xArr.push((xEdges[j] + xEdges[j + 1]) / 2);
      yArr.push((yEdges[i] + yEdges[i + 1]) / 2);
      zArr.push(counts[j * nyBins + i]);
    }
  }

  // Set width to the smaller bin spacing so bars don't overlap
  const xBinWidth = xEdges[1] - xEdges[0];
  const yBinWidth = yEdges[1] - yEdges[0];
  const width = Math.min(xBinWidth, yBinWidth);

  return {
    x: xArr,
    y: yArr,
    z: zArr,
    rows: nyBins,
    cols: nxBins,
    width,
  };
}

function findBin(v: number, edges: number[]): number {
  const nBins = edges.length - 1;
  for (let i = 0; i < nBins; i++) {
    if (i < nBins - 1) {
      if (v >= edges[i] && v < edges[i + 1]) return i;
    } else {
      if (v >= edges[i] && v <= edges[i + 1]) return i;
    }
  }
  return -1;
}

// ── boxchart argument parser ────────────────────────────────────────────

/** Compute percentile using linear interpolation. */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/** Compute box statistics for an array of values. */
function computeBoxStats(
  values: number[],
  x: number,
  width: number,
  color?: [number, number, number]
): BoxTrace | null {
  const data = values.filter(v => isFinite(v));
  if (data.length === 0) return null;
  data.sort((a, b) => a - b);

  const median = percentile(data, 50);
  const q1 = percentile(data, 25);
  const q3 = percentile(data, 75);
  const iqr = q3 - q1;
  const lowerFence = q1 - 1.5 * iqr;
  const upperFence = q3 + 1.5 * iqr;

  let whiskerLow = q1;
  let whiskerHigh = q3;
  const outliers: number[] = [];
  for (const v of data) {
    if (v < lowerFence || v > upperFence) {
      outliers.push(v);
    } else {
      if (v < whiskerLow) whiskerLow = v;
      if (v > whiskerHigh) whiskerHigh = v;
    }
  }

  const trace: BoxTrace = {
    x,
    median,
    q1,
    q3,
    whiskerLow,
    whiskerHigh,
    outliers,
    width,
  };
  if (color) trace.color = color;
  return trace;
}

/**
 * Parse boxchart() arguments.
 *
 * Supported forms:
 *   boxchart(ydata)           — one box per column of ydata
 *   boxchart(xgroupdata, ydata) — group ydata by unique values in xgroupdata
 */
export function parseBoxchartArgs(args: RuntimeValue[]): BoxTrace[] {
  if (args.length === 0)
    throw new Error("boxchart requires at least 1 argument");

  let pos = 0;
  const first = args[pos++];

  if (pos < args.length && isNumericArg(args[pos])) {
    // boxchart(xgroupdata, ydata)
    const xGroup = toNumberArray(first);
    const yData = toNumberArray(args[pos++]);
    const n = Math.min(xGroup.length, yData.length);

    // Group y values by unique x values
    const groups = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
      const key = xGroup[i];
      if (!isFinite(key)) continue;
      let arr = groups.get(key);
      if (!arr) {
        arr = [];
        groups.set(key, arr);
      }
      arr.push(yData[i]);
    }

    const traces: BoxTrace[] = [];
    const sortedKeys = [...groups.keys()].sort((a, b) => a - b);
    for (const key of sortedKeys) {
      const t = computeBoxStats(groups.get(key)!, key, 0.5);
      if (t) traces.push(t);
    }
    return traces;
  }

  // boxchart(ydata) — one box per column
  if (
    isRuntimeTensor(first) &&
    first.shape.length >= 2 &&
    first.shape[0] > 1 &&
    first.shape[1] > 1
  ) {
    // Matrix: one box per column
    const traces: BoxTrace[] = [];
    for (let j = 0; j < first.shape[1]; j++) {
      const col = tensorColumn(first, j);
      const t = computeBoxStats(col, j + 1, 0.5);
      if (t) traces.push(t);
    }
    return traces;
  }

  // Vector: single box at x=1
  const data = toNumberArray(first);
  const t = computeBoxStats(data, 1, 0.5);
  return t ? [t] : [];
}

// ── swarmchart argument parser ──────────────────────────────────────────

/**
 * Apply jitter to points at each unique x position based on density of y values.
 * Returns jittered x positions.
 */
function applySwarmJitter(
  x: number[],
  y: number[],
  maxJitter: number
): number[] {
  const jittered = new Array(x.length);

  // Group points by unique x value
  const groups = new Map<number, number[]>();
  for (let i = 0; i < x.length; i++) {
    let arr = groups.get(x[i]);
    if (!arr) {
      arr = [];
      groups.set(x[i], arr);
    }
    arr.push(i);
  }

  for (const [, indices] of groups) {
    if (indices.length <= 1) {
      for (const idx of indices) jittered[idx] = x[idx];
      continue;
    }

    // Sort by y value within this group
    const sorted = indices.slice().sort((a, b) => y[a] - y[b]);
    const yVals = sorted.map(i => y[i]);
    const yMin = yVals[0];
    const yMax = yVals[yVals.length - 1];
    const yRange = yMax - yMin || 1;

    // Estimate density using a simple Gaussian KDE
    const bandwidth = yRange / Math.max(3, Math.sqrt(sorted.length));
    const densities = new Array(sorted.length);
    for (let i = 0; i < sorted.length; i++) {
      let d = 0;
      for (let j = 0; j < sorted.length; j++) {
        const u = (yVals[i] - yVals[j]) / bandwidth;
        d += Math.exp(-0.5 * u * u);
      }
      densities[i] = d;
    }
    const maxDensity = Math.max(...densities);

    // Assign jitter: alternate left/right at each y level
    for (let i = 0; i < sorted.length; i++) {
      const relDensity = maxDensity > 0 ? densities[i] / maxDensity : 0;
      const jitterRange = maxJitter * relDensity;
      // Place points in a beeswarm pattern
      const rank = i;
      const side = rank % 2 === 0 ? 1 : -1;
      const offset =
        (Math.ceil(rank / 2) / Math.max(1, sorted.length / 2)) * jitterRange;
      jittered[sorted[i]] = x[sorted[i]] + side * offset;
    }
  }

  return jittered;
}

/**
 * Parse swarmchart() arguments.
 *
 * Supported forms:
 *   swarmchart(x, y)
 *   swarmchart(x, y, sz)
 *   swarmchart(x, y, sz, c)
 *   swarmchart(..., 'filled')
 *   swarmchart(..., markertype)
 */
export function parseSwarmchartArgs(args: RuntimeValue[]): PlotTrace[] {
  if (args.length < 2)
    throw new Error("swarmchart requires at least 2 arguments");

  const xData = toNumberArray(args[0]);
  const yData = toNumberArray(args[1]);

  // Apply jitter based on density
  const jitteredX = applySwarmJitter(xData, yData, 0.3);

  // Build synthetic scatter args with jittered x
  const synthArgs: RuntimeValue[] = [
    tensorFromArray(jitteredX),
    args[1],
    ...args.slice(2),
  ];
  return parseScatterArgs(synthArgs);
}

/**
 * Parse swarmchart3() arguments.
 *
 * Supported forms:
 *   swarmchart3(x, y, z)
 *   swarmchart3(x, y, z, sz)
 *   swarmchart3(x, y, z, sz, c)
 *   swarmchart3(..., 'filled')
 *   swarmchart3(..., markertype)
 */
export function parseSwarmchart3Args(args: RuntimeValue[]): Plot3Trace[] {
  if (args.length < 3)
    throw new Error("swarmchart3 requires at least 3 arguments");

  const xData = toNumberArray(args[0]);
  const yData = toNumberArray(args[1]);
  const zData = toNumberArray(args[2]);

  // Apply jitter in x based on z density at each (x,y) position
  const jitteredX = applySwarmJitter(xData, zData, 0.3);
  // Apply jitter in y based on z density at each (x,y) position
  const jitteredY = applySwarmJitter(yData, zData, 0.3);

  const synthArgs: RuntimeValue[] = [
    tensorFromArray(jitteredX),
    tensorFromArray(jitteredY),
    args[2],
    ...args.slice(3),
  ];
  return parseScatter3Args(synthArgs);
}

/** Create a runtime tensor from a plain number array (for synthetic args). */
function tensorFromArray(arr: number[]): RuntimeValue {
  return {
    kind: "tensor",
    data: allocFloat64Array(arr),
    shape: [1, arr.length],
  } as RuntimeTensor;
}

// ── piechart / donutchart argument parser ────────────────────────────────

/**
 * Parse piechart() / donutchart() arguments.
 *
 * Supported forms:
 *   piechart(data)
 *   piechart(data, names)
 *
 * @param innerRadius 0 for pie, >0 for donut
 */
export function parsePiechartArgs(
  args: RuntimeValue[],
  innerRadius: number
): PieTrace {
  if (args.length === 0)
    throw new Error("piechart requires at least 1 argument");

  let pos = 0;
  const values = toNumberArray(args[pos++]).filter(v => isFinite(v) && v > 0);

  let names: string[] | undefined;
  if (pos < args.length && !isNumericArg(args[pos])) {
    // Collect string names from a cell array or individual string args
    if (isRuntimeTensor(args[pos]) || isRuntimeNumber(args[pos])) {
      // Not names, skip
    } else {
      names = [];
      // Could be a cell array of strings or individual string args
      const nameArg = args[pos];
      if (
        nameArg &&
        typeof nameArg === "object" &&
        "kind" in nameArg &&
        (nameArg as { kind: string }).kind === "cell"
      ) {
        // Cell array of strings
        const cell = nameArg as { data: RuntimeValue[] };
        for (const item of cell.data) {
          names.push(getStringValue(item));
        }
      } else if (isStringArg(nameArg)) {
        // Single string — try collecting multiple string args
        names.push(getStringValue(nameArg));
        pos++;
        while (pos < args.length && isStringArg(args[pos])) {
          names.push(getStringValue(args[pos]));
          pos++;
        }
        // Already advanced pos, skip the increment below
        return { values, names, innerRadius };
      }
      pos++;
    }
  }

  return { values, names, innerRadius };
}

// ── heatmap argument parser ─────────────────────────────────────────────

/**
 * Parse heatmap() arguments.
 *
 * Supported forms:
 *   heatmap(cdata)                — matrix of values
 *   heatmap(xvalues, yvalues, cdata) — with axis labels
 */
export function parseHeatmapArgs(args: RuntimeValue[]): HeatmapTrace {
  if (args.length === 0)
    throw new Error("heatmap requires at least 1 argument");

  let pos = 0;

  // Count leading args to determine form
  if (args.length >= 3 && !isNumericArg(args[0])) {
    // Possibly heatmap(xvalues_strings, yvalues_strings, cdata) — skip for now
  }

  if (
    args.length >= 3 &&
    isNumericArg(args[0]) &&
    isNumericArg(args[1]) &&
    isNumericArg(args[2])
  ) {
    // heatmap(xvalues, yvalues, cdata)
    const xArr = toNumberArray(args[pos++]);
    const yArr = toNumberArray(args[pos++]);
    const info = getMatrixInfo(args[pos++]);
    return {
      data: info.data,
      rows: info.rows,
      cols: info.cols,
      xLabels: xArr.map(v => String(v)),
      yLabels: yArr.map(v => String(v)),
    };
  }

  // heatmap(cdata)
  const info = getMatrixInfo(args[pos++]);
  return {
    data: info.data,
    rows: info.rows,
    cols: info.cols,
  };
}

// ── Quiver argument parser ──────────────────────────────────────────────

const QUIVER_NAME_VALUE_KEYS = new Set([
  "color",
  "linestyle",
  "linewidth",
  "showarrowhead",
  "autoscale",
  "autoscalefactor",
  "marker",
]);

function isQuiverNameValueKey(v: RuntimeValue): string | null {
  if (!isRuntimeString(v) && !isRuntimeChar(v)) return null;
  const lower = isRuntimeString(v)
    ? v.toLocaleLowerCase()
    : v.value.toLowerCase();
  if (QUIVER_NAME_VALUE_KEYS.has(lower)) return lower;
  return null;
}

function getStringIfString(v: unknown): string | undefined {
  if (typeof v === "string") return v;
  if (v && typeof v === "object" && "kind" in v) {
    const mv = v as RuntimeValue;
    if (isRuntimeString(mv)) return mv;
    if (isRuntimeChar(mv)) return mv.value;
  }
  return undefined;
}

/**
 * Parse quiver() arguments.
 *
 * Supported forms:
 *   quiver(U, V)
 *   quiver(X, Y, U, V)
 *   quiver(..., scale)        — scale: nonnegative number or 'off'
 *   quiver(..., LineSpec)
 *   quiver(..., LineSpec, 'filled')
 *   quiver(..., Name, Value)  — Color, LineStyle, LineWidth, ShowArrowHead, etc.
 *
 * X, Y, U, V can be vectors or matrices. If U,V are matrices and X,Y are
 * vectors, X and Y are expanded to a meshgrid.
 */
export function parseQuiverArgs(args: RuntimeValue[]): QuiverTrace[] {
  if (args.length < 2) throw new Error("quiver requires at least 2 arguments");

  // Count leading numeric args
  let numericCount = 0;
  for (let i = 0; i < args.length; i++) {
    if (isNumericArg(args[i])) numericCount++;
    else break;
  }

  let pos = 0;
  let xData: number[];
  let yData: number[];
  let uData: number[];
  let vData: number[];
  let rows: number;
  let cols: number;

  // Determine arity: (U,V) vs (X,Y,U,V), with optional trailing scale
  // We treat quiver(U,V) and quiver(U,V,scale) and quiver(X,Y,U,V) and quiver(X,Y,U,V,scale)
  let arity: 2 | 4;
  if (numericCount === 2) {
    arity = 2;
  } else if (numericCount === 3) {
    // quiver(U, V, scale)
    arity = 2;
  } else if (numericCount >= 4) {
    arity = 4;
  } else {
    throw new Error("quiver requires at least 2 numeric arguments");
  }

  if (arity === 2) {
    const U = args[pos++];
    const V = args[pos++];
    const uInfo = getMatrixInfo(U);
    const vInfo = getMatrixInfo(V);
    rows = uInfo.rows;
    cols = uInfo.cols;
    uData = uInfo.data;
    vData = vInfo.data;
    // Default coordinates
    if (rows === 1 || cols === 1) {
      // Vector: X = 1:n, Y all 1
      const n = uData.length;
      xData = new Array(n);
      yData = new Array(n);
      for (let i = 0; i < n; i++) {
        xData[i] = i + 1;
        yData[i] = 1;
      }
    } else {
      const gen = generateMeshgrid(rows, cols);
      xData = gen.x;
      yData = gen.y;
    }
  } else {
    const X = args[pos++];
    const Y = args[pos++];
    const U = args[pos++];
    const V = args[pos++];
    const uInfo = getMatrixInfo(U);
    const vInfo = getMatrixInfo(V);
    rows = uInfo.rows;
    cols = uInfo.cols;
    uData = uInfo.data;
    vData = vInfo.data;
    const expanded = expandXY(X, Y, rows, cols);
    xData = expanded.x;
    yData = expanded.y;
  }

  // Trailing scale: numeric (positional after the X/Y/U/V tuple) or 'off'
  let autoScale = true;
  let autoScaleFactor = 0.9;

  // Numeric scale immediately after the data args
  if (pos < args.length && isNumericArg(args[pos]) && !isStringArg(args[pos])) {
    const s = toNumber(args[pos] as RuntimeValue);
    if (s === 0) {
      autoScale = false;
    } else {
      autoScaleFactor = s * 0.9;
    }
    pos++;
  }

  const trace: QuiverTrace = {
    x: xData,
    y: yData,
    u: uData,
    v: vData,
    showArrowHead: true,
  };

  // Trailing string args: 'off' for scale, or LineSpec, or 'filled'
  while (
    pos < args.length &&
    isStringArg(args[pos]) &&
    !isQuiverNameValueKey(args[pos])
  ) {
    const s = getStringIfString(args[pos]);
    if (s === undefined) break;
    if (s === "off") {
      autoScale = false;
      pos++;
      continue;
    }
    if (s === "filled") {
      trace.markerFilled = true;
      pos++;
      continue;
    }
    const spec = parseLineSpec(s);
    if (spec) {
      if (spec.color) trace.color = COLOR_SHORT[spec.color];
      if (spec.lineStyle) trace.lineStyle = spec.lineStyle;
      if (spec.marker) {
        trace.marker = spec.marker;
        // When a marker is specified via LineSpec, MATLAB hides arrowheads
        trace.showArrowHead = false;
      }
      pos++;
      continue;
    }
    // Maybe a color name like 'red'
    const c = resolveColor(s);
    if (c) {
      trace.color = c;
      pos++;
      continue;
    }
    break;
  }

  // Name-Value pairs
  while (pos < args.length) {
    const key = isQuiverNameValueKey(args[pos]);
    if (!key) break;
    pos++;
    if (pos >= args.length) break;
    const value = args[pos++];
    switch (key) {
      case "color": {
        const c = resolveColor(value);
        if (c) trace.color = c;
        break;
      }
      case "linestyle":
        trace.lineStyle = getStringValue(value);
        break;
      case "linewidth":
        trace.lineWidth = typeof value === "number" ? value : toNumber(value);
        break;
      case "marker": {
        const s = getStringValue(value);
        trace.marker = s === "none" ? undefined : s;
        break;
      }
      case "showarrowhead": {
        const s = getStringValue(value).toLowerCase();
        trace.showArrowHead = !(s === "off" || s === "false" || s === "0");
        break;
      }
      case "autoscale": {
        const s = getStringValue(value).toLowerCase();
        autoScale = !(s === "off" || s === "false" || s === "0");
        break;
      }
      case "autoscalefactor": {
        const n = typeof value === "number" ? value : toNumber(value);
        autoScaleFactor = n * 0.9;
        break;
      }
    }
  }

  // Apply autoscale: scale (u, v) so max arrow length ≈ characteristic
  // grid spacing × autoScaleFactor.
  if (autoScale) {
    const factor = computeQuiverAutoScale(
      xData,
      yData,
      uData,
      vData,
      rows,
      cols,
      autoScaleFactor
    );
    if (factor !== 1) {
      const u2 = new Array(uData.length);
      const v2 = new Array(vData.length);
      for (let i = 0; i < uData.length; i++) u2[i] = uData[i] * factor;
      for (let i = 0; i < vData.length; i++) v2[i] = vData[i] * factor;
      trace.u = u2;
      trace.v = v2;
    }
  }

  return [trace];
}

/**
 * Compute a scaling factor so the longest arrow is roughly `factor` times
 * the characteristic point spacing.
 *
 * Combines two estimators and takes the max:
 *  1. Mean of distances between row/column-adjacent grid points — accurate
 *     for regular and polar/curvilinear grids.
 *  2. Area-based sqrt(bbox_area / N) — accurate for points scattered along
 *     a curve (e.g., chunker nodes), where curve-adjacency is much smaller
 *     than the typical 2D point spacing.
 */
function computeQuiverAutoScale(
  x: number[],
  y: number[],
  u: number[],
  v: number[],
  rows: number,
  cols: number,
  factor: number
): number {
  const n = u.length;
  if (n === 0) return 1;

  // Estimator 1: mean adjacent-grid-point distance.
  let total = 0;
  let count = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols - 1; j++) {
      const i0 = j * rows + i;
      const i1 = (j + 1) * rows + i;
      const ddx = x[i1] - x[i0];
      const ddy = y[i1] - y[i0];
      const d = Math.sqrt(ddx * ddx + ddy * ddy);
      if (isFinite(d) && d > 0) {
        total += d;
        count++;
      }
    }
  }
  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < cols; j++) {
      const i0 = j * rows + i;
      const i1 = j * rows + i + 1;
      const ddx = x[i1] - x[i0];
      const ddy = y[i1] - y[i0];
      const d = Math.sqrt(ddx * ddx + ddy * ddy);
      if (isFinite(d) && d > 0) {
        total += d;
        count++;
      }
    }
  }
  const adjacencySpacing = count > 0 ? total / count : 0;

  // Estimator 2: bounding-box area / N.
  let xMin = Infinity,
    xMax = -Infinity,
    yMin = Infinity,
    yMax = -Infinity;
  for (let i = 0; i < x.length; i++) {
    if (isFinite(x[i])) {
      if (x[i] < xMin) xMin = x[i];
      if (x[i] > xMax) xMax = x[i];
    }
  }
  for (let i = 0; i < y.length; i++) {
    if (isFinite(y[i])) {
      if (y[i] < yMin) yMin = y[i];
      if (y[i] > yMax) yMax = y[i];
    }
  }
  const xRange = xMax - xMin;
  const yRange = yMax - yMin;
  let bboxSpacing: number;
  if (xRange > 0 && yRange > 0) {
    bboxSpacing = Math.sqrt((xRange * yRange) / n);
  } else if (xRange > 0) {
    bboxSpacing = xRange / Math.max(1, n - 1);
  } else if (yRange > 0) {
    bboxSpacing = yRange / Math.max(1, n - 1);
  } else {
    bboxSpacing = 0;
  }

  const spacing = Math.max(adjacencySpacing, bboxSpacing) || 1;

  // Maximum vector magnitude
  let maxMag = 0;
  for (let i = 0; i < n; i++) {
    const ux = u[i];
    const vy = v[i];
    if (!isFinite(ux) || !isFinite(vy)) continue;
    const m = Math.sqrt(ux * ux + vy * vy);
    if (m > maxMag) maxMag = m;
  }

  if (maxMag === 0) return 1;
  return (factor * spacing) / maxMag;
}

/**
 * Parse quiver3() arguments.
 *
 * Supported forms:
 *   quiver3(Z, U, V, W)
 *   quiver3(X, Y, Z, U, V, W)
 *   quiver3(..., scale)         — scale: nonnegative number or 'off'
 *   quiver3(..., LineSpec)
 *   quiver3(..., LineSpec, 'filled')
 *   quiver3(..., Name, Value)   — Color, LineStyle, LineWidth, ShowArrowHead, …
 */
export function parseQuiver3Args(args: RuntimeValue[]): Quiver3Trace {
  if (args.length < 4) throw new Error("quiver3 requires at least 4 arguments");

  let numericCount = 0;
  for (let i = 0; i < args.length; i++) {
    if (isNumericArg(args[i])) numericCount++;
    else break;
  }

  let pos = 0;
  let xData: number[];
  let yData: number[];
  let zData: number[];
  let uData: number[];
  let vData: number[];
  let wData: number[];
  let rows: number;
  let cols: number;

  // arity 4 = quiver3(Z,U,V,W); arity 6 = quiver3(X,Y,Z,U,V,W).
  // (counts of 5 / 7 carry a trailing positional scale.)
  const arity: 4 | 6 = numericCount >= 6 ? 6 : 4;

  if (arity === 4) {
    const zInfo = getMatrixInfo(args[pos++]);
    rows = zInfo.rows;
    cols = zInfo.cols;
    zData = zInfo.data;
    uData = getMatrixInfo(args[pos++]).data;
    vData = getMatrixInfo(args[pos++]).data;
    wData = getMatrixInfo(args[pos++]).data;
    // Base grid: x along columns, y along rows (1-based). For a vector Z,
    // x = 1..n and y = 1. For an N-D Z, treat the trailing dimensions as
    // additional columns so every element gets a base position.
    const n = zData.length;
    if (rows === 1 || cols === 1) {
      xData = new Array(n);
      yData = new Array(n);
      for (let i = 0; i < n; i++) {
        xData[i] = i + 1;
        yData[i] = 1;
      }
    } else {
      const gen = generateMeshgrid(rows, n / rows);
      xData = gen.x;
      yData = gen.y;
    }
  } else {
    const X = args[pos++];
    const Y = args[pos++];
    const zInfo = getMatrixInfo(args[pos++]);
    rows = zInfo.rows;
    cols = zInfo.cols;
    zData = zInfo.data;
    uData = getMatrixInfo(args[pos++]).data;
    vData = getMatrixInfo(args[pos++]).data;
    wData = getMatrixInfo(args[pos++]).data;
    // One arrow per element: when X and Y already have as many elements as
    // U (full matrices/N-D arrays), use them directly. Only expand when they
    // are the shorter meshgrid vectors (size [length(Y)] / [length(X)]).
    const n = uData.length;
    const xFlat = toNumberArray(X);
    const yFlat = toNumberArray(Y);
    if (xFlat.length === n && yFlat.length === n) {
      xData = xFlat;
      yData = yFlat;
    } else {
      const expanded = expandXY(X, Y, rows, cols);
      xData = expanded.x;
      yData = expanded.y;
    }
  }

  let autoScale = true;
  let autoScaleFactor = 0.9; // internal multiplier (≈ reported × 0.9)
  let reportedASF = 0.9; // MATLAB AutoScaleFactor (default 0.9)

  // Trailing positional numeric scale.
  if (pos < args.length && isNumericArg(args[pos]) && !isStringArg(args[pos])) {
    const s = toNumber(args[pos] as RuntimeValue);
    if (s === 0) {
      autoScale = false;
    } else {
      autoScaleFactor = s * 0.9;
      reportedASF = s;
    }
    pos++;
  }

  const trace: Quiver3Trace = {
    x: xData,
    y: yData,
    z: zData,
    u: uData,
    v: vData,
    w: wData,
    showArrowHead: true,
  };

  // Trailing string args: 'off' (scale), LineSpec, 'filled', or a color name.
  while (
    pos < args.length &&
    isStringArg(args[pos]) &&
    !isQuiverNameValueKey(args[pos])
  ) {
    const s = getStringIfString(args[pos]);
    if (s === undefined) break;
    if (s === "off") {
      autoScale = false;
      pos++;
      continue;
    }
    if (s === "filled") {
      trace.markerFilled = true;
      pos++;
      continue;
    }
    const spec = parseLineSpec(s);
    if (spec) {
      if (spec.color) trace.color = COLOR_SHORT[spec.color];
      if (spec.lineStyle) trace.lineStyle = spec.lineStyle;
      if (spec.marker) {
        trace.marker = spec.marker;
        trace.showArrowHead = false; // LineSpec marker hides arrowheads
      }
      pos++;
      continue;
    }
    const c = resolveColor(s);
    if (c) {
      trace.color = c;
      pos++;
      continue;
    }
    break;
  }

  // Name-Value pairs.
  while (pos < args.length) {
    const key = isQuiverNameValueKey(args[pos]);
    if (!key) break;
    pos++;
    if (pos >= args.length) break;
    const value = args[pos++];
    switch (key) {
      case "color": {
        const c = resolveColor(value);
        if (c) trace.color = c;
        break;
      }
      case "linestyle":
        trace.lineStyle = getStringValue(value);
        break;
      case "linewidth":
        trace.lineWidth = typeof value === "number" ? value : toNumber(value);
        break;
      case "marker": {
        const s = getStringValue(value);
        trace.marker = s === "none" ? undefined : s;
        break;
      }
      case "showarrowhead": {
        const s = getStringValue(value).toLowerCase();
        trace.showArrowHead = !(s === "off" || s === "false" || s === "0");
        break;
      }
      case "autoscale": {
        const s = getStringValue(value).toLowerCase();
        autoScale = !(s === "off" || s === "false" || s === "0");
        break;
      }
      case "autoscalefactor": {
        const n = typeof value === "number" ? value : toNumber(value);
        autoScaleFactor = n * 0.9;
        reportedASF = n;
        break;
      }
    }
  }

  trace.autoScale = autoScale;
  trace.autoScaleFactor = reportedASF;

  if (autoScale) {
    const factor = computeQuiver3AutoScale(
      xData,
      yData,
      zData,
      uData,
      vData,
      wData,
      autoScaleFactor
    );
    if (factor !== 1) {
      trace.u = uData.map(x => x * factor);
      trace.v = vData.map(x => x * factor);
      trace.w = wData.map(x => x * factor);
    }
  }

  return trace;
}

/** 3-D analogue of computeQuiverAutoScale: scale arrows so the longest is
 *  roughly `factor` times the characteristic point spacing in (x,y,z). */
function computeQuiver3AutoScale(
  x: number[],
  y: number[],
  z: number[],
  u: number[],
  v: number[],
  w: number[],
  factor: number
): number {
  const n = u.length;
  if (n === 0) return 1;

  const range = (a: number[]) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const t of a) {
      if (isFinite(t)) {
        if (t < lo) lo = t;
        if (t > hi) hi = t;
      }
    }
    return hi > lo ? hi - lo : 0;
  };
  const xr = range(x);
  const yr = range(y);
  const zr = range(z);
  // Bounding-box volume / N, falling back to lower dimensions when flat.
  const dims = [xr, yr, zr].filter(d => d > 0);
  let spacing: number;
  if (dims.length === 3) spacing = Math.cbrt((xr * yr * zr) / n);
  else if (dims.length === 2) spacing = Math.sqrt((dims[0] * dims[1]) / n);
  else if (dims.length === 1) spacing = dims[0] / Math.max(1, n - 1);
  else spacing = 1;
  spacing = spacing || 1;

  let maxMag = 0;
  for (let i = 0; i < n; i++) {
    const m = Math.sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
    if (isFinite(m) && m > maxMag) maxMag = m;
  }
  if (maxMag === 0) return 1;
  return (factor * spacing) / maxMag;
}
