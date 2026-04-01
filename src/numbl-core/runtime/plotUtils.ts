/**
 * Utilities for parsing plot() arguments at runtime.
 *
 * Handles: plot(Y), plot(X,Y), plot(X1,Y1,...,Xn,Yn),
 * plot(X1,Y1,LineSpec1,...), and Name-Value pairs like 'Color','r','LineWidth',2.
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
  FloatXArray,
} from "./types.js";
import { toNumber, toString } from "./convert.js";

// Re-export trace types from the shared graphics types module.
export type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
  ImagescTrace,
  ContourTrace,
  BarTrace,
  Bar3Trace,
  ErrorBarTrace,
  BoxTrace,
} from "../../graphics/types.js";

import type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
  ImagescTrace,
  ContourTrace,
  BarTrace,
  Bar3Trace,
  ErrorBarTrace,
  BoxTrace,
} from "../../graphics/types.js";

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

function resolveColor(
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
    // contour(Z, N)
    const info = getMatrixInfo(args[pos++]);
    rows = info.rows;
    cols = info.cols;
    zData = info.data;
    nLevels =
      typeof args[pos] === "number"
        ? (args[pos] as number)
        : toNumber(args[pos]);
    pos++;
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
    // contour(X, Y, Z, N)
    const x = args[pos++];
    const y = args[pos++];
    const zInfo = getMatrixInfo(args[pos++]);
    rows = zInfo.rows;
    cols = zInfo.cols;
    zData = zInfo.data;
    nLevels =
      typeof args[pos] === "number"
        ? (args[pos] as number)
        : toNumber(args[pos]);
    pos++;
    const expanded = expandXY(x, y, rows, cols);
    xData = expanded.x;
    yData = expanded.y;
  } else {
    throw new Error("contour requires at least 1 argument");
  }

  // Skip name-value pairs
  return { x: xData, y: yData, z: zData, rows, cols, nLevels, filled };
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
    data: new FloatXArray(arr),
    shape: [1, arr.length],
  } as RuntimeTensor;
}
