/**
 * Argument parsers for gridded data: surf, mesh, pcolor, imagesc, contour
 * (including the MATLAB contour matrix), and heatmap.
 */

import {
  type RuntimeValue,
  isRuntimeString,
  isRuntimeChar,
} from "../runtime/types.js";
import { toNumber } from "../runtime/convert.js";
import type {
  SurfTrace,
  ImagescTrace,
  PcolorTrace,
  ContourTrace,
  HeatmapTrace,
} from "./types.js";
import {
  expandXY,
  generateMeshgrid,
  getMatrixInfo,
  getStringValue,
  getStringValueIfString,
  isNumericArg,
  isStringArg,
  resolveColor,
  toNumberArray,
} from "./parseArgsCommon.js";

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

// ── Imagesc argument parser ──────────────────────────────────────────────

/**
 * Parse imagesc() arguments.
 *
 * Supported forms:
 *   imagesc(C)         — C is m×n matrix
 *   imagesc(x, y, C)   — x and y are vectors defining axis limits
 *   imagesc(..., clims) — clims = [low high] sets the axes' color limits
 */
export function parseImagescArgs(args: RuntimeValue[]): {
  trace: ImagescTrace;
  clims?: [number, number];
} {
  if (args.length === 2 || args.length === 4) {
    const c = toNumberArray(args[args.length - 1]);
    if (c.length !== 2 || !(c[0] < c[1]))
      throw new Error("imagesc: clims must be a two-element vector [low high]");
    return {
      trace: parseImagescTrace(args.slice(0, -1)),
      clims: [c[0], c[1]],
    };
  }
  return { trace: parseImagescTrace(args) };
}

function parseImagescTrace(args: RuntimeValue[]): ImagescTrace {
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
