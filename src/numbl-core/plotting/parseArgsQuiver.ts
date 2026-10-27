/**
 * Argument parsers for vector fields: quiver and quiver3.
 */

import {
  type RuntimeValue,
  isRuntimeString,
  isRuntimeChar,
} from "../runtime/types.js";
import { toNumber } from "../runtime/convert.js";
import type { QuiverTrace, Quiver3Trace } from "./types.js";
import {
  COLOR_SHORT,
  expandXY,
  generateMeshgrid,
  getMatrixInfo,
  getStringValue,
  isNumericArg,
  isStringArg,
  parseLineSpec,
  resolveColor,
  toNumberArray,
} from "./parseArgsCommon.js";

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
