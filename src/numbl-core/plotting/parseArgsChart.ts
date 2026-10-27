/**
 * Argument parsers for statistical charts: bar, barh, bar3, histogram,
 * histogram2, boxchart, swarmchart, and piechart/donutchart.
 */

import {
  type RuntimeValue,
  isRuntimeTensor,
  isRuntimeNumber,
} from "../runtime/types.js";
import type {
  PlotTrace,
  Plot3Trace,
  BarTrace,
  Bar3Trace,
  BoxTrace,
  PieTrace,
} from "./types.js";
import {
  getMatrixInfo,
  getStringValue,
  isNumericArg,
  isStringArg,
  oneBasedIndices,
  resolveColor,
  tensorColumn,
  tensorFromArray,
  toNumberArray,
} from "./parseArgsCommon.js";
import { parseScatter3Args, parseScatterArgs } from "./parseArgsLine.js";

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
