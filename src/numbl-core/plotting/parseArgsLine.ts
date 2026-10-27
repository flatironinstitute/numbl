/**
 * Argument parsers for line-like plots: plot, plot3, line, scatter, scatter3,
 * stairs, errorbar, and area.
 *
 * Handles: plot(Y), plot(X,Y), plot(X1,Y1,...,Xn,Yn),
 * plot(X1,Y1,LineSpec1,...), and Name-Value pairs like 'Color','r','LineWidth',2.
 */

import {
  type RuntimeValue,
  isRuntimeTensor,
  isRuntimeNumber,
} from "../runtime/types.js";
import { toNumber } from "../runtime/convert.js";
import type { PlotTrace, Plot3Trace, ErrorBarTrace } from "./types.js";
import {
  COLOR_SHORT,
  type ParsedLineSpec,
  expandColumns,
  getStringValue,
  isAxesHandleArg,
  isNameValueKey,
  isNumericArg,
  isStringArg,
  numColumns,
  oneBasedIndices,
  parseLineSpec,
  resolveColor,
  toNumberArray,
} from "./parseArgsCommon.js";

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
