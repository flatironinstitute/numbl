/**
 * Plot-related operations for the runtime.
 */

import { type RuntimeValue, toNumber, toString } from "../runtime/index.js";
import type { PlotInstruction } from "../../graphics/types.js";
import type { ExecOptions } from "../executeCode.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeFunction,
  isRuntimeCell,
  FloatXArray,
  type RuntimeTensor,
  type RuntimeCell,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import type { Runtime } from "./runtime.js";

/** Convert RuntimeValue to number array for plotting */
function runtimeValueToNumberArray(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeTensor(v)) {
    if (v.imag) {
      throw new Error("Cannot convert complex tensor to number array");
    }
    return Array.from(v.data);
  }
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  return [0];
}
import {
  parsePlotArgs,
  parsePlot3Args,
  parseSurfArgs,
  parseScatterArgs,
  parseImagescArgs,
  parsePcolorArgs,
  parseContourArgs,
  parseMeshArgs,
  parseBarArgs,
  parseBarHArgs,
  parseBar3Args,
  parseStairsArgs,
  parseErrorBarArgs,
  parseAreaArgs,
  parseScatter3Args,
  parseHistogramArgs,
  parseHistogram2Args,
  parseBoxchartArgs,
  parseSwarmchartArgs,
  parseSwarmchart3Args,
  parsePiechartArgs,
  parseHeatmapArgs,
} from "../runtime/plotUtils.js";
import { ensureRuntimeValue } from "./runtimeHelpers.js";
import { syncSleep } from "./syncChannel.js";

/** Resolve an unknown value (possibly a RuntimeValue) to a string. */
function resolveStr(val: unknown): string {
  return typeof val === "string" ? val : toString(ensureRuntimeValue(val));
}

/** Resolve an unknown value to a number, with a fallback. */
function resolveNum(val: unknown, fallback?: number): number {
  if (typeof val === "number") return val;
  try {
    return toNumber(ensureRuntimeValue(val));
  } catch {
    return fallback ?? 0;
  }
}

/** Resolve an on/off value to a boolean. */
function resolveOnOff(val: unknown): boolean {
  if (typeof val === "boolean") return val;
  return resolveStr(val) === "on";
}

export function plotInstr(
  plotInstructions: PlotInstruction[],
  instr:
    | { type: "set_figure_handle"; handle: unknown }
    | { type: "plot"; x: unknown; y: unknown }
    | { type: "set_hold"; value: unknown }
    | { type: "set_title"; text: unknown }
    | { type: "set_xlabel"; text: unknown }
    | { type: "set_ylabel"; text: unknown }
    | { type: "set_shading"; shading: unknown }
    | { type: "close" }
    | { type: "close_all" }
    | { type: "clf" }
    | { type: "set_subplot"; rows: unknown; cols: unknown; index: unknown }
    | { type: "set_sgtitle"; text: unknown }
    | { type: "set_grid"; value: unknown }
    | { type: "set_zlabel"; text: unknown }
    | { type: "set_colorbar"; value: unknown; location?: unknown }
    | { type: "set_colormap"; name: unknown; data?: number[][] }
    | { type: "set_axis"; value: unknown }
    | { type: "set_caxis"; limits: [number, number] }
): void {
  switch (instr.type) {
    case "set_figure_handle":
      plotInstructions.push({
        type: "set_figure_handle",
        handle: resolveNum(instr.handle, 1),
      });
      break;
    case "plot":
      plotInstructions.push({
        type: "plot",
        traces: [
          {
            x: runtimeValueToNumberArray(ensureRuntimeValue(instr.x)),
            y: runtimeValueToNumberArray(ensureRuntimeValue(instr.y)),
          },
        ],
      });
      break;
    case "set_hold":
      plotInstructions.push({
        type: "set_hold",
        value: resolveOnOff(instr.value),
      });
      break;
    case "set_title":
    case "set_xlabel":
    case "set_ylabel":
    case "set_zlabel":
    case "set_sgtitle":
      plotInstructions.push({ type: instr.type, text: resolveStr(instr.text) });
      break;
    case "set_shading": {
      const shading = resolveStr(instr.shading).replace(/^'|'$/g, "") as
        | "faceted"
        | "flat"
        | "interp";
      plotInstructions.push({ type: "set_shading", shading });
      break;
    }
    case "close":
    case "close_all":
    case "clf":
      plotInstructions.push({ type: instr.type });
      break;
    case "set_subplot":
      plotInstructions.push({
        type: "set_subplot",
        rows: resolveNum(instr.rows),
        cols: resolveNum(instr.cols),
        index: resolveNum(instr.index),
      });
      break;
    case "set_grid":
      plotInstructions.push({
        type: "set_grid",
        value: resolveOnOff(instr.value),
      });
      break;
    case "set_colorbar": {
      const cbInstr: {
        type: "set_colorbar";
        value: string;
        location?: string;
      } = {
        type: "set_colorbar",
        value: resolveStr(instr.value),
      };
      if (instr.location !== undefined) {
        cbInstr.location = resolveStr(instr.location);
      }
      plotInstructions.push(cbInstr);
      break;
    }
    case "set_colormap": {
      const cmInstr: PlotInstruction & { type: "set_colormap" } = {
        type: "set_colormap",
        name: resolveStr(instr.name).replace(/^"|"$/g, ""),
      };
      if (instr.data) cmInstr.data = instr.data;
      plotInstructions.push(cmInstr);
      break;
    }
    case "set_axis":
      plotInstructions.push({
        type: "set_axis",
        value: resolveStr(instr.value).replace(/^"|"$/g, ""),
      });
      break;
    case "set_caxis":
      plotInstructions.push({
        type: "set_caxis",
        limits: instr.limits,
      });
      break;
  }
}

export function viewCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  let az: number;
  let el: number;
  if (args.length === 1) {
    // view(3) → default 3D view: az=-37.5, el=30
    // view(2) → default 2D view: az=0, el=90
    const n = typeof args[0] === "number" ? args[0] : toNumber(args[0]);
    if (n === 2) {
      az = 0;
      el = 90;
    } else {
      az = -37.5;
      el = 30;
    }
  } else if (args.length >= 2) {
    az = typeof args[0] === "number" ? args[0] : toNumber(args[0]);
    el = typeof args[1] === "number" ? args[1] : toNumber(args[1]);
  } else {
    return;
  }
  plotInstructions.push({ type: "set_view", az, el });
}

export function plotCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parsePlotArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }
}

export function plot3Call(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parsePlot3Args(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot3", traces });
  }
}

export function surfCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parseSurfArgs(args);
  plotInstructions.push({ type: "surf", trace });
}

export function imagescCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parseImagescArgs(args);
  plotInstructions.push({ type: "imagesc", trace });
}

export function pcolorCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parsePcolorArgs(args);
  plotInstructions.push({ type: "pcolor", trace });
}

export function contourCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[],
  filled: boolean
): void {
  const trace = parseContourArgs(args, filled);
  plotInstructions.push({ type: "contour", trace });
}

export function meshCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parseMeshArgs(args);
  plotInstructions.push({ type: "mesh", trace });
}

export function scatterCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseScatterArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }
}

export function barCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseBarArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "bar", traces });
  }
}

export function barhCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseBarHArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "barh", traces });
  }
}

export function bar3Call(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parseBar3Args(args);
  plotInstructions.push({ type: "bar3", trace });
}

export function bar3hCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parseBar3Args(args);
  plotInstructions.push({ type: "bar3h", trace });
}

export function stairsCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseStairsArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }
}

export function errorbarCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseErrorBarArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "errorbar", traces });
  }
}

export function semilogxCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  plotInstructions.push({ type: "set_axis_scale", value: "semilogx" });
  const traces = parsePlotArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }
}

export function semilogyCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  plotInstructions.push({ type: "set_axis_scale", value: "semilogy" });
  const traces = parsePlotArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }
}

export function areaCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const { traces, baseValue } = parseAreaArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "area", traces, baseValue });
  }
}

export function loglogCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  plotInstructions.push({ type: "set_axis_scale", value: "loglog" });
  const traces = parsePlotArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }
}

/** Evaluate a function handle at a scalar value, returning a number. */
function evalFnAtScalar(rt: Runtime, fn: RuntimeValue, x: number): number {
  const result = rt.index(fn, [x], 1);
  if (typeof result === "number") return result;
  if (isRuntimeTensor(result as RuntimeValue)) {
    return (result as { data: Float64Array }).data[0];
  }
  return toNumber(ensureRuntimeValue(result));
}

/** Check if a value is a function handle or string (for fplot). */
function isFnArg(v: RuntimeValue): boolean {
  return isRuntimeFunction(v);
}

/** Check if a value is a numeric 2-element vector (interval). */
function is2ElementVector(v: RuntimeValue): boolean {
  if (isRuntimeTensor(v) && v.data.length === 2 && !v.imag) return true;
  return false;
}

/**
 * fplot(f), fplot(f, xinterval), fplot(funx, funy),
 * fplot(funx, funy, tinterval), fplot(..., LineSpec)
 */
export function fplotCall(
  rt: Runtime,
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  if (args.length === 0) throw new Error("fplot requires at least 1 argument");

  let pos = 0;
  const fn1 = args[pos++];
  if (!isFnArg(fn1))
    throw new Error("fplot: first argument must be a function handle");

  let fn2: RuntimeValue | undefined;
  let interval: [number, number] = [-5, 5];
  let restStart: number;

  // Check if second arg is also a function handle (parametric mode)
  if (pos < args.length && isFnArg(args[pos])) {
    fn2 = args[pos++];
    // Check for interval
    if (pos < args.length && is2ElementVector(args[pos])) {
      const iv = args[pos++];
      const arr = runtimeValueToNumberArray(iv);
      interval = [arr[0], arr[1]];
    }
    restStart = pos;
  } else {
    // Non-parametric: check for interval
    if (pos < args.length && is2ElementVector(args[pos])) {
      const iv = args[pos++];
      const arr = runtimeValueToNumberArray(iv);
      interval = [arr[0], arr[1]];
    }
    restStart = pos;
  }

  // Generate sample points
  const nPoints = 200;
  const t = new Float64Array(nPoints);
  for (let i = 0; i < nPoints; i++) {
    t[i] = interval[0] + ((interval[1] - interval[0]) * i) / (nPoints - 1);
  }

  // Evaluate function(s)
  const xArr = new Float64Array(nPoints);
  const yArr = new Float64Array(nPoints);

  if (fn2) {
    // Parametric: x = fn1(t), y = fn2(t)
    for (let i = 0; i < nPoints; i++) {
      xArr[i] = evalFnAtScalar(rt, fn1, t[i]);
      yArr[i] = evalFnAtScalar(rt, fn2, t[i]);
    }
  } else {
    // Non-parametric: x = t, y = fn1(t)
    for (let i = 0; i < nPoints; i++) {
      xArr[i] = t[i];
      yArr[i] = evalFnAtScalar(rt, fn1, t[i]);
    }
  }

  // Build synthetic args: X tensor, Y tensor, then any remaining LineSpec/Name-Value args
  const xTensor = RTV.tensor(new FloatXArray(xArr), [1, nPoints]);
  const yTensor = RTV.tensor(new FloatXArray(yArr), [1, nPoints]);
  const synthArgs: RuntimeValue[] = [
    xTensor,
    yTensor,
    ...args.slice(restStart),
  ];

  const traces = parsePlotArgs(synthArgs);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }
}

/**
 * fplot3(funx, funy, funz), fplot3(funx, funy, funz, tinterval),
 * fplot3(..., LineSpec)
 */
export function fplot3Call(
  rt: Runtime,
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  if (args.length < 3) throw new Error("fplot3 requires at least 3 arguments");

  let pos = 0;
  const funx = args[pos++];
  const funy = args[pos++];
  const funz = args[pos++];

  if (!isFnArg(funx) || !isFnArg(funy) || !isFnArg(funz)) {
    throw new Error("fplot3: first three arguments must be function handles");
  }

  let interval: [number, number] = [-5, 5];

  if (pos < args.length && is2ElementVector(args[pos])) {
    const iv = args[pos++];
    const arr = runtimeValueToNumberArray(iv);
    interval = [arr[0], arr[1]];
  }
  const restStart = pos;

  // Generate sample points
  const nPoints = 200;
  const tArr = new Float64Array(nPoints);
  for (let i = 0; i < nPoints; i++) {
    tArr[i] = interval[0] + ((interval[1] - interval[0]) * i) / (nPoints - 1);
  }

  // Evaluate functions
  const xArr = new Float64Array(nPoints);
  const yArr = new Float64Array(nPoints);
  const zArr = new Float64Array(nPoints);

  for (let i = 0; i < nPoints; i++) {
    xArr[i] = evalFnAtScalar(rt, funx, tArr[i]);
    yArr[i] = evalFnAtScalar(rt, funy, tArr[i]);
    zArr[i] = evalFnAtScalar(rt, funz, tArr[i]);
  }

  // Build synthetic args for plot3
  const xTensor = RTV.tensor(new FloatXArray(xArr), [1, nPoints]);
  const yTensor = RTV.tensor(new FloatXArray(yArr), [1, nPoints]);
  const zTensor = RTV.tensor(new FloatXArray(zArr), [1, nPoints]);
  const synthArgs: RuntimeValue[] = [
    xTensor,
    yTensor,
    zTensor,
    ...args.slice(restStart),
  ];

  const traces = parsePlot3Args(synthArgs);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot3", traces });
  }
}

export function scatter3Call(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseScatter3Args(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot3", traces });
  }
}

export function histogramCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseHistogramArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "bar", traces });
  }
}

export function histogram2Call(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parseHistogram2Args(args);
  if (trace.x.length > 0) {
    plotInstructions.push({ type: "bar3", trace });
  }
}

export function boxchartCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseBoxchartArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "boxchart", traces });
  }
}

export function swarmchartCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseSwarmchartArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }
}

export function swarmchart3Call(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseSwarmchart3Args(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "plot3", traces });
  }
}

export function piechartCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parsePiechartArgs(args, 0);
  if (trace.values.length > 0) {
    plotInstructions.push({ type: "piechart", trace });
  }
}

export function donutchartCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parsePiechartArgs(args, 0.5);
  if (trace.values.length > 0) {
    plotInstructions.push({ type: "piechart", trace });
  }
}

export function heatmapCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const trace = parseHeatmapArgs(args);
  plotInstructions.push({ type: "heatmap", trace });
}

export function legendCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  // Collect string arguments as labels, ignoring name-value pairs like 'Location','best'
  const labels: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const s = toString(args[i]);
    // Skip known name-value pairs
    if (
      s === "Location" ||
      s === "Orientation" ||
      s === "FontSize" ||
      s === "Box" ||
      s === "Color" ||
      s === "EdgeColor" ||
      s === "TextColor"
    ) {
      i++; // skip the value
      continue;
    }
    labels.push(s);
  }
  if (labels.length > 0) {
    plotInstructions.push({ type: "set_legend", labels });
  }
}

export function drawnow(
  plotInstructions: PlotInstruction[],
  options: ExecOptions
): void {
  if (options.onDrawnow && plotInstructions.length > 0) {
    options.onDrawnow([...plotInstructions]);
    plotInstructions.length = 0;
  }
}

export function pause(seconds: unknown): void {
  const s =
    typeof seconds === "number"
      ? seconds
      : toNumber(ensureRuntimeValue(seconds));
  syncSleep(s * 1000);
}

// ── Streamline helpers ────────────────────────────────────────────────

/** Extract a flat number[] from a RuntimeValue that is a number or tensor. */
function toNumArr(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  throw new Error("streamline: expected numeric array");
}

/**
 * Bilinear interpolation of a scalar field on a meshgrid.
 *
 * X, Y are the grid coordinates (column-major, rows × cols).
 * F is the field values on that grid.
 * Returns interpolated value at (px, py), or NaN if out of bounds.
 */
function interpGrid(
  xVec: number[],
  yVec: number[],
  F: number[],
  rows: number,
  cols: number,
  px: number,
  py: number
): number {
  // Find column index (x direction)
  let ci = -1;
  for (let j = 0; j < cols - 1; j++) {
    if (
      (xVec[j] <= px && px <= xVec[j + 1]) ||
      (xVec[j] >= px && px >= xVec[j + 1])
    ) {
      ci = j;
      break;
    }
  }
  if (ci < 0) return NaN;

  // Find row index (y direction)
  let ri = -1;
  for (let i = 0; i < rows - 1; i++) {
    if (
      (yVec[i] <= py && py <= yVec[i + 1]) ||
      (yVec[i] >= py && py >= yVec[i + 1])
    ) {
      ri = i;
      break;
    }
  }
  if (ri < 0) return NaN;

  // Bilinear weights
  const dx = xVec[ci + 1] - xVec[ci];
  const dy = yVec[ri + 1] - yVec[ri];
  if (dx === 0 || dy === 0) return NaN;
  const tx = (px - xVec[ci]) / dx;
  const ty = (py - yVec[ri]) / dy;

  // Column-major: index = row + col * rows
  const f00 = F[ri + ci * rows];
  const f10 = F[ri + 1 + ci * rows];
  const f01 = F[ri + (ci + 1) * rows];
  const f11 = F[ri + 1 + (ci + 1) * rows];

  return (
    f00 * (1 - tx) * (1 - ty) +
    f01 * tx * (1 - ty) +
    f10 * (1 - tx) * ty +
    f11 * tx * ty
  );
}

/**
 * Compute 2D streamline vertices from a meshgrid vector field.
 *
 * X, Y: grid coordinate arrays from meshgrid (rows × cols, column-major)
 * U, V: velocity component arrays (same layout)
 * startX, startY: starting position arrays
 * step: integration step size (default 0.1)
 * maxVert: max vertices per streamline (default 10000)
 *
 * Returns array of vertex lists, one per starting point.
 * Each vertex list is {x: number[], y: number[]}.
 */
function computeStreamlines2D(
  X: RuntimeTensor,
  Y: RuntimeTensor,
  U: RuntimeTensor,
  V: RuntimeTensor,
  startX: number[],
  startY: number[],
  step: number,
  maxVert: number
): { x: number[]; y: number[] }[] {
  const rows = X.shape[0];
  const cols = X.shape[1];

  // Extract unique x and y vectors from the meshgrid.
  // X varies along columns (same value in each row), Y varies along rows.
  const xVec: number[] = new Array(cols);
  for (let j = 0; j < cols; j++) xVec[j] = X.data[j * rows]; // first row
  const yVec: number[] = new Array(rows);
  for (let i = 0; i < rows; i++) yVec[i] = Y.data[i]; // first column

  const Udata = Array.from(U.data);
  const Vdata = Array.from(V.data);

  // Characteristic cell size for scaling the step
  const hx = Math.abs(xVec[cols - 1] - xVec[0]) / (cols - 1);
  const hy = Math.abs(yVec[rows - 1] - yVec[0]) / (rows - 1);
  const cellSize = Math.sqrt(hx * hx + hy * hy);

  const results: { x: number[]; y: number[] }[] = [];

  for (let k = 0; k < startX.length; k++) {
    const xs: number[] = [startX[k]];
    const ys: number[] = [startY[k]];
    let px = startX[k];
    let py = startY[k];

    for (let iter = 0; iter < maxVert - 1; iter++) {
      // RK4 integration
      const u1 = interpGrid(xVec, yVec, Udata, rows, cols, px, py);
      const v1 = interpGrid(xVec, yVec, Vdata, rows, cols, px, py);
      if (isNaN(u1) || isNaN(v1)) break;

      const speed1 = Math.sqrt(u1 * u1 + v1 * v1);
      if (speed1 === 0) break;
      const ds = step * cellSize;
      const dt1 = ds / speed1;

      const px2 = px + 0.5 * dt1 * u1;
      const py2 = py + 0.5 * dt1 * v1;
      const u2 = interpGrid(xVec, yVec, Udata, rows, cols, px2, py2);
      const v2 = interpGrid(xVec, yVec, Vdata, rows, cols, px2, py2);
      if (isNaN(u2) || isNaN(v2)) break;

      const px3 = px + 0.5 * dt1 * u2;
      const py3 = py + 0.5 * dt1 * v2;
      const u3 = interpGrid(xVec, yVec, Udata, rows, cols, px3, py3);
      const v3 = interpGrid(xVec, yVec, Vdata, rows, cols, px3, py3);
      if (isNaN(u3) || isNaN(v3)) break;

      const px4 = px + dt1 * u3;
      const py4 = py + dt1 * v3;
      const u4 = interpGrid(xVec, yVec, Udata, rows, cols, px4, py4);
      const v4 = interpGrid(xVec, yVec, Vdata, rows, cols, px4, py4);
      if (isNaN(u4) || isNaN(v4)) break;

      const dx = (dt1 / 6) * (u1 + 2 * u2 + 2 * u3 + u4);
      const dy = (dt1 / 6) * (v1 + 2 * v2 + 2 * v3 + v4);

      px += dx;
      py += dy;
      xs.push(px);
      ys.push(py);
    }

    results.push({ x: xs, y: ys });
  }

  return results;
}

/**
 * Parse streamline/stream2 arguments.
 *
 * Supported forms:
 *   streamline(X,Y,U,V,startX,startY)
 *   streamline(X,Y,U,V,startX,startY,options)
 *   streamline(verts)  — cell array of Nx2 vertex matrices
 */
function parseStreamlineArgs(args: RuntimeValue[]):
  | {
      mode: "field";
      X: RuntimeTensor;
      Y: RuntimeTensor;
      U: RuntimeTensor;
      V: RuntimeTensor;
      startX: number[];
      startY: number[];
      step: number;
      maxVert: number;
    }
  | {
      mode: "verts";
      verts: RuntimeCell;
    } {
  if (args.length === 1 && isRuntimeCell(args[0])) {
    return { mode: "verts", verts: args[0] };
  }

  if (args.length < 4) {
    throw new Error("streamline: expected (X,Y,U,V,startX,startY) or (verts)");
  }

  let step = 0.1;
  let maxVert = 10000;

  // Check for trailing options vector
  let nPositional = args.length;
  if (nPositional === 5 || nPositional === 7) {
    const last = args[nPositional - 1];
    if (
      isRuntimeTensor(last) &&
      last.data.length <= 2 &&
      last.data.length >= 1
    ) {
      step = last.data[0];
      if (last.data.length === 2) maxVert = last.data[1];
      nPositional--;
    }
  }

  if (nPositional === 4) {
    // streamline(U,V,startX,startY) — default grid coordinates
    const U = ensureRuntimeValue(args[0]) as RuntimeTensor;
    const V = ensureRuntimeValue(args[1]) as RuntimeTensor;
    if (!isRuntimeTensor(U) || !isRuntimeTensor(V)) {
      throw new Error("streamline: U and V must be numeric arrays");
    }
    const rows = U.shape[0];
    const cols = U.shape[1];
    // Default coordinates: X = 1:cols, Y = 1:rows (meshgrid convention)
    const xData = new FloatXArray(rows * cols);
    const yData = new FloatXArray(rows * cols);
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        xData[i + j * rows] = j + 1;
        yData[i + j * rows] = i + 1;
      }
    }
    const X = RTV.tensor(xData, [rows, cols]);
    const Y = RTV.tensor(yData, [rows, cols]);
    const startX = toNumArr(args[2]);
    const startY = toNumArr(args[3]);
    return { mode: "field", X, Y, U, V, startX, startY, step, maxVert };
  }

  if (nPositional >= 6) {
    // streamline(X,Y,U,V,startX,startY)
    const X = ensureRuntimeValue(args[0]) as RuntimeTensor;
    const Y = ensureRuntimeValue(args[1]) as RuntimeTensor;
    const U = ensureRuntimeValue(args[2]) as RuntimeTensor;
    const V = ensureRuntimeValue(args[3]) as RuntimeTensor;
    if (
      !isRuntimeTensor(X) ||
      !isRuntimeTensor(Y) ||
      !isRuntimeTensor(U) ||
      !isRuntimeTensor(V)
    ) {
      throw new Error("streamline: X, Y, U, V must be numeric arrays");
    }
    const startX = toNumArr(args[4]);
    const startY = toNumArr(args[5]);
    return { mode: "field", X, Y, U, V, startX, startY, step, maxVert };
  }

  throw new Error("streamline: expected (X,Y,U,V,startX,startY) or (verts)");
}

/**
 * streamline(X,Y,U,V,startX,startY) — compute and plot 2D streamlines.
 * streamline(verts) — plot from pre-computed cell array of vertex matrices.
 * Returns a dummy graphics handle.
 */
export function streamlineCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): RuntimeValue {
  const parsed = parseStreamlineArgs(args);

  if (parsed.mode === "verts") {
    // Plot from cell array of vertex matrices
    const traces = cellVertsToTraces(parsed.verts);
    if (traces.length > 0) {
      plotInstructions.push({ type: "plot", traces });
    }
    return RTV.dummyHandle();
  }

  // Compute streamlines
  const streamlines = computeStreamlines2D(
    parsed.X,
    parsed.Y,
    parsed.U,
    parsed.V,
    parsed.startX,
    parsed.startY,
    parsed.step,
    parsed.maxVert
  );

  const traces = streamlines
    .filter(s => s.x.length > 1)
    .map(s => ({ x: s.x, y: s.y }));

  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }

  return RTV.dummyHandle();
}

/** Convert a cell array of Nx2 (or Nx3) vertex matrices into PlotTraces. */
function cellVertsToTraces(cell: RuntimeCell): { x: number[]; y: number[] }[] {
  const traces: { x: number[]; y: number[] }[] = [];
  for (const elem of cell.data) {
    if (isRuntimeTensor(elem) && elem.shape.length === 2) {
      const nRows = elem.shape[0];
      const nCols = elem.shape[1];
      if (nCols >= 2 && nRows > 0) {
        const x: number[] = new Array(nRows);
        const y: number[] = new Array(nRows);
        for (let i = 0; i < nRows; i++) {
          x[i] = elem.data[i]; // column 0
          y[i] = elem.data[i + nRows]; // column 1
        }
        traces.push({ x, y });
      }
    }
  }
  return traces;
}

/**
 * stream2(X,Y,U,V,startX,startY) — compute 2D streamline vertices.
 * Returns a cell array of Nx2 vertex matrices.
 */
export function stream2Call(args: RuntimeValue[]): RuntimeValue {
  const parsed = parseStreamlineArgs(args);

  if (parsed.mode === "verts") {
    // Already vertices — just return them
    return parsed.verts;
  }

  const streamlines = computeStreamlines2D(
    parsed.X,
    parsed.Y,
    parsed.U,
    parsed.V,
    parsed.startX,
    parsed.startY,
    parsed.step,
    parsed.maxVert
  );

  // Convert to cell array of Nx2 matrices
  const cellData: RuntimeValue[] = streamlines.map(s => {
    const n = s.x.length;
    const data = new FloatXArray(n * 2);
    for (let i = 0; i < n; i++) {
      data[i] = s.x[i]; // column 0
      data[i + n] = s.y[i]; // column 1
    }
    return RTV.tensor(data, [n, 2]);
  });

  return RTV.cell(cellData, [1, cellData.length]);
}
