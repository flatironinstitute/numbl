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
  FloatXArray,
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
  parseContourArgs,
  parseMeshArgs,
  parseBarArgs,
  parseBarHArgs,
  parseBar3Args,
  parseStairsArgs,
  parseErrorBarArgs,
  parseAreaArgs,
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
    | { type: "set_colorbar"; value: unknown }
    | { type: "set_colormap"; name: unknown }
    | { type: "set_axis"; value: unknown }
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
    case "set_colorbar":
      plotInstructions.push({
        type: "set_colorbar",
        value: resolveStr(instr.value),
      });
      break;
    case "set_colormap":
      plotInstructions.push({
        type: "set_colormap",
        name: resolveStr(instr.name).replace(/^"|"$/g, ""),
      });
      break;
    case "set_axis":
      plotInstructions.push({
        type: "set_axis",
        value: resolveStr(instr.value).replace(/^"|"$/g, ""),
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
