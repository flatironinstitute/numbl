/**
 * Plot-related operations for the runtime.
 */

import { type RuntimeValue, toNumber, toString } from "../runtime/index.js";
import { runtimeValueToNumberArray } from "../executor/helpers.js";
import type { PlotInstruction, ExecOptions } from "../executor/types.js";
import {
  parsePlotArgs,
  parsePlot3Args,
  parseSurfArgs,
  parseScatterArgs,
  parseImagescArgs,
  parseContourArgs,
  parseMeshArgs,
} from "../runtime/plotUtils.js";
import { ensureRuntimeValue } from "./runtimeHelpers.js";
import { syncSleep } from "./syncChannel.js";

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
  if (instr.type === "set_figure_handle") {
    let handle: number;
    if (typeof instr.handle === "number") {
      handle = instr.handle;
    } else {
      // figure() may be called with name-value pairs like figure('Name', '...')
      // In that case the first arg is a string — just use default handle 1.
      try {
        handle = toNumber(ensureRuntimeValue(instr.handle));
      } catch {
        handle = 1;
      }
    }
    plotInstructions.push({ type: "set_figure_handle", handle });
  } else if (instr.type === "plot") {
    plotInstructions.push({
      type: "plot",
      traces: [
        {
          x: runtimeValueToNumberArray(ensureRuntimeValue(instr.x)),
          y: runtimeValueToNumberArray(ensureRuntimeValue(instr.y)),
        },
      ],
    });
  } else if (instr.type === "set_hold") {
    const val = instr.value;
    let on: boolean;
    if (typeof val === "string") on = val === "on";
    else {
      const mv = ensureRuntimeValue(val);
      on = toString(mv) === "on";
    }
    plotInstructions.push({ type: "set_hold", value: on });
  } else if (instr.type === "set_title") {
    const text =
      typeof instr.text === "string"
        ? instr.text
        : toString(ensureRuntimeValue(instr.text));
    plotInstructions.push({ type: "set_title", text });
  } else if (instr.type === "set_xlabel") {
    const text =
      typeof instr.text === "string"
        ? instr.text
        : toString(ensureRuntimeValue(instr.text));
    plotInstructions.push({ type: "set_xlabel", text });
  } else if (instr.type === "set_ylabel") {
    const text =
      typeof instr.text === "string"
        ? instr.text
        : toString(ensureRuntimeValue(instr.text));
    plotInstructions.push({ type: "set_ylabel", text });
  } else if (instr.type === "set_shading") {
    const raw =
      typeof instr.shading === "string"
        ? instr.shading
        : toString(ensureRuntimeValue(instr.shading));
    const shading = raw.replace(/^'|'$/g, "") as "faceted" | "flat" | "interp";
    plotInstructions.push({ type: "set_shading", shading });
  } else if (instr.type === "close") {
    plotInstructions.push({ type: "close" });
  } else if (instr.type === "close_all") {
    plotInstructions.push({ type: "close_all" });
  } else if (instr.type === "clf") {
    plotInstructions.push({ type: "clf" });
  } else if (instr.type === "set_subplot") {
    const rows =
      typeof instr.rows === "number"
        ? instr.rows
        : toNumber(ensureRuntimeValue(instr.rows));
    const cols =
      typeof instr.cols === "number"
        ? instr.cols
        : toNumber(ensureRuntimeValue(instr.cols));
    const index =
      typeof instr.index === "number"
        ? instr.index
        : toNumber(ensureRuntimeValue(instr.index));
    plotInstructions.push({ type: "set_subplot", rows, cols, index });
  } else if (instr.type === "set_sgtitle") {
    const text =
      typeof instr.text === "string"
        ? instr.text
        : toString(ensureRuntimeValue(instr.text));
    plotInstructions.push({ type: "set_sgtitle", text });
  } else if (instr.type === "set_grid") {
    const val = instr.value;
    let on: boolean;
    if (typeof val === "string") on = val === "on";
    else if (typeof val === "boolean") on = val;
    else {
      const mv = ensureRuntimeValue(val);
      on = toString(mv) === "on";
    }
    plotInstructions.push({ type: "set_grid", value: on });
  } else if (instr.type === "set_zlabel") {
    const text =
      typeof instr.text === "string"
        ? instr.text
        : toString(ensureRuntimeValue(instr.text));
    plotInstructions.push({ type: "set_zlabel", text });
  } else if (instr.type === "set_colorbar") {
    const val =
      typeof instr.value === "string"
        ? instr.value
        : toString(ensureRuntimeValue(instr.value));
    plotInstructions.push({ type: "set_colorbar", value: val });
  } else if (instr.type === "set_colormap") {
    const name =
      typeof instr.name === "string"
        ? instr.name
        : toString(ensureRuntimeValue(instr.name));
    plotInstructions.push({
      type: "set_colormap",
      name: name.replace(/^"|"$/g, ""),
    });
  } else if (instr.type === "set_axis") {
    const val =
      typeof instr.value === "string"
        ? instr.value
        : toString(ensureRuntimeValue(instr.value));
    plotInstructions.push({
      type: "set_axis",
      value: val.replace(/^"|"$/g, ""),
    });
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
