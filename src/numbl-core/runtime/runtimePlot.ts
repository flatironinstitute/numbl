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
} from "../runtime/plotUtils.js";
import { ensureRuntimeValue } from "./runtimeHelpers.js";
import { syncSleep } from "./syncChannel.js";

export function plotInstr(
  plotInstructions: PlotInstruction[],
  instr:
    | { type: "set_figure_handle"; handle: unknown }
    | { type: "plot"; x: unknown; y: unknown }
    | { type: "set_hold"; value: unknown }
    | { type: "close" }
    | { type: "close_all" }
    | { type: "clf" }
): void {
  if (instr.type === "set_figure_handle") {
    plotInstructions.push({
      type: "set_figure_handle",
      handle:
        typeof instr.handle === "number"
          ? instr.handle
          : toNumber(ensureRuntimeValue(instr.handle)),
    });
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
  } else if (instr.type === "close") {
    plotInstructions.push({ type: "close" });
  } else if (instr.type === "close_all") {
    plotInstructions.push({ type: "close_all" });
  } else if (instr.type === "clf") {
    plotInstructions.push({ type: "clf" });
  }
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
