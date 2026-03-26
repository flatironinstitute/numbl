import type { PlotInstruction } from "./types.js";

/** Restore NaN/Infinity values that were converted to null by JSON serialization. */
function restoreNullsToNaN(arr: number[]): void {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === null) (arr as number[])[i] = NaN;
  }
}

export function restoreNaNs(instr: PlotInstruction): void {
  if (instr.type === "plot") {
    for (const t of instr.traces) {
      restoreNullsToNaN(t.x);
      restoreNullsToNaN(t.y);
    }
  } else if (instr.type === "plot3") {
    for (const t of instr.traces) {
      restoreNullsToNaN(t.x);
      restoreNullsToNaN(t.y);
      restoreNullsToNaN(t.z);
    }
  } else if (instr.type === "surf" || instr.type === "mesh") {
    restoreNullsToNaN(instr.trace.x);
    restoreNullsToNaN(instr.trace.y);
    restoreNullsToNaN(instr.trace.z);
    if (instr.trace.c) restoreNullsToNaN(instr.trace.c);
  } else if (instr.type === "imagesc") {
    restoreNullsToNaN(instr.trace.z);
  } else if (instr.type === "contour") {
    restoreNullsToNaN(instr.trace.x);
    restoreNullsToNaN(instr.trace.y);
    restoreNullsToNaN(instr.trace.z);
  }
}
