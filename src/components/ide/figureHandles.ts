import type { FiguresState } from "../../numbl-core/plotting/figuresReducer.js";

/** Figure handles in tab order. */
export function sortedHandles(figs: FiguresState["figs"]): number[] {
  return Object.keys(figs)
    .map(Number)
    .sort((a, b) => a - b);
}
