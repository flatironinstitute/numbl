/**
 * Entry point for all reduction-related builtins.
 */

import { registerBasicReductions } from "./basic-reductions.js";
import { registerMinMax } from "./min-max.js";
import { registerLogical } from "./logical.js";
import { registerCumulative } from "./cumulative.js";
import { registerSortUnique } from "./sort-unique.js";
import { registerSetOperations } from "./set-operations.js";

export function registerReductionFunctions(): void {
  registerBasicReductions();
  registerMinMax();
  registerLogical();
  registerCumulative();
  registerSortUnique();
  registerSetOperations();
}
