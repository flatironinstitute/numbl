/**
 * `and(a, b)` — the function form of `&`. See `or.ts` for why this
 * routes through the short-circuit scaffold.
 */

import { defineShortCircuit } from "./_shortcircuit.js";

export const andBuiltin = defineShortCircuit("and", "&", "and");
