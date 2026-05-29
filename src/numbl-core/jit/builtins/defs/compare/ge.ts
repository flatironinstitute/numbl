import type { Builtin } from "../../registry.js";
import { defineCompare } from "./_compare.js";

export const ge: Builtin = defineCompare("ge", ">=", (a, b) => a >= b);
