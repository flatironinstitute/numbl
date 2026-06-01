import type { Builtin } from "../../registry.js";
import { defineCompare } from "./_compare.js";

export const lt: Builtin = defineCompare("lt", "<", (a, b) => a < b);
