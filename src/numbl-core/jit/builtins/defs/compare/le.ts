import type { Builtin } from "../../registry.js";
import { defineCompare } from "./_compare.js";

export const le: Builtin = defineCompare("le", "<=", (a, b) => a <= b);
