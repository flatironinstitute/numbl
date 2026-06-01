import type { Builtin } from "../../registry.js";
import { defineCompare } from "./_compare.js";

export const eq: Builtin = defineCompare("eq", "==", (a, b) => a === b, "eq");
