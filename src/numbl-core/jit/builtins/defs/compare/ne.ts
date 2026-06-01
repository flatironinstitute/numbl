import type { Builtin } from "../../registry.js";
import { defineCompare } from "./_compare.js";

export const ne: Builtin = defineCompare("ne", "!=", (a, b) => a !== b, "ne");
