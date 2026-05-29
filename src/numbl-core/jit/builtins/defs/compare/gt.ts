import type { Builtin } from "../../registry.js";
import { defineCompare } from "./_compare.js";

export const gt: Builtin = defineCompare("gt", ">", (a, b) => a > b);
