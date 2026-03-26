/**
 * Helper functions for value conversion and wrapping.
 */

import { RTV, RuntimeValue } from "../runtime/index.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
} from "../runtime/types.js";

/** Extract number from value if possible, for fast-path operations */
export function asNumber(v: unknown): number | null {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (v && typeof v === "object" && "kind" in v) {
    const mv = v as RuntimeValue;
    if (isRuntimeNumber(mv)) return mv;
    if (isRuntimeLogical(mv)) return mv ? 1 : 0;
  }
  return null;
}

/** Convert RuntimeValue to number array for plotting */
export function runtimeValueToNumberArray(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeTensor(v)) {
    if (v.imag) {
      throw new Error("Cannot convert complex tensor to number array");
    }
    return Array.from(v.data);
  }
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  return [0];
}

/** Wrap return value as RuntimeValue */
export function wrapReturnValue(r: unknown): RuntimeValue {
  if (typeof r === "number") return RTV.num(r);
  if (r === undefined) return RTV.num(0);
  return r as RuntimeValue;
}
