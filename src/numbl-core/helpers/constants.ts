/**
 * Built-in constant values
 */

import { RuntimeValue, RTV } from "../runtime/index.js";

const CONSTANTS: Record<string, number> = {
  pi: Math.PI,
  inf: Infinity,
  Inf: Infinity,
  nan: NaN,
  NaN: NaN,
  realmax: Number.MAX_VALUE,
  realmin: 2.2250738585072014e-308,
};

export function getConstant(name: string): RuntimeValue | undefined {
  if (name === "true") return RTV.logical(true);
  if (name === "false") return RTV.logical(false);
  if (name in CONSTANTS) return RTV.num(CONSTANTS[name]);
  if (name === "i" || name === "j") {
    return RTV.complex(0, 1);
  }
  return undefined;
}

export function getAllConstantNames(): string[] {
  return [...Object.keys(CONSTANTS), "true", "false", "i", "j"];
}
