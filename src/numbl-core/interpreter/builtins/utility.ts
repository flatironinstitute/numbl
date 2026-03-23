/**
 * Utility builtins for JIT: isequal, assert, abs.
 */

import {
  isRuntimeComplexNumber,
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
} from "../../runtime/types.js";
import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
import { registerIBuiltin } from "./types.js";

// ── isequal ──────────────────────────────────────────────────────────────

function valuesEqualSimple(a: RuntimeValue, b: RuntimeValue): boolean {
  // number == number
  if (typeof a === "number" && typeof b === "number") return a === b;
  // logical coercion: logical true == 1, logical false == 0
  if (
    (isRuntimeNumber(a) || isRuntimeLogical(a)) &&
    (isRuntimeNumber(b) || isRuntimeLogical(b))
  ) {
    const av = isRuntimeLogical(a) ? (a ? 1 : 0) : (a as number);
    const bv = isRuntimeLogical(b) ? (b ? 1 : 0) : (b as number);
    return av === bv;
  }
  // complex == complex
  if (isRuntimeComplexNumber(a) && isRuntimeComplexNumber(b))
    return a.re === b.re && a.im === b.im;
  // complex == number
  if (isRuntimeComplexNumber(a) && typeof b === "number")
    return a.re === b && a.im === 0;
  if (typeof a === "number" && isRuntimeComplexNumber(b))
    return b.re === a && b.im === 0;
  // tensor == tensor
  if (isRuntimeTensor(a) && isRuntimeTensor(b)) return tensorsEqual(a, b);
  // tensor == scalar (1x1 tensor vs number)
  if (isRuntimeTensor(a) && typeof b === "number")
    return a.data.length === 1 && !a.imag && a.data[0] === b;
  if (typeof a === "number" && isRuntimeTensor(b))
    return b.data.length === 1 && !b.imag && b.data[0] === a;
  // tensor == complex scalar
  if (isRuntimeTensor(a) && isRuntimeComplexNumber(b))
    return (
      a.data.length === 1 &&
      a.data[0] === b.re &&
      (a.imag ? a.imag[0] === b.im : b.im === 0)
    );
  if (isRuntimeComplexNumber(a) && isRuntimeTensor(b))
    return (
      b.data.length === 1 &&
      b.data[0] === a.re &&
      (b.imag ? b.imag[0] === a.im : a.im === 0)
    );
  return false;
}

function tensorsEqual(a: RuntimeTensor, b: RuntimeTensor): boolean {
  if (a.data.length !== b.data.length) return false;
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return false;
  }
  for (let i = 0; i < a.data.length; i++) {
    if (a.data[i] !== b.data[i]) return false;
  }
  if (a.imag || b.imag) {
    if (!a.imag || !b.imag) return false;
    for (let i = 0; i < a.imag.length; i++) {
      if (a.imag[i] !== b.imag[i]) return false;
    }
  }
  return true;
}

registerIBuiltin({
  name: "isequal",
  typeRule: argTypes => {
    if (argTypes.length < 2) return null;
    return [{ kind: "number", nonneg: true }];
  },
  apply: args => {
    for (let i = 1; i < args.length; i++) {
      if (!valuesEqualSimple(args[0], args[i])) return false;
    }
    return true;
  },
});

// ── assert ───────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "assert",
  typeRule: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    return [{ kind: "number" }];
  },
  apply: args => {
    const v = args[0];
    let pass = false;
    if (typeof v === "boolean") pass = v;
    else if (typeof v === "number") pass = v !== 0;
    else if (isRuntimeLogical(v)) pass = v;
    else if (isRuntimeTensor(v)) {
      pass = v.data.length > 0;
      for (let i = 0; i < v.data.length; i++) {
        if (v.data[i] === 0) {
          pass = false;
          break;
        }
      }
    }
    if (!pass) {
      throw new Error(args.length > 1 ? String(args[1]) : "Assertion failed");
    }
    return 0;
  },
});
