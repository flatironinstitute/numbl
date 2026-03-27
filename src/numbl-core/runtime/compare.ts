/**
 * Value conversion helpers and equality.
 */

import {
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  type RuntimeValue,
} from "./types.js";

export const valuesAreEqual = (a: RuntimeValue, b: RuntimeValue): boolean => {
  if (isRuntimeNumber(a)) {
    if (!isRuntimeNumber(b)) return false;
    return a === b;
  }
  if (isRuntimeLogical(a)) {
    if (!isRuntimeLogical(b)) return false;
    return a === b;
  }
  if (isRuntimeString(a)) {
    if (!isRuntimeString(b)) return false;
    return a === b;
  }
  if (isRuntimeNumber(b) || isRuntimeLogical(b) || isRuntimeString(b))
    return false;
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case "char":
      return a.value === (b as typeof a).value;
    case "tensor": {
      const tb = b as typeof a;
      if (a.data.length !== tb.data.length) return false;
      for (let i = 0; i < a.data.length; i++) {
        if (a.data[i] !== tb.data[i]) return false;
      }
      if (!!a.imag !== !!tb.imag) return false;
      if (a.imag && tb.imag) {
        for (let i = 0; i < a.imag.length; i++) {
          if (a.imag[i] !== tb.imag[i]) return false;
        }
      }
      return true;
    }
    case "cell": {
      const cb = b as typeof a;
      if (a.data.length !== cb.data.length) return false;
      for (let i = 0; i < a.data.length; i++) {
        if (!valuesAreEqual(a.data[i], cb.data[i])) return false;
      }
      return true;
    }
    case "struct": {
      // not handling struct equality for now
      return a === b;
    }
    case "class_instance": {
      // not handling class instance equality for now
      return a === b;
    }
    case "complex_number": {
      const cb = b as typeof a;
      return a.re === cb.re && a.im === cb.im;
    }
    case "dictionary":
      return a === b;
    default:
      return false;
  }
};
