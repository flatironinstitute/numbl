/**
 * Value conversion helpers and equality.
 */

import {
  isRuntimeChar,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  kstr,
  type RuntimeValue,
} from "./types.js";
import { RuntimeError } from "./error.js";

/** Extract a JS number from an RuntimeValue (scalar num, logical, or 1x1 tensor) */
export function toNumber(v: RuntimeValue): number {
  if (isRuntimeNumber(v)) return v;
  if (isRuntimeLogical(v)) return v ? 1 : 0;
  if (isRuntimeChar(v)) {
    if (v.value.length === 1) return v.value.charCodeAt(0);
    throw new RuntimeError(`Cannot convert multi-char to number`);
  }
  if (isRuntimeString(v)) {
    const n = Number(v);
    if (!isNaN(n)) return n;
    throw new RuntimeError(`Cannot convert string "${v}" to number`);
  }
  switch (v.kind) {
    case "tensor":
      if (v.data.length === 1) return v.data[0];
      throw new RuntimeError("Cannot convert non-scalar tensor to number");
    case "complex_number":
      if (v.im !== 0)
        throw new RuntimeError(
          "Complex value cannot be converted to real number"
        );
      return v.re;
    default:
      throw new RuntimeError(`Cannot convert ${v.kind} to number`);
  }
}

/** Extract a boolean from an RuntimeValue */
export function toBool(v: RuntimeValue): boolean {
  if (isRuntimeNumber(v)) {
    return v !== 0;
  }
  if (isRuntimeLogical(v)) {
    return v;
  }
  if (isRuntimeString(v)) {
    return v.length > 0;
  }
  switch (v.kind) {
    case "tensor": {
      // All elements must be nonzero for truthy
      if (!v.imag) {
        for (let i = 0; i < v.data.length; i++) {
          if (v.data[i] === 0) return false;
        }
      } else {
        for (let i = 0; i < v.data.length; i++) {
          if (v.data[i] === 0 && v.imag[i] === 0) return false;
        }
      }
      return v.data.length > 0;
    }
    case "sparse_matrix": {
      // All m*n elements must be nonzero — true only when nnz == m*n
      const nnz = v.jc[v.n];
      return v.m * v.n > 0 && nnz === v.m * v.n;
    }
    case "char":
      return v.value.length > 0;
    case "complex_number":
      return v.re !== 0 || v.im !== 0;
    default:
      return true;
  }
}

/** Extract a string from an RuntimeValue */
export function toString(v: RuntimeValue): string {
  if (isRuntimeString(v)) {
    return v;
  }
  if (isRuntimeNumber(v)) {
    return String(v);
  }
  if (isRuntimeLogical(v)) {
    return v ? "1" : "0";
  }
  switch (v.kind) {
    case "char":
      return v.value;
    default:
      throw new RuntimeError(`Cannot convert ${kstr(v)} to string`);
  }
}
