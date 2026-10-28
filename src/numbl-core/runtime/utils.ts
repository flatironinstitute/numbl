/**
 * Tensor shape/indexing utilities.
 */

import {
  type RuntimeTensor,
  type RuntimeValue,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeClassInstanceArray,
  isRuntimeSparseMatrix,
  isRuntimeStringArray,
  isRuntimeStructArray,
  isRuntimeTensor,
} from "./types.js";

// ── Tensor shape utilities ──────────────────────────────────────────────

/** The MATLAB size of any runtime value, with at least two dimensions.
 *  Scalars, strings, structs and objects are 1×1. */
export function valueShape(v: RuntimeValue): number[] {
  if (isRuntimeTensor(v)) {
    if (v.shape.length >= 2) return v.shape;
    return v.shape.length === 1 ? [1, v.shape[0]] : [1, 1];
  }
  if (isRuntimeChar(v))
    return v.shape ?? (v.value.length === 0 ? [0, 0] : [1, v.value.length]);
  if (isRuntimeSparseMatrix(v)) return [v.m, v.n];
  if (isRuntimeCell(v)) return v.shape;
  if (isRuntimeStructArray(v)) return v.shape;
  if (isRuntimeClassInstanceArray(v)) return [...v.shape];
  if (isRuntimeStringArray(v)) return [...v.shape];
  return [1, 1];
}

/** Get 2D size (rows, cols) from shape, padding with 1s as needed.
 *  For N-D tensors (N>2), trailing dimensions are collapsed into cols,
 *  matching MATLAB's behavior when fewer subscripts than dimensions. */
export function tensorSize2D(t: RuntimeTensor): [number, number] {
  const s = t.shape;
  if (s.length === 0) return [1, 1];
  if (s.length === 1) return [1, s[0]];
  if (s.length === 2) {
    return [s[0], s[1]];
  }
  // N-D: collapse trailing dims into cols
  let cols = 1;
  for (let i = 1; i < s.length; i++) cols *= s[i];
  return [s[0], cols];
}

/** Compute total number of elements from shape */
export function numel(shape: number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/** Column-major index for 2D: element at (row, col) in a matrix with `rows` rows */
export function colMajorIndex(row: number, col: number, rows: number): number {
  return col * rows + row;
}

/** Convert linear index to subscripts (column-major) */
export function ind2sub(shape: number[], idx: number): number[] {
  const subs: number[] = [];
  let remaining = idx;
  for (const dim of shape) {
    subs.push(remaining % dim);
    remaining = Math.floor(remaining / dim);
  }
  return subs;
}

/** Convert subscripts to linear index (column-major) */
export function sub2ind(shape: number[], subs: number[]): number {
  let idx = 0;
  let stride = 1;
  for (let i = 0; i < shape.length; i++) {
    idx += subs[i] * stride;
    stride *= shape[i];
  }
  return idx;
}

/** MATLAB num2str/string(double) default formatting: integers exact,
 *  otherwise 5 significant digits (%.5g with MATLAB exponent padding). */
export function matlabNumToString(n: number): string {
  if (n === Infinity) return "Inf";
  if (n === -Infinity) return "-Inf";
  if (isNaN(n)) return "NaN";
  if (n === 0) return "0";
  if (Number.isInteger(n)) return String(n);
  const prec = 5;
  const exp = Math.floor(Math.log10(Math.abs(n)));
  let s: string;
  if (exp < -4 || exp >= prec) {
    s = n.toExponential(prec - 1);
    const ePos = s.indexOf("e");
    let mantissa = s.slice(0, ePos);
    const expPart0 = s.slice(ePos);
    if (mantissa.includes(".")) mantissa = mantissa.replace(/\.?0+$/, "");
    const expPart = expPart0.replace(/([eE][+-])(\d)$/, "$1" + "0$2");
    s = mantissa + expPart;
  } else {
    s = n.toPrecision(prec);
    if (s.includes(".")) s = s.replace(/\.?0+$/, "");
  }
  return s;
}
