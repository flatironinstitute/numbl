/**
 * Tensor shape/indexing utilities and copy-on-write helpers.
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  type RuntimeCell,
  type RuntimeClassInstance,
  isRuntimeClassInstance,
  isRuntimeCell,
  isRuntimeTensor,
} from "./types.js";

// ── Tensor shape utilities ──────────────────────────────────────────────

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

// ── Copy-on-write helpers ───────────────────────────────────────────────

/**
 * Share an RuntimeValue for assignment (COW): increments refcount on tensors/cells
 * so that mIndexStore knows to copy before mutating.
 * Returns a new wrapper object that shares the same underlying data.
 */
export function shareRuntimeValue(v: RuntimeValue): RuntimeValue {
  if (isRuntimeTensor(v)) {
    v._rc++;
    return {
      kind: "tensor",
      data: v.data,
      imag: v.imag,
      shape: v.shape,
      _isLogical: v._isLogical,
      _rc: v._rc,
    };
  }
  if (isRuntimeCell(v)) {
    v._rc++;
    return {
      kind: "cell",
      data: v.data,
      shape: v.shape,
      _rc: v._rc,
    } as RuntimeCell;
  }
  // Value class instances: create a new instance with its own fields Map,
  // sharing each field value via shareRuntimeValue for COW semantics.
  if (isRuntimeClassInstance(v) && !v.isHandleClass) {
    const newFields = new Map<string, RuntimeValue>();
    for (const [k, fv] of v.fields) {
      newFields.set(k, shareRuntimeValue(fv));
    }
    const copy: RuntimeClassInstance = {
      kind: "class_instance",
      className: v.className,
      fields: newFields,
      isHandleClass: false,
    };
    if (v._builtinData) copy._builtinData = shareRuntimeValue(v._builtinData);
    return copy;
  }
  // Scalars, strings, logicals, functions, structs, handle classes are immutable or already copied on mutation
  return v;
}
