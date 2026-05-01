/**
 * Tensor shape/indexing utilities and the deep-clone helper used at every
 * value-semantics seam (function call boundary, var-to-var assignment).
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  type RuntimeCell,
  type RuntimeStruct,
  type RuntimeStructArray,
  type RuntimeClassInstance,
  type RuntimeClassInstanceArray,
  type RuntimeSparseMatrix,
  type RuntimeDictionary,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeCell,
  isRuntimeTensor,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeSparseMatrix,
  isRuntimeDictionary,
} from "./types.js";
import { copyFloat64, disposeFloat64, disposeFloatX } from "./alloc.js";

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

// ── Deep clone ──────────────────────────────────────────────────────────

/**
 * Recursively deep-clone a RuntimeValue so that the result can be mutated
 * without affecting the original. Handle classes, function handles, dummy
 * handles, and graphics handles are deliberately NOT cloned — they have
 * reference semantics.
 *
 * Used at every value-semantics seam: function call entry (caller's args
 * become callee's locals) and plain var-to-var assignment (`a = b`).
 *
 * This is the safe-but-inefficient replacement for the previous COW
 * machinery — every assignment / call gets its own buffer.
 */
export function deepCloneValue(v: RuntimeValue): RuntimeValue {
  if (v === null || v === undefined) return v;
  if (typeof v !== "object") return v;
  if (isRuntimeTensor(v)) return cloneTensor(v);
  if (isRuntimeCell(v)) return cloneCell(v);
  if (isRuntimeStruct(v)) return cloneStruct(v);
  if (isRuntimeStructArray(v)) return cloneStructArray(v);
  if (isRuntimeSparseMatrix(v)) return cloneSparse(v);
  if (isRuntimeDictionary(v)) return cloneDictionary(v);
  if (isRuntimeClassInstance(v)) {
    if (v.isHandleClass) return v;
    return cloneClassInstance(v);
  }
  if (isRuntimeClassInstanceArray(v)) {
    if (v.elements.length > 0 && v.elements[0].isHandleClass) return v;
    return cloneClassInstanceArray(v);
  }
  // RuntimeChar (immutable string), RuntimeComplexNumber (primitive fields),
  // RuntimeFunction (closure — keep reference), RuntimeDummyHandle,
  // RuntimeGraphicsHandle: pass through.
  return v;
}

function cloneTensor(t: RuntimeTensor): RuntimeTensor {
  const out: RuntimeTensor = {
    kind: "tensor",
    data: t.data.slice() as typeof t.data,
    shape: t.shape.slice(),
  };
  if (t.imag) out.imag = t.imag.slice() as typeof t.imag;
  if (t._isLogical) out._isLogical = true;
  return out;
}

function cloneCell(c: RuntimeCell): RuntimeCell {
  const data = new Array(c.data.length);
  for (let i = 0; i < c.data.length; i++) data[i] = deepCloneValue(c.data[i]);
  return { kind: "cell", data, shape: c.shape.slice() };
}

function cloneStruct(s: RuntimeStruct): RuntimeStruct {
  const fields = new Map<string, RuntimeValue>();
  for (const [k, v] of s.fields) fields.set(k, deepCloneValue(v));
  return { kind: "struct", fields };
}

function cloneStructArray(sa: RuntimeStructArray): RuntimeStructArray {
  return {
    kind: "struct_array",
    fieldNames: sa.fieldNames.slice(),
    elements: sa.elements.map(cloneStruct),
  };
}

function cloneSparse(s: RuntimeSparseMatrix): RuntimeSparseMatrix {
  return {
    kind: "sparse_matrix",
    m: s.m,
    n: s.n,
    ir: new Int32Array(s.ir),
    jc: new Int32Array(s.jc),
    pr: copyFloat64(s.pr),
    pi: s.pi ? copyFloat64(s.pi) : undefined,
  };
}

function cloneDictionary(d: RuntimeDictionary): RuntimeDictionary {
  const entries = new Map<string, { key: RuntimeValue; value: RuntimeValue }>();
  for (const [k, e] of d.entries) {
    entries.set(k, {
      key: deepCloneValue(e.key),
      value: deepCloneValue(e.value),
    });
  }
  return {
    kind: "dictionary",
    entries,
    keyType: d.keyType,
    valueType: d.valueType,
  };
}

function cloneClassInstance(ci: RuntimeClassInstance): RuntimeClassInstance {
  const fields = new Map<string, RuntimeValue>();
  for (const [k, v] of ci.fields) fields.set(k, deepCloneValue(v));
  const out: RuntimeClassInstance = {
    kind: "class_instance",
    className: ci.className,
    fields,
    isHandleClass: false,
  };
  if (ci._builtinData) out._builtinData = deepCloneValue(ci._builtinData);
  return out;
}

function cloneClassInstanceArray(
  cia: RuntimeClassInstanceArray
): RuntimeClassInstanceArray {
  return {
    kind: "class_instance_array",
    className: cia.className,
    elements: cia.elements.map(cloneClassInstance),
  };
}

/** Back-compat alias — kept so existing call sites keep working while we
 *  finish the migration off the COW API name. */
export const shareRuntimeValue = deepCloneValue;

// ── Dispose ─────────────────────────────────────────────────────────────

/**
 * Recursively dispose a RuntimeValue, returning every dense float buffer
 * inside it to the allocator pool. Caller asserts the value (and every
 * tensor / buffer reachable from it) has no other live references — a
 * stray alias becomes a use-after-free once the pool hands the buffer
 * out again.
 *
 * Skipped (left to GC):
 *   - Handle-class instances (and arrays of them): shared by reference.
 *   - Function handles: closure may keep the captured snapshot live.
 *   - Graphics handles, dummy handles: refer to host-side state.
 *   - RuntimeChar (no buffer), primitives, complex numbers (numeric fields).
 */
export function disposeValue(v: RuntimeValue): void {
  if (v === null || v === undefined) return;
  if (typeof v !== "object") return;
  if (isRuntimeTensor(v)) {
    disposeFloatX(v.data);
    if (v.imag) disposeFloatX(v.imag);
    return;
  }
  if (isRuntimeCell(v)) {
    for (const e of v.data) disposeValue(e);
    return;
  }
  if (isRuntimeStruct(v)) {
    for (const fv of v.fields.values()) disposeValue(fv);
    return;
  }
  if (isRuntimeStructArray(v)) {
    for (const e of v.elements) disposeValue(e);
    return;
  }
  if (isRuntimeSparseMatrix(v)) {
    disposeFloat64(v.pr);
    if (v.pi) disposeFloat64(v.pi);
    return;
  }
  if (isRuntimeDictionary(v)) {
    for (const e of v.entries.values()) {
      disposeValue(e.key);
      disposeValue(e.value);
    }
    return;
  }
  if (isRuntimeClassInstance(v)) {
    if (v.isHandleClass) return;
    for (const fv of v.fields.values()) disposeValue(fv);
    if (v._builtinData) disposeValue(v._builtinData);
    return;
  }
  if (isRuntimeClassInstanceArray(v)) {
    if (v.elements.length > 0 && v.elements[0].isHandleClass) return;
    for (const e of v.elements) disposeValue(e);
    return;
  }
  // RuntimeChar, RuntimeComplexNumber, RuntimeFunction, RuntimeDummyHandle,
  // RuntimeGraphicsHandle: nothing to dispose.
}
