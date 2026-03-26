/**
 * Runtime helper types and standalone utility functions.
 *
 * Types and pure functions
 * These have no dependency on the Runtime class.
 */

import {
  type RuntimeValue,
  RTV,
  toNumber,
  valuesAreEqual,
  RuntimeError,
} from "../runtime/index.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeSparseMatrix,
  isRuntimeChar,
  isRuntimeString,
  FloatXArray,
} from "../runtime/types.js";
import { sparseToDense } from "../../numbl-core/helpers/sparse-arithmetic.js";
import { END_SENTINEL } from "./sentinels.js";

// ── Deferred Range ──────────────────────────────────────────────────────

export type DeferredRange = {
  _deferredRange: true;
  start: RuntimeValue | typeof END_SENTINEL;
  step: RuntimeValue | typeof END_SENTINEL;
  end: RuntimeValue | typeof END_SENTINEL;
};

export function isDeferredRange(v: unknown): v is DeferredRange {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as Record<string, unknown>)["_deferredRange"] === true
  );
}

export type DeferredHorzcat = {
  _deferredHorzcat: true;
  elems: (RuntimeValue | typeof END_SENTINEL | DeferredRange)[];
};

export function isDeferredHorzcat(v: unknown): v is DeferredHorzcat {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as Record<string, unknown>)["_deferredHorzcat"] === true
  );
}

// ── Helpers ─────────────────────────────────────────────────────────────

function isRuntimeValue(v: unknown): v is RuntimeValue {
  if (typeof v === "number") return true;
  if (typeof v === "boolean") return true;
  if (typeof v === "string") return true;
  if (v && typeof v === "object" && "kind" in v) return true;
  return false;
}

export function ensureRuntimeValue(v: unknown): RuntimeValue {
  if (v === undefined) return RTV.tensor(new FloatXArray(0), [0]);
  if (!isRuntimeValue(v)) {
    throw new RuntimeError(
      `Expected a runtime value, got ${JSON.stringify(v)}`
    );
  }
  return v;
}

export function switchValuesMatch(a: RuntimeValue, b: RuntimeValue): boolean {
  if (
    (isRuntimeChar(a) || isRuntimeString(a)) &&
    (isRuntimeChar(b) || isRuntimeString(b))
  ) {
    return (
      (typeof a === "string" ? a : a.value) ===
      (typeof b === "string" ? b : b.value)
    );
  }
  return valuesAreEqual(a, b);
}

export function isNumericKind(v: RuntimeValue): boolean {
  return (
    isRuntimeTensor(v) ||
    isRuntimeNumber(v) ||
    isRuntimeLogical(v) ||
    isRuntimeSparseMatrix(v)
  );
}

export function elementWiseLogicalOp(
  a: RuntimeValue,
  b: RuntimeValue,
  op: (x: number, y: number) => number
): RuntimeValue {
  if (isRuntimeSparseMatrix(a))
    return elementWiseLogicalOp(sparseToDense(a), b, op);
  if (isRuntimeSparseMatrix(b))
    return elementWiseLogicalOp(a, sparseToDense(b), op);
  const aIsT = isRuntimeTensor(a);
  const bIsT = isRuntimeTensor(b);
  const aData = aIsT ? a.data : new FloatXArray([toNumber(a)]);
  const bData = bIsT ? b.data : new FloatXArray([toNumber(b)]);
  const aShape = aIsT ? a.shape : [1, 1];
  const bShape = bIsT ? b.shape : [1, 1];
  if (aData.length === bData.length) {
    const result = new FloatXArray(aData.length);
    for (let i = 0; i < aData.length; i++) result[i] = op(aData[i], bData[i]);
    const t = RTV.tensor(result, aIsT ? aShape : bShape);
    t._isLogical = true;
    return t;
  }
  if (aData.length === 1) {
    const result = new FloatXArray(bData.length);
    for (let i = 0; i < bData.length; i++) result[i] = op(aData[0], bData[i]);
    const t = RTV.tensor(result, bShape);
    t._isLogical = true;
    return t;
  }
  if (bData.length === 1) {
    const result = new FloatXArray(aData.length);
    for (let i = 0; i < aData.length; i++) result[i] = op(aData[i], bData[0]);
    const t = RTV.tensor(result, aShape);
    t._isLogical = true;
    return t;
  }
  throw new RuntimeError("Matrix dimensions must agree for logical operation");
}

// ── Call site type ──────────────────────────────────────────────────────

/** Where a function call originates (passed from codegen into runtime). */
export type CallSite = {
  file: string; // source file name (e.g., "script.m", "foo.m")
  className?: string; // set when calling from inside a class method
  methodName?: string; // set when calling from inside a class method
  targetClassName?: string; // set when explicitly calling a method on a known class
};
