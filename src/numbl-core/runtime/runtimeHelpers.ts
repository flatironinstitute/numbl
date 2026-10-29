/**
 * Runtime helper types and standalone utility functions.
 *
 * Types and pure functions
 * These have no dependency on the Runtime class.
 */

import {
  type RuntimeValue,
  RTV,
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
} from "../runtime/types.js";
import { END_SENTINEL } from "./sentinels.js";
import { allocFloat64Array } from "./alloc.js";
import { isEnumInstance, enumScalarEquals } from "./runtimeEnum.js";

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
  if (v === undefined) return RTV.tensor(allocFloat64Array(0), [0]);
  if (!isRuntimeValue(v)) {
    throw new RuntimeError(
      `Expected a runtime value, got ${JSON.stringify(v)}`
    );
  }
  return v;
}

export function switchValuesMatch(a: RuntimeValue, b: RuntimeValue): boolean {
  // Enumeration member in a `switch`: match by underlying value (member vs
  // member / numeric) or by name (member vs char case label).
  if (isEnumInstance(a)) return enumScalarEquals(a, b);
  if (isEnumInstance(b)) return enumScalarEquals(b, a);
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

// ── Call site type ──────────────────────────────────────────────────────

/** Where a function call originates (passed from codegen into runtime). */
export type CallSite = {
  file: string; // source file name (e.g., "script.m", "foo.m")
  className?: string; // set when calling from inside a class method
  methodName?: string; // set when calling from inside a class method
  targetClassName?: string; // set when explicitly calling a method on a known class
};
