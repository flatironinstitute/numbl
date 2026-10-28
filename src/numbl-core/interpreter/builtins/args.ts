/**
 * Argument checking and coercion helpers for builtins.
 *
 * Errors carry MATLAB's message text and identifier. The coercion helpers
 * reject what they cannot represent (for example complex input to a
 * real-only argument) instead of silently dropping data.
 */

import { RuntimeError } from "../../runtime/error.js";
import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
  type RuntimeComplexNumber,
  type RuntimeSparseMatrix,
  type RuntimeTensor,
  type RuntimeValue,
} from "../../runtime/types.js";
import { charToNumericTensor } from "../../runtime/indexing.js";
import {
  imagAllZero,
  stripZeroImagTensor,
} from "../../helpers/effectively-real.js";

// ── Errors ──────────────────────────────────────────────────────────────

/** A RuntimeError with a MATLAB identifier (caught as `ME.identifier`). */
export function builtinError(
  identifier: string,
  message: string
): RuntimeError {
  return new RuntimeError(message, null, { identifier });
}

/** Raise MATLAB's arity error when `n` input arguments are outside
 *  [min, max]. MATLAB uses different "too many" identifiers for built-in
 *  functions and for functions it implements as M-files (`mfile`). */
export function checkNargin(
  n: number,
  min: number,
  max: number,
  mfile = false
): void {
  if (n < min)
    throw builtinError("MATLAB:minrhs", "Not enough input arguments.");
  if (n > max)
    throw builtinError(
      mfile ? "MATLAB:TooManyInputs" : "MATLAB:maxrhs",
      "Too many input arguments."
    );
}

/** Raise MATLAB's arity error when `n` requested outputs are outside
 *  [min, max]. `mfile` as for checkNargin. */
export function checkNargout(
  n: number,
  min: number,
  max: number,
  mfile = false
): void {
  if (n > max)
    throw builtinError(
      mfile ? "MATLAB:TooManyOutputs" : "MATLAB:maxlhs",
      "Too many output arguments."
    );
  if (n < min)
    throw builtinError("MATLAB:minlhs", "Not enough output arguments.");
}

/** MATLAB's error for a non-numeric argument to a math function. */
export function invalidDataTypeError(): RuntimeError {
  return builtinError(
    "MATLAB:math:mustBeNumericCharOrLogical",
    "Invalid data type. Argument must be numeric, char, or logical."
  );
}

/** MATLAB's error for complex input to a real-only argument. */
export function mustBeRealError(): RuntimeError {
  return builtinError("MATLAB:math:mustBeReal", "Argument must be real.");
}

// ── Coercion ────────────────────────────────────────────────────────────

/** Logical scalar to double; other values unchanged. */
export function boolToNumber<T>(v: T | boolean): T | number {
  return typeof v === "boolean" ? (v ? 1 : 0) : v;
}

export type NumericArg =
  | number
  | RuntimeComplexNumber
  | RuntimeTensor
  | RuntimeSparseMatrix;

/** A numeric argument as MATLAB math functions see it: a logical scalar
 *  becomes a double, a char becomes its character codes (a scalar for a
 *  single character). Logical tensors are returned as-is. Anything else
 *  (cell, struct, string, object, ...) raises MATLAB's invalid-data-type
 *  error. */
export function numericArg(v: RuntimeValue): NumericArg {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (
    isRuntimeTensor(v) ||
    isRuntimeComplexNumber(v) ||
    isRuntimeSparseMatrix(v)
  )
    return v;
  if (isRuntimeChar(v)) {
    if (v.value.length === 1) return v.value.charCodeAt(0);
    return charToNumericTensor(v);
  }
  throw invalidDataTypeError();
}

/** Like `numericArg`, but complex input raises `err` (by default MATLAB's
 *  "Argument must be real."). A tensor whose imaginary part is all zero
 *  counts as real. */
export function realNumericArg(
  v: RuntimeValue,
  err: () => RuntimeError = mustBeRealError
): number | RuntimeTensor | RuntimeSparseMatrix {
  const a = numericArg(v);
  if (typeof a === "number") return a;
  if (isRuntimeComplexNumber(a)) {
    if (a.im !== 0) throw err();
    return a.re;
  }
  if (isRuntimeTensor(a)) {
    if (!a.imag) return a;
    if (!imagAllZero(a.imag)) throw err();
    return stripZeroImagTensor(a);
  }
  if (a.pi && !imagAllZero(a.pi)) throw err();
  return a;
}

/** A real scalar argument as a number. Accepts a double or logical scalar,
 *  a single character, and 1x1 tensors and sparse matrices. Complex,
 *  non-scalar, and non-numeric values raise "<what> must be a real
 *  scalar." */
export function realScalarArg(v: RuntimeValue, what = "Argument"): number {
  const fail = () => new RuntimeError(`${what} must be a real scalar.`);
  if (
    !(
      typeof v === "number" ||
      typeof v === "boolean" ||
      isRuntimeTensor(v) ||
      isRuntimeComplexNumber(v) ||
      isRuntimeSparseMatrix(v) ||
      isRuntimeChar(v)
    )
  )
    throw fail();
  const a = realNumericArg(v, fail);
  if (typeof a === "number") return a;
  if (isRuntimeTensor(a)) {
    if (a.data.length !== 1) throw fail();
    return a.data[0];
  }
  if (a.m * a.n !== 1) throw fail();
  return a.pr.length > 0 ? a.pr[0] : 0;
}
