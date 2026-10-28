/**
 * Argument helpers shared by the numeric builtins (math, arithmetic,
 * reductions, special functions, ...). They complement args.ts with the
 * stricter checks MATLAB's floating-point functions apply.
 */

import { RuntimeError } from "../../runtime/error.js";
import {
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
  RuntimeTensor,
  type RuntimeValue,
} from "../../runtime/types.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import { builtinError, numericArg, type NumericArg } from "./args.js";

/** MATLAB's error for a non-floating-point argument (sin('a'), exp({})). */
export function mustBeFloatError(): RuntimeError {
  return builtinError(
    "MATLAB:math:mustBeFloat",
    "Invalid data type. Argument must be double or single."
  );
}

/** A numeric argument to a floating-point function (sin, exp, gamma, ...).
 *  MATLAB rejects char, logical, and non-numeric input; numbl keeps
 *  accepting logical (as double), since it has no single type and code
 *  relies on it, but rejects everything else. */
export function floatArg(v: RuntimeValue): NumericArg {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (
    isRuntimeTensor(v) ||
    isRuntimeComplexNumber(v) ||
    isRuntimeSparseMatrix(v)
  )
    return v;
  throw mustBeFloatError();
}

/** True when a numeric argument has a nonzero imaginary part. */
export function hasImag(a: NumericArg): boolean {
  if (typeof a === "number") return false;
  if (isRuntimeComplexNumber(a)) return a.im !== 0;
  if (isRuntimeTensor(a)) return !!a.imag && a.imag.some(x => x !== 0);
  return !!a.pi && a.pi.some(x => x !== 0);
}

/** A real floating-point argument: like `floatArg`, but complex input
 *  raises `err()`. The result is a number or a real dense tensor. */
export function realFloatArg(
  v: RuntimeValue,
  err: () => RuntimeError
): number | RuntimeTensor {
  return toRealDense(floatArg(v), err);
}

/** A real numeric (or char/logical) argument as a number or a real dense
 *  tensor; complex input raises `err()`. */
export function realDenseArg(
  v: RuntimeValue,
  err: () => RuntimeError
): number | RuntimeTensor {
  return toRealDense(numericArg(v), err);
}

function toRealDense(
  a: NumericArg,
  err: () => RuntimeError
): number | RuntimeTensor {
  if (hasImag(a)) throw err();
  if (typeof a === "number") return a;
  if (isRuntimeComplexNumber(a)) return a.re;
  const t = isRuntimeSparseMatrix(a) ? sparseToDense(a) : a;
  if (!t.imag) return t;
  // an all-zero imaginary part
  return new RuntimeTensor(t.data, t.shape);
}

/** MATLAB's "Input must be real and full." for erf, gamma, ... */
export function notFullRealError(name: string): RuntimeError {
  return builtinError(
    `MATLAB:${name}:notFullReal`,
    "Input must be real and full."
  );
}

/** The real data of a number or real dense tensor, as an array. */
export function realData(a: number | RuntimeTensor): ArrayLike<number> {
  return typeof a === "number" ? [a] : a.data;
}
