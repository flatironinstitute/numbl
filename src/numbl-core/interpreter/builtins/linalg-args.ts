/**
 * Argument helpers for the linear-algebra and sparse builtins.
 *
 * MATLAB's built-in matrix functions (norm, svd, eig, ...) take floating-point
 * input and reject char and non-numeric values with a function-specific
 * error. As in the other numeric builtins, numbl also accepts logical input
 * (as double), since it has no single type and code relies on it.
 */

import { RuntimeError } from "../../runtime/error.js";
import {
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
  RuntimeTensor,
  type RuntimeSparseMatrix,
  type RuntimeValue,
} from "../../runtime/types.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import { builtinError, numericArg, realScalarArg } from "./args.js";

/** A thunk raising MATLAB's error `identifier` / `message`. */
export function errorFn(
  identifier: string,
  message: string
): () => RuntimeError {
  return () => builtinError(identifier, message);
}

/** A scalar as a 1x1 tensor (complex when `im` is given). */
export function scalarTensor(re: number, im?: number): RuntimeTensor {
  return new RuntimeTensor(
    allocFloat64Array([re]),
    [1, 1],
    im === undefined ? undefined : allocFloat64Array([im])
  );
}

/** A floating-point matrix argument: a double or logical value (scalars
 *  become 1x1 tensors) or a sparse matrix. Char and non-numeric values
 *  raise `err()`. */
export function floatMatrixArg(
  v: RuntimeValue,
  err: () => RuntimeError
): RuntimeTensor | RuntimeSparseMatrix {
  if (typeof v === "number") return scalarTensor(v);
  if (typeof v === "boolean") return scalarTensor(v ? 1 : 0);
  if (isRuntimeComplexNumber(v)) return scalarTensor(v.re, v.im);
  if (isRuntimeTensor(v) || isRuntimeSparseMatrix(v)) return v;
  throw err();
}

/** Like `floatMatrixArg`, but a sparse matrix is converted to a dense one,
 *  or raises `sparseErr()` when given. */
export function denseFloatMatrixArg(
  v: RuntimeValue,
  err: () => RuntimeError,
  sparseErr?: () => RuntimeError
): RuntimeTensor {
  const a = floatMatrixArg(v, err);
  if (isRuntimeTensor(a)) return a;
  if (sparseErr) throw sparseErr();
  return sparseToDense(a);
}

/** A numeric argument (char and logical allowed, as for MATLAB's M-file
 *  matrix functions such as dot and kron) as a tensor or sparse matrix. */
export function numericMatrixArg(
  v: RuntimeValue
): RuntimeTensor | RuntimeSparseMatrix {
  const a = numericArg(v);
  if (typeof a === "number") return scalarTensor(a);
  if (isRuntimeComplexNumber(a)) return scalarTensor(a.re, a.im);
  return a;
}

/** Like `numericMatrixArg`, with sparse input converted to dense. */
export function denseNumericMatrixArg(v: RuntimeValue): RuntimeTensor {
  const a = numericMatrixArg(v);
  return isRuntimeTensor(a) ? a : sparseToDense(a);
}

/** A real scalar argument; anything else raises `err()`. */
export function realScalarOr(v: RuntimeValue, err: () => RuntimeError): number {
  try {
    return realScalarArg(v);
  } catch {
    throw err();
  }
}

/** A dimension argument: a positive integer scalar, else `err()`. */
export function dimArgOr(v: RuntimeValue, err: () => RuntimeError): number {
  const d = realScalarOr(v, err);
  if (!(Number.isInteger(d) && d >= 1)) throw err();
  return d;
}

/** Package results for `nargout` outputs: the first value when at most one
 *  output is requested, else the first `nargout` values. */
export function outputs(
  values: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  return nargout <= 1 ? values[0] : values.slice(0, nargout);
}
