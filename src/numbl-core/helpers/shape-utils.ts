/**
 * Shared shape/size argument parsing and value coercion for builtins.
 */

import { allocFloat64Array } from "../runtime/alloc.js";
import {
  type RuntimeValue,
  RTV,
  RuntimeError,
  toNumber,
} from "../runtime/index.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  type RuntimeTensor,
} from "../runtime/types.js";

/** Promote a scalar (number, logical, complex) to a 1×1 tensor. Tensors pass through. */
export function coerceToTensor(v: RuntimeValue, name: string): RuntimeTensor {
  if (isRuntimeTensor(v)) return v;
  if (isRuntimeNumber(v))
    return RTV.tensor(allocFloat64Array([v]), [1, 1]) as RuntimeTensor;
  if (isRuntimeLogical(v))
    return RTV.tensor(allocFloat64Array([v ? 1 : 0]), [1, 1]) as RuntimeTensor;
  if (isRuntimeComplexNumber(v))
    return RTV.tensor(
      allocFloat64Array([v.re]),
      [1, 1],
      allocFloat64Array([v.im])
    ) as RuntimeTensor;
  throw new RuntimeError(`${name}: argument must be numeric`);
}

/** Validate one dimension value: MATLAB rejects NaN, Inf, and non-integer
 *  sizes; negative integers are silently clamped to 0. */
export function validateDim(x: number): number {
  if (!Number.isFinite(x) || !Number.isInteger(x))
    throw new RuntimeError("Size inputs must be nonnegative integers.");
  return Math.max(0, x);
}

/** Extract a single scalar dimension. With multiple size args MATLAB requires
 *  each to be a scalar — a 1×1 array counts, but a multi-element vector errors
 *  with "Size inputs must be scalar." */
export function toScalarDim(v: RuntimeValue): number {
  if (isRuntimeTensor(v) && v.data.length !== 1)
    throw new RuntimeError("Size inputs must be scalar.");
  return validateDim(toNumber(v));
}

/** Parse shape arguments: zeros(2,3) or zeros([2,3]) -> [2, 3].
 *  Negative dimensions are clamped to 0. */
export function parseShapeArgs(args: RuntimeValue[]): number[] {
  if (args.length === 1 && isRuntimeTensor(args[0])) {
    const t = args[0];
    const shape: number[] = [];
    for (let i = 0; i < t.data.length; i++) shape.push(validateDim(t.data[i]));
    return shape;
  }
  return args.map(toScalarDim);
}
