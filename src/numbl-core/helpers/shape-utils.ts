/**
 * Shared shape/size argument parsing and value coercion for builtins.
 */

import { type RuntimeValue, RTV, RuntimeError } from "../runtime/index.js";
import {
  FloatXArray,
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
    return RTV.tensor(new FloatXArray([v]), [1, 1]) as RuntimeTensor;
  if (isRuntimeLogical(v))
    return RTV.tensor(new FloatXArray([v ? 1 : 0]), [1, 1]) as RuntimeTensor;
  if (isRuntimeComplexNumber(v))
    return RTV.tensor(
      new FloatXArray([v.re]),
      [1, 1],
      new FloatXArray([v.im])
    ) as RuntimeTensor;
  throw new RuntimeError(`${name}: argument must be numeric`);
}
