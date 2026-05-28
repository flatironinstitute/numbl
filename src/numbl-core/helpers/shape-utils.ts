/**
 * Shared shape/size argument parsing and value coercion for builtins.
 */

import { allocFloat64Array } from "../runtime/alloc.js";
import { type RuntimeValue, RTV, RuntimeError } from "../runtime/index.js";
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
