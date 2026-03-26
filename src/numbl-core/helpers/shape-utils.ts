/**
 * Shared shape/size argument parsing and value coercion for builtins.
 */

import {
  type RuntimeValue,
  RTV,
  toNumber,
  RuntimeError,
} from "../runtime/index.js";
import {
  FloatXArray,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  type RuntimeTensor,
} from "../runtime/types.js";

/** Parse shape arguments: zeros(2,3) or zeros([2,3]) -> [2, 3]
 *  Negative dimensions are clamped to 0 */
export function parseShapeArgs(args: RuntimeValue[]): number[] {
  if (args.length === 1 && isRuntimeTensor(args[0])) {
    const t = args[0];
    const shape: number[] = [];
    for (let i = 0; i < t.data.length; i++)
      shape.push(Math.max(0, Math.round(t.data[i])));
    return shape;
  }
  return args.map(a => Math.max(0, Math.round(toNumber(a))));
}

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

/** Extract a numeric vector from a scalar or tensor. */
export function toNumericVector(v: RuntimeValue, name: string): number[] {
  if (isRuntimeNumber(v)) return [v as number];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  throw new RuntimeError(`${name}: argument must be numeric`);
}
