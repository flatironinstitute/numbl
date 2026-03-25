/**
 * Cumulative and difference builtins: cumsum, cumprod, cummax, cummin, diff.
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  RuntimeError,
} from "../../runtime/index.js";
import { ItemType } from "../../lowering/itemTypes.js";
import { register } from "../registry.js";
import {
  FloatXArray,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { sparseToDense } from "../sparse-arithmetic.js";
import { preserveTypeCheck } from "./helpers.js";

// ── Generic cumulative operation ───────────────────────────────────────

/** Generic cumulative operation along a specified or default dimension.
 *  Supports cumsum(A), cumsum(A, dim), etc. for arbitrary N-D arrays. */
export function cumOp(
  name: string,
  args: RuntimeValue[],
  accumFn: (acc: number, val: number) => number,
  initial?: number,
  complexAccumFn?: (
    accRe: number,
    accIm: number,
    valRe: number,
    valIm: number
  ) => [number, number]
): RuntimeValue {
  if (args.length < 1)
    throw new RuntimeError(`${name} requires at least 1 argument`);
  let v = args[0];
  if (isRuntimeSparseMatrix(v)) v = sparseToDense(v);
  if (isRuntimeNumber(v)) return v;
  if (isRuntimeTensor(v)) {
    const shape = v.shape;
    const hasImag = v.imag !== undefined;

    // Determine dimension to accumulate along (1-based)
    let dim: number;
    if (args.length >= 2) {
      dim = Math.round(toNumber(args[1]));
    } else {
      const idx = shape.findIndex(d => d > 1);
      dim = idx >= 0 ? idx + 1 : 1;
    }
    const dimIdx = dim - 1;

    // If dim exceeds dimensions, return a copy (dim has size 1)
    if (dimIdx >= shape.length) {
      return RTV.tensor(
        new FloatXArray(v.data),
        [...shape],
        hasImag ? new FloatXArray(v.imag!) : undefined
      );
    }

    const dimSize = shape[dimIdx];
    const result = new FloatXArray(v.data.length);
    const resultImag = hasImag ? new FloatXArray(v.data.length) : undefined;

    const accumOne = (
      acc: number,
      accIm: number,
      valRe: number,
      valIm: number
    ): [number, number] => {
      if (hasImag && complexAccumFn) {
        return complexAccumFn(acc, accIm, valRe, valIm);
      }
      return [accumFn(acc, valRe), hasImag ? accumFn(accIm, valIm) : 0];
    };

    // Helper to accumulate along a fiber starting at `base` with given stride
    const accumFiber = (base: number, stride: number) => {
      if (initial !== undefined) {
        let acc = initial;
        let accIm = 0;
        for (let k = 0; k < dimSize; k++) {
          const idx = base + k * stride;
          [acc, accIm] = accumOne(
            acc,
            accIm,
            v.data[idx],
            v.imag ? v.imag[idx] : 0
          );
          result[idx] = acc;
          if (resultImag) resultImag[idx] = accIm;
        }
      } else {
        const startIdx = base;
        result[startIdx] = v.data[startIdx];
        if (resultImag) resultImag[startIdx] = v.imag![startIdx];
        let acc = v.data[startIdx];
        let accIm = hasImag ? v.imag![startIdx] : 0;
        for (let k = 1; k < dimSize; k++) {
          const idx = base + k * stride;
          [acc, accIm] = accumOne(
            acc,
            accIm,
            v.data[idx],
            v.imag ? v.imag[idx] : 0
          );
          result[idx] = acc;
          if (resultImag) resultImag[idx] = accIm;
        }
      }
    };

    if (dimIdx === 0) {
      // Fast path: dim 1 is contiguous in column-major layout (stride = 1)
      const numSlices = v.data.length / dimSize;
      for (let slice = 0; slice < numSlices; slice++) {
        accumFiber(slice * dimSize, 1);
      }
    } else {
      // General case: accumulate along non-contiguous dimension using stride arithmetic
      let strideDim = 1;
      for (let d = 0; d < dimIdx; d++) strideDim *= shape[d];
      const slabSize = strideDim * dimSize;
      let numOuter = 1;
      for (let d = dimIdx + 1; d < shape.length; d++) numOuter *= shape[d];

      for (let outer = 0; outer < numOuter; outer++) {
        for (let inner = 0; inner < strideDim; inner++) {
          accumFiber(outer * slabSize + inner, strideDim);
        }
      }
    }

    const imOut =
      resultImag && resultImag.some(x => x !== 0) ? resultImag : undefined;
    return RTV.tensor(result, shape, imOut);
  }
  throw new RuntimeError(`${name}: argument must be numeric`);
}

// ── diff ───────────────────────────────────────────────────────────────

export function diffOnce(v: RuntimeValue, dim?: number): RuntimeValue {
  if (isRuntimeNumber(v)) {
    return RTV.tensor(new FloatXArray(0), [0, 0]);
  }
  if (isRuntimeTensor(v)) {
    const shape = v.shape;

    let opDim: number;
    if (dim !== undefined) {
      opDim = dim - 1;
    } else if (shape.length <= 1 || (shape.length === 2 && shape[0] === 1)) {
      opDim = shape.length === 2 && shape[0] === 1 ? 1 : 0;
    } else {
      opDim = 0;
    }

    const dimSize = opDim < shape.length ? shape[opDim] : 1;
    if (dimSize <= 1) {
      const newShape = [...shape];
      if (opDim < newShape.length) newShape[opDim] = 0;
      return RTV.tensor(new FloatXArray(0), newShape);
    }

    const newShape = [...shape];
    newShape[opDim] = dimSize - 1;
    const totalOut = newShape.reduce((a, b) => a * b, 1);
    const result = new FloatXArray(totalOut);
    const resultImag = v.imag ? new FloatXArray(totalOut) : undefined;

    let innerCount = 1;
    for (let d = 0; d < opDim; d++) innerCount *= shape[d];
    let outerCount = 1;
    for (let d = opDim + 1; d < shape.length; d++) outerCount *= shape[d];

    let outIdx = 0;
    for (let outer = 0; outer < outerCount; outer++) {
      for (let k = 0; k < dimSize - 1; k++) {
        for (let inner = 0; inner < innerCount; inner++) {
          const base = outer * (dimSize * innerCount) + k * innerCount + inner;
          result[outIdx] = v.data[base + innerCount] - v.data[base];
          if (resultImag && v.imag) {
            resultImag[outIdx] = v.imag[base + innerCount] - v.imag[base];
          }
          outIdx++;
        }
      }
    }

    return RTV.tensor(result, newShape, resultImag);
  }
  throw new RuntimeError("diff: argument must be numeric");
}

// ── Registration ───────────────────────────────────────────────────────

export function registerCumulative(): void {
  register("cumsum", [
    {
      check: preserveTypeCheck,
      apply: args => cumOp("cumsum", args, (acc, val) => acc + val, 0),
    },
  ]);

  register("cumprod", [
    {
      check: preserveTypeCheck,
      apply: args =>
        cumOp(
          "cumprod",
          args,
          (acc, val) => acc * val,
          1,
          (aRe, aIm, bRe, bIm) => [aRe * bRe - aIm * bIm, aRe * bIm + aIm * bRe]
        ),
    },
  ]);

  register("cummax", [
    {
      check: preserveTypeCheck,
      apply: args => cumOp("cummax", args, Math.max),
    },
  ]);

  register("cummin", [
    {
      check: preserveTypeCheck,
      apply: args => cumOp("cummin", args, Math.min),
    },
  ]);

  register("diff", [
    {
      check: (_argTypes: ItemType[], nargout: number) => {
        if (nargout !== 1) return null;
        return {
          outputTypes: [
            { kind: "Tensor" as const, ndim: 2, shape: "unknown" as const },
          ],
        };
      },
      apply: args => {
        if (args.length < 1)
          throw new RuntimeError("diff requires at least 1 argument");
        if (isRuntimeSparseMatrix(args[0]))
          args = [sparseToDense(args[0]), ...args.slice(1)];
        const n = args.length >= 2 ? Math.round(toNumber(args[1])) : 1;
        const dimArg =
          args.length >= 3 ? Math.round(toNumber(args[2])) : undefined;

        let result = args[0];
        for (let i = 0; i < n; i++) {
          result = diffOnce(result, dimArg);
        }
        return result;
      },
    },
  ]);
}
