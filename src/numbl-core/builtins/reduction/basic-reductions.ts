/**
 * Basic reduction builtins: sum, prod, mean, std, var, median, mode.
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  toString,
  RuntimeError,
} from "../../runtime/index.js";
import { register } from "../registry.js";
import {
  isRuntimeChar,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { sparseToDense } from "../sparse-arithmetic.js";
import {
  firstReduceDim,
  reductionCheck,
  makeReduction,
  accumKernel,
  accumKernelOmitNaN,
  sliceKernel,
  sliceKernelOmitNaN,
  sparseSum,
  complexProd,
  parseNanFlag,
  filterNaN,
} from "./helpers.js";

export function registerBasicReductions(): void {
  // ── sum ──────────────────────────────────────────────────────────────

  register("sum", [
    {
      check: reductionCheck,
      apply: rawArgs => {
        if (rawArgs.length < 1)
          throw new RuntimeError("sum requires at least 1 argument");
        const { args, omitNaN } = parseNanFlag(rawArgs);
        const v = args[0];
        if (isRuntimeSparseMatrix(v)) {
          const dim =
            args.length >= 2
              ? Math.round(toNumber(args[1]))
              : v.m > 1
                ? 1
                : v.n > 1
                  ? 2
                  : 1;
          return sparseSum(v, dim);
        }
        const kernel = omitNaN
          ? accumKernelOmitNaN((acc, val) => acc + val, 0)
          : accumKernel((acc, val) => acc + val, 0);
        if (isRuntimeNumber(v)) return v;
        if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
        if (isRuntimeTensor(v)) {
          if (args.length >= 2) {
            if (isRuntimeChar(args[1]) && toString(args[1]) === "all")
              return kernel.reduceAll(v);
            return kernel.reduceDim(v, Math.round(toNumber(args[1])));
          }
          const d = firstReduceDim(v.shape);
          return d === 0 ? kernel.reduceAll(v) : kernel.reduceDim(v, d);
        }
        throw new RuntimeError("sum: argument must be numeric");
      },
    },
  ]);

  // ── prod ─────────────────────────────────────────────────────────────

  register("prod", [
    {
      check: reductionCheck,
      apply: rawArgs => {
        if (rawArgs.length < 1)
          throw new RuntimeError("prod requires at least 1 argument");
        const { args: parsedArgs, omitNaN } = parseNanFlag(rawArgs);
        let args = parsedArgs;
        let v = args[0];
        if (isRuntimeSparseMatrix(v)) {
          v = sparseToDense(v);
          args = [v, ...args.slice(1)];
        }
        if (isRuntimeNumber(v)) return v;
        if (isRuntimeTensor(v)) {
          if (v.imag) {
            return complexProd(
              v,
              args.length >= 2 ? Math.round(toNumber(args[1])) : undefined
            );
          }
          const kernel = omitNaN
            ? accumKernelOmitNaN((acc, val) => acc * val, 1)
            : accumKernel((acc, val) => acc * val, 1);
          if (args.length >= 2) {
            if (isRuntimeChar(args[1]) && toString(args[1]) === "all")
              return kernel.reduceAll(v);
            return kernel.reduceDim(v, Math.round(toNumber(args[1])));
          }
          const d = firstReduceDim(v.shape);
          return d === 0 ? kernel.reduceAll(v) : kernel.reduceDim(v, d);
        }
        throw new RuntimeError("prod: argument must be numeric");
      },
    },
  ]);

  // ── mean ─────────────────────────────────────────────────────────────

  register("mean", [
    makeReduction(
      "mean",
      accumKernel(
        (acc, val) => acc + val,
        0,
        (sum, count) => sum / count
      ),
      accumKernelOmitNaN(
        (acc, val) => acc + val,
        0,
        (sum, count) => (count === 0 ? NaN : sum / count)
      )
    ),
  ]);

  // ── std / var ────────────────────────────────────────────────────────

  const varianceOf = (
    slice: ArrayLike<number>,
    w: number,
    omitNaN: boolean
  ): number => {
    let data: ArrayLike<number> = slice;
    if (omitNaN) data = filterNaN(slice);
    const n = data.length;
    if (n === 0) return NaN;
    if (n <= 1 && w === 0) return 0;
    let s = 0;
    for (let i = 0; i < n; i++) s += data[i];
    const m = s / n;
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (data[i] - m) ** 2;
    const denom = w === 1 ? n : n - 1;
    return ss / denom;
  };

  const stdVarApply = (
    name: string,
    transform: (variance: number) => number
  ) => {
    return (rawArgs: RuntimeValue[]): RuntimeValue => {
      if (rawArgs.length < 1)
        throw new RuntimeError(`${name} requires at least 1 argument`);
      const { args, omitNaN } = parseNanFlag(rawArgs);
      const v = args[0];
      const w = args.length >= 2 ? toNumber(args[1]) : 0;
      const dimArg = args.length >= 3 ? Math.round(toNumber(args[2])) : 0;
      if (isRuntimeNumber(v)) return RTV.num(0);
      if (isRuntimeTensor(v)) {
        const kernel = sliceKernel((slice: ArrayLike<number>) =>
          transform(varianceOf(slice, w, omitNaN))
        );
        if (dimArg > 0) return kernel.reduceDim(v, dimArg);
        const d = firstReduceDim(v.shape);
        return d === 0 ? kernel.reduceAll(v) : kernel.reduceDim(v, d);
      }
      throw new RuntimeError(`${name}: argument must be numeric`);
    };
  };

  register("std", [
    {
      check: reductionCheck,
      apply: stdVarApply("std", v => Math.sqrt(v)),
    },
  ]);

  register("var", [
    {
      check: reductionCheck,
      apply: stdVarApply("var", v => v),
    },
  ]);

  // ── median ───────────────────────────────────────────────────────────

  const medianOf = (arr: ArrayLike<number>): number => {
    const sorted = Array.from(arr).sort((a, b) => a - b);
    const n = sorted.length;
    if (n === 0) return NaN;
    if (n % 2 === 1) return sorted[(n - 1) / 2];
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  };

  register("median", [
    makeReduction(
      "median",
      sliceKernel(medianOf),
      sliceKernelOmitNaN(medianOf)
    ),
  ]);

  // ── mode ─────────────────────────────────────────────────────────────

  const modeOf = (arr: ArrayLike<number>): number => {
    const counts = new Map<number, number>();
    for (let i = 0; i < arr.length; i++) {
      counts.set(arr[i], (counts.get(arr[i]) || 0) + 1);
    }
    let bestVal = arr[0],
      bestCount = 0;
    for (const [val, count] of counts) {
      if (count > bestCount || (count === bestCount && val < bestVal)) {
        bestVal = val;
        bestCount = count;
      }
    }
    return bestVal;
  };

  register("mode", [
    makeReduction("mode", sliceKernel(modeOf), sliceKernelOmitNaN(modeOf)),
  ]);
}
