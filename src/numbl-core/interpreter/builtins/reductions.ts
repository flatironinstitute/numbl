/**
 * Interpreter builtins for reduction operations:
 * sum, prod, mean, std, var, median, mode, any, all,
 * cumsum, cumprod, cummax, cummin, diff, xor.
 */

import type { RuntimeTensor, RuntimeValue } from "../../runtime/types.js";
import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeTensor,
  withElementClass,
} from "../../runtime/types.js";
import { RTV, RuntimeError, toString } from "../../runtime/index.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import {
  firstReduceDim,
  accumKernel,
  accumKernelOmitNaN,
  sliceKernel,
  sliceKernelOmitNaN,
  filterNaN,
  complexProd,
  scanLogical,
  logicalAlongDim,
  sparseSum,
  type ReductionKernel,
} from "../../helpers/reduction-helpers.js";
import { cumOp, diffOnce } from "../../helpers/reduction/cumulative.js";
import { mLogical } from "../../helpers/arithmetic.js";
import { OpReduce, OpCmp } from "../../ops/index.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { builtin, type SingleOutputs } from "./builtin.js";
import {
  builtinError,
  mustBeRealError,
  numericArg,
  realScalarArg,
  type NumericArg,
} from "./args.js";
import { hasImag, realDenseArg } from "./math-args.js";

// ── Argument helpers ───────────────────────────────────────────────────

const isText = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);
const textOf = (v: RuntimeValue) =>
  (isRuntimeChar(v) ? v.value : String(v)).toLowerCase();
const isAll = (v: RuntimeValue) => isText(v) && textOf(v) === "all";

/** The data argument of a reduction: numeric, logical, or char (as its
 *  character codes). Anything else raises MATLAB's
 *  "<id>:wrongInput" error. */
function dataArg(
  v: RuntimeValue,
  id: string,
  msg = "Invalid data type. First argument must be numeric or logical."
): NumericArg {
  if (
    typeof v === "number" ||
    typeof v === "boolean" ||
    isRuntimeTensor(v) ||
    isRuntimeComplexNumber(v) ||
    isRuntimeSparseMatrix(v) ||
    isRuntimeChar(v)
  )
    return numericArg(v);
  throw builtinError(`MATLAB:${id}:wrongInput`, msg);
}

/** A dimension argument: a positive integer scalar. */
function dimArg(v: RuntimeValue, name: string): number {
  if (isRuntimeTensor(v) && v.data.length > 1)
    throw new RuntimeError(
      `${name}: vector dimension arguments are not supported`
    );
  const err = () =>
    builtinError(
      "MATLAB:getdimarg:invalidDim",
      "Dimension argument must be a positive integer scalar, a vector of unique positive integers, or 'all'."
    );
  let d: number;
  try {
    d = realScalarArg(v);
  } catch {
    throw err();
  }
  if (!Number.isInteger(d) || d < 1) throw err();
  return d;
}

/** Strip trailing option flags. Accepts the nan flags ('omitnan' /
 *  'includenan'), and the output-type flags ('double', 'default',
 *  'native') when `outtype` is set; any other text raises an error. */
function parseFlags(
  name: string,
  rest: RuntimeValue[],
  opts: { outtype?: boolean; nanflag?: boolean } = {}
): { rest: RuntimeValue[]; omitNaN: boolean } {
  const out = rest.slice();
  let omitNaN = false;
  while (out.length > 0 && isText(out[out.length - 1])) {
    const f = textOf(out[out.length - 1]);
    if (f === "all") break;
    if ((opts.nanflag ?? true) && (f === "omitnan" || f === "includenan")) {
      omitNaN = f === "omitnan";
    } else if (
      opts.outtype &&
      (f === "double" || f === "default" || f === "native")
    ) {
      // numbl has only double data
    } else {
      throw new RuntimeError(`${name}: unsupported option '${f}'`);
    }
    out.pop();
  }
  return { rest: out, omitNaN };
}

/** Densify a sparse argument. */
function dense(v: NumericArg): number | RuntimeValue {
  return isRuntimeSparseMatrix(v) ? sparseToDense(v) : v;
}

// ── Generic reduction ──────────────────────────────────────────────────

/** Reduce `v` (after dataArg) along `rest[0]` (dim or 'all'), or along the
 *  first non-singleton dimension. */
function reduceWith(
  name: string,
  v: NumericArg,
  rest: RuntimeValue[],
  k: ReductionKernel
): RuntimeValue {
  if (rest.length > 1) throw new RuntimeError(`${name}: too many arguments`);
  const t = dense(v);
  if (!isRuntimeTensor(t)) return t as RuntimeValue;
  if (rest.length === 1) {
    if (isAll(rest[0])) return k.reduceAll(t);
    return k.reduceDim(t, dimArg(rest[0], name));
  }
  const d = firstReduceDim(t.shape);
  return d === 0 ? k.reduceAll(t) : k.reduceDim(t, d);
}

// ── sum ────────────────────────────────────────────────────────────────

const sumKernel = accumKernel(
  (acc, val) => acc + val,
  0,
  undefined,
  OpReduce.SUM
);
const sumOmitNanKernel = accumKernelOmitNaN((acc, val) => acc + val, 0);

/** The outputs of sum, prod and mean that take a single argument's
 *  class: none for the 'double' output type. */
function outTypeOutputs(args: RuntimeValue[]): SingleOutputs {
  for (const a of args)
    if (
      (isRuntimeChar(a) || isRuntimeString(a)) &&
      toString(a).toLowerCase() === "double"
    )
      return [];
  return true;
}

builtin({
  name: "sum",
  singleOutputs: outTypeOutputs,
  accepts: { sparse: true },
  // sum(A, dim|vecdim|'all', outtype, nanflag)
  nargin: [1, 4],
  apply: args => {
    const v = dataArg(args[0], "sum");
    const { rest, omitNaN } = parseFlags("sum", args.slice(1), {
      outtype: true,
    });
    if (isRuntimeSparseMatrix(v) && !(rest.length > 0 && isAll(rest[0]))) {
      const dim =
        rest.length >= 1
          ? dimArg(rest[0], "sum")
          : v.m > 1
            ? 1
            : v.n > 1
              ? 2
              : 1;
      return sparseSum(v, dim);
    }
    return reduceWith("sum", v, rest, omitNaN ? sumOmitNanKernel : sumKernel);
  },
});

// ── prod ───────────────────────────────────────────────────────────────

const prodKernel = accumKernel(
  (acc, val) => acc * val,
  1,
  undefined,
  OpReduce.PROD
);
const prodOmitNanKernel = accumKernelOmitNaN((acc, val) => acc * val, 1);

builtin({
  name: "prod",
  singleOutputs: outTypeOutputs,
  accepts: { sparse: true },
  nargin: [1, 4],
  apply: args => {
    if (isRuntimeChar(args[0]))
      throw builtinError(
        "MATLAB:prod:charInput",
        "PROD does not work with char inputs."
      );
    const v = dense(dataArg(args[0], "prod"));
    const { rest, omitNaN } = parseFlags("prod", args.slice(1), {
      outtype: true,
    });
    if (isRuntimeTensor(v) && v.imag) {
      if (rest.length > 1) throw new RuntimeError("prod: too many arguments");
      if (rest.length === 0) return complexProd(v);
      if (isAll(rest[0]))
        return complexProd(RTV.tensor(v.data, [1, v.data.length], v.imag));
      return complexProd(v, dimArg(rest[0], "prod"));
    }
    return reduceWith(
      "prod",
      v as NumericArg,
      rest,
      omitNaN ? prodOmitNanKernel : prodKernel
    );
  },
});

// ── mean ───────────────────────────────────────────────────────────────

const meanKernel = accumKernel(
  (acc, val) => acc + val,
  0,
  (sum, count) => sum / count,
  OpReduce.SUM
);
const meanOmitNanKernel = accumKernelOmitNaN(
  (acc, val) => acc + val,
  0,
  (sum, count) => (count === 0 ? NaN : sum / count)
);

builtin({
  name: "mean",
  singleOutputs: outTypeOutputs,
  accepts: { sparse: true },
  nargin: [1, 4],
  mfile: true,
  apply: args => {
    const v = dataArg(args[0], "sum");
    const { rest, omitNaN } = parseFlags("mean", args.slice(1), {
      outtype: true,
    });
    return reduceWith(
      "mean",
      v,
      rest,
      omitNaN ? meanOmitNanKernel : meanKernel
    );
  },
});

// ── std / var ──────────────────────────────────────────────────────────

/** Variance of a real slice with normalization weight w (0 or 1). */
function varianceOf(
  slice: ArrayLike<number>,
  w: number,
  omitNaN: boolean
): number {
  const data = omitNaN ? filterNaN(slice) : slice;
  const n = data.length;
  if (n === 0) return NaN;
  if (n <= 1 && w === 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += data[i];
  const m = s / n;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (data[i] - m) ** 2;
  return ss / (w === 1 ? n : n - 1);
}

/** Apply `fn` element-wise to a reduction result. */
function mapResult(r: RuntimeValue, fn: (x: number) => number): RuntimeValue {
  if (typeof r === "number") return fn(r);
  if (isRuntimeTensor(r)) {
    const out = allocFloat64Array(r.data.length);
    for (let i = 0; i < out.length; i++) out[i] = fn(r.data[i]);
    return RTV.tensor(out, r.shape.slice());
  }
  return r;
}

/** Element-wise sum of two real reduction results of the same shape. */
function addResults(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (typeof a === "number" && typeof b === "number") return a + b;
  if (isRuntimeTensor(a) && isRuntimeTensor(b)) {
    const out = allocFloat64Array(a.data.length);
    for (let i = 0; i < out.length; i++) out[i] = a.data[i] + b.data[i];
    return RTV.tensor(out, a.shape.slice());
  }
  throw new RuntimeError("var: internal shape mismatch");
}

function stdVarBuiltin(name: "std" | "var"): void {
  const transform = name === "std" ? Math.sqrt : (x: number) => x;
  builtin({
    name,
    singleOutputs: true,
    // std(A, w, dim|vecdim|'all', nanflag); [S, M] = std(...)
    nargin: [1, 4],
    nargout: 2,
    mfile: true,
    apply: (args, nargout) => {
      const v = dense(dataArg(args[0], "sum"));
      const { rest, omitNaN } = parseFlags(name, args.slice(1));
      if (rest.length > 2)
        throw new RuntimeError(`${name}: too many arguments`);
      let w = 0;
      if (rest.length >= 1) {
        const wa = rest[0];
        if (isRuntimeTensor(wa) && wa.data.length > 1)
          throw new RuntimeError(`${name}: weight vectors are not supported`);
        if (!(isRuntimeTensor(wa) && wa.data.length === 0)) {
          w = realScalarArg(wa);
          if (w !== 0 && w !== 1)
            throw builtinError(
              `MATLAB:${name}:invalidWgts`,
              "W must be a double or single vector of nonnegative weights, or a scalar 0 or 1."
            );
        }
      }
      const dimSpec = rest.length >= 2 ? [rest[1]] : [];
      const mean = () =>
        reduceWith(
          name,
          v as NumericArg,
          dimSpec,
          omitNaN ? meanOmitNanKernel : meanKernel
        );
      let result: RuntimeValue;
      if (!isRuntimeTensor(v)) {
        result = 0;
      } else if (!v.imag) {
        // (dim > ndims leaves the input unchanged; the JIT matches this)
        const kernel = sliceKernel(s => transform(varianceOf(s, w, omitNaN)));
        result = reduceWith(name, v, dimSpec, kernel);
      } else {
        // var(z) = var(real(z)) + var(imag(z)) for complex z
        const kernel = sliceKernel(s => varianceOf(s, w, omitNaN));
        const variance = (t: RuntimeTensor) =>
          reduceWith(name, t, dimSpec, kernel);
        result = mapResult(
          addResults(
            variance(RTV.tensor(v.data, v.shape)),
            variance(RTV.tensor(v.imag, v.shape))
          ),
          transform
        );
      }
      if (nargout >= 2) return [result, mean()];
      return result;
    },
  });
}

stdVarBuiltin("std");
stdVarBuiltin("var");

// ── median / mode ──────────────────────────────────────────────────────

function medianOf(arr: ArrayLike<number>): number {
  // A typed array sorts numerically with NaN last; a NaN gives NaN
  // ('includenan', the default; 'omitnan' filters them first).
  const sorted = Float64Array.from(arr).sort();
  const n = sorted.length;
  if (n === 0 || isNaN(sorted[n - 1])) return NaN;
  if (n % 2 === 1) return sorted[(n - 1) / 2];
  return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

function modeOf(arr: ArrayLike<number>): number {
  const counts = new Map<number, number>();
  for (let i = 0; i < arr.length; i++) {
    counts.set(arr[i], (counts.get(arr[i]) ?? 0) + 1);
  }
  let bestVal = (arr as number[])[0],
    bestCount = 0;
  for (const [val, count] of counts) {
    if (count > bestCount || (count === bestCount && val < bestVal)) {
      bestVal = val;
      bestCount = count;
    }
  }
  return bestVal;
}

/** median and mode: real numeric or logical input. MATLAB also accepts
 *  char (returning char) and complex input; numbl rejects both. */
function sliceReductionBuiltin(
  name: string,
  fn: (arr: ArrayLike<number>) => number,
  nargout: number
): void {
  const kernel = sliceKernel(fn);
  const omitNanKernel = sliceKernelOmitNaN(fn);
  builtin({
    name,
    singleOutputs: [1],
    nargin: [1, 4],
    nargout,
    mfile: true,
    accepts: { sparse: true },
    apply: (args, nout) => {
      if (nout > 1)
        throw new RuntimeError(`${name}: only the first output is supported`);
      if (isRuntimeChar(args[0]))
        throw new RuntimeError(`${name}: char input is not supported`);
      const v = dataArg(args[0], name);
      if (hasImag(v))
        throw new RuntimeError(`${name}: complex input is not supported`);
      const { rest, omitNaN } = parseFlags(name, args.slice(1), {
        outtype: name === "median",
      });
      return reduceWith(name, v, rest, omitNaN ? omitNanKernel : kernel);
    },
  });
}

sliceReductionBuiltin("median", medianOf, 1);
sliceReductionBuiltin("mode", modeOf, 3);

// ── any / all ──────────────────────────────────────────────────────────

function anyAllBuiltin(mode: "any" | "all"): void {
  builtin({
    name: mode,
    nargin: [1, 2],
    accepts: { sparse: true },
    apply: args => {
      const v = dense(dataArg(args[0], mode));
      if (typeof v === "number")
        return RTV.logical(scanLogical([v], undefined, mode));
      if (isRuntimeComplexNumber(v))
        return RTV.logical(scanLogical([v.re], [v.im], mode));
      if (!isRuntimeTensor(v))
        throw new RuntimeError(`${mode}: invalid arguments`);
      if (args.length === 1) {
        // Empties are NOT all scalars: MATLAB reduces along the first
        // non-singleton dimension, so any(zeros(4,0)) is 1x0 and
        // any(zeros(0,4)) is 1x4. firstReduceDim returns 0 only for the
        // genuine scalar-result empties (0x0, row/col vectors).
        const d = firstReduceDim(v.shape);
        if (d === 0) return RTV.logical(scanLogical(v.data, v.imag, mode));
        return logicalAlongDim(v, d, mode);
      }
      const arg2 = args[1];
      if (isAll(arg2)) {
        if (v.data.length === 0) return RTV.logical(mode === "all");
        return RTV.logical(scanLogical(v.data, v.imag, mode));
      }
      if (isRuntimeTensor(arg2) && arg2.data.length > 1) {
        const dims = Array.from(arg2.data).map(d => Math.round(d));
        let result: RuntimeValue = v;
        for (const dim of dims) {
          if (isRuntimeTensor(result)) {
            result = logicalAlongDim(result, dim, mode);
          }
        }
        return result;
      }
      return logicalAlongDim(v, dimArg(arg2, mode), mode);
    },
  });
}

anyAllBuiltin("any");
anyAllBuiltin("all");

// ── cumulative ─────────────────────────────────────────────────────────

/** Parse cumsum(A, dim, direction, nanflag) etc.: returns the dim args
 *  for cumOp. Only the default 'forward' direction and 'includenan' are
 *  supported. */
function cumArgs(name: string, rest: RuntimeValue[]): RuntimeValue[] {
  const out = rest.slice();
  while (out.length > 0 && isText(out[out.length - 1])) {
    const f = textOf(out[out.length - 1]);
    if (f !== "forward" && f !== "includenan")
      throw new RuntimeError(`${name}: the '${f}' option is not supported`);
    out.pop();
  }
  if (out.length > 1) throw new RuntimeError(`${name}: too many arguments`);
  return out.length === 1 ? [dimArg(out[0], name)] : [];
}

function cumBuiltin(
  name: string,
  accumFn: (acc: number, val: number) => number,
  initial: number | undefined,
  complexAccumFn:
    | ((aRe: number, aIm: number, bRe: number, bIm: number) => [number, number])
    | undefined,
  keepLogical: boolean
): void {
  builtin({
    name,
    singleOutputs: [1],
    nargin: [1, 4],
    apply: args => {
      if (isRuntimeChar(args[0]))
        throw builtinError(
          `MATLAB:${name}:wrongInput`,
          "Invalid data type. First input argument must be numeric or logical."
        );
      const a0 = args[0];
      const v = dense(
        dataArg(
          a0,
          name,
          "Invalid data type. First input argument must be numeric or logical."
        )
      );
      const dims = cumArgs(name, args.slice(1));
      if (typeof v === "number")
        return keepLogical && typeof a0 === "boolean" ? a0 : v;
      if (isRuntimeComplexNumber(v)) return v;
      const r = cumOp(
        name,
        [v as RuntimeValue, ...dims],
        accumFn,
        initial,
        complexAccumFn
      );
      if (
        keepLogical &&
        isRuntimeTensor(a0) &&
        a0.cls === "logical" &&
        isRuntimeTensor(r)
      )
        withElementClass(r, "logical");
      return r;
    },
  });
}

cumBuiltin("cumsum", (acc, val) => acc + val, 0, undefined, false);
cumBuiltin(
  "cumprod",
  (acc, val) => acc * val,
  1,
  (aRe, aIm, bRe, bIm) => [aRe * bRe - aIm * bIm, aRe * bIm + aIm * bRe],
  false
);

// cummax/cummin order complex values by magnitude, then phase angle. A NaN
// propagates once it enters the running value (unlike MATLAB, which skips
// NaN); the JIT matches this, see numbl_test_scripts/reduction/
// jit_cummax_cummin.m.
function cumExtreme(
  name: string,
  realFn: (a: number, b: number) => number,
  better: (a: number, b: number) => boolean
) {
  const complexFn = (
    aRe: number,
    aIm: number,
    bRe: number,
    bIm: number
  ): [number, number] => {
    if (aRe !== aRe || aIm !== aIm || bRe !== bRe || bIm !== bIm)
      return [NaN, 0];
    const absA = Math.hypot(aRe, aIm);
    const absB = Math.hypot(bRe, bIm);
    const takeB =
      absA !== absB
        ? better(absB, absA)
        : better(Math.atan2(bIm, bRe), Math.atan2(aIm, aRe));
    return takeB ? [bRe, bIm] : [aRe, aIm];
  };
  cumBuiltin(name, realFn, undefined, complexFn, true);
}

cumExtreme("cummax", Math.max, (a, b) => a > b);
cumExtreme("cummin", Math.min, (a, b) => a < b);

// ── diff ───────────────────────────────────────────────────────────────

builtin({
  name: "diff",
  singleOutputs: true,
  nargin: [1, 3],
  apply: (args): RuntimeValue => {
    const v = dense(numericArg(args[0]));
    let n = 1;
    if (
      args.length >= 2 &&
      !(isRuntimeTensor(args[1]) && args[1].data.length === 0)
    ) {
      const err = () =>
        builtinError(
          "MATLAB:diff:differenceOrderMustBePositiveInteger",
          "Difference order N must be a positive integer scalar."
        );
      try {
        n = realScalarArg(args[1]);
      } catch {
        throw err();
      }
      if (!Number.isInteger(n) || n < 1) throw err();
    }
    const dim = args.length >= 3 ? dimArg(args[2], "diff") : undefined;
    // A scalar has no differences.
    if (typeof v === "number" || isRuntimeComplexNumber(v))
      return RTV.tensor(allocFloat64Array(0), [0, 0]);
    let result = v as RuntimeValue;
    for (let i = 0; i < n; i++) result = diffOnce(result, dim);
    return result;
  },
});

// ── xor ────────────────────────────────────────────────────────────────

builtin({
  name: "xor",
  nargin: 2,
  apply: (args): RuntimeValue => {
    const a = realDenseArg(args[0], mustBeRealError);
    const b = realDenseArg(args[1], mustBeRealError);
    return mLogical(OpCmp.XOR, a, b);
  },
});
