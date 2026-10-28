/**
 * Array construction builtins: zeros, ones, nan/NaN, inf/Inf, eye,
 * linspace, logspace.
 */

import {
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeFunction,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeStruct,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { toString, numel } from "../../runtime/index.js";
import { makeTensor, mkc } from "./types.js";
import { builtin } from "./builtin.js";
import { builtinError, numericArg } from "./args.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { parseShapeArgs, validateDim } from "../../helpers/shape-utils.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";

// ── Helpers ──────────────────────────────────────────────────────────────

// MATLAB's size constructors (zeros, ones, eye, rand, …) accept a trailing
// class-name string — `zeros(2,3,'int32')` — and the `'like'` prototype form
// — `zeros(sz,'like',p)`. numbl only has the double type, so we accept and
// ignore the class spec rather than erroring on the extra arg.

/** Drop a trailing class-name spec from the runtime args (used in `apply`). */
export function stripClassNameArgs(args: RuntimeValue[]): RuntimeValue[] {
  const n = args.length;
  if (n === 0) return args;
  const isStr = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);
  // `'like'` keyword: drop it and the prototype (and anything after).
  for (let i = 0; i < n; i++) {
    if (isStr(args[i]) && toString(args[i]).toLowerCase() === "like")
      return args.slice(0, i);
  }
  if (isStr(args[n - 1])) return args.slice(0, n - 1);
  return args;
}

const isText = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);

const ALL_NUMERIC_CLASSES = [
  "double",
  "single",
  "logical",
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
];
const FLOAT_CLASSES = ["double", "single"];

interface SizeCtorSpec {
  /** MATLAB's spelling in error identifiers (zeros, ones, eye, NaN, Inf). */
  idName: string;
  /** Class names accepted (and ignored) after the size arguments. */
  classes: string[];
}

/** Split a size constructor's arguments into the size arguments, checking
 *  a trailing class name or `'like', prototype` pair the way MATLAB does.
 *  The class itself is ignored (numbl only has doubles). Each size
 *  argument must be real numeric. */
function sizeCtorArgs(
  args: RuntimeValue[],
  spec: SizeCtorSpec
): RuntimeValue[] {
  const { idName } = spec;
  const upper = idName.toUpperCase();
  const badClass = () =>
    idName === "NaN" || idName === "Inf"
      ? builtinError(
          `MATLAB:${idName}:invalidClassName`,
          `Class name must be a class that supports ${upper}, such as "double" or "single".`
        )
      : builtinError(
          `MATLAB:${idName}:invalidStringInput`,
          idName === "eye"
            ? `Class name must be a class that supports ${upper}, such as "single" or "double".`
            : `Class name must be a class that supports ${upper}, such as "double" or "single".`
        );
  let n = args.length;
  if (
    n >= 2 &&
    isText(args[n - 2]) &&
    toString(args[n - 2]).toLowerCase() === "like"
  ) {
    const proto = args[n - 1];
    if (
      isRuntimeCell(proto) ||
      isRuntimeStruct(proto) ||
      isRuntimeFunction(proto) ||
      isText(proto)
    )
      throw builtinError(
        `MATLAB:${idName}:invalidInputClass`,
        `Argument following "like" must be a variable with a data type that supports ${upper}, such as a double or single variable.`
      );
    n -= 2;
  } else if (n >= 1 && isText(args[n - 1])) {
    if (!spec.classes.includes(toString(args[n - 1]))) throw badClass();
    n -= 1;
  }
  const sizeArgs = args.slice(0, n);
  for (const a of sizeArgs) {
    if (
      typeof a === "number" ||
      typeof a === "boolean" ||
      isRuntimeTensor(a) ||
      isRuntimeSparseMatrix(a)
    ) {
      if (isRuntimeTensor(a) && a.imag)
        throw builtinError(
          "MATLAB:obsoleteComplexInput",
          "Size inputs must be real."
        );
      continue;
    }
    if (isRuntimeComplexNumber(a))
      throw builtinError(
        "MATLAB:obsoleteComplexInput",
        "Size inputs must be real."
      );
    throw builtinError(
      `MATLAB:${idName}:invalidNumericInput`,
      "Size inputs must be numeric."
    );
  }
  return sizeArgs.map(a => (isRuntimeSparseMatrix(a) ? sparseToDense(a) : a));
}

/** The shape requested by size arguments: none → 1×1, one scalar n → n×n,
 *  one vector → that shape, several scalars → those dims. */
function sizeCtorShape(sizeArgs: RuntimeValue[]): number[] {
  if (sizeArgs.length === 0) return [1, 1];
  const shape = parseShapeArgs(sizeArgs);
  if (shape.length === 0) return [0, 0];
  if (shape.length === 1) shape.push(shape[0]);
  return shape;
}

function filled(shape: number[], value: number): RuntimeValue {
  const data = allocFloat64Array(numel(shape));
  if (value !== 0) data.fill(value);
  return makeTensor(data, undefined, shape);
}

function defineFill(name: string, idName: string, value: number): void {
  const spec: SizeCtorSpec = {
    idName,
    classes:
      Number.isNaN(value) || value === Infinity
        ? FLOAT_CLASSES
        : ALL_NUMERIC_CLASSES,
  };
  builtin({
    name,
    apply: args => {
      const sizeArgs = sizeCtorArgs(args, spec);
      if (sizeArgs.length === 0) return value;
      return filled(sizeCtorShape(sizeArgs), value);
    },
  });
}

defineFill("zeros", "zeros", 0);
defineFill("ones", "ones", 1);
defineFill("nan", "NaN", NaN);
defineFill("NaN", "NaN", NaN);
defineFill("inf", "Inf", Infinity);
defineFill("Inf", "Inf", Infinity);

// ── eye ──────────────────────────────────────────────────────────────────

builtin({
  name: "eye",
  apply: args => {
    const sizeArgs = sizeCtorArgs(args, {
      idName: "eye",
      classes: ALL_NUMERIC_CLASSES,
    });
    if (sizeArgs.length === 0) return 1;
    const shape = sizeCtorShape(sizeArgs);
    while (shape.length > 2 && shape[shape.length - 1] === 1) shape.pop();
    if (shape.length > 2)
      throw builtinError(
        "MATLAB:eye:unsupportedNd",
        "N-dimensional arrays are not supported."
      );
    const [rows, cols] = shape.map(validateDim);
    const data = allocFloat64Array(rows * cols);
    const minDim = Math.min(rows, cols);
    for (let i = 0; i < minDim; i++) data[i * rows + i] = 1;
    return makeTensor(data, undefined, [rows, cols]);
  },
});

// ── linspace / logspace ──────────────────────────────────────────────────

/** A scalar endpoint for linspace/logspace as [re, im]. */
function scalarEndpoint(v: RuntimeValue): [number, number] {
  const a = numericArg(v);
  if (typeof a === "number") return [a, 0];
  if (isRuntimeComplexNumber(a)) return [a.re, a.im];
  const t = isRuntimeSparseMatrix(a) ? sparseToDense(a) : a;
  if (t.data.length !== 1)
    throw builtinError(
      "MATLAB:linspace:scalarInputs",
      "Inputs must be scalars."
    );
  return [t.data[0], t.imag ? t.imag[0] : 0];
}

/** The point count argument: a real scalar, floored as MATLAB does. */
function pointCount(v: RuntimeValue): number {
  const [re, im] = isRuntimeString(v) ? [Number(v), 0] : scalarEndpoint(v);
  if (im !== 0)
    throw builtinError(
      "MATLAB:colon:operandsNotRealScalar",
      "Colon operands must be real scalars."
    );
  return Math.floor(re);
}

/** n evenly spaced points from start to end (real parts; the imaginary
 *  parts are spaced the same way when either endpoint is complex). */
function linspaceData(start: number, end: number, n: number): Float64Array {
  const data = allocFloat64Array(n);
  if (n === 1) {
    data[0] = end;
    return data;
  }
  // MATLAB preserves both endpoints exactly (so NaN/Inf at one end don't
  // contaminate the other).
  data[0] = start;
  data[n - 1] = end;
  for (let i = 1; i < n - 1; i++) {
    data[i] = start + ((end - start) * i) / (n - 1);
  }
  // Opposite-sign infinite endpoints: MATLAB places 0 at the exact center
  // for odd n.
  if (
    (n & 1) === 1 &&
    !Number.isFinite(start) &&
    !Number.isFinite(end) &&
    Math.sign(start) !== Math.sign(end)
  ) {
    data[(n - 1) / 2] = 0;
  }
  return data;
}

builtin({
  name: "linspace",
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    const [s, sIm] = scalarEndpoint(args[0]);
    const [e, eIm] = scalarEndpoint(args[1]);
    const n = args.length === 3 ? pointCount(args[2]) : 100;
    if (Number.isNaN(n)) return NaN;
    if (n <= 0) return makeTensor(allocFloat64Array(0), undefined, [1, 0]);
    const re = linspaceData(s, e, n);
    if (sIm === 0 && eIm === 0) return makeTensor(re, undefined, [1, n]);
    const im = linspaceData(sIm, eIm, n);
    if (n === 1) return mkc(re[0], im[0]);
    return makeTensor(re, im, [1, n]);
  },
});

builtin({
  name: "logspace",
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    const [a, aIm] = scalarEndpoint(args[0]);
    const [b, bIm] = scalarEndpoint(args[1]);
    const n = args.length === 3 ? pointCount(args[2]) : 50;
    if (n <= 0) return makeTensor(allocFloat64Array(0), undefined, [1, 0]);
    if (aIm !== 0 || bIm !== 0) {
      // 10.^linspace(a, b, n) on complex exponents.
      const tRe = linspaceData(a, b, n);
      const tIm = linspaceData(aIm, bIm, n);
      const re = allocFloat64Array(n);
      const im = allocFloat64Array(n);
      for (let i = 0; i < n; i++) {
        const mag = Math.pow(10, tRe[i]);
        const ang = tIm[i] * Math.LN10;
        re[i] = mag * Math.cos(ang);
        im[i] = mag * Math.sin(ang);
      }
      if (n === 1) return mkc(re[0], im[0]);
      return makeTensor(re, im, [1, n]);
    }
    const isPi = b === Math.PI;
    const endVal = isPi ? Math.PI : Math.pow(10, b);
    const startVal = Math.pow(10, a);
    if (n === 1)
      return makeTensor(allocFloat64Array([endVal]), undefined, [1, 1]);
    const data = allocFloat64Array(n);
    if (isPi) {
      const logStart = Math.log10(startVal);
      const logEnd = Math.log10(Math.PI);
      for (let i = 0; i < n; i++) {
        const t = logStart + ((logEnd - logStart) * i) / (n - 1);
        data[i] = Math.pow(10, t);
      }
    } else {
      for (let i = 0; i < n; i++) {
        const t = a + ((b - a) * i) / (n - 1);
        data[i] = Math.pow(10, t);
      }
    }
    return makeTensor(data, undefined, [1, n]);
  },
});
