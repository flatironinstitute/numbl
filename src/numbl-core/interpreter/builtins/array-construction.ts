/**
 * Array construction builtins: zeros, ones, nan/NaN, eye, linspace, logspace.
 */

import {
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
} from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import {
  toNumber,
  toString,
  numel,
  RuntimeError,
} from "../../runtime/index.js";
import type { JitType, SignCategory } from "../../jitTypes.js";
import { defineBuiltin, type BuiltinCase, makeTensor } from "./types.js";
import { allocFloat64Array } from "../../runtime/alloc.js";

// ── Helpers ──────────────────────────────────────────────────────────────

// MATLAB's size constructors (zeros, ones, eye, rand, …) accept a trailing
// class-name string — `zeros(2,3,'int32')` — and the `'like'` prototype form
// — `zeros(sz,'like',p)`. numbl only has the double type, so we accept and
// silently ignore the class spec rather than erroring on the extra arg.

function isClassNameType(t: JitType): boolean {
  return t.kind === "char" || t.kind === "string";
}

/** Drop a trailing class-name spec from the JIT arg types (used in `match`
 *  so the size cases still apply when a class string is present). */
export function stripClassNameTypes(argTypes: JitType[]): JitType[] {
  const n = argTypes.length;
  if (n === 0) return argTypes;
  // `..., 'like', proto`: a string followed by a non-string prototype.
  if (
    n >= 2 &&
    isClassNameType(argTypes[n - 2]) &&
    !isClassNameType(argTypes[n - 1])
  )
    return argTypes.slice(0, n - 2);
  // `..., classname`
  if (isClassNameType(argTypes[n - 1])) return argTypes.slice(0, n - 1);
  return argTypes;
}

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

/** Validate one dimension value: MATLAB rejects NaN, Inf, and non-integer
 *  sizes; negative integers are silently clamped to 0. */
function validateDim(x: number): number {
  if (!Number.isFinite(x) || !Number.isInteger(x))
    throw new RuntimeError("Size inputs must be nonnegative integers.");
  return Math.max(0, x);
}

/** Extract a single scalar dimension. With multiple size args MATLAB requires
 *  each to be a scalar — a 1×1 array counts, but a multi-element vector errors
 *  with "Size inputs must be scalar." */
function toScalarDim(v: RuntimeValue): number {
  if (isRuntimeTensor(v) && v.data.length !== 1)
    throw new RuntimeError("Size inputs must be scalar.");
  return validateDim(toNumber(v));
}

/** Parse shape arguments: zeros(2,3) or zeros([2,3]) -> [2, 3].
 *  Negative dimensions are clamped to 0. */
function parseShapeArgs(args: RuntimeValue[]): number[] {
  if (args.length === 1 && isRuntimeTensor(args[0])) {
    const t = args[0];
    const shape: number[] = [];
    for (let i = 0; i < t.data.length; i++) shape.push(validateDim(t.data[i]));
    return shape;
  }
  return args.map(toScalarDim);
}

/** Build cases for array constructors that take shape args and return a real tensor. */
function arrayConstructorCases(
  fillFn: (shape: number[]) => RuntimeValue,
  scalarValue: RuntimeValue,
  opts?: { scalarSign?: SignCategory; nonneg?: boolean }
): BuiltinCase[] {
  const cases: BuiltinCase[] = [
    // No args: return scalar
    {
      match: argTypes => {
        if (argTypes.length !== 0) return null;
        return [
          {
            kind: "number" as const,
            ...(opts?.scalarSign ? { sign: opts.scalarSign } : {}),
          },
        ];
      },
      apply: () => scalarValue,
    },
    // Single arg: number/boolean → n×n, tensor → shape from tensor
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        if (
          a.kind === "number" ||
          a.kind === "boolean" ||
          a.kind === "tensor"
        ) {
          // Propagate literal `exact` to the n×n static shape when known.
          const n =
            a.kind === "number" && typeof a.exact === "number" ? a.exact : -1;
          return [
            {
              kind: "tensor" as const,
              isComplex: false,
              shape: [n, n],
              ...(opts?.nonneg ? { nonneg: true } : {}),
            },
          ];
        }
        return null;
      },
      apply: args => {
        const shape = parseShapeArgs(args);
        if (shape.length === 1) shape.push(shape[0]);
        return fillFn(shape);
      },
    },
    // Multiple scalar args. A 1×1 array counts as a scalar dimension, so
    // accept `tensor` here too (apply-time validates each is scalar).
    {
      match: argTypes => {
        if (argTypes.length <= 1) return null;
        for (const a of argTypes) {
          if (
            a.kind !== "number" &&
            a.kind !== "boolean" &&
            a.kind !== "tensor"
          )
            return null;
        }
        // Propagate literal `exact` per-dim so e.g. zeros(2, 3) gets
        // [2, 3] instead of [-1, -1].
        const shape = argTypes.map(a =>
          a.kind === "number" && typeof a.exact === "number" ? a.exact : -1
        );
        return [
          {
            kind: "tensor" as const,
            isComplex: false,
            shape,
            ...(opts?.nonneg ? { nonneg: true } : {}),
          },
        ];
      },
      apply: args => {
        const shape = parseShapeArgs(args);
        return fillFn(shape);
      },
    },
  ];
  // Accept (and ignore) a trailing class-name spec on every case.
  return cases.map(c => ({
    match: (argTypes: JitType[], nargout: number) =>
      c.match(stripClassNameTypes(argTypes), nargout),
    apply: (args: RuntimeValue[], nargout: number) =>
      c.apply(stripClassNameArgs(args), nargout),
  }));
}

// ── zeros ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "zeros",
  cases: arrayConstructorCases(
    shape => makeTensor(allocFloat64Array(numel(shape)), undefined, shape),
    0,
    { scalarSign: "nonneg", nonneg: true }
  ),
});

// ── ones ─────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "ones",
  cases: arrayConstructorCases(
    shape => {
      const data = allocFloat64Array(numel(shape));
      data.fill(1);
      return makeTensor(data, undefined, shape);
    },
    1,
    { scalarSign: "positive", nonneg: true }
  ),
});

// ── nan / NaN ────────────────────────────────────────────────────────────

function nanFill(shape: number[]): RuntimeValue {
  const data = allocFloat64Array(numel(shape));
  data.fill(NaN);
  return makeTensor(data, undefined, shape);
}

defineBuiltin({
  name: "nan",
  cases: arrayConstructorCases(nanFill, NaN),
});

defineBuiltin({
  name: "NaN",
  cases: arrayConstructorCases(nanFill, NaN),
});

// ── inf / Inf ────────────────────────────────────────────────────────────

function infFill(shape: number[]): RuntimeValue {
  const data = allocFloat64Array(numel(shape));
  data.fill(Infinity);
  return makeTensor(data, undefined, shape);
}

defineBuiltin({
  name: "inf",
  cases: arrayConstructorCases(infFill, Infinity),
});

defineBuiltin({
  name: "Inf",
  cases: arrayConstructorCases(infFill, Infinity),
});

// ── eye ──────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "eye",
  cases: arrayConstructorCases(
    () => {
      throw new Error("eye: unreachable — uses custom apply");
    },
    1,
    { scalarSign: "nonneg", nonneg: true }
  ).map((c, i) =>
    i === 0
      ? c
      : {
          ...c,
          apply: (rawArgs: RuntimeValue[]) => {
            const args = stripClassNameArgs(rawArgs);
            let rows: number, cols: number;
            if (args.length === 1) {
              const shape = parseShapeArgs(args);
              if (shape.length >= 2) {
                rows = shape[0];
                cols = shape[1];
              } else {
                rows = shape[0];
                cols = rows;
              }
            } else {
              rows = validateDim(toNumber(args[0]));
              cols = validateDim(toNumber(args[1]));
            }
            const data = allocFloat64Array(rows * cols);
            const minDim = Math.min(rows, cols);
            for (let i = 0; i < minDim; i++) {
              data[i * rows + i] = 1;
            }
            return makeTensor(data, undefined, [rows, cols]);
          },
        }
  ),
});

// ── linspace ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "linspace",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        for (const a of argTypes) {
          if (a.kind !== "number" && a.kind !== "boolean") return null;
        }
        return [{ kind: "tensor", isComplex: false, shape: [1, -1] }];
      },
      apply: args => {
        if (args.length < 2 || args.length > 3)
          throw new Error("linspace requires 2 or 3 arguments");
        const start = toNumber(args[0]);
        const end = toNumber(args[1]);
        const n = args.length === 3 ? Math.round(toNumber(args[2])) : 100;
        if (n <= 0) return makeTensor(allocFloat64Array(0), undefined, [1, 0]);
        if (n === 1)
          return makeTensor(allocFloat64Array([end]), undefined, [1, 1]);
        const data = allocFloat64Array(n);
        // MATLAB preserves both endpoints exactly (so NaN/Inf at one end
        // don't contaminate the other).
        data[0] = start;
        data[n - 1] = end;
        for (let i = 1; i < n - 1; i++) {
          data[i] = start + ((end - start) * i) / (n - 1);
        }
        // Opposite-sign infinite endpoints: MATLAB places 0 at the exact
        // center for odd n.
        if (
          (n & 1) === 1 &&
          !Number.isFinite(start) &&
          !Number.isFinite(end) &&
          Math.sign(start) !== Math.sign(end)
        ) {
          data[(n - 1) / 2] = 0;
        }
        return makeTensor(data, undefined, [1, n]);
      },
    },
  ],
});

// ── logspace ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "logspace",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        for (const a of argTypes) {
          if (a.kind !== "number" && a.kind !== "boolean") return null;
        }
        return [{ kind: "tensor", isComplex: false, shape: [1, -1] }];
      },
      apply: args => {
        if (args.length < 2 || args.length > 3)
          throw new Error("logspace requires 2 or 3 arguments");
        const a = toNumber(args[0]);
        const b = toNumber(args[1]);
        const n = args.length === 3 ? Math.round(toNumber(args[2])) : 50;
        if (n <= 0) return makeTensor(allocFloat64Array(0), undefined, [1, 0]);
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
    },
  ],
});
