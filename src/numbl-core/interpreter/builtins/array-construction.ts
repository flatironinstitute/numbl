/**
 * Array construction builtins: zeros, ones, nan/NaN, eye, linspace, logspace.
 */

import { FloatXArray, isRuntimeTensor } from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { toNumber, numel } from "../../runtime/index.js";
import type { SignCategory } from "../jit/jitTypes.js";
import { defineBuiltin, type BuiltinCase, makeTensor } from "./types.js";

// ── Helpers ──────────────────────────────────────────────────────────────

/** Parse shape arguments: zeros(2,3) or zeros([2,3]) -> [2, 3].
 *  Negative dimensions are clamped to 0. */
function parseShapeArgs(args: RuntimeValue[]): number[] {
  if (args.length === 1 && isRuntimeTensor(args[0])) {
    const t = args[0];
    const shape: number[] = [];
    for (let i = 0; i < t.data.length; i++)
      shape.push(Math.max(0, Math.round(t.data[i])));
    return shape;
  }
  return args.map(a => Math.max(0, Math.round(toNumber(a))));
}

/** Build cases for array constructors that take shape args and return a real tensor. */
function arrayConstructorCases(
  fillFn: (shape: number[]) => RuntimeValue,
  scalarValue: RuntimeValue,
  opts?: { scalarSign?: SignCategory; nonneg?: boolean }
): BuiltinCase[] {
  return [
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
        if (a.kind === "number" || a.kind === "boolean" || a.kind === "tensor")
          return [
            {
              kind: "tensor" as const,
              isComplex: false,
              shape: [-1, -1],
              ...(opts?.nonneg ? { nonneg: true } : {}),
            },
          ];
        return null;
      },
      apply: args => {
        const shape = parseShapeArgs(args);
        if (shape.length === 1) shape.push(shape[0]);
        return fillFn(shape);
      },
    },
    // Multiple scalar args
    {
      match: argTypes => {
        if (argTypes.length <= 1) return null;
        for (const a of argTypes) {
          if (a.kind !== "number" && a.kind !== "boolean") return null;
        }
        return [
          {
            kind: "tensor" as const,
            isComplex: false,
            shape: new Array(argTypes.length).fill(-1),
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
}

// ── zeros ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "zeros",
  cases: arrayConstructorCases(
    shape => makeTensor(new FloatXArray(numel(shape)), undefined, shape),
    0,
    { scalarSign: "nonneg", nonneg: true }
  ),
});

// ── ones ─────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "ones",
  cases: arrayConstructorCases(
    shape => {
      const data = new FloatXArray(numel(shape));
      data.fill(1);
      return makeTensor(data, undefined, shape);
    },
    1,
    { scalarSign: "positive", nonneg: true }
  ),
});

// ── nan / NaN ────────────────────────────────────────────────────────────

function nanFill(shape: number[]): RuntimeValue {
  const data = new FloatXArray(numel(shape));
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
          apply: (args: RuntimeValue[]) => {
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
              rows = Math.max(0, Math.round(toNumber(args[0])));
              cols = Math.max(0, Math.round(toNumber(args[1])));
            }
            const data = new FloatXArray(rows * cols);
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
        if (n <= 0) return makeTensor(new FloatXArray(0), undefined, [1, 0]);
        if (n === 1)
          return makeTensor(new FloatXArray([end]), undefined, [1, 1]);
        const data = new FloatXArray(n);
        for (let i = 0; i < n; i++) {
          data[i] = start + ((end - start) * i) / (n - 1);
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
        if (n <= 0) return makeTensor(new FloatXArray(0), undefined, [1, 0]);
        const isPi = b === Math.PI;
        const endVal = isPi ? Math.PI : Math.pow(10, b);
        const startVal = Math.pow(10, a);
        if (n === 1)
          return makeTensor(new FloatXArray([endVal]), undefined, [1, 1]);
        const data = new FloatXArray(n);
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
