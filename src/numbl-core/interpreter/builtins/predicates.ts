/**
 * Predicate builtins: isnan, isinf, isfinite, isreal.
 */

import {
  FloatXArray,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { JitType } from "../jit/jitTypes.js";
import { registerIBuiltin, makeTensor } from "./types.js";

/** Type rule for predicates: any numeric type → produces logical */
function predicateType(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 1) return null;
  const a = argTypes[0];
  switch (a.kind) {
    case "number":
    case "logical":
    case "complex":
      return [{ kind: "logical" }];
    case "realTensor":
      return [
        { kind: "realTensor", shape: a.shape, nonneg: true, isLogical: true },
      ];
    case "complexTensor":
      return [
        { kind: "realTensor", shape: a.shape, nonneg: true, isLogical: true },
      ];
    default:
      return null;
  }
}

function makeBoolTensor(flags: boolean[], shape: number[]) {
  const out = new FloatXArray(flags.length);
  for (let i = 0; i < flags.length; i++) out[i] = flags[i] ? 1 : 0;
  const t = makeTensor(out, undefined, shape);
  t._isLogical = true;
  return t;
}

// ── isnan ───────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "isnan",
  typeRule: argTypes => predicateType(argTypes),
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return Number.isNaN(v);
    if (isRuntimeComplexNumber(v))
      return Number.isNaN(v.re) || Number.isNaN(v.im);
    if (isRuntimeTensor(v)) {
      const n = v.data.length;
      const flags: boolean[] = new Array(n);
      if (!v.imag) {
        for (let i = 0; i < n; i++) flags[i] = Number.isNaN(v.data[i]);
      } else {
        for (let i = 0; i < n; i++)
          flags[i] = Number.isNaN(v.data[i]) || Number.isNaN(v.imag[i]);
      }
      return makeBoolTensor(flags, v.shape.slice());
    }
    throw new Error("isnan: unsupported argument type");
  },
  jitEmit: (args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical")
      return `(Number.isNaN(${args[0]}) ? 1 : 0)`;
    return null;
  },
});

// ── isinf ───────────────────────────────────────────────────────────────

function isInfVal(x: number): boolean {
  return !isFinite(x) && !Number.isNaN(x);
}

registerIBuiltin({
  name: "isinf",
  typeRule: argTypes => predicateType(argTypes),
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return isInfVal(v);
    if (isRuntimeComplexNumber(v)) return isInfVal(v.re) || isInfVal(v.im);
    if (isRuntimeTensor(v)) {
      const n = v.data.length;
      const flags: boolean[] = new Array(n);
      if (!v.imag) {
        for (let i = 0; i < n; i++) flags[i] = isInfVal(v.data[i]);
      } else {
        for (let i = 0; i < n; i++)
          flags[i] = isInfVal(v.data[i]) || isInfVal(v.imag[i]);
      }
      return makeBoolTensor(flags, v.shape.slice());
    }
    throw new Error("isinf: unsupported argument type");
  },
  jitEmit: (args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical")
      return `(Math.abs(${args[0]}) === Infinity ? 1 : 0)`;
    return null;
  },
});

// ── isfinite ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "isfinite",
  typeRule: argTypes => predicateType(argTypes),
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return isFinite(v);
    if (isRuntimeComplexNumber(v)) return isFinite(v.re) && isFinite(v.im);
    if (isRuntimeTensor(v)) {
      const n = v.data.length;
      const flags: boolean[] = new Array(n);
      if (!v.imag) {
        for (let i = 0; i < n; i++) flags[i] = isFinite(v.data[i]);
      } else {
        for (let i = 0; i < n; i++)
          flags[i] = isFinite(v.data[i]) && isFinite(v.imag[i]);
      }
      return makeBoolTensor(flags, v.shape.slice());
    }
    throw new Error("isfinite: unsupported argument type");
  },
  jitEmit: (args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical")
      return `(isFinite(${args[0]}) ? 1 : 0)`;
    return null;
  },
});

// ── isreal ──────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "isreal",
  typeRule: argTypes => {
    if (argTypes.length !== 1) return null;
    return [{ kind: "logical" }];
  },
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return true;
    if (typeof v === "boolean") return true;
    if (isRuntimeComplexNumber(v)) return v.im === 0;
    if (isRuntimeTensor(v)) return !v.imag;
    if (isRuntimeSparseMatrix(v)) return !v.pi;
    return true;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical") return "1";
    if (k === "realTensor") return "1";
    return null;
  },
});
