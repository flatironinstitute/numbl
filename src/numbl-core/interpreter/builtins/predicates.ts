/**
 * Predicate builtins: isnan, isinf, isfinite, isreal.
 */

import {
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { JitType } from "../../jit/jitTypes.js";
import {
  defineBuiltin,
  predicateCases,
  scalarConstantJitEmitC,
  unaryPredicateJitEmitC,
} from "./types.js";

// ── isnan ───────────────────────────────────────────────────────────────

defineBuiltin({
  name: "isnan",
  cases: predicateCases(
    Number.isNaN,
    (re, im) => Number.isNaN(re) || Number.isNaN(im),
    Number.isNaN,
    (re, im) => Number.isNaN(re) || Number.isNaN(im),
    "isnan"
  ),
  jitEmit: (args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "boolean") return `Number.isNaN(${args[0]})`;
    return null;
  },
  jitEmitC: unaryPredicateJitEmitC("numbl_is_nan"),
});

// ── isinf ───────────────────────────────────────────────────────────────

function isInfVal(x: number): boolean {
  return !isFinite(x) && !Number.isNaN(x);
}

defineBuiltin({
  name: "isinf",
  cases: predicateCases(
    isInfVal,
    (re, im) => isInfVal(re) || isInfVal(im),
    isInfVal,
    (re, im) => isInfVal(re) || isInfVal(im),
    "isinf"
  ),
  jitEmit: (args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "boolean")
      return `(Math.abs(${args[0]}) === Infinity)`;
    return null;
  },
  jitEmitC: unaryPredicateJitEmitC("numbl_is_inf"),
});

// ── isfinite ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "isfinite",
  cases: predicateCases(
    isFinite,
    (re, im) => isFinite(re) && isFinite(im),
    isFinite,
    (re, im) => isFinite(re) && isFinite(im),
    "isfinite"
  ),
  jitEmit: (args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "boolean") return `isFinite(${args[0]})`;
    return null;
  },
  jitEmitC: unaryPredicateJitEmitC("numbl_is_finite"),
});

// ── isreal ──────────────────────────────────────────────────────────────

defineBuiltin({
  name: "isreal",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const k = argTypes[0].kind;
        if (k === "unknown") return null;
        return [{ kind: "boolean" }];
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
    },
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "boolean") return "true";
    if (
      k === "tensor" &&
      (types[0] as Extract<JitType, { kind: "tensor" }>).isComplex === false
    )
      return "true";
    return null;
  },
  // Real scalars (C-JIT tensors are always real too) are always real.
  // Complex never reaches the C-JIT scalar path.
  jitEmitC: scalarConstantJitEmitC({ number: "1.0", boolean: "1.0" }),
});
