/**
 * Predicate builtins: isnan, isinf, isfinite, isreal.
 */

import {
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeString,
  isRuntimeFunction,
} from "../../runtime/types.js";
import { defineBuiltin, predicateCases } from "./types.js";
import { imagAllZero } from "../../helpers/effectively-real.js";

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
        // A complex tensor whose imaginary lane is entirely zero is real
        // in value (consistent with the complex-scalar `v.im === 0` test
        // above, and with the JIT, which routinely produces such tensors
        // when it cannot prove realness at compile time).
        if (isRuntimeTensor(v)) return imagAllZero(v.imag);
        if (isRuntimeSparseMatrix(v)) return !v.pi || imagAllZero(v.pi);
        // MATLAB: isreal is false for cells, structs, strings, and function
        // handles. char and other numeric-ish values are real (fall through).
        if (
          isRuntimeCell(v) ||
          isRuntimeStruct(v) ||
          isRuntimeStructArray(v) ||
          isRuntimeString(v) ||
          isRuntimeFunction(v)
        )
          return false;
        return true;
      },
    },
  ],
});
