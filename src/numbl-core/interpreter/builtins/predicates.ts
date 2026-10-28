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
  isRuntimeStringArray,
  isRuntimeFunction,
} from "../../runtime/types.js";
import { builtin } from "./builtin.js";
import { unaryPredicate } from "./types.js";
import { imagAllZero } from "../../helpers/effectively-real.js";

// ── isnan ───────────────────────────────────────────────────────────────

builtin({
  name: "isnan",
  nargin: 1,
  apply: unaryPredicate(
    Number.isNaN,
    (re, im) => Number.isNaN(re) || Number.isNaN(im)
  ),
});

// ── isinf ───────────────────────────────────────────────────────────────

function isInfVal(x: number): boolean {
  return !isFinite(x) && !Number.isNaN(x);
}

builtin({
  name: "isinf",
  nargin: 1,
  apply: unaryPredicate(isInfVal, (re, im) => isInfVal(re) || isInfVal(im)),
});

// ── isfinite ────────────────────────────────────────────────────────────

builtin({
  name: "isfinite",
  nargin: 1,
  apply: unaryPredicate(isFinite, (re, im) => isFinite(re) && isFinite(im)),
});

// ── isreal ──────────────────────────────────────────────────────────────

builtin({
  name: "isreal",
  nargin: 1,
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
      isRuntimeStringArray(v) ||
      isRuntimeFunction(v)
    )
      return false;
    return true;
  },
});
