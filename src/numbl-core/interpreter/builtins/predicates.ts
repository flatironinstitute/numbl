/**
 * Predicate builtins: isnan, isinf, isfinite, isreal.
 */

import { isRealValue } from "../../runtime/kinds.js";
import { builtin } from "./builtin.js";
import { unaryPredicate } from "./types.js";

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
  accepts: { sparse: true },
  nargin: 1,
  apply: args => isRealValue(args[0]),
});
