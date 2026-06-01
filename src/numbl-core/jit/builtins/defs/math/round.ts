import { defineUnaryRealMath, roundingSignRule } from "./_unary_real.js";
import { cRound } from "./_complex_fold.js";

/** MATLAB-style `round`: half-away-from-zero (NOT JS `Math.round`, which
 *  rounds half toward +Inf). The compile-time JS fold uses the same
 *  formula numbl uses; the runtime calls a `mtoc2_round_half_away`
 *  helper backed by C99's `round()` (which is half-away-from-zero).
 *  Both `positive` and `negative` inputs may round to 0 (e.g.
 *  `round(±0.4) = 0`).
 *
 *  Complex inputs round each component independently (MATLAB
 *  convention). */
function matlabRound(x: number): number {
  return Math.sign(x) * Math.round(Math.abs(x));
}

export const round = defineUnaryRealMath({
  name: "round",
  cFnReal: "mtoc2_round_half_away",
  jsFn: matlabRound,
  // MATLAB-style half-away-from-zero — JS `Math.round` is half-toward-
  // +Inf, so we render the matlabRound formula inline. (Kept inline
  // rather than as a paired `.js` snippet because it's a single
  // expression and avoids snippet activation overhead.)
  jsExpr: arg => `(Math.sign(${arg}) * Math.round(Math.abs(${arg})))`,
  signRule: roundingSignRule(true, true),
  complex: { cFnComplex: "mtoc2_cround", jsFnComplex: cRound },
});
