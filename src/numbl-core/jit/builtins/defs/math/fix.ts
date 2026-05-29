import { defineUnaryRealMath, roundingSignRule } from "./_unary_real.js";
import { cFix } from "./_complex_fold.js";

/** `fix(x)` truncates toward zero (C99 `trunc`). Complex inputs truncate
 *  each component independently. Both `positive` and `negative` inputs
 *  may collapse to 0 (e.g. `fix(±0.5) = 0`). */
export const fix = defineUnaryRealMath({
  name: "fix",
  cFnReal: "trunc",
  jsFn: Math.trunc,
  // JS has no `Math.fix`; the textual form for emitJs uses `Math.trunc`.
  jsExpr: arg => `Math.trunc(${arg})`,
  signRule: roundingSignRule(true, true),
  complex: { cFnComplex: "mtoc2_cfix", jsFnComplex: cFix },
});
