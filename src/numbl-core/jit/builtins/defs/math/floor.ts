import { defineUnaryRealMath, roundingSignRule } from "./_unary_real.js";
import { cFloor } from "./_complex_fold.js";

/** `floor(x)` rounds toward -Inf. Complex inputs round each component
 *  independently (MATLAB convention). Only `positive` inputs may land
 *  on 0 (e.g. `floor(0.5) = 0`); `negative` inputs are bounded away. */
export const floor = defineUnaryRealMath({
  name: "floor",
  cFnReal: "floor",
  jsFn: Math.floor,
  signRule: roundingSignRule(true, false),
  complex: { cFnComplex: "mtoc2_cfloor", jsFnComplex: cFloor },
});
