import { defineUnaryRealMath } from "./_unary_real.js";
import { signIsNonneg } from "../../../lowering/types.js";
import { cLog2 } from "./_complex_fold.js";

/** Base-2 log. Same non-negative real-domain rule as `log`; negative
 *  inputs lift to the complex path. Single-output form only; two-output
 *  frexp form `[f,e] = log2(x)` is deferred.
 */
export const log2 = defineUnaryRealMath({
  name: "log2",
  cFnReal: "log2",
  jsFn: Math.log2,
  signRule: () => "unknown",
  realDomainOk: t => signIsNonneg(t.sign),
  complex: {
    cFnComplex: "mtoc2_clog2",
    jsFnComplex: cLog2,
    liftOnDomainMiss: true,
  },
});
