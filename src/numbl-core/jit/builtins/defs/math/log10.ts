import { defineUnaryRealMath } from "./_unary_real.js";
import { signIsNonneg } from "../../../lowering/types.js";
import { cLog10 } from "./_complex_fold.js";

/** Base-10 log. Same non-negative real-domain rule as `log`; negative
 *  inputs lift to the complex path. */
export const log10 = defineUnaryRealMath({
  name: "log10",
  cFnReal: "log10",
  jsFn: Math.log10,
  signRule: () => "unknown",
  realDomainOk: t => signIsNonneg(t.sign),
  complex: {
    cFnComplex: "mtoc2_clog10",
    jsFnComplex: cLog10,
    liftOnDomainMiss: true,
  },
});
