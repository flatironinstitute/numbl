import { defineUnaryRealMath } from "./_unary_real.js";
import { signIsNonneg } from "../../../lowering/types.js";
import { cLog } from "./_complex_fold.js";

/** Natural log. Provably non-negative real inputs stay on the real
 *  path (`log(0) = -Inf` is still real-typed in MATLAB); anything that
 *  could be negative lifts to the complex path. */
export const log = defineUnaryRealMath({
  name: "log",
  cFnReal: "log",
  jsFn: Math.log,
  signRule: () => "unknown",
  realDomainOk: t => signIsNonneg(t.sign),
  complex: {
    cFnComplex: "mtoc2_clog",
    jsFnComplex: cLog,
    liftOnDomainMiss: true,
  },
});
