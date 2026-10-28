import { defineUnaryRealMath } from "./_unary_real.js";
import { signIsNonneg } from "../../../lowering/types.js";
import { jit_clog2 } from "../../runtime/snippets.gen.js";

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
    cFnComplex: "jit_clog2",
    jsFnComplex: jit_clog2,
    liftOnDomainMiss: true,
  },
});
