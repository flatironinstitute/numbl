import { defineUnaryRealMath } from "./_unary_real.js";
import { cSinh } from "./_complex_fold.js";

// sinh is odd and monotonic increasing, so it preserves the input's
// sign. Entire function: real input → real output (no domain miss).
export const sinh = defineUnaryRealMath({
  name: "sinh",
  cFnReal: "sinh",
  jsFn: Math.sinh,
  signRule: t => t.sign,
  complex: { cFnComplex: "mtoc2_csinh", jsFnComplex: cSinh },
});
