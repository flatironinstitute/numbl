import { defineUnaryRealMath } from "./_unary_real.js";
import { cCos } from "./_complex_fold.js";

export const cos = defineUnaryRealMath({
  name: "cos",
  cFnReal: "cos",
  jsFn: Math.cos,
  signRule: () => "unknown",
  complex: { cFnComplex: "mtoc2_ccos", jsFnComplex: cCos },
});
