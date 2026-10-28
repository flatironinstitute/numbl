import { defineUnaryRealMath } from "./_unary_real.js";
import { cAtan } from "./_complex_fold.js";

export const atan = defineUnaryRealMath({
  name: "atan",
  cFnReal: "atan",
  jsFn: Math.atan,
  signRule: t => t.sign,
  complex: { cFnComplex: "jit_catan", jsFnComplex: cAtan },
});
