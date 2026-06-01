import { defineUnaryRealMath } from "./_unary_real.js";
import { cExp } from "./_complex_fold.js";

export const exp = defineUnaryRealMath({
  name: "exp",
  cFnReal: "exp",
  jsFn: Math.exp,
  signRule: () => "positive",
  complex: { cFnComplex: "mtoc2_cexp", jsFnComplex: cExp },
});
