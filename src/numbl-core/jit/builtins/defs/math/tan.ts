import { defineUnaryRealMath } from "./_unary_real.js";
import { cTan } from "./_complex_fold.js";

export const tan = defineUnaryRealMath({
  name: "tan",
  cFnReal: "tan",
  jsFn: Math.tan,
  signRule: () => "unknown",
  complex: { cFnComplex: "mtoc2_ctan", jsFnComplex: cTan },
});
