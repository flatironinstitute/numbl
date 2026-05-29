import { defineUnaryRealMath } from "./_unary_real.js";
import { cSin } from "./_complex_fold.js";

export const sin = defineUnaryRealMath({
  name: "sin",
  cFnReal: "sin",
  jsFn: Math.sin,
  signRule: () => "unknown",
  complex: { cFnComplex: "mtoc2_csin", jsFnComplex: cSin },
});
