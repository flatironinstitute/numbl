import { defineUnaryRealMath } from "./_unary_real.js";
import { cSin } from "./_complex_fold.js";

export const sin = defineUnaryRealMath({
  name: "sin",
  cFnReal: "sin",
  jsFn: Math.sin,
  signRule: () => "unknown",
  complex: { cFnComplex: "jit_csin", jsFnComplex: cSin },
});
