import { defineUnaryRealMath } from "./_unary_real.js";
import { cCosh } from "./_complex_fold.js";

// cosh(x) >= 1 for all real x, so the result is strictly positive.
// Entire function: real input → real output (no domain miss).
export const cosh = defineUnaryRealMath({
  name: "cosh",
  cFnReal: "cosh",
  jsFn: Math.cosh,
  signRule: () => "positive",
  complex: { cFnComplex: "mtoc2_ccosh", jsFnComplex: cCosh },
});
