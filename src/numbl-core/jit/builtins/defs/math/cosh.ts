import { defineUnaryRealMath } from "./_unary_real.js";
import { jit_ccosh } from "../../runtime/snippets.gen.js";

// cosh(x) >= 1 for all real x, so the result is strictly positive.
// Entire function: real input → real output (no domain miss).
export const cosh = defineUnaryRealMath({
  name: "cosh",
  cFnReal: "cosh",
  jsFn: Math.cosh,
  signRule: () => "positive",
  complex: { cFnComplex: "jit_ccosh", jsFnComplex: jit_ccosh },
});
