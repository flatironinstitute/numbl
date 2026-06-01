import { defineUnaryRealMath } from "./_unary_real.js";

// asinh is odd and monotonic increasing → preserves the input's sign,
// and is defined for every real x (entire on the real line), so real
// input always yields real output. Real-only: complex asinh has branch
// cuts and falls back to the interpreter.
export const asinh = defineUnaryRealMath({
  name: "asinh",
  cFnReal: "asinh",
  jsFn: Math.asinh,
  signRule: t => t.sign,
});
