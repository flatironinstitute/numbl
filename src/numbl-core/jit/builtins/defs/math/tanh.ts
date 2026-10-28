import { defineUnaryRealMath } from "./_unary_real.js";
import { jit_ctanh } from "../../runtime/snippets.gen.js";

// tanh is odd and monotonic increasing, so it preserves the input's
// sign. Entire on the real line: real input → real output.
export const tanh = defineUnaryRealMath({
  name: "tanh",
  cFnReal: "tanh",
  jsFn: Math.tanh,
  signRule: t => t.sign,
  complex: { cFnComplex: "jit_ctanh", jsFnComplex: jit_ctanh },
});
