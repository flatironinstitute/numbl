import { defineUnaryRealMath } from "./_unary_real.js";
import { jit_ccos } from "../../runtime/snippets.gen.js";

export const cos = defineUnaryRealMath({
  name: "cos",
  cFnReal: "cos",
  jsFn: Math.cos,
  signRule: () => "unknown",
  complex: { cFnComplex: "jit_ccos", jsFnComplex: jit_ccos },
});
