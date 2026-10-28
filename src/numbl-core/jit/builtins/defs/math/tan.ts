import { defineUnaryRealMath } from "./_unary_real.js";
import { jit_ctan } from "../../runtime/snippets.gen.js";

export const tan = defineUnaryRealMath({
  name: "tan",
  cFnReal: "tan",
  jsFn: Math.tan,
  signRule: () => "unknown",
  complex: { cFnComplex: "jit_ctan", jsFnComplex: jit_ctan },
});
