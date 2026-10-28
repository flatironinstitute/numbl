import { defineUnaryRealMath } from "./_unary_real.js";
import { jit_catan } from "../../runtime/snippets.gen.js";

export const atan = defineUnaryRealMath({
  name: "atan",
  cFnReal: "atan",
  jsFn: Math.atan,
  signRule: t => t.sign,
  complex: { cFnComplex: "jit_catan", jsFnComplex: jit_catan },
});
