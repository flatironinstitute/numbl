import { defineUnaryRealMath } from "./_unary_real.js";
import { jit_cexp } from "../../runtime/snippets.gen.js";

export const exp = defineUnaryRealMath({
  name: "exp",
  cFnReal: "exp",
  jsFn: Math.exp,
  signRule: () => "positive",
  complex: { cFnComplex: "jit_cexp", jsFnComplex: jit_cexp },
});
