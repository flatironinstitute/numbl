import { defineUnaryRealMath } from "./_unary_real.js";
import { jit_csin } from "../../runtime/snippets.gen.js";

export const sin = defineUnaryRealMath({
  name: "sin",
  cFnReal: "sin",
  jsFn: Math.sin,
  signRule: () => "unknown",
  complex: { cFnComplex: "jit_csin", jsFnComplex: jit_csin },
});
