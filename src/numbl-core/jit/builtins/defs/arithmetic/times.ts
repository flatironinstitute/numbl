import type { Builtin } from "../../registry.js";
import { jit_cmul_mixed } from "../../runtime/snippets.gen.js";
import { defineElemwiseRealBinary, signProd } from "./_elemwise.js";

export const times: Builtin = defineElemwiseRealBinary({
  name: "times",
  cOp: "*",
  helperBase: "times",
  commutative: true,
  fold: (a, b) => a * b,
  signRule: signProd,
  // MATLAB's mixed real/complex rule: an operand with zero imag part acts
  // as real and scales each part of the other (no NaN*0 from promotion).
  complexFold: jit_cmul_mixed,
});
