import type { Builtin } from "../../registry.js";
import { defineElemwiseRealBinary, signSum } from "./_elemwise.js";

export const plus: Builtin = defineElemwiseRealBinary({
  name: "plus",
  cOp: "+",
  helperBase: "plus",
  commutative: true,
  fold: (a, b) => a + b,
  signRule: signSum,
  complexFold: (a, b) => ({ re: a.re + b.re, im: a.im + b.im }),
});
