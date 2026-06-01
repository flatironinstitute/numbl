import type { Builtin } from "../../registry.js";
import { defineElemwiseRealBinary, signDiff } from "./_elemwise.js";

export const minus: Builtin = defineElemwiseRealBinary({
  name: "minus",
  cOp: "-",
  helperBase: "minus",
  commutative: false,
  fold: (a, b) => a - b,
  signRule: signDiff,
  complexFold: (a, b) => ({ re: a.re - b.re, im: a.im - b.im }),
});
