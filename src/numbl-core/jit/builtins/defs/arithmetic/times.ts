import type { Builtin } from "../../registry.js";
import { defineElemwiseRealBinary, signProd } from "./_elemwise.js";

export const times: Builtin = defineElemwiseRealBinary({
  name: "times",
  cOp: "*",
  helperBase: "times",
  commutative: true,
  fold: (a, b) => a * b,
  signRule: signProd,
  complexFold: (a, b) => ({
    re: a.re * b.re - a.im * b.im,
    im: a.re * b.im + a.im * b.re,
  }),
});
