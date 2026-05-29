import { defineElemwiseRealBinaryFn } from "../arithmetic/_elemwise.js";

/** `rem(a, b)`: same as JS `%` and C `fmod` — result sign tracks `a`'s
 *  sign. */
export const rem = defineElemwiseRealBinaryFn({
  name: "rem",
  cFn: "fmod",
  helperBase: "rem",
  commutative: false,
  fold: (a, b) => a % b,
  // JS `%` matches C `fmod` semantics (result sign tracks `a`'s sign).
  jsScalarExpr: (a, b) => `((${a}) % (${b}))`,
  signRule: a => a.sign,
  runtimeDep: "mtoc2_tensor_elemwise_real_fn",
});
