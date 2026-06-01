import { defineElemwiseRealBinaryFn } from "../arithmetic/_elemwise.js";

/** `hypot(a, b) = sqrt(a^2 + b^2)`. Always `nonneg` (C99 `hypot`
 *  guarantees no spurious overflow for finite inputs). */
export const hypot = defineElemwiseRealBinaryFn({
  name: "hypot",
  cFn: "hypot",
  helperBase: "hypot",
  commutative: true,
  fold: Math.hypot,
  jsScalarExpr: (a, b) => `Math.hypot(${a}, ${b})`,
  signRule: () => "nonneg",
  runtimeDep: "mtoc2_tensor_elemwise_real_fn",
});
