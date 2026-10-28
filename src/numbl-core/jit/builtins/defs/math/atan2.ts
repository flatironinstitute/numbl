import { defineElemwiseRealBinaryFn } from "../arithmetic/_elemwise.js";

/** Two-argument arctangent. Result is in (-π, π]; sign is generally
 *  `unknown`. */
export const atan2 = defineElemwiseRealBinaryFn({
  name: "atan2",
  cFn: "atan2",
  helperBase: "atan2",
  commutative: false,
  fold: Math.atan2,
  jsScalarExpr: (a, b) => `Math.atan2(${a}, ${b})`,
  signRule: () => "unknown",
  runtimeDep: "jit_tensor_elemwise_real_fn",
});
