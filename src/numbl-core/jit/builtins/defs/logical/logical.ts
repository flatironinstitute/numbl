/**
 * `logical(x)` — convert a real value to logical: nonzero → 1, zero →
 * 0, elementwise. Mirrors numbl's `logical` (a NaN is nonzero, so it
 * maps to 1; numbl does not raise on NaN). Complex input is rejected
 * by `requireRealDouble`.
 */
import { defineUnaryPred } from "./_unary_pred.js";

export const logical = defineUnaryPred({
  name: "logical",
  cScalar: arg => `((${arg}) != 0.0)`,
  jsScalar: arg => `((${arg}) !== 0)`,
  jsFn: x => x !== 0,
  tensorHelper: "mtoc2_tensor_predicate",
});
