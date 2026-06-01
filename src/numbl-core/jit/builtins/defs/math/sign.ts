import { defineUnaryRealMath } from "./_unary_real.js";
import { cSign } from "./_complex_fold.js";

/** `sign(x)`:
 *   - Real input ∈ {-1, 0, 1}; the result's sign mirrors the input's.
 *   - Complex input → `z / |z|` for nonzero, `0 + 0i` for `0 + 0i`.
 *     The complex sign is itself complex. */
export const signBuiltin = defineUnaryRealMath({
  name: "sign",
  cFnReal: "mtoc2_signum",
  jsFn: Math.sign,
  signRule: t => t.sign,
  complex: { cFnComplex: "mtoc2_csign", jsFnComplex: cSign },
});
