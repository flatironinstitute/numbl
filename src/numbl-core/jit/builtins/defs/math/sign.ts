import { defineUnaryRealMath } from "./_unary_real.js";
import { jit_csign } from "../../runtime/snippets.gen.js";

/** `sign(x)`:
 *   - Real input ∈ {-1, 0, 1}; the result's sign mirrors the input's.
 *   - Complex input → `z / |z|` for nonzero, `0 + 0i` for `0 + 0i`.
 *     The complex sign is itself complex. */
export const signBuiltin = defineUnaryRealMath({
  name: "sign",
  cFnReal: "jit_signum",
  // `+ 0` maps Math.sign(-0) = -0 to MATLAB's sign(-0) = 0.
  jsFn: x => Math.sign(x) + 0,
  jsExpr: a => `(Math.sign(${a}) + 0)`,
  signRule: t => t.sign,
  complex: { cFnComplex: "jit_csign", jsFnComplex: jit_csign },
});
