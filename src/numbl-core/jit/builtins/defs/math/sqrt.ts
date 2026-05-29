import { defineUnaryRealMath } from "./_unary_real.js";
import { signIsNonneg, signIsPositive } from "../../../lowering/types.js";
import { cSqrt } from "./_complex_fold.js";

/** `sqrt(x)`: real input that's provably non-negative stays on the
 *  real path; everything else lifts to the complex path
 *  (`mtoc2_csqrt` for scalars, `mtoc2_tensor_sqrt_complex` for
 *  tensors). Matches MATLAB: `sqrt(-1)` returns `0 + 1i`,
 *  `sqrt([-1 4])` returns `[0+1i, 2]`.
 *
 *  Sign rule on the (real-path) input:
 *   - `positive` → `positive` (sqrt of strictly positive is strictly positive)
 *   - everything else (`nonneg` / `zero`) → `nonneg`.
 */
export const sqrt = defineUnaryRealMath({
  name: "sqrt",
  cFnReal: "sqrt",
  jsFn: x => Math.sqrt(x),
  signRule: t => (signIsPositive(t.sign) ? "positive" : "nonneg"),
  realDomainOk: t => signIsNonneg(t.sign),
  complex: {
    cFnComplex: "mtoc2_csqrt",
    jsFnComplex: cSqrt,
    liftOnDomainMiss: true,
  },
});
