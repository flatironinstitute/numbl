import { defineUnaryRealMath } from "./_unary_real.js";
import { signIsPositive } from "../../../lowering/types.js";
import { exactDouble, exactRealArray } from "../_shared.js";
import { jit_realsqrt } from "../../runtime/snippets.gen.js";

/** `realsqrt(x)`: square root that stays real. Unlike `sqrt`, an input
 *  of unknown sign stays on the real path; a negative element raises
 *  MATLAB's error at run time instead of lifting to complex. An input
 *  known to be negative (sign or exact value) declines, so the
 *  interpreter raises the error. */
export const realsqrt = defineUnaryRealMath({
  name: "realsqrt",
  cFnReal: "jit_realsqrt",
  jsFnReal: "jit_realsqrt",
  jsFn: jit_realsqrt as (x: number) => number,
  signRule: t => (signIsPositive(t.sign) ? "positive" : "nonneg"),
  realDomainOk: t => {
    if (t.sign === "negative") return false;
    const x = exactDouble(t);
    if (x !== undefined) return !(x < 0);
    const arr = exactRealArray(t);
    return arr === undefined || !arr.some(v => v < 0);
  },
});
