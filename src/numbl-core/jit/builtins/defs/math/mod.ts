import { defineElemwiseRealBinaryFn } from "../arithmetic/_elemwise.js";
import type { NumericType, Sign } from "../../../lowering/types.js";

/** MATLAB `mod(a, b)`: result has the sign of `b`. Differs from C's
 *  `fmod` (which has the sign of `a`); the runtime calls our own
 *  `mtoc2_mod_real` helper. Special case: `mod(a, 0) = a`. */
function modFn(a: number, b: number): number {
  if (b === 0) return a;
  let r = a % b;
  if (r !== 0 && r < 0 !== b < 0) r += b;
  return r;
}

function modSign(_a: NumericType, b: NumericType): Sign {
  return b.sign;
}

/** MATLAB-style mod expressed as a JS one-liner. The same formula
 *  as `modFn` above, rendered as a string for inline emitJs. The
 *  IIFE keeps the `r = a % b` temp from being recomputed in the
 *  sign-fixup branch. */
function modJsExpr(aJs: string, bJs: string): string {
  return (
    `(((a, b) => { if (b===0) return a; let r = a % b; ` +
    `if (r !== 0 && (r < 0) !== (b < 0)) r += b; return r; })` +
    `(${aJs}, ${bJs}))`
  );
}

export const mod = defineElemwiseRealBinaryFn({
  name: "mod",
  cFn: "mtoc2_mod_real",
  helperBase: "mod",
  commutative: false,
  fold: modFn,
  jsScalarExpr: modJsExpr,
  signRule: modSign,
  runtimeDep: "mtoc2_tensor_elemwise_real_fn",
});
