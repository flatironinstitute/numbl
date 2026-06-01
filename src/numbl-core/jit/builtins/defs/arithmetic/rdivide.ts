import type { Builtin } from "../../registry.js";
import { defineElemwiseRealBinary, signQuot } from "./_elemwise.js";

/** Complex scalar division uses Smith's algorithm under the hood. The
 *  `mtoc2_cdiv` runtime helper centralizes the math so numbl-compatible
 *  ±0 / ±Inf handling lives in one place. */
function cdivFold(
  a: { re: number; im: number },
  b: { re: number; im: number }
): { re: number; im: number } {
  // Smith's algorithm: pick the scaling so the divisor's larger
  // magnitude component goes in the denominator. This matches the
  // runtime helper's behavior for finite inputs.
  if (Math.abs(b.re) >= Math.abs(b.im)) {
    const r = b.im / b.re;
    const den = b.re + r * b.im;
    return { re: (a.re + a.im * r) / den, im: (a.im - a.re * r) / den };
  }
  const r = b.re / b.im;
  const den = b.im + r * b.re;
  return { re: (a.re * r + a.im) / den, im: (a.im * r - a.re) / den };
}

export const rdivide: Builtin = defineElemwiseRealBinary({
  name: "rdivide",
  cOp: "/",
  helperBase: "rdivide",
  commutative: false,
  fold: (a, b) => a / b,
  signRule: signQuot,
  complexFold: cdivFold,
  // Complex scalar `/` routes through the runtime helper so signed-Inf
  // edges match numbl. The real-typed scalar path keeps the bare `/`.
  complexScalarExpr: (a, b) => `mtoc2_cdiv(${a}, ${b})`,
  complexRuntimeDeps: ["mtoc2_cdiv"],
});
