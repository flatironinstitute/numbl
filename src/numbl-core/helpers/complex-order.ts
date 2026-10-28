/**
 * MATLAB's sort order for numeric values, shared by sort, sortrows, unique,
 * the set operations and groupcounts.
 *
 * Real values compare by value. Complex values compare by magnitude, then by
 * phase angle in (-pi, pi]. In both cases a value with a NaN part sorts
 * after every other value (ascending order).
 */

/** Ascending comparison of two real values, NaN last. */
export function cmpReal(a: number, b: number): number {
  const an = a !== a;
  const bn = b !== b;
  if (an || bn) return an === bn ? 0 : an ? 1 : -1;
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Ascending comparison of two complex values (abs, then angle), NaN last. */
export function cmpComplex(
  aRe: number,
  aIm: number,
  bRe: number,
  bIm: number
): number {
  const an = aRe !== aRe || aIm !== aIm;
  const bn = bRe !== bRe || bIm !== bIm;
  if (an || bn) return an === bn ? 0 : an ? 1 : -1;
  const ma = Math.hypot(aRe, aIm);
  const mb = Math.hypot(bRe, bIm);
  if (ma !== mb) return ma < mb ? -1 : 1;
  // A zero imaginary part counts as +0, so a negative real has angle pi.
  const pa = Math.atan2(aIm === 0 ? 0 : aIm, aRe);
  const pb = Math.atan2(bIm === 0 ? 0 : bIm, bRe);
  return pa < pb ? -1 : pa > pb ? 1 : 0;
}

/** True when a value has a NaN real or imaginary part. */
export function isNaNValue(re: number, im = 0): boolean {
  return re !== re || im !== im;
}
