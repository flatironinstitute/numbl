/**
 * linsolve(A, B) — solve the linear system A * X = B.
 *
 * If A is square (m × m), uses LU factorisation (dgesv / dgetrf+solve).
 * If A is non-square, uses QR / LQ factorisation (dgels / dgeqrf+solve):
 *   overdetermined  (m > n): least-squares solution minimising ||A*X - B||₂
 *   underdetermined (m < n): minimum-norm solution minimising ||X||₂
 */

import { getLapackBridge } from "../native/lapack-bridge.js";
import { toF64 } from "./check-helpers.js";

// ── LAPACK helpers ────────────────────────────────────────────────────────────

/** Real A \ B through the LAPACK bridge. */
export function linsolveLapack(
  A: Float64Array,
  m: number,
  n: number,
  B: Float64Array,
  nrhs: number
): Float64Array {
  return getLapackBridge().linsolve(toF64(A), m, n, toF64(B), nrhs);
}

/** Complex A \ B through the LAPACK bridge. */
export function linsolveComplexLapack(
  ARe: Float64Array,
  AIm: Float64Array,
  m: number,
  n: number,
  BRe: Float64Array,
  BIm: Float64Array,
  nrhs: number
): { re: Float64Array; im: Float64Array } {
  return getLapackBridge().linsolveComplex(
    toF64(ARe),
    toF64(AIm),
    m,
    n,
    toF64(BRe),
    toF64(BIm),
    nrhs
  );
}

// ── Builtin registration ──────────────────────────────────────────────────────
