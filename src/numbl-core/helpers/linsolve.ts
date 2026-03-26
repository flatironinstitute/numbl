/**
 * linsolve(A, B) — solve the linear system A * X = B.
 *
 * If A is square (m × m), uses LU factorisation (dgesv / dgetrf+solve).
 * If A is non-square, uses QR / LQ factorisation (dgels / dgeqrf+solve):
 *   overdetermined  (m > n): least-squares solution minimising ||A*X - B||₂
 *   underdetermined (m < n): minimum-norm solution minimising ||X||₂
 *
 * Both the native LAPACK addon and the ts-lapack TypeScript fallback are
 * supported; the native addon is preferred when available.
 */

import { RuntimeError } from "../runtime/index.js";
import { FloatXArrayType } from "../runtime/types.js";
import { getEffectiveBridge } from "../native/bridge-resolve.js";
import { toF64 } from "./check-helpers.js";

// ── LAPACK helpers ────────────────────────────────────────────────────────────

/**
 * Dispatch linsolve to the native addon (preferred) or ts-lapack fallback.
 * Returns null only when neither bridge exposes `linsolve` (should not happen
 * in practice since ts-lapack always has it).
 */
export function linsolveLapack(
  A: FloatXArrayType,
  m: number,
  n: number,
  B: FloatXArrayType,
  nrhs: number
): Float64Array | null {
  const bridge = getEffectiveBridge("linsolve", "linsolve");
  if (!bridge?.linsolve) return null;

  return bridge.linsolve(toF64(A), m, n, toF64(B), nrhs);
}

/**
 * Dispatch complex linsolve to the native addon (preferred) or ts-lapack fallback.
 * The ts-lapack fallback throws — native addon is required for complex linsolve.
 */
export function linsolveComplexLapack(
  ARe: FloatXArrayType,
  AIm: FloatXArrayType,
  m: number,
  n: number,
  BRe: FloatXArrayType,
  BIm: FloatXArrayType,
  nrhs: number
): { re: Float64Array; im: Float64Array } {
  const bridge = getEffectiveBridge("linsolveComplex", "linsolveComplex");
  if (!bridge?.linsolveComplex)
    throw new RuntimeError(
      "linsolveComplex: no bridge available (should not happen)"
    );

  return bridge.linsolveComplex(
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
