/**
 * Pure-TS implementation of Bessel tensor ops.
 * Mirrors native/ops/bessel.c — fallback when the native addon is unavailable.
 *
 * Delegates the per-element math to src/numbl-core/helpers/bessel.ts so both
 * the scalar interpreter path and the tensor path share the same algorithm.
 */

import { OpBessel } from "./opCodes.js";
import { besselj, bessely, besseli, besselk } from "../helpers/bessel.js";

/**
 * out[i] = bessel<OP>(nu, z[i]).  scale mirrors the behavior of numbl_bessel_real.
 */
export function tsBesselReal(
  op: number,
  nu: number,
  n: number,
  z: Float64Array,
  scale: number,
  out: Float64Array
): void {
  const scaled = scale !== 0;
  switch (op) {
    case OpBessel.J:
      if (scaled) {
        for (let i = 0; i < n; i++)
          out[i] = besselj(nu, z[i]) * Math.exp(-Math.abs(z[i]));
      } else {
        for (let i = 0; i < n; i++) out[i] = besselj(nu, z[i]);
      }
      return;
    case OpBessel.Y:
      if (scaled) {
        for (let i = 0; i < n; i++)
          out[i] = bessely(nu, z[i]) * Math.exp(-Math.abs(z[i]));
      } else {
        for (let i = 0; i < n; i++) out[i] = bessely(nu, z[i]);
      }
      return;
    case OpBessel.I:
      if (scaled) {
        for (let i = 0; i < n; i++)
          out[i] = besseli(nu, z[i]) * Math.exp(-Math.abs(z[i]));
      } else {
        for (let i = 0; i < n; i++) out[i] = besseli(nu, z[i]);
      }
      return;
    case OpBessel.K:
      if (scaled) {
        for (let i = 0; i < n; i++) out[i] = besselk(nu, z[i]) * Math.exp(z[i]);
      } else {
        for (let i = 0; i < n; i++) out[i] = besselk(nu, z[i]);
      }
      return;
    default:
      throw new Error(`tsBesselReal: unknown op ${op}`);
  }
}

/**
 * Hankel function for real z:
 *   kKind=1 → out[i] = J_nu(z[i]) + i * Y_nu(z[i])
 *   kKind=2 → out[i] = J_nu(z[i]) - i * Y_nu(z[i])
 * scaled=1 multiplies by exp(-i*z) (k=1) or exp(+i*z) (k=2).
 */
export function tsBesselH(
  kKind: number,
  nu: number,
  n: number,
  z: Float64Array,
  scale: number,
  outRe: Float64Array,
  outIm: Float64Array
): void {
  const ysign = kKind === 1 ? 1 : -1;
  if (scale === 0) {
    for (let i = 0; i < n; i++) {
      outRe[i] = besselj(nu, z[i]);
      outIm[i] = ysign * bessely(nu, z[i]);
    }
    return;
  }
  const ssign = kKind === 1 ? -1 : 1;
  for (let i = 0; i < n; i++) {
    const zi = z[i];
    const J = besselj(nu, zi);
    const Y = ysign * bessely(nu, zi);
    const c = Math.cos(zi);
    const s = ssign * Math.sin(zi);
    outRe[i] = J * c - Y * s;
    outIm[i] = J * s + Y * c;
  }
}
