/**
 * Pure-TS implementation of complex binary element-wise ops, split storage.
 * Mirrors native/ops/complex_binary_elemwise.c.
 *
 * Caller-allocated input/output buffers; never copies.
 * aIm/bIm/arrIm may be null → treat as zero.
 * outIm is always required (caller allocates even if result might be real).
 */

import { OpComplexBin } from "./opCodes.js";

function cdivzRe(r: number, i: number): number {
  if (r === 0 && i === 0) return NaN;
  return (r > 0 ? 1 : r < 0 ? -1 : 0) / 0;
}
function cdivzIm(r: number, i: number): number {
  if (r === 0 && i === 0) return 0;
  return (i > 0 ? 1 : i < 0 ? -1 : 0) / 0;
}

export function tsComplexBinaryElemwise(
  op: number,
  n: number,
  aRe: Float64Array,
  aIm: Float64Array | null,
  bRe: Float64Array,
  bIm: Float64Array | null,
  outRe: Float64Array,
  outIm: Float64Array
): void {
  switch (op) {
    case OpComplexBin.ADD:
      for (let i = 0; i < n; i++) {
        outRe[i] = aRe[i] + bRe[i];
        outIm[i] = (aIm ? aIm[i] : 0) + (bIm ? bIm[i] : 0);
      }
      return;
    case OpComplexBin.SUB:
      for (let i = 0; i < n; i++) {
        outRe[i] = aRe[i] - bRe[i];
        outIm[i] = (aIm ? aIm[i] : 0) - (bIm ? bIm[i] : 0);
      }
      return;
    case OpComplexBin.MUL:
      for (let i = 0; i < n; i++) {
        const ar = aRe[i],
          ai = aIm ? aIm[i] : 0;
        const br = bRe[i],
          bi = bIm ? bIm[i] : 0;
        outRe[i] = ar * br - ai * bi;
        outIm[i] = ar * bi + ai * br;
      }
      return;
    case OpComplexBin.DIV:
      for (let i = 0; i < n; i++) {
        const ar = aRe[i],
          ai = aIm ? aIm[i] : 0;
        const br = bRe[i],
          bi = bIm ? bIm[i] : 0;
        const denom = br * br + bi * bi;
        if (denom === 0) {
          outRe[i] = cdivzRe(ar, ai);
          outIm[i] = cdivzIm(ar, ai);
        } else {
          outRe[i] = (ar * br + ai * bi) / denom;
          outIm[i] = (ai * br - ar * bi) / denom;
        }
      }
      return;
    default:
      throw new Error(`tsComplexBinaryElemwise: unknown op ${op}`);
  }
}

export function tsComplexScalarBinaryElemwise(
  op: number,
  n: number,
  sRe: number,
  sIm: number,
  arrRe: Float64Array,
  arrIm: Float64Array | null,
  scalarOnLeft: boolean,
  outRe: Float64Array,
  outIm: Float64Array
): void {
  switch (op) {
    case OpComplexBin.ADD:
      for (let i = 0; i < n; i++) {
        outRe[i] = sRe + arrRe[i];
        outIm[i] = sIm + (arrIm ? arrIm[i] : 0);
      }
      return;
    case OpComplexBin.SUB:
      if (scalarOnLeft) {
        for (let i = 0; i < n; i++) {
          outRe[i] = sRe - arrRe[i];
          outIm[i] = sIm - (arrIm ? arrIm[i] : 0);
        }
      } else {
        for (let i = 0; i < n; i++) {
          outRe[i] = arrRe[i] - sRe;
          outIm[i] = (arrIm ? arrIm[i] : 0) - sIm;
        }
      }
      return;
    case OpComplexBin.MUL:
      for (let i = 0; i < n; i++) {
        const ar = arrRe[i];
        const ai = arrIm ? arrIm[i] : 0;
        outRe[i] = sRe * ar - sIm * ai;
        outIm[i] = sRe * ai + sIm * ar;
      }
      return;
    case OpComplexBin.DIV:
      if (scalarOnLeft) {
        for (let i = 0; i < n; i++) {
          const ar = arrRe[i];
          const ai = arrIm ? arrIm[i] : 0;
          const denom = ar * ar + ai * ai;
          if (denom === 0) {
            outRe[i] = cdivzRe(sRe, sIm);
            outIm[i] = cdivzIm(sRe, sIm);
          } else {
            outRe[i] = (sRe * ar + sIm * ai) / denom;
            outIm[i] = (sIm * ar - sRe * ai) / denom;
          }
        }
      } else {
        const denom = sRe * sRe + sIm * sIm;
        if (denom === 0) {
          for (let i = 0; i < n; i++) {
            const ar = arrRe[i];
            const ai = arrIm ? arrIm[i] : 0;
            outRe[i] = cdivzRe(ar, ai);
            outIm[i] = cdivzIm(ar, ai);
          }
        } else {
          const invDenom = 1 / denom;
          for (let i = 0; i < n; i++) {
            const ar = arrRe[i];
            const ai = arrIm ? arrIm[i] : 0;
            outRe[i] = (ar * sRe + ai * sIm) * invDenom;
            outIm[i] = (ai * sRe - ar * sIm) * invDenom;
          }
        }
      }
      return;
    default:
      throw new Error(`tsComplexScalarBinaryElemwise: unknown op ${op}`);
  }
}
