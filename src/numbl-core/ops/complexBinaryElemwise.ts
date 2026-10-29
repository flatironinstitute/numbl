/**
 * Pure-TS implementation of complex binary element-wise ops, split storage.
 * Mirrors native/ops/complex_binary_elemwise.c.
 *
 * Caller-allocated input/output buffers; never copies.
 * aIm/bIm/arrIm may be null → treat as zero.
 * outIm is always required (caller allocates even if result might be real).
 */

import { OpBin } from "./opCodes.js";
import { cdivResult, complexDivide } from "./complexDivide.js";

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
    case OpBin.ADD:
      for (let i = 0; i < n; i++) {
        outRe[i] = aRe[i] + bRe[i];
        outIm[i] = (aIm ? aIm[i] : 0) + (bIm ? bIm[i] : 0);
      }
      return;
    case OpBin.SUB:
      for (let i = 0; i < n; i++) {
        outRe[i] = aRe[i] - bRe[i];
        outIm[i] = (aIm ? aIm[i] : 0) - (bIm ? bIm[i] : 0);
      }
      return;
    case OpBin.MUL:
      for (let i = 0; i < n; i++) {
        const ar = aRe[i],
          ai = aIm ? aIm[i] : 0;
        const br = bRe[i],
          bi = bIm ? bIm[i] : 0;
        outRe[i] = ar * br - ai * bi;
        outIm[i] = ar * bi + ai * br;
      }
      return;
    case OpBin.DIV:
      for (let i = 0; i < n; i++) {
        const ar = aRe[i],
          ai = aIm ? aIm[i] : 0;
        complexDivide(ar, ai, bRe[i], bIm ? bIm[i] : 0);
        outRe[i] = cdivResult.re;
        outIm[i] = cdivResult.im;
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
    case OpBin.ADD:
      for (let i = 0; i < n; i++) {
        outRe[i] = sRe + arrRe[i];
        outIm[i] = sIm + (arrIm ? arrIm[i] : 0);
      }
      return;
    case OpBin.SUB:
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
    case OpBin.MUL:
      for (let i = 0; i < n; i++) {
        const ar = arrRe[i];
        const ai = arrIm ? arrIm[i] : 0;
        outRe[i] = sRe * ar - sIm * ai;
        outIm[i] = sRe * ai + sIm * ar;
      }
      return;
    case OpBin.DIV:
      for (let i = 0; i < n; i++) {
        const ar = arrRe[i];
        const ai = arrIm ? arrIm[i] : 0;
        if (scalarOnLeft) complexDivide(sRe, sIm, ar, ai);
        else complexDivide(ar, ai, sRe, sIm);
        outRe[i] = cdivResult.re;
        outIm[i] = cdivResult.im;
      }
      return;
    default:
      throw new Error(`tsComplexScalarBinaryElemwise: unknown op ${op}`);
  }
}
