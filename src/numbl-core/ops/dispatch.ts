/**
 * Tensor-ops entry points: thin wrappers over the LAPACK bridge's tensorOp*
 * kernels (native addon when installed, the TS kernels in this directory
 * otherwise; see native/lapack-bridge.ts).
 *
 * Caller owns input AND output memory; functions never allocate output.
 * Element-wise ops skip the bridge entirely for empty inputs.
 */

import { getLapackBridge } from "../native/lapack-bridge.js";

export const tensorOps = {
  realBinaryElemwise(
    op: number,
    n: number,
    a: Float64Array,
    b: Float64Array,
    out: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpRealBinary(op, n, a, b, out);
  },

  realScalarBinaryElemwise(
    op: number,
    n: number,
    scalar: number,
    arr: Float64Array,
    scalarOnLeft: boolean,
    out: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpRealScalarBinary(
      op,
      n,
      scalar,
      arr,
      scalarOnLeft,
      out
    );
  },

  complexBinaryElemwise(
    op: number,
    n: number,
    aRe: Float64Array,
    aIm: Float64Array | null,
    bRe: Float64Array,
    bIm: Float64Array | null,
    outRe: Float64Array,
    outIm: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpComplexBinary(
      op,
      n,
      aRe,
      aIm,
      bRe,
      bIm,
      outRe,
      outIm
    );
  },

  complexScalarBinaryElemwise(
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
    if (n === 0) return;
    getLapackBridge().tensorOpComplexScalarBinary(
      op,
      n,
      sRe,
      sIm,
      arrRe,
      arrIm,
      scalarOnLeft,
      outRe,
      outIm
    );
  },

  realUnaryElemwise(
    op: number,
    n: number,
    a: Float64Array,
    out: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpRealUnary(op, n, a, out);
  },

  complexUnaryElemwise(
    op: number,
    n: number,
    aRe: Float64Array,
    aIm: Float64Array | null,
    outRe: Float64Array,
    outIm: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpComplexUnary(op, n, aRe, aIm, outRe, outIm);
  },

  complexAbs(
    n: number,
    aRe: Float64Array,
    aIm: Float64Array | null,
    out: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpComplexAbs(n, aRe, aIm, out);
  },

  realComparison(
    op: number,
    n: number,
    a: Float64Array,
    b: Float64Array,
    out: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpRealComparison(op, n, a, b, out);
  },

  realScalarComparison(
    op: number,
    n: number,
    scalar: number,
    arr: Float64Array,
    scalarOnLeft: boolean,
    out: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpRealScalarComparison(
      op,
      n,
      scalar,
      arr,
      scalarOnLeft,
      out
    );
  },

  complexComparison(
    op: number,
    n: number,
    aRe: Float64Array,
    aIm: Float64Array | null,
    bRe: Float64Array,
    bIm: Float64Array | null,
    out: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpComplexComparison(op, n, aRe, aIm, bRe, bIm, out);
  },

  complexScalarComparison(
    op: number,
    n: number,
    sRe: number,
    sIm: number,
    arrRe: Float64Array,
    arrIm: Float64Array | null,
    scalarOnLeft: boolean,
    out: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpComplexScalarComparison(
      op,
      n,
      sRe,
      sIm,
      arrRe,
      arrIm,
      scalarOnLeft,
      out
    );
  },

  realFlatReduce(
    op: number,
    n: number,
    a: Float64Array,
    out: Float64Array
  ): void {
    getLapackBridge().tensorOpRealFlatReduce(op, n, a, out);
  },

  complexFlatReduce(
    op: number,
    n: number,
    aRe: Float64Array,
    aIm: Float64Array | null,
    outRe: Float64Array,
    outIm: Float64Array | null
  ): void {
    getLapackBridge().tensorOpComplexFlatReduce(op, n, aRe, aIm, outRe, outIm);
  },

  besselReal(
    op: number,
    nu: number,
    n: number,
    z: Float64Array,
    scale: number,
    out: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpBesselReal(op, nu, n, z, scale, out);
  },

  besselH(
    kKind: number,
    nu: number,
    n: number,
    z: Float64Array,
    scale: number,
    outRe: Float64Array,
    outIm: Float64Array
  ): void {
    if (n === 0) return;
    getLapackBridge().tensorOpBesselH(kKind, nu, n, z, scale, outRe, outIm);
  },
};
