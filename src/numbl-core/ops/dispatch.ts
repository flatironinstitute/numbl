/**
 * Tensor-ops dispatcher: chooses native bridge or pure-TS fallback.
 *
 * All entry points have identical signatures across native and TS impls.
 * Caller owns input AND output memory; functions never allocate output.
 *
 * Selection happens once on first use per category and is cached
 * (the bridge identity is captured at module load via getLapackBridge()).
 */

import { getLapackBridge } from "../native/lapack-bridge.js";
import {
  tsRealBinaryElemwise,
  tsRealScalarBinaryElemwise,
} from "./realBinaryElemwise.js";
import {
  tsComplexBinaryElemwise,
  tsComplexScalarBinaryElemwise,
} from "./complexBinaryElemwise.js";
import { tsRealUnaryElemwise } from "./realUnaryElemwise.js";
import {
  tsComplexUnaryElemwise,
  tsComplexAbs,
} from "./complexUnaryElemwise.js";

export const tensorOps = {
  realBinaryElemwise(
    op: number,
    n: number,
    a: Float64Array,
    b: Float64Array,
    out: Float64Array
  ): void {
    const bridge = getLapackBridge();
    if (bridge?.tensorOpRealBinary) {
      bridge.tensorOpRealBinary(op, n, a, b, out);
    } else {
      tsRealBinaryElemwise(op, n, a, b, out);
    }
  },

  realScalarBinaryElemwise(
    op: number,
    n: number,
    scalar: number,
    arr: Float64Array,
    scalarOnLeft: boolean,
    out: Float64Array
  ): void {
    const bridge = getLapackBridge();
    if (bridge?.tensorOpRealScalarBinary) {
      bridge.tensorOpRealScalarBinary(op, n, scalar, arr, scalarOnLeft, out);
    } else {
      tsRealScalarBinaryElemwise(op, n, scalar, arr, scalarOnLeft, out);
    }
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
    const bridge = getLapackBridge();
    if (bridge?.tensorOpComplexBinary) {
      bridge.tensorOpComplexBinary(op, n, aRe, aIm, bRe, bIm, outRe, outIm);
    } else {
      tsComplexBinaryElemwise(op, n, aRe, aIm, bRe, bIm, outRe, outIm);
    }
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
    const bridge = getLapackBridge();
    if (bridge?.tensorOpComplexScalarBinary) {
      bridge.tensorOpComplexScalarBinary(
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
    } else {
      tsComplexScalarBinaryElemwise(
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
    }
  },

  realUnaryElemwise(
    op: number,
    n: number,
    a: Float64Array,
    out: Float64Array
  ): void {
    const bridge = getLapackBridge();
    if (bridge?.tensorOpRealUnary) {
      bridge.tensorOpRealUnary(op, n, a, out);
    } else {
      tsRealUnaryElemwise(op, n, a, out);
    }
  },

  complexUnaryElemwise(
    op: number,
    n: number,
    aRe: Float64Array,
    aIm: Float64Array | null,
    outRe: Float64Array,
    outIm: Float64Array
  ): void {
    const bridge = getLapackBridge();
    if (bridge?.tensorOpComplexUnary) {
      bridge.tensorOpComplexUnary(op, n, aRe, aIm, outRe, outIm);
    } else {
      tsComplexUnaryElemwise(op, n, aRe, aIm, outRe, outIm);
    }
  },

  complexAbs(
    n: number,
    aRe: Float64Array,
    aIm: Float64Array | null,
    out: Float64Array
  ): void {
    const bridge = getLapackBridge();
    if (bridge?.tensorOpComplexAbs) {
      bridge.tensorOpComplexAbs(n, aRe, aIm, out);
    } else {
      tsComplexAbs(n, aRe, aIm, out);
    }
  },
};
