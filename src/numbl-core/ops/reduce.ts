/**
 * Pure-TS flat reductions.  Mirrors native/ops/reduce.c.
 * Caller-allocated input and output (1-element).
 */

import { OpReduce } from "./opCodes.js";

export function tsRealFlatReduce(
  op: number,
  n: number,
  a: Float64Array,
  out: Float64Array
): void {
  switch (op) {
    case OpReduce.SUM: {
      let s = 0;
      for (let i = 0; i < n; i++) s += a[i];
      out[0] = s;
      return;
    }
    case OpReduce.PROD: {
      let p = 1;
      for (let i = 0; i < n; i++) p *= a[i];
      out[0] = p;
      return;
    }
    case OpReduce.MAX: {
      let m = -Infinity;
      for (let i = 0; i < n; i++) {
        const v = a[i];
        if (Number.isNaN(v)) {
          m = v;
          break;
        }
        if (v > m) m = v;
      }
      out[0] = m;
      return;
    }
    case OpReduce.MIN: {
      let m = Infinity;
      for (let i = 0; i < n; i++) {
        const v = a[i];
        if (Number.isNaN(v)) {
          m = v;
          break;
        }
        if (v < m) m = v;
      }
      out[0] = m;
      return;
    }
    case OpReduce.ANY: {
      let r = 0;
      for (let i = 0; i < n; i++) {
        if (a[i] !== 0 || Number.isNaN(a[i])) {
          r = 1;
          break;
        }
      }
      out[0] = r;
      return;
    }
    case OpReduce.ALL: {
      let r = 1;
      for (let i = 0; i < n; i++) {
        if (a[i] === 0) {
          r = 0;
          break;
        }
      }
      out[0] = r;
      return;
    }
    case OpReduce.MEAN: {
      if (n === 0) {
        out[0] = NaN;
        return;
      }
      let s = 0;
      for (let i = 0; i < n; i++) s += a[i];
      out[0] = s / n;
      return;
    }
    default:
      throw new Error(`tsRealFlatReduce: unknown op ${op}`);
  }
}

export function tsComplexFlatReduce(
  op: number,
  n: number,
  aRe: Float64Array,
  aIm: Float64Array | null,
  outRe: Float64Array,
  outIm: Float64Array | null
): void {
  switch (op) {
    case OpReduce.SUM: {
      if (!outIm)
        throw new Error("tsComplexFlatReduce(SUM): outIm is required");
      let sr = 0,
        si = 0;
      for (let i = 0; i < n; i++) {
        sr += aRe[i];
        if (aIm) si += aIm[i];
      }
      outRe[0] = sr;
      outIm[0] = si;
      return;
    }
    case OpReduce.PROD: {
      if (!outIm)
        throw new Error("tsComplexFlatReduce(PROD): outIm is required");
      let arAcc = 1,
        aiAcc = 0;
      for (let i = 0; i < n; i++) {
        const ar = aRe[i];
        const ai = aIm ? aIm[i] : 0;
        const nr = arAcc * ar - aiAcc * ai;
        const ni = arAcc * ai + aiAcc * ar;
        arAcc = nr;
        aiAcc = ni;
      }
      outRe[0] = arAcc;
      outIm[0] = aiAcc;
      return;
    }
    case OpReduce.ANY: {
      let r = 0;
      for (let i = 0; i < n; i++) {
        const ar = aRe[i];
        const ai = aIm ? aIm[i] : 0;
        if (ar !== 0 || ai !== 0 || Number.isNaN(ar) || Number.isNaN(ai)) {
          r = 1;
          break;
        }
      }
      outRe[0] = r;
      return;
    }
    case OpReduce.ALL: {
      let r = 1;
      for (let i = 0; i < n; i++) {
        const ar = aRe[i];
        const ai = aIm ? aIm[i] : 0;
        if (ar === 0 && ai === 0) {
          r = 0;
          break;
        }
      }
      outRe[0] = r;
      return;
    }
    default:
      throw new Error(`tsComplexFlatReduce: unsupported op ${op}`);
  }
}
