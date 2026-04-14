/**
 * Pure-TS implementation of real unary element-wise ops.
 * Mirrors native/ops/real_unary_elemwise.c.
 */

import { OpUnary } from "./opCodes.js";

function rsign(x: number): number {
  return x > 0 ? 1 : x < 0 ? -1 : 0;
}

export function tsRealUnaryElemwise(
  op: number,
  n: number,
  a: Float64Array,
  out: Float64Array
): void {
  switch (op) {
    case OpUnary.EXP:
      for (let i = 0; i < n; i++) out[i] = Math.exp(a[i]);
      return;
    case OpUnary.LOG:
      for (let i = 0; i < n; i++) out[i] = Math.log(a[i]);
      return;
    case OpUnary.LOG2:
      for (let i = 0; i < n; i++) out[i] = Math.log2(a[i]);
      return;
    case OpUnary.LOG10:
      for (let i = 0; i < n; i++) out[i] = Math.log10(a[i]);
      return;
    case OpUnary.SQRT:
      for (let i = 0; i < n; i++) out[i] = Math.sqrt(a[i]);
      return;
    case OpUnary.ABS:
      for (let i = 0; i < n; i++) out[i] = Math.abs(a[i]);
      return;
    case OpUnary.FLOOR:
      for (let i = 0; i < n; i++) out[i] = Math.floor(a[i]);
      return;
    case OpUnary.CEIL:
      for (let i = 0; i < n; i++) out[i] = Math.ceil(a[i]);
      return;
    case OpUnary.ROUND:
      // MATLAB round: half-away-from-zero. Math.round is half-up (towards +Inf),
      // which differs for negative halves (e.g. Math.round(-0.5)=0 vs MATLAB -1).
      for (let i = 0; i < n; i++) {
        const x = a[i];
        out[i] = x >= 0 ? Math.round(x) : -Math.round(-x);
      }
      return;
    case OpUnary.TRUNC:
      for (let i = 0; i < n; i++) out[i] = Math.trunc(a[i]);
      return;
    case OpUnary.SIN:
      for (let i = 0; i < n; i++) out[i] = Math.sin(a[i]);
      return;
    case OpUnary.COS:
      for (let i = 0; i < n; i++) out[i] = Math.cos(a[i]);
      return;
    case OpUnary.TAN:
      for (let i = 0; i < n; i++) out[i] = Math.tan(a[i]);
      return;
    case OpUnary.ASIN:
      for (let i = 0; i < n; i++) out[i] = Math.asin(a[i]);
      return;
    case OpUnary.ACOS:
      for (let i = 0; i < n; i++) out[i] = Math.acos(a[i]);
      return;
    case OpUnary.ATAN:
      for (let i = 0; i < n; i++) out[i] = Math.atan(a[i]);
      return;
    case OpUnary.SINH:
      for (let i = 0; i < n; i++) out[i] = Math.sinh(a[i]);
      return;
    case OpUnary.COSH:
      for (let i = 0; i < n; i++) out[i] = Math.cosh(a[i]);
      return;
    case OpUnary.TANH:
      for (let i = 0; i < n; i++) out[i] = Math.tanh(a[i]);
      return;
    case OpUnary.SIGN:
      for (let i = 0; i < n; i++) out[i] = rsign(a[i]);
      return;
    default:
      throw new Error(`tsRealUnaryElemwise: unknown op ${op}`);
  }
}
