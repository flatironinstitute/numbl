/**
 * Pure-TS implementation of comparison ops.
 * Mirrors native/ops/comparison.c.
 *
 * Output is 0.0 / 1.0 stored in a Float64Array (numbl's logical-tensor
 * convention — caller wraps the result with _isLogical = true).
 */

import { OpCmp } from "./opCodes.js";

function rcmp(op: number, a: number, b: number): number {
  switch (op) {
    case OpCmp.EQ:
      return a === b ? 1 : 0;
    case OpCmp.NE:
      return a !== b ? 1 : 0;
    case OpCmp.LT:
      return a < b ? 1 : 0;
    case OpCmp.LE:
      return a <= b ? 1 : 0;
    case OpCmp.GT:
      return a > b ? 1 : 0;
    case OpCmp.GE:
      return a >= b ? 1 : 0;
    default:
      throw new Error(`tsRealComparison: unknown op ${op}`);
  }
}

export function tsRealComparison(
  op: number,
  n: number,
  a: Float64Array,
  b: Float64Array,
  out: Float64Array
): void {
  switch (op) {
    case OpCmp.EQ:
      for (let i = 0; i < n; i++) out[i] = a[i] === b[i] ? 1 : 0;
      return;
    case OpCmp.NE:
      for (let i = 0; i < n; i++) out[i] = a[i] !== b[i] ? 1 : 0;
      return;
    case OpCmp.LT:
      for (let i = 0; i < n; i++) out[i] = a[i] < b[i] ? 1 : 0;
      return;
    case OpCmp.LE:
      for (let i = 0; i < n; i++) out[i] = a[i] <= b[i] ? 1 : 0;
      return;
    case OpCmp.GT:
      for (let i = 0; i < n; i++) out[i] = a[i] > b[i] ? 1 : 0;
      return;
    case OpCmp.GE:
      for (let i = 0; i < n; i++) out[i] = a[i] >= b[i] ? 1 : 0;
      return;
    default:
      throw new Error(`tsRealComparison: unknown op ${op}`);
  }
}

export function tsRealScalarComparison(
  op: number,
  n: number,
  scalar: number,
  arr: Float64Array,
  scalarOnLeft: boolean,
  out: Float64Array
): void {
  if (scalarOnLeft) {
    for (let i = 0; i < n; i++) out[i] = rcmp(op, scalar, arr[i]);
  } else {
    for (let i = 0; i < n; i++) out[i] = rcmp(op, arr[i], scalar);
  }
}

export function tsComplexComparison(
  op: number,
  n: number,
  aRe: Float64Array,
  aIm: Float64Array | null,
  bRe: Float64Array,
  bIm: Float64Array | null,
  out: Float64Array
): void {
  switch (op) {
    case OpCmp.EQ:
      for (let i = 0; i < n; i++) {
        const ar = aRe[i],
          ai = aIm ? aIm[i] : 0;
        const br = bRe[i],
          bi = bIm ? bIm[i] : 0;
        out[i] = ar === br && ai === bi ? 1 : 0;
      }
      return;
    case OpCmp.NE:
      for (let i = 0; i < n; i++) {
        const ar = aRe[i],
          ai = aIm ? aIm[i] : 0;
        const br = bRe[i],
          bi = bIm ? bIm[i] : 0;
        out[i] = ar !== br || ai !== bi ? 1 : 0;
      }
      return;
    // MATLAB semantics: ordering ops use real parts only.
    case OpCmp.LT:
      for (let i = 0; i < n; i++) out[i] = aRe[i] < bRe[i] ? 1 : 0;
      return;
    case OpCmp.LE:
      for (let i = 0; i < n; i++) out[i] = aRe[i] <= bRe[i] ? 1 : 0;
      return;
    case OpCmp.GT:
      for (let i = 0; i < n; i++) out[i] = aRe[i] > bRe[i] ? 1 : 0;
      return;
    case OpCmp.GE:
      for (let i = 0; i < n; i++) out[i] = aRe[i] >= bRe[i] ? 1 : 0;
      return;
    default:
      throw new Error(`tsComplexComparison: unknown op ${op}`);
  }
}

export function tsComplexScalarComparison(
  op: number,
  n: number,
  sRe: number,
  sIm: number,
  arrRe: Float64Array,
  arrIm: Float64Array | null,
  scalarOnLeft: boolean,
  out: Float64Array
): void {
  switch (op) {
    case OpCmp.EQ:
      if (scalarOnLeft) {
        for (let i = 0; i < n; i++) {
          const ar = arrRe[i],
            ai = arrIm ? arrIm[i] : 0;
          out[i] = sRe === ar && sIm === ai ? 1 : 0;
        }
      } else {
        for (let i = 0; i < n; i++) {
          const ar = arrRe[i],
            ai = arrIm ? arrIm[i] : 0;
          out[i] = ar === sRe && ai === sIm ? 1 : 0;
        }
      }
      return;
    case OpCmp.NE:
      if (scalarOnLeft) {
        for (let i = 0; i < n; i++) {
          const ar = arrRe[i],
            ai = arrIm ? arrIm[i] : 0;
          out[i] = sRe !== ar || sIm !== ai ? 1 : 0;
        }
      } else {
        for (let i = 0; i < n; i++) {
          const ar = arrRe[i],
            ai = arrIm ? arrIm[i] : 0;
          out[i] = ar !== sRe || ai !== sIm ? 1 : 0;
        }
      }
      return;
    case OpCmp.LT:
    case OpCmp.LE:
    case OpCmp.GT:
    case OpCmp.GE:
      if (scalarOnLeft) {
        for (let i = 0; i < n; i++) out[i] = rcmp(op, sRe, arrRe[i]);
      } else {
        for (let i = 0; i < n; i++) out[i] = rcmp(op, arrRe[i], sRe);
      }
      return;
    default:
      throw new Error(`tsComplexScalarComparison: unknown op ${op}`);
  }
}
