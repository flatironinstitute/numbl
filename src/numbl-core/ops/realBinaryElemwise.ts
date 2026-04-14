/**
 * Pure-TS implementation of real binary element-wise ops.
 * Mirrors native/ops/real_binary_elemwise.c.
 *
 * Caller-allocated input/output buffers; never copies.
 */

import { OpRealBin } from "./opCodes.js";

export function tsRealBinaryElemwise(
  op: number,
  n: number,
  a: Float64Array,
  b: Float64Array,
  out: Float64Array
): void {
  switch (op) {
    case OpRealBin.ADD:
      for (let i = 0; i < n; i++) out[i] = a[i] + b[i];
      return;
    case OpRealBin.SUB:
      for (let i = 0; i < n; i++) out[i] = a[i] - b[i];
      return;
    case OpRealBin.MUL:
      for (let i = 0; i < n; i++) out[i] = a[i] * b[i];
      return;
    case OpRealBin.DIV:
      for (let i = 0; i < n; i++) out[i] = a[i] / b[i];
      return;
    default:
      throw new Error(`tsRealBinaryElemwise: unknown op ${op}`);
  }
}

export function tsRealScalarBinaryElemwise(
  op: number,
  n: number,
  scalar: number,
  arr: Float64Array,
  scalarOnLeft: boolean,
  out: Float64Array
): void {
  if (scalarOnLeft) {
    switch (op) {
      case OpRealBin.ADD:
        for (let i = 0; i < n; i++) out[i] = scalar + arr[i];
        return;
      case OpRealBin.SUB:
        for (let i = 0; i < n; i++) out[i] = scalar - arr[i];
        return;
      case OpRealBin.MUL:
        for (let i = 0; i < n; i++) out[i] = scalar * arr[i];
        return;
      case OpRealBin.DIV:
        for (let i = 0; i < n; i++) out[i] = scalar / arr[i];
        return;
      default:
        throw new Error(`tsRealScalarBinaryElemwise: unknown op ${op}`);
    }
  } else {
    switch (op) {
      case OpRealBin.ADD:
        for (let i = 0; i < n; i++) out[i] = arr[i] + scalar;
        return;
      case OpRealBin.SUB:
        for (let i = 0; i < n; i++) out[i] = arr[i] - scalar;
        return;
      case OpRealBin.MUL:
        for (let i = 0; i < n; i++) out[i] = arr[i] * scalar;
        return;
      case OpRealBin.DIV:
        for (let i = 0; i < n; i++) out[i] = arr[i] / scalar;
        return;
      default:
        throw new Error(`tsRealScalarBinaryElemwise: unknown op ${op}`);
    }
  }
}
