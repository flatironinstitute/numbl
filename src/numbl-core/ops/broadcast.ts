/**
 * Implicit expansion (broadcasting) of two N-d shapes, and the one
 * size-mismatch error every element-wise operation raises.
 */

import { RuntimeError } from "../runtime/error.js";

/** MATLAB's error for operands whose sizes do not broadcast. */
export function sizeMismatchError(): RuntimeError {
  return new RuntimeError(
    "Arrays have incompatible sizes for this operation.",
    null,
    { identifier: "MATLAB:sizeDimensionsMustMatch" }
  );
}

/** A shape as at least 2-D, padded with trailing 1s to `ndim` dims. */
function padShape(shape: readonly number[], ndim: number): number[] {
  const s =
    shape.length === 0
      ? [1, 1]
      : shape.length === 1
        ? [1, shape[0]]
        : [...shape];
  while (s.length < ndim) s.push(1);
  return s;
}

/** The broadcast shape of `a` and `b` (each dimension equal, or one of
 *  them 1), or null when they are incompatible. */
export function tryBroadcastShape(
  a: readonly number[],
  b: readonly number[]
): number[] | null {
  const ndim = Math.max(a.length, b.length, 2);
  const pa = padShape(a, ndim);
  const pb = padShape(b, ndim);
  const out: number[] = new Array(ndim);
  for (let i = 0; i < ndim; i++) {
    const da = pa[i];
    const db = pb[i];
    if (da === db || db === 1) out[i] = da;
    else if (da === 1) out[i] = db;
    else return null;
  }
  return out;
}

/** The broadcast shape of `a` and `b`; raises `sizeMismatchError` when
 *  they are incompatible. */
export function broadcastShape(
  a: readonly number[],
  b: readonly number[]
): number[] {
  const s = tryBroadcastShape(a, b);
  if (s === null) throw sizeMismatchError();
  return s;
}

/** Strides of `shape` padded to `ndim`, zero along broadcast (size-1) dims. */
function broadcastStrides(shape: readonly number[], ndim: number): number[] {
  const p = padShape(shape, ndim);
  const strides: number[] = new Array(ndim);
  let s = 1;
  for (let d = 0; d < ndim; d++) {
    strides[d] = p[d] === 1 ? 0 : s;
    s *= p[d];
  }
  return strides;
}

/**
 * Walk a broadcast result one column (first-dimension run) at a time.
 * `row(ia, sa, ib, sb, io, len)` covers outputs `io .. io+len-1`, reading
 * `a` at `ia, ia+sa, ...` and `b` at `ib, ib+sb, ...` (a stride is 0 along
 * a broadcast dimension). `outShape` comes from `broadcastShape`.
 */
export function forEachBroadcastRow(
  aShape: readonly number[],
  bShape: readonly number[],
  outShape: readonly number[],
  row: (
    ia: number,
    sa: number,
    ib: number,
    sb: number,
    io: number,
    len: number
  ) => void
): void {
  const ndim = outShape.length;
  let total = 1;
  for (let d = 0; d < ndim; d++) total *= outShape[d];
  if (total === 0) return;
  const as = broadcastStrides(aShape, ndim);
  const bs = broadcastStrides(bShape, ndim);
  const len = outShape[0];
  const rows = total / len;
  const subs: number[] = new Array(ndim).fill(0);
  let ia = 0;
  let ib = 0;
  for (let r = 0; r < rows; r++) {
    row(ia, as[0], ib, bs[0], r * len, len);
    for (let d = 1; d < ndim; d++) {
      subs[d]++;
      ia += as[d];
      ib += bs[d];
      if (subs[d] < outShape[d]) break;
      ia -= subs[d] * as[d];
      ib -= subs[d] * bs[d];
      subs[d] = 0;
    }
  }
}

/** Visit every element of a broadcast result in column-major order with
 *  its source indices into `a` and `b`. */
export function broadcastIterate(
  aShape: readonly number[],
  bShape: readonly number[],
  outShape: readonly number[],
  visit: (aIdx: number, bIdx: number, outIdx: number) => void
): void {
  forEachBroadcastRow(aShape, bShape, outShape, (ia, sa, ib, sb, io, len) => {
    for (let k = 0; k < len; k++) visit(ia + k * sa, ib + k * sb, io + k);
  });
}
