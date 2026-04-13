/**
 * Tensor indexing helpers for JIT-compiled code.
 *
 * Three tiers of helpers, from most general to most specialized:
 * 1. Generic (idx1, idx2, idxN) — handle any base type, real or complex
 * 2. Real-tensor fast path (idx1r, idx2r, idx3r) — skip type/imag checks
 * 3. Hoisted-base (idx*r_h, set*r_h) — take pre-extracted data/len/shape
 *
 * All helpers use 1-based MATLAB indexing and include per-dimension bounds
 * checks to ensure consistency with the interpreter.
 */

import {
  type FloatXArrayType,
  type RuntimeTensor,
} from "../../runtime/types.js";
import { mkc } from "./jitHelpersComplex.js";

function isComplex(
  v: unknown
): v is import("../../runtime/types.js").RuntimeComplexNumber {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as { kind?: string }).kind === "complex_number"
  );
}

function isTensor(v: unknown): v is RuntimeTensor {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as { kind?: string }).kind === "tensor"
  );
}

// ── Bounds check error ─────────────────────────────────────────────────

export function bce(): never {
  throw new Error("Index exceeds array bounds");
}

// ── Generic helpers (any base type, real or complex) ───────────────────

export function idx1(base: unknown, i: number): unknown {
  if (isTensor(base)) {
    const idx = Math.round(i) - 1;
    if (idx < 0 || idx >= base.data.length)
      throw new Error("Index exceeds array bounds");
    if (base.imag !== undefined) {
      const imVal = base.imag[idx];
      return imVal === 0 ? base.data[idx] : mkc(base.data[idx], imVal);
    }
    return base.data[idx];
  }
  if (typeof base === "number") {
    if (Math.round(i) !== 1) throw new Error("Index exceeds array bounds");
    return base;
  }
  if (isComplex(base)) {
    if (Math.round(i) !== 1) throw new Error("Index exceeds array bounds");
    return base;
  }
  throw new Error("JIT index: unsupported base type");
}

export function idx2(base: unknown, ri: number, ci: number): unknown {
  if (isTensor(base)) {
    const s = base.shape;
    const rows = s.length === 0 ? 1 : s.length === 1 ? 1 : s[0];
    const cols = s.length === 0 ? 1 : s.length === 1 ? s[0] : s[1];
    const r = Math.round(ri) - 1;
    const c = Math.round(ci) - 1;
    if (r < 0 || r >= rows || c < 0 || c >= cols)
      throw new Error("Index exceeds array bounds");
    const lin = c * rows + r;
    if (base.imag !== undefined) {
      const imVal = base.imag[lin];
      return imVal === 0 ? base.data[lin] : mkc(base.data[lin], imVal);
    }
    return base.data[lin];
  }
  throw new Error("JIT index: unsupported base type for 2D indexing");
}

export function idxN(base: unknown, indices: number[]): unknown {
  if (isTensor(base)) {
    const s = base.shape;
    let lin = 0;
    let stride = 1;
    for (let k = 0; k < indices.length; k++) {
      const dimSize = k < s.length ? s[k] : 1;
      const sub = Math.round(indices[k]) - 1;
      if (sub < 0 || sub >= dimSize)
        throw new Error("Index exceeds array bounds");
      lin += sub * stride;
      stride *= dimSize;
    }
    if (base.imag !== undefined) {
      const imVal = base.imag[lin];
      return imVal === 0 ? base.data[lin] : mkc(base.data[lin], imVal);
    }
    return base.data[lin];
  }
  throw new Error("JIT index: unsupported base type for N-D indexing");
}

// ── Specialized real-tensor fast-path helpers ──────────────────────────
//
// Emitted when the JIT codegen has statically proven the base is a
// real-valued tensor. They skip isTensor/imag checks and use `|0`
// instead of Math.round (callers always pass integer indices).

export function idx1r(base: RuntimeTensor, i: number): number {
  const idx = (i - 1) | 0;
  if (idx >>> 0 >= base.data.length) bce();
  return base.data[idx];
}

export function idx2r(base: RuntimeTensor, ri: number, ci: number): number {
  const s = base.shape;
  const rows = s.length >= 2 ? s[0] : 1;
  const r = (ri - 1) | 0;
  const c = (ci - 1) | 0;
  if (r >>> 0 >= rows) bce();
  const cols = base.data.length / rows;
  if (c >>> 0 >= cols) bce();
  const lin = c * rows + r;
  return base.data[lin];
}

export function idx3r(
  base: RuntimeTensor,
  i1: number,
  i2: number,
  i3: number
): number {
  const s = base.shape;
  const d0 = s[0];
  const d1 = s.length >= 2 ? s[1] : 1;
  const k0 = (i1 - 1) | 0;
  const k1 = (i2 - 1) | 0;
  const k2 = (i3 - 1) | 0;
  if (k0 >>> 0 >= d0) bce();
  if (k1 >>> 0 >= d1) bce();
  const d2 = base.data.length / (d0 * d1);
  if (k2 >>> 0 >= d2) bce();
  const lin = k2 * d0 * d1 + k1 * d0 + k0;
  return base.data[lin];
}

// ── Hoisted-base read helpers ──────────────────────────────────────────
//
// Take the tensor's .data, .length, and dimension sizes as separate
// scalar arguments. The JIT codegen hoists these reads ONCE at the
// top of a loop function, so per-iter cost is just register reads.

export function idx1r_h(data: FloatXArrayType, len: number, i: number): number {
  const idx = (i - 1) | 0;
  if (idx >>> 0 >= len) bce();
  return data[idx];
}

export function idx2r_h(
  data: FloatXArrayType,
  len: number,
  rows: number,
  ri: number,
  ci: number
): number {
  const r = (ri - 1) | 0;
  const c = (ci - 1) | 0;
  if (r >>> 0 >= rows) bce();
  const cols = len / rows;
  if (c >>> 0 >= cols) bce();
  const lin = c * rows + r;
  return data[lin];
}

export function idx3r_h(
  data: FloatXArrayType,
  len: number,
  d0: number,
  d1: number,
  i1: number,
  i2: number,
  i3: number
): number {
  const k0 = (i1 - 1) | 0;
  const k1 = (i2 - 1) | 0;
  const k2 = (i3 - 1) | 0;
  if (k0 >>> 0 >= d0) bce();
  if (k1 >>> 0 >= d1) bce();
  const d2 = len / (d0 * d1);
  if (k2 >>> 0 >= d2) bce();
  const lin = k2 * d0 * d1 + k1 * d0 + k0;
  return data[lin];
}

// ── Hoisted-base write helpers ─────────────────────────────────────────

export function set1r_h(
  data: FloatXArrayType,
  len: number,
  i: number,
  v: number
): void {
  const idx = (i - 1) | 0;
  if (idx >>> 0 >= len) bce();
  data[idx] = v;
}

export function set2r_h(
  data: FloatXArrayType,
  len: number,
  rows: number,
  ri: number,
  ci: number,
  v: number
): void {
  const r = (ri - 1) | 0;
  const c = (ci - 1) | 0;
  if (r >>> 0 >= rows) bce();
  const cols = len / rows;
  if (c >>> 0 >= cols) bce();
  const lin = c * rows + r;
  data[lin] = v;
}

export function set3r_h(
  data: FloatXArrayType,
  len: number,
  d0: number,
  d1: number,
  i1: number,
  i2: number,
  i3: number,
  v: number
): void {
  const k0 = (i1 - 1) | 0;
  const k1 = (i2 - 1) | 0;
  const k2 = (i3 - 1) | 0;
  if (k0 >>> 0 >= d0) bce();
  if (k1 >>> 0 >= d1) bce();
  const d2 = len / (d0 * d1);
  if (k2 >>> 0 >= d2) bce();
  const lin = k2 * d0 * d1 + k1 * d0 + k0;
  data[lin] = v;
}

// ── Range-slice write helper ───────────────────────────────────────────
//
// dst(a:b) = src(c:d) where both are real tensors with 1-based MATLAB
// ranges. Uses TypedArray.prototype.set which handles overlapping memory.

export function setRange1r_h(
  dstData: FloatXArrayType,
  dstLen: number,
  dstStart: number,
  dstEnd: number,
  srcData: FloatXArrayType,
  srcLen: number,
  srcStart: number,
  srcEnd: number
): void {
  const dStart = (dstStart - 1) | 0;
  const dEnd = (dstEnd - 1) | 0;
  const sStart = (srcStart - 1) | 0;
  const sEnd = (srcEnd - 1) | 0;
  const dN = dEnd - dStart + 1;
  const sN = sEnd - sStart + 1;
  if (dN !== sN) {
    throw new Error(
      "Unable to perform assignment because the indices on the left side are not compatible with the size of the right side."
    );
  }
  if (dN <= 0) return;
  if (dStart >>> 0 >= dstLen) bce();
  if (dEnd >>> 0 >= dstLen) bce();
  if (sStart >>> 0 >= srcLen) bce();
  if (sEnd >>> 0 >= srcLen) bce();
  dstData.set(srcData.subarray(sStart, sEnd + 1), dStart);
}
