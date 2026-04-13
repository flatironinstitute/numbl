/**
 * Tensor operation helpers for JIT-compiled code.
 * Handles element-wise binary/unary ops, comparison ops, concat, and
 * tensor coercion for struct-array field access.
 */

import {
  FloatXArray,
  type FloatXArrayType,
  type RuntimeTensor,
} from "../../runtime/types.js";
import type { RuntimeComplexNumber } from "../../runtime/types.js";
import { re, im, mkc, cAdd, cSub, cMul, cDiv } from "./jitHelpersComplex.js";

// ── Type checks ────────────────────────────────────────────────────────

function isTensor(v: unknown): v is RuntimeTensor {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeTensor).kind === "tensor"
  );
}

function isComplex(v: unknown): v is RuntimeComplexNumber {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeComplexNumber).kind === "complex_number"
  );
}

// ── Tensor creation ────────────────────────────────────────────────────

export function makeTensor(
  data: FloatXArrayType,
  imag: FloatXArrayType | undefined,
  shape: number[]
): RuntimeTensor {
  const s = [...shape];
  while (s.length > 2 && s[s.length - 1] === 1) s.pop();
  const t: RuntimeTensor = { kind: "tensor", data, shape: s, _rc: 1 };
  if (imag) t.imag = imag;
  return t;
}

// ── Element-wise tensor binary operations ──────────────────────────────

type ScalarVal = number | RuntimeComplexNumber;

export function tensorBinaryOp(
  a: unknown,
  b: unknown,
  realOp: (x: number, y: number) => number,
  complexOp: (a: ScalarVal, b: ScalarVal) => ScalarVal
): RuntimeTensor | ScalarVal {
  const aIsT = isTensor(a);
  const bIsT = isTensor(b);

  if (aIsT && bIsT) {
    const at = a as RuntimeTensor;
    const bt = b as RuntimeTensor;
    if (at.data.length !== bt.data.length)
      throw new Error("Matrix dimensions must agree.");
    const n = at.data.length;
    const aHasImag = !!at.imag;
    const bHasImag = !!bt.imag;

    if (!aHasImag && !bHasImag) {
      const out = new FloatXArray(n);
      for (let i = 0; i < n; i++) out[i] = realOp(at.data[i], bt.data[i]);
      return makeTensor(out, undefined, at.shape.slice());
    }
    const outR = new FloatXArray(n);
    const outI = new FloatXArray(n);
    for (let i = 0; i < n; i++) {
      const av: ScalarVal = aHasImag
        ? mkc(at.data[i], at.imag![i])
        : at.data[i];
      const bv: ScalarVal = bHasImag
        ? mkc(bt.data[i], bt.imag![i])
        : bt.data[i];
      const r = complexOp(av, bv);
      outR[i] = re(r);
      outI[i] = im(r);
    }
    return makeTensor(outR, outI, at.shape.slice());
  }

  if (aIsT) {
    const at = a as RuntimeTensor;
    const n = at.data.length;
    const aHasImag = !!at.imag;
    const bIsC = isComplex(b);

    if (!aHasImag && !bIsC) {
      const bv = b as number;
      const out = new FloatXArray(n);
      for (let i = 0; i < n; i++) out[i] = realOp(at.data[i], bv);
      return makeTensor(out, undefined, at.shape.slice());
    }
    const outR = new FloatXArray(n);
    const outI = new FloatXArray(n);
    for (let i = 0; i < n; i++) {
      const av: ScalarVal = aHasImag
        ? mkc(at.data[i], at.imag![i])
        : at.data[i];
      const r = complexOp(av, b as ScalarVal);
      outR[i] = re(r);
      outI[i] = im(r);
    }
    return makeTensor(outR, outI, at.shape.slice());
  }

  if (bIsT) {
    const bt = b as RuntimeTensor;
    const n = bt.data.length;
    const bHasImag = !!bt.imag;
    const aIsC = isComplex(a);

    if (!bHasImag && !aIsC) {
      const av = a as number;
      const out = new FloatXArray(n);
      for (let i = 0; i < n; i++) out[i] = realOp(av, bt.data[i]);
      return makeTensor(out, undefined, bt.shape.slice());
    }
    const outR = new FloatXArray(n);
    const outI = new FloatXArray(n);
    for (let i = 0; i < n; i++) {
      const bv: ScalarVal = bHasImag
        ? mkc(bt.data[i], bt.imag![i])
        : bt.data[i];
      const r = complexOp(a as ScalarVal, bv);
      outR[i] = re(r);
      outI[i] = im(r);
    }
    return makeTensor(outR, outI, bt.shape.slice());
  }

  throw new Error("JIT tensor binary: unexpected argument types");
}

// ── Element-wise tensor comparison ─────────────────────────────────────

export function tensorCompareOp(
  a: unknown,
  b: unknown,
  cmp: (x: number, y: number) => boolean
): RuntimeTensor {
  const aIsT = isTensor(a);
  const bIsT = isTensor(b);

  if (aIsT && bIsT) {
    const at = a as RuntimeTensor;
    const bt = b as RuntimeTensor;
    if (at.data.length !== bt.data.length)
      throw new Error("Matrix dimensions must agree.");
    const n = at.data.length;
    const out = new FloatXArray(n);
    for (let i = 0; i < n; i++) out[i] = cmp(at.data[i], bt.data[i]) ? 1 : 0;
    return makeTensor(out, undefined, at.shape.slice());
  }
  if (aIsT) {
    const at = a as RuntimeTensor;
    const n = at.data.length;
    const bv = b as number;
    const out = new FloatXArray(n);
    for (let i = 0; i < n; i++) out[i] = cmp(at.data[i], bv) ? 1 : 0;
    return makeTensor(out, undefined, at.shape.slice());
  }
  if (!bIsT) {
    throw new Error(
      "JIT tensor compare: at least one operand must be a tensor"
    );
  }
  const bt = b as RuntimeTensor;
  const n = bt.data.length;
  const av = a as number;
  const out = new FloatXArray(n);
  for (let i = 0; i < n; i++) out[i] = cmp(av, bt.data[i]) ? 1 : 0;
  return makeTensor(out, undefined, bt.shape.slice());
}

// ── Element-wise tensor unary ──────────────────────────────────────────

export function tensorUnary(
  a: RuntimeTensor,
  fn: (x: number) => number
): RuntimeTensor {
  const n = a.data.length;
  const out = new FloatXArray(n);
  for (let i = 0; i < n; i++) out[i] = fn(a.data[i]);
  return makeTensor(out, undefined, a.shape.slice());
}

export function tensorNeg(a: RuntimeTensor): RuntimeTensor {
  const n = a.data.length;
  const outR = new FloatXArray(n);
  for (let i = 0; i < n; i++) outR[i] = -a.data[i];
  let outI: FloatXArrayType | undefined;
  if (a.imag) {
    outI = new FloatXArray(n);
    for (let i = 0; i < n; i++) outI[i] = -a.imag[i];
  }
  return makeTensor(outR, outI, a.shape.slice());
}

// ── Vertical concat growth ─────────────────────────────────────────────

export function vconcatGrow1r(base: unknown, v: number): RuntimeTensor {
  const bt = base as RuntimeTensor;
  const baseLen = bt.data.length;
  if (baseLen === 0) {
    const out = new FloatXArray(1);
    out[0] = v;
    return makeTensor(out, undefined, [1, 1]);
  }
  const s = bt.shape;
  const isColOrScalar =
    (s.length === 2 && s[1] === 1) ||
    (s.length === 1 && s[0] >= 1) ||
    (s.length === 2 && s[0] === 1 && s[1] === 1);
  if (!isColOrScalar) {
    throw new Error(
      "Dimensions of arrays being concatenated are not consistent."
    );
  }
  const out = new FloatXArray(baseLen + 1);
  out.set(bt.data);
  out[baseLen] = v;
  return makeTensor(out, undefined, [baseLen + 1, 1]);
}

// ── Copy-on-write unsharing ────────────────────────────────────────────

export function unshare(t: unknown): RuntimeTensor {
  const tt = t as RuntimeTensor;
  if (tt._rc <= 1) return tt;
  tt._rc--;
  const newData = new FloatXArray(tt.data);
  const newImag = tt.imag ? new FloatXArray(tt.imag) : undefined;
  const copy: RuntimeTensor = {
    kind: "tensor",
    data: newData,
    shape: tt.shape.slice(),
    _rc: 1,
  };
  if (newImag) copy.imag = newImag;
  if (tt._isLogical) copy._isLogical = tt._isLogical;
  return copy;
}

// ── Scalar → 1x1 tensor coercion ──────────────────────────────────────

export function asTensor(v: unknown): RuntimeTensor {
  if (typeof v === "number") {
    return makeTensor(new FloatXArray([v]), undefined, [1, 1]);
  }
  if (typeof v === "boolean") {
    return makeTensor(new FloatXArray([v ? 1 : 0]), undefined, [1, 1]);
  }
  return v as RuntimeTensor;
}

// ── Tensor binary op wrappers (re-exported via jitHelpers) ─────────────

export const tAdd = (a: unknown, b: unknown) =>
  tensorBinaryOp(a, b, (x, y) => x + y, cAdd);
export const tSub = (a: unknown, b: unknown) =>
  tensorBinaryOp(a, b, (x, y) => x - y, cSub);
export const tMul = (a: unknown, b: unknown) =>
  tensorBinaryOp(a, b, (x, y) => x * y, cMul);
export const tDiv = (a: unknown, b: unknown) =>
  tensorBinaryOp(a, b, (x, y) => x / y, cDiv);
export const tPow = (a: unknown, b: unknown) =>
  tensorBinaryOp(a, b, Math.pow, () => {
    throw new Error("JIT tPow: complex power not supported");
  });

export const tEq = (a: unknown, b: unknown) =>
  tensorCompareOp(a, b, (x, y) => x === y);
export const tNeq = (a: unknown, b: unknown) =>
  tensorCompareOp(a, b, (x, y) => x !== y);
export const tLt = (a: unknown, b: unknown) =>
  tensorCompareOp(a, b, (x, y) => x < y);
export const tLe = (a: unknown, b: unknown) =>
  tensorCompareOp(a, b, (x, y) => x <= y);
export const tGt = (a: unknown, b: unknown) =>
  tensorCompareOp(a, b, (x, y) => x > y);
export const tGe = (a: unknown, b: unknown) =>
  tensorCompareOp(a, b, (x, y) => x >= y);
