/**
 * Runtime helpers for JIT-compiled tensor and complex operations.
 * Passed as `$h` to generated functions.
 */

import {
  FloatXArray,
  type FloatXArrayType,
  type RuntimeTensor,
  type RuntimeComplexNumber,
} from "../../runtime/types.js";

// ── Type checks ─────────────────────────────────────────────────────────

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

function re(v: unknown): number {
  if (typeof v === "number") return v;
  return (v as RuntimeComplexNumber).re;
}

function im(v: unknown): number {
  if (typeof v === "number") return 0;
  return (v as RuntimeComplexNumber).im;
}

function mkc(r: number, i: number): number | RuntimeComplexNumber {
  if (i === 0) return r;
  return { kind: "complex_number", re: r, im: i };
}

// ── Tensor creation ─────────────────────────────────────────────────────

function makeTensor(
  data: FloatXArrayType,
  imag: FloatXArrayType | undefined,
  shape: number[]
): RuntimeTensor {
  // Strip trailing singleton dimensions (always keep minimum 2D)
  const s = [...shape];
  while (s.length > 2 && s[s.length - 1] === 1) s.pop();
  const t: RuntimeTensor = { kind: "tensor", data, shape: s, _rc: 1 };
  if (imag) t.imag = imag;
  return t;
}

// ── Complex scalar operations ───────────────────────────────────────────

function cAdd(a: unknown, b: unknown): number | RuntimeComplexNumber {
  return mkc(re(a) + re(b), im(a) + im(b));
}

function cSub(a: unknown, b: unknown): number | RuntimeComplexNumber {
  return mkc(re(a) - re(b), im(a) - im(b));
}

function cMul(a: unknown, b: unknown): number | RuntimeComplexNumber {
  const ar = re(a),
    ai = im(a),
    br = re(b),
    bi = im(b);
  return mkc(ar * br - ai * bi, ar * bi + ai * br);
}

function cDiv(a: unknown, b: unknown): number | RuntimeComplexNumber {
  const ar = re(a),
    ai = im(a),
    br = re(b),
    bi = im(b);
  const d = br * br + bi * bi;
  return mkc((ar * br + ai * bi) / d, (ai * br - ar * bi) / d);
}

function cNeg(a: unknown): number | RuntimeComplexNumber {
  return mkc(-re(a), -im(a));
}

function cConj(a: unknown): number | RuntimeComplexNumber {
  return mkc(re(a), -im(a));
}

function cAngle(a: unknown): number {
  return Math.atan2(im(a), re(a));
}

function mod(a: number, b: number): number {
  return ((a % b) + b) % b;
}

// ── Element-wise tensor operations (real and complex) ───────────────────

type ScalarVal = number | RuntimeComplexNumber;

function tensorBinaryOp(
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
    const n = at.data.length;
    const aHasImag = !!at.imag;
    const bHasImag = !!bt.imag;

    if (!aHasImag && !bHasImag) {
      // Both real
      const out = new FloatXArray(n);
      for (let i = 0; i < n; i++) out[i] = realOp(at.data[i], bt.data[i]);
      return makeTensor(out, undefined, at.shape.slice());
    }
    // At least one complex
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

function tensorCompareOp(
  a: unknown,
  b: unknown,
  cmp: (x: number, y: number) => boolean
): RuntimeTensor {
  const aIsT = isTensor(a);
  const bIsT = isTensor(b);

  if (aIsT && bIsT) {
    const at = a as RuntimeTensor;
    const bt = b as RuntimeTensor;
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
  const bt = b as RuntimeTensor;
  const n = bt.data.length;
  const av = a as number;
  const out = new FloatXArray(n);
  for (let i = 0; i < n; i++) out[i] = cmp(av, bt.data[i]) ? 1 : 0;
  return makeTensor(out, undefined, bt.shape.slice());
}

function tensorUnary(
  a: RuntimeTensor,
  fn: (x: number) => number
): RuntimeTensor {
  const n = a.data.length;
  const out = new FloatXArray(n);
  for (let i = 0; i < n; i++) out[i] = fn(a.data[i]);
  return makeTensor(out, undefined, a.shape.slice());
}

function tensorNeg(a: RuntimeTensor): RuntimeTensor {
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

// ── Tensor indexing ─────────────────────────────────────────────────────

function idx1(base: unknown, i: number): unknown {
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
  // Scalar indexing: x(1) = x
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

function idx2(base: unknown, ri: number, ci: number): unknown {
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

function idxN(base: unknown, indices: number[]): unknown {
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

// ── Complex truthiness ──────────────────────────────────────────────────

function cTruthy(v: unknown): boolean {
  if (typeof v === "number") return v !== 0;
  const c = v as RuntimeComplexNumber;
  return c.re !== 0 || c.im !== 0;
}

// ── Exported helpers object ─────────────────────────────────────────────

export const jitHelpers = {
  // Complex truthiness
  cTruthy,

  // Complex scalar ops
  cAdd,
  cSub,
  cMul,
  cDiv,
  cNeg,
  cConj,
  cAngle,

  // Scalar math
  mod,

  // Tensor binary ops (handles real + complex + mixed)
  tAdd: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x + y, cAdd),
  tSub: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x - y, cSub),
  tMul: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x * y, cMul),
  tDiv: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x / y, cDiv),

  // Tensor power (element-wise)
  tPow: (a: unknown, b: unknown) =>
    tensorBinaryOp(a, b, Math.pow, () => {
      // Complex power not supported in JIT — should not reach here
      throw new Error("JIT tPow: complex power not supported");
    }),

  // Tensor comparisons (real only, returns logical tensor)
  tEq: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x === y),
  tNeq: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x !== y),
  tLt: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x < y),
  tLe: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x <= y),
  tGt: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x > y),
  tGe: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x >= y),

  // Tensor unary
  tNeg: tensorNeg,

  // Tensor math (real only for now)
  tSin: (a: RuntimeTensor) => tensorUnary(a, Math.sin),
  tCos: (a: RuntimeTensor) => tensorUnary(a, Math.cos),
  tTan: (a: RuntimeTensor) => tensorUnary(a, Math.tan),
  tAsin: (a: RuntimeTensor) => tensorUnary(a, Math.asin),
  tAcos: (a: RuntimeTensor) => tensorUnary(a, Math.acos),
  tAtan: (a: RuntimeTensor) => tensorUnary(a, Math.atan),
  tSinh: (a: RuntimeTensor) => tensorUnary(a, Math.sinh),
  tCosh: (a: RuntimeTensor) => tensorUnary(a, Math.cosh),
  tTanh: (a: RuntimeTensor) => tensorUnary(a, Math.tanh),
  tSqrt: (a: RuntimeTensor) => tensorUnary(a, Math.sqrt),
  tAbs: (a: RuntimeTensor) => tensorUnary(a, Math.abs),
  tFloor: (a: RuntimeTensor) => tensorUnary(a, Math.floor),
  tCeil: (a: RuntimeTensor) => tensorUnary(a, Math.ceil),
  tRound: (a: RuntimeTensor) => tensorUnary(a, Math.round),
  tFix: (a: RuntimeTensor) => tensorUnary(a, Math.trunc),
  tExp: (a: RuntimeTensor) => tensorUnary(a, Math.exp),
  tLog: (a: RuntimeTensor) => tensorUnary(a, Math.log),
  tLog2: (a: RuntimeTensor) => tensorUnary(a, Math.log2),
  tLog10: (a: RuntimeTensor) => tensorUnary(a, Math.log10),
  tSign: (a: RuntimeTensor) => tensorUnary(a, Math.sign),

  // Tensor literal construction
  mkTensor: (data: number[], shape: number[]) =>
    makeTensor(new FloatXArray(data), undefined, shape),
  mkTensorC: (reData: number[], imData: number[], shape: number[]) =>
    makeTensor(new FloatXArray(reData), new FloatXArray(imData), shape),

  // Tensor indexing
  idx1,
  idx2,
  idxN,

  // Scalar accessors
  re,
  im,

  // User function call with call frame tracking
  callUser: (
    rt: {
      pushCallFrame: (name: string) => void;
      popCallFrame: () => void;
      annotateError: (e: unknown) => void;
    },
    name: string,
    fn: (...args: unknown[]) => unknown,
    ...args: unknown[]
  ) => {
    rt.pushCallFrame(name);
    try {
      return fn(...args);
    } catch (e) {
      rt.annotateError(e);
      throw e;
    } finally {
      rt.popCallFrame();
    }
  },

  // IBuiltin apply functions (populated by buildJitHelpers)
} as Record<string, unknown>;

import {
  buildIBuiltinHelpers,
  setDynamicRegisterHook,
} from "../builtins/index.js";
Object.assign(jitHelpers, buildIBuiltinHelpers());

// Hook: when a dynamic IBuiltin is registered, add its ib_* entry to jitHelpers
import type { IBuiltin } from "../builtins/index.js";
import { inferJitType as _ijt } from "../builtins/index.js";
setDynamicRegisterHook((b: IBuiltin) => {
  const h = jitHelpers as Record<string, unknown>;
  h[`ib_${b.name}`] = (...args: unknown[]) => {
    const pe = h._profileEnter as (...a: unknown[]) => void;
    const pl = h._profileLeave as (...a: unknown[]) => void;
    pe("builtin:jit:" + b.name);
    const rtArgs = args as import("../../runtime/types.js").RuntimeValue[];
    const argTypes = rtArgs.map(_ijt);
    const res = b.resolve(argTypes, 1);
    if (!res) {
      pl();
      throw new Error(`JIT ib_${b.name}: resolve failed`);
    }
    const result = res.apply(rtArgs, 1);
    pl();
    return typeof result === "boolean" ? (result ? 1 : 0) : result;
  };
});
