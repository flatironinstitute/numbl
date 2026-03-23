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
  const t: RuntimeTensor = { kind: "tensor", data, shape, _rc: 1 };
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

// ── Exported helpers object ─────────────────────────────────────────────

export const jitHelpers = {
  // Complex scalar ops
  cAdd,
  cSub,
  cMul,
  cDiv,
  cNeg,

  // Tensor binary ops (handles real + complex + mixed)
  tAdd: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x + y, cAdd),
  tSub: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x - y, cSub),
  tMul: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x * y, cMul),
  tDiv: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x / y, cDiv),

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
  tFix: (a: RuntimeTensor) => tensorUnary(a, x => x | 0),
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

  // Scalar accessors (for complex tensor literal construction)
  re,
  im,

  // IBuiltin apply functions (populated by buildJitHelpers)
} as Record<string, unknown>;

import { buildIBuiltinHelpers } from "../builtins/index.js";
Object.assign(jitHelpers, buildIBuiltinHelpers());
