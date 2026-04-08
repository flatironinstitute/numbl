/**
 * Arithmetic, comparison, and related internal helpers.
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  type RuntimeCell,
  type FloatXArrayType,
  FloatXArray,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  kstr,
} from "../runtime/types.js";
import { RuntimeError } from "../runtime/error.js";
import { RTV } from "../runtime/constructors.js";
import { tensorSize2D, colMajorIndex } from "../runtime/utils.js";
import { toNumber } from "../runtime/convert.js";
import { getEffectiveBridge } from "../native/bridge-resolve.js";
import { getLapackBridge } from "../native/lapack-bridge.js";
import { linsolveLapack, linsolveComplexLapack } from "./linsolve.js";
import { applyBuiltin as applyBuiltinFn } from "./check-helpers.js";
import { coerceToTensor } from "./shape-utils.js";
import { num2strScalar } from "./string.js";
import {
  mAddSparse,
  mSubSparse,
  mMulSparse,
  mElemMulSparse,
  mElemDivSparse,
  sparseNeg,
  sparseTranspose,
  sparseConjugateTranspose,
  sparseToDense,
} from "./sparse-arithmetic.js";

// ── Complex helpers ──────────────────────────────────────────────────────

function toComplex(v: RuntimeValue): { re: number; im: number } {
  if (isRuntimeComplexNumber(v)) return { re: v.re, im: v.im };
  if (isRuntimeNumber(v)) return { re: v, im: 0 };
  if (isRuntimeLogical(v)) return { re: v ? 1 : 0, im: 0 };
  if (isRuntimeTensor(v) && v.data.length === 1) {
    return { re: v.data[0], im: v.imag ? v.imag[0] : 0 };
  }
  if (isRuntimeSparseMatrix(v)) return toComplex(densify(v));
  throw new RuntimeError(`Cannot use ${kstr(v)} in complex arithmetic`);
}

function isComplexOrMixed(a: RuntimeValue, b: RuntimeValue): boolean {
  return (
    isRuntimeComplexNumber(a) ||
    isRuntimeComplexNumber(b) ||
    (isRuntimeTensor(a) && a.imag !== undefined) ||
    (isRuntimeTensor(b) && b.imag !== undefined) ||
    (isRuntimeSparseMatrix(a) && a.pi !== undefined) ||
    (isRuntimeSparseMatrix(b) && b.pi !== undefined)
  );
}

function complexResult(re: number, im: number): RuntimeValue {
  return im === 0 ? RTV.num(re) : RTV.complex(re, im);
}

/** Extract complex representation from RuntimeValue (works for scalars and tensors) */
function toComplexParts(v: RuntimeValue):
  | {
      scalar: true;
      re: number;
      im: number;
    }
  | {
      scalar: false;
      re: FloatXArrayType;
      im: FloatXArrayType;
      shape: number[];
    } {
  if (isRuntimeComplexNumber(v)) {
    return { scalar: true, re: v.re, im: v.im };
  }
  if (isRuntimeNumber(v)) {
    return { scalar: true, re: v, im: 0 };
  }
  if (isRuntimeLogical(v)) {
    return { scalar: true, re: v ? 1 : 0, im: 0 };
  }
  if (isRuntimeTensor(v)) {
    if (v.data.length === 1) {
      return { scalar: true, re: v.data[0], im: v.imag ? v.imag[0] : 0 };
    }
    const im = v.imag || new FloatXArray(v.data.length); // allocate zeros if needed
    return { scalar: false, re: v.data, im, shape: v.shape };
  }
  if (isRuntimeSparseMatrix(v)) return toComplexParts(densify(v));
  throw new RuntimeError(`Cannot use ${kstr(v)} in complex arithmetic`);
}

/** Complex comparison operation supporting scalars and tensors */
function complexComparisonOp(
  a: RuntimeValue,
  b: RuntimeValue,
  op: (aRe: number, aIm: number, bRe: number, bIm: number) => boolean
): RuntimeValue {
  const ac = toComplexParts(a);
  const bc = toComplexParts(b);

  // scalar op scalar
  if (ac.scalar && bc.scalar) {
    return RTV.logical(op(ac.re, ac.im, bc.re, bc.im));
  }

  // scalar op tensor
  if (ac.scalar && !bc.scalar) {
    const len = bc.re.length;
    const result = new FloatXArray(len);
    for (let i = 0; i < len; i++) {
      result[i] = op(ac.re, ac.im, bc.re[i], bc.im[i]) ? 1 : 0;
    }
    const t = RTV.tensor(result, bc.shape);
    t._isLogical = true;
    return t;
  }

  // tensor op scalar
  if (!ac.scalar && bc.scalar) {
    const len = ac.re.length;
    const result = new FloatXArray(len);
    for (let i = 0; i < len; i++) {
      result[i] = op(ac.re[i], ac.im[i], bc.re, bc.im) ? 1 : 0;
    }
    const t = RTV.tensor(result, ac.shape);
    t._isLogical = true;
    return t;
  }

  // tensor op tensor
  const at = ac as {
    scalar: false;
    re: FloatXArrayType;
    im: FloatXArrayType;
    shape: number[];
  };
  const bt = bc as {
    scalar: false;
    re: FloatXArrayType;
    im: FloatXArrayType;
    shape: number[];
  };

  // Check if shapes are identical (fast path)
  if (
    at.re.length === bt.re.length &&
    at.shape.length === bt.shape.length &&
    at.shape.every((d, i) => d === bt.shape[i])
  ) {
    const len = at.re.length;
    const result = new FloatXArray(len);
    for (let i = 0; i < len; i++) {
      result[i] = op(at.re[i], at.im[i], bt.re[i], bt.im[i]) ? 1 : 0;
    }
    const t = RTV.tensor(result, at.shape);
    t._isLogical = true;
    return t;
  }

  // Try broadcasting (broadcastComparisonComplex already sets _isLogical)
  const broadcastShape = getBroadcastShape(at.shape, bt.shape);
  if (broadcastShape !== null) {
    return broadcastComparisonComplex(at, bt, broadcastShape, op);
  }

  // Incompatible shapes
  throw new RuntimeError(
    `Matrix dimensions must agree for comparison: [${at.shape.join(",")}] vs [${bt.shape.join(",")}]`
  );
}

/** Complex binary operation supporting scalars and tensors */
function signedInf(x: number): number {
  return x > 0 ? Infinity : x < 0 ? -Infinity : 0;
}

function complexDivide(
  aRe: number,
  aIm: number,
  bRe: number,
  bIm: number
): { re: number; im: number } {
  const denom = bRe * bRe + bIm * bIm;
  if (denom === 0) {
    if (aRe === 0 && aIm === 0) {
      return { re: NaN, im: 0 };
    }
    return { re: signedInf(aRe), im: signedInf(aIm) };
  }
  return {
    re: (aRe * bRe + aIm * bIm) / denom,
    im: (aIm * bRe - aRe * bIm) / denom,
  };
}

function complexBinaryOp(
  a: RuntimeValue,
  b: RuntimeValue,
  op: (
    aRe: number,
    aIm: number,
    bRe: number,
    bIm: number
  ) => { re: number; im: number }
): RuntimeValue {
  const ac = toComplexParts(a);
  const bc = toComplexParts(b);

  // scalar op scalar
  if (ac.scalar && bc.scalar) {
    const result = op(ac.re, ac.im, bc.re, bc.im);
    return complexResult(result.re, result.im);
  }

  // scalar op tensor
  if (ac.scalar && !bc.scalar) {
    const len = bc.re.length;
    const resultRe = new FloatXArray(len);
    const resultIm = new FloatXArray(len);
    for (let i = 0; i < len; i++) {
      const r = op(ac.re, ac.im, bc.re[i], bc.im[i]);
      resultRe[i] = r.re;
      resultIm[i] = r.im;
    }
    // Check if result is purely real
    const isReal = resultIm.every(x => x === 0);
    return RTV.tensor(resultRe, bc.shape, isReal ? undefined : resultIm);
  }

  // tensor op scalar
  if (!ac.scalar && bc.scalar) {
    const len = ac.re.length;
    const resultRe = new FloatXArray(len);
    const resultIm = new FloatXArray(len);
    for (let i = 0; i < len; i++) {
      const r = op(ac.re[i], ac.im[i], bc.re, bc.im);
      resultRe[i] = r.re;
      resultIm[i] = r.im;
    }
    // Check if result is purely real
    const isReal = resultIm.every(x => x === 0);
    return RTV.tensor(resultRe, ac.shape, isReal ? undefined : resultIm);
  }

  // tensor op tensor
  const at = ac as {
    scalar: false;
    re: FloatXArrayType;
    im: FloatXArrayType;
    shape: number[];
  };
  const bt = bc as {
    scalar: false;
    re: FloatXArrayType;
    im: FloatXArrayType;
    shape: number[];
  };

  // Check if shapes are identical (fast path)
  if (
    at.re.length === bt.re.length &&
    at.shape.length === bt.shape.length &&
    at.shape.every((d, i) => d === bt.shape[i])
  ) {
    const len = at.re.length;
    const resultRe = new FloatXArray(len);
    const resultIm = new FloatXArray(len);
    for (let i = 0; i < len; i++) {
      const r = op(at.re[i], at.im[i], bt.re[i], bt.im[i]);
      resultRe[i] = r.re;
      resultIm[i] = r.im;
    }
    // Check if result is purely real
    const isReal = resultIm.every(x => x === 0);
    return RTV.tensor(resultRe, at.shape, isReal ? undefined : resultIm);
  }

  // Try broadcasting
  const broadcastShape = getBroadcastShape(at.shape, bt.shape);
  if (broadcastShape !== null) {
    return broadcastBinaryComplex(at, bt, broadcastShape, op);
  }

  // Incompatible shapes
  throw new RuntimeError(
    `Matrix dimensions must agree: [${at.shape.join(",")}] vs [${bt.shape.join(",")}]`
  );
}

// ── Tensor element-wise fast path ────────────────────────────────────────

/** Element-wise op codes matching native addon convention. */
const ELEMWISE_ADD = 0;
const ELEMWISE_SUB = 1;
const ELEMWISE_MUL = 2;
const ELEMWISE_DIV = 3;

/**
 * Check whether two RuntimeValues are same-shape tensors.
 * Returns [tensorA, tensorB] on match, null otherwise.
 */
function matchSameShapeTensors(
  a: RuntimeValue,
  b: RuntimeValue
): [RuntimeTensor, RuntimeTensor] | null {
  if (
    typeof a !== "object" ||
    a === null ||
    (a as RuntimeTensor).kind !== "tensor" ||
    typeof b !== "object" ||
    b === null ||
    (b as RuntimeTensor).kind !== "tensor"
  )
    return null;
  const at = a as RuntimeTensor;
  const bt = b as RuntimeTensor;
  if (
    at.data.length !== bt.data.length ||
    at.shape.length !== bt.shape.length ||
    at.shape.some((d, i) => d !== bt.shape[i])
  )
    return null;
  return [at, bt];
}

/**
 * Try native element-wise op on two same-shape real tensors.
 * Returns null if native addon is unavailable (caller should use inline JS loop).
 */
function tryNativeElemwiseReal(
  at: RuntimeTensor,
  bt: RuntimeTensor,
  opCode: number
): RuntimeValue | null {
  const bridge = getLapackBridge();
  if (!bridge?.elemwise) return null;
  const result = bridge.elemwise(
    at.data as Float64Array,
    bt.data as Float64Array,
    opCode
  );
  return RTV.tensorRaw(result, at.shape);
}

/**
 * Try native scalar-tensor element-wise op.
 * Returns null if native addon is unavailable.
 */
function tryNativeElemwiseScalar(
  scalar: number,
  tensor: RuntimeTensor,
  opCode: number,
  scalarOnLeft: boolean
): RuntimeValue | null {
  if (tensor.imag) return null;
  const bridge = getLapackBridge();
  if (!bridge?.elemwiseScalar) return null;
  const result = bridge.elemwiseScalar(
    scalar,
    tensor.data as Float64Array,
    opCode,
    scalarOnLeft
  );
  return RTV.tensorRaw(result, tensor.shape);
}

/**
 * Element-wise op on two same-shape tensors where at least one is complex.
 * Uses native addon when available, otherwise JS callback loop.
 */
function tensorElemwiseComplex(
  at: RuntimeTensor,
  bt: RuntimeTensor,
  opCode: number,
  jsOp: (
    aRe: number,
    aIm: number,
    bRe: number,
    bIm: number
  ) => { re: number; im: number }
): RuntimeValue {
  const bridge = getLapackBridge();
  if (bridge?.elemwiseComplex) {
    const r = bridge.elemwiseComplex(
      at.data as Float64Array,
      (at.imag as Float64Array) ?? null,
      bt.data as Float64Array,
      (bt.imag as Float64Array) ?? null,
      opCode
    );
    if (r.im) return RTV.tensor(r.re, at.shape, r.im);
    return RTV.tensorRaw(r.re, at.shape);
  }
  const len = at.data.length;
  const aIm = at.imag;
  const bIm = bt.imag;
  const resultRe = new FloatXArray(len);
  const resultIm = new FloatXArray(len);
  if (aIm && bIm) {
    for (let i = 0; i < len; i++) {
      const r = jsOp(at.data[i], aIm[i], bt.data[i], bIm[i]);
      resultRe[i] = r.re;
      resultIm[i] = r.im;
    }
  } else if (aIm) {
    for (let i = 0; i < len; i++) {
      const r = jsOp(at.data[i], aIm[i], bt.data[i], 0);
      resultRe[i] = r.re;
      resultIm[i] = r.im;
    }
  } else {
    for (let i = 0; i < len; i++) {
      const r = jsOp(at.data[i], 0, bt.data[i], bIm![i]);
      resultRe[i] = r.re;
      resultIm[i] = r.im;
    }
  }
  const isReal = resultIm.every(x => x === 0);
  return RTV.tensor(resultRe, at.shape, isReal ? undefined : resultIm);
}

// ── Arithmetic operators ────────────────────────────────────────────────

/** Format a RuntimeValue for string concatenation via `+`.
 *  MATLAB converts numbers and logicals to their num2str form, and
 *  strings/chars to their raw text.  Used only by mAdd when one operand
 *  is a RuntimeString. */
function coerceToConcatString(v: RuntimeValue): string | null {
  if (isRuntimeString(v)) return v;
  if (isRuntimeChar(v)) return v.value;
  if (isRuntimeLogical(v)) return v ? "true" : "false";
  if (isRuntimeNumber(v)) return num2strScalar(v);
  // Scalar tensor (including logical-typed scalar tensors) — match
  // MATLAB by formatting the single value.
  if (isRuntimeTensor(v) && v.data.length === 1 && !v.imag) {
    const x = v.data[0];
    if (v._isLogical === true) return x ? "true" : "false";
    return num2strScalar(x);
  }
  return null;
}

/** Add two RuntimeValues */
export function mAdd(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  // MATLAB string `+` is concatenation, not numeric addition.  When
  // either operand is a string (and the other is convertible to text),
  // return a concatenated string.
  if (isRuntimeString(a) || isRuntimeString(b)) {
    const aStr = coerceToConcatString(a);
    const bStr = coerceToConcatString(b);
    if (aStr !== null && bStr !== null) {
      return RTV.string(aStr + bStr);
    }
    // Fall through to numeric path, which will raise a more descriptive
    // error (e.g. when trying to add a tensor to a string).
  }
  const m = matchSameShapeTensors(a, b);
  if (m) {
    const [at, bt] = m;
    if (!at.imag && !bt.imag) {
      const nr = tryNativeElemwiseReal(at, bt, ELEMWISE_ADD);
      if (nr) return nr;
      const len = at.data.length;
      const result = new FloatXArray(len);
      for (let i = 0; i < len; i++) result[i] = at.data[i] + bt.data[i];
      return RTV.tensorRaw(result, at.shape);
    }
    return tensorElemwiseComplex(
      at,
      bt,
      ELEMWISE_ADD,
      (aRe, aIm, bRe, bIm) => ({
        re: aRe + bRe,
        im: aIm + bIm,
      })
    );
  }
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mAddSparse(a, b);
  if (isComplexOrMixed(a, b)) {
    return complexBinaryOp(a, b, (aRe, aIm, bRe, bIm) => ({
      re: aRe + bRe,
      im: aIm + bIm,
    }));
  }
  // Scalar-tensor native fast path
  if (isRuntimeNumber(a) && isRuntimeTensor(b)) {
    const nr = tryNativeElemwiseScalar(a as number, b, ELEMWISE_ADD, true);
    if (nr) return nr;
  } else if (isRuntimeTensor(a) && isRuntimeNumber(b)) {
    const nr = tryNativeElemwiseScalar(b as number, a, ELEMWISE_ADD, false);
    if (nr) return nr;
  }
  return binaryOp(a, b, (x, y) => x + y);
}

/** Subtract two RuntimeValues */
export function mSub(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  const m = matchSameShapeTensors(a, b);
  if (m) {
    const [at, bt] = m;
    if (!at.imag && !bt.imag) {
      const nr = tryNativeElemwiseReal(at, bt, ELEMWISE_SUB);
      if (nr) return nr;
      const len = at.data.length;
      const result = new FloatXArray(len);
      for (let i = 0; i < len; i++) result[i] = at.data[i] - bt.data[i];
      return RTV.tensorRaw(result, at.shape);
    }
    return tensorElemwiseComplex(
      at,
      bt,
      ELEMWISE_SUB,
      (aRe, aIm, bRe, bIm) => ({
        re: aRe - bRe,
        im: aIm - bIm,
      })
    );
  }
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mSubSparse(a, b);
  if (isComplexOrMixed(a, b)) {
    return complexBinaryOp(a, b, (aRe, aIm, bRe, bIm) => ({
      re: aRe - bRe,
      im: aIm - bIm,
    }));
  }
  if (isRuntimeNumber(a) && isRuntimeTensor(b)) {
    const nr = tryNativeElemwiseScalar(a as number, b, ELEMWISE_SUB, true);
    if (nr) return nr;
  } else if (isRuntimeTensor(a) && isRuntimeNumber(b)) {
    const nr = tryNativeElemwiseScalar(b as number, a, ELEMWISE_SUB, false);
    if (nr) return nr;
  }
  return binaryOp(a, b, (x, y) => x - y);
}

/** Multiply two RuntimeValues (matrix multiply for 2D tensors, scalar otherwise) */
export function mMul(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mMulSparse(a, b);
  // Unwrap 1×1 tensors to scalars so scalar×tensor uses element-wise multiply
  if (isRuntimeTensor(a) && a.data.length === 1) a = unwrap1x1(a);
  if (isRuntimeTensor(b) && b.data.length === 1) b = unwrap1x1(b);
  // Matrix multiply if both are tensors
  if (isRuntimeTensor(a) && isRuntimeTensor(b)) {
    return matMul(a, b);
  }
  // Complex scalar multiplication
  if (isComplexOrMixed(a, b)) {
    return complexBinaryOp(a, b, (aRe, aIm, bRe, bIm) => ({
      re: aRe * bRe - aIm * bIm,
      im: aRe * bIm + aIm * bRe,
    }));
  }
  if (isRuntimeNumber(a) && isRuntimeTensor(b)) {
    const nr = tryNativeElemwiseScalar(a as number, b, ELEMWISE_MUL, true);
    if (nr) return nr;
  } else if (isRuntimeTensor(a) && isRuntimeNumber(b)) {
    const nr = tryNativeElemwiseScalar(b as number, a, ELEMWISE_MUL, false);
    if (nr) return nr;
  }
  return binaryOp(a, b, (x, y) => x * y);
}

/** Element-wise multiply */
export function mElemMul(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  const m = matchSameShapeTensors(a, b);
  if (m) {
    const [at, bt] = m;
    if (!at.imag && !bt.imag) {
      const nr = tryNativeElemwiseReal(at, bt, ELEMWISE_MUL);
      if (nr) return nr;
      const len = at.data.length;
      const result = new FloatXArray(len);
      for (let i = 0; i < len; i++) result[i] = at.data[i] * bt.data[i];
      return RTV.tensorRaw(result, at.shape);
    }
    return tensorElemwiseComplex(
      at,
      bt,
      ELEMWISE_MUL,
      (aRe, aIm, bRe, bIm) => ({
        re: aRe * bRe - aIm * bIm,
        im: aRe * bIm + aIm * bRe,
      })
    );
  }
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mElemMulSparse(a, b);
  if (isComplexOrMixed(a, b)) {
    return complexBinaryOp(a, b, (aRe, aIm, bRe, bIm) => ({
      re: aRe * bRe - aIm * bIm,
      im: aRe * bIm + aIm * bRe,
    }));
  }
  if (isRuntimeNumber(a) && isRuntimeTensor(b)) {
    const nr = tryNativeElemwiseScalar(a as number, b, ELEMWISE_MUL, true);
    if (nr) return nr;
  } else if (isRuntimeTensor(a) && isRuntimeNumber(b)) {
    const nr = tryNativeElemwiseScalar(b as number, a, ELEMWISE_MUL, false);
    if (nr) return nr;
  }
  return binaryOp(a, b, (x, y) => x * y);
}

/** Divide */
export function mDiv(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  // Matrix right division: A / B = (B' \ A')' (uses mldivide)
  // When B is a matrix, always use matrix division (scalar A is promoted);
  // mLeftDiv will error on dimension mismatch, matching MATLAB behaviour.
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b)) {
    return mDiv(densify(a), densify(b));
  }
  // Unwrap 1×1 tensors to scalars so scalar division path is used
  if (isRuntimeTensor(a) && a.data.length === 1) a = unwrap1x1(a);
  if (isRuntimeTensor(b) && b.data.length === 1) b = unwrap1x1(b);
  if (isRuntimeTensor(b)) {
    const at = mConjugateTranspose(coerceToTensor(a, "mrdivide"));
    const bt = mConjugateTranspose(b);
    const result = mLeftDiv(bt, at);
    return mConjugateTranspose(result);
  }
  // Scalar or element-wise division (b is not a tensor)
  if (isComplexOrMixed(a, b)) {
    return complexBinaryOp(a, b, complexDivide);
  }
  return binaryOp(a, b, (x, y) => x / y);
}

/** Element-wise divide */
export function mElemDiv(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  const m = matchSameShapeTensors(a, b);
  if (m) {
    const [at, bt] = m;
    if (!at.imag && !bt.imag) {
      const nr = tryNativeElemwiseReal(at, bt, ELEMWISE_DIV);
      if (nr) return nr;
      const len = at.data.length;
      const result = new FloatXArray(len);
      for (let i = 0; i < len; i++) result[i] = at.data[i] / bt.data[i];
      return RTV.tensorRaw(result, at.shape);
    }
    return tensorElemwiseComplex(at, bt, ELEMWISE_DIV, complexDivide);
  }
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mElemDivSparse(a, b);
  if (isComplexOrMixed(a, b)) {
    return complexBinaryOp(a, b, complexDivide);
  }
  if (isRuntimeNumber(a) && isRuntimeTensor(b)) {
    const nr = tryNativeElemwiseScalar(a as number, b, ELEMWISE_DIV, true);
    if (nr) return nr;
  } else if (isRuntimeTensor(a) && isRuntimeNumber(b)) {
    const nr = tryNativeElemwiseScalar(b as number, a, ELEMWISE_DIV, false);
    if (nr) return nr;
  }
  return binaryOp(a, b, (x, y) => x / y);
}

/** Left division (mldivide): a \ b — for scalars b/a, for matrices solve a*x = b */
export function mLeftDiv(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  // Densify sparse operands for linear solve
  if (isRuntimeSparseMatrix(a)) return mLeftDiv(densify(a), b);
  if (isRuntimeSparseMatrix(b)) return mLeftDiv(a, densify(b));
  // Unwrap 1×1 tensors to scalars so scalar division path is used
  if (isRuntimeTensor(a) && a.data.length === 1) a = unwrap1x1(a);
  if (isRuntimeTensor(b) && b.data.length === 1) b = unwrap1x1(b);
  // Scalar A: A \ B is element-wise B / A
  if (isRuntimeNumber(a) || isRuntimeLogical(a)) {
    return mElemDiv(b, a);
  }
  // Matrix A: solve A * X = B  (mldivide / linsolve semantics)
  const tensorA = a;
  const tensorB =
    isRuntimeNumber(b) || isRuntimeLogical(b)
      ? RTV.tensor(new FloatXArray([toNumber(b)]), [1, 1])
      : b;
  if (!isRuntimeTensor(tensorA) || !isRuntimeTensor(tensorB))
    throw new RuntimeError("mldivide (\\): operands must be numeric matrices");
  const [mA, nA] = tensorSize2D(tensorA);
  const [mB, nB] = tensorSize2D(tensorB);
  if (mB !== mA)
    throw new RuntimeError(
      `mldivide (\\): A is ${mA}×${nA} but B has ${mB} rows`
    );
  // Empty matrix: return empty result (nA × nB)
  if (mA === 0 || nA === 0 || nB === 0) {
    return RTV.tensor(new FloatXArray(nA * nB), [nA, nB]);
  }
  if (tensorA.imag || tensorB.imag) {
    const ARe = tensorA.data;
    const AIm = tensorA.imag ?? new FloatXArray(tensorA.data.length);
    const BRe = tensorB.data;
    const BIm = tensorB.imag ?? new FloatXArray(tensorB.data.length);
    const X = linsolveComplexLapack(ARe, AIm, mA, nA, BRe, BIm, nB);
    return unwrap1x1(
      RTV.tensor(new FloatXArray(X.re), [nA, nB], new FloatXArray(X.im))
    );
  }
  const X = linsolveLapack(tensorA.data, mA, nA, tensorB.data, nB);
  if (!X) throw new RuntimeError("mldivide (\\): LAPACK bridge unavailable");
  return unwrap1x1(RTV.tensor(new FloatXArray(X), [nA, nB]));
}

/** Element-wise left division (ldivide): a .\ b = b ./ a */
export function mElemLeftDiv(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  return mElemDiv(b, a);
}

/** Power */
export function mPow(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isComplexOrMixed(a, b)) {
    const ac = toComplex(a),
      bc = toComplex(b);
    // z^w = exp(w * ln(z)), using polar form: ln(z) = ln|z| + i*arg(z)
    const r = Math.sqrt(ac.re * ac.re + ac.im * ac.im);
    if (r === 0) {
      // 0^w: result is 0 for positive real exponent, 1 for w=0
      if (bc.re === 0 && bc.im === 0) return complexResult(1, 0);
      if (bc.re > 0) return complexResult(0, 0);
      return complexResult(Infinity, 0);
    }
    const theta = Math.atan2(ac.im, ac.re);
    const lnR = Math.log(r);
    // w * ln(z) = (bre + bi*i) * (lnR + theta*i) = (bre*lnR - bi*theta) + (bre*theta + bi*lnR)*i
    const newRe = bc.re * lnR - bc.im * theta;
    const newIm = bc.re * theta + bc.im * lnR;
    const expR = Math.exp(newRe);
    return complexResult(expR * Math.cos(newIm), expR * Math.sin(newIm));
  }
  // Matrix power if a is a matrix and b is scalar integer
  if (isRuntimeTensor(a) && (isRuntimeNumber(b) || isRuntimeLogical(b))) {
    const exp = toNumber(b);
    const [rows, cols] = tensorSize2D(a);
    if (rows === cols && Number.isInteger(exp)) {
      if (exp === 0) {
        // A^0 = eye(n)
        const data = new FloatXArray(rows * cols);
        for (let i = 0; i < rows; i++) {
          data[colMajorIndex(i, i, rows)] = 1;
        }
        return RTV.tensor(data, [rows, cols]);
      }
      let base: RuntimeTensor = a;
      let n = exp;
      if (n < 0) {
        // A^(-n) = inv(A)^n
        const invA = applyBuiltinFn("mpower", "inv", [a], 1);
        if (!isRuntimeTensor(invA))
          throw new RuntimeError("mpower: inv returned non-tensor");
        base = invA;
        n = -n;
      }
      // Repeated squaring for A^n
      let result: RuntimeValue = base;
      for (let i = 1; i < n; i++) {
        result = matMul(result as RuntimeTensor, base);
      }
      return result;
    }
  }
  // Scalar negative base with non-integer exponent → complex result
  if (
    isRuntimeNumber(a) &&
    a < 0 &&
    isRuntimeNumber(b) &&
    !Number.isInteger(b)
  ) {
    const r = Math.abs(a);
    const theta = Math.PI; // arg of negative real
    const lnR = Math.log(r);
    const bVal = b as number;
    const newRe = bVal * lnR;
    const newIm = bVal * theta;
    const expR = Math.exp(newRe);
    return complexResult(expR * Math.cos(newIm), expR * Math.sin(newIm));
  }
  return binaryOp(a, b, (x, y) => Math.pow(x, y));
}

/** Element-wise power */
export function mElemPow(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  // Fast path: real tensor .^ scalar integer
  if (isRuntimeTensor(a) && !a.imag && isRuntimeNumber(b)) {
    const exp = b as number;
    if (exp === 2) {
      const result = new FloatXArray(a.data.length);
      for (let i = 0; i < result.length; i++) result[i] = a.data[i] * a.data[i];
      return RTV.tensor(result, a.shape);
    }
    if (Number.isInteger(exp) || exp >= 0) {
      // No negative bases with fractional exponent concern if exp is integer or base could be negative
      let hasNeg = false;
      if (!Number.isInteger(exp)) {
        for (let i = 0; i < a.data.length; i++) {
          if (a.data[i] < 0) {
            hasNeg = true;
            break;
          }
        }
      }
      if (!hasNeg) {
        const result = new FloatXArray(a.data.length);
        for (let i = 0; i < result.length; i++)
          result[i] = Math.pow(a.data[i], exp);
        return RTV.tensor(result, a.shape);
      }
    }
  }
  const complexPow = (aRe: number, aIm: number, bRe: number, bIm: number) => {
    // z^w = exp(w * ln(z)), using polar form: ln(z) = ln|z| + i*arg(z)
    const r = Math.sqrt(aRe * aRe + aIm * aIm);
    if (r === 0) {
      // 0^w: result is 0 for positive real exponent, 1 for w=0
      if (bRe === 0 && bIm === 0) return { re: 1, im: 0 };
      if (bRe > 0) return { re: 0, im: 0 };
      return { re: Infinity, im: 0 };
    }
    const theta = Math.atan2(aIm, aRe);
    const lnR = Math.log(r);
    const newRe = bRe * lnR - bIm * theta;
    const newIm = bRe * theta + bIm * lnR;
    const expR = Math.exp(newRe);
    return { re: expR * Math.cos(newIm), im: expR * Math.sin(newIm) };
  };
  if (isComplexOrMixed(a, b)) {
    return complexBinaryOp(a, b, complexPow);
  }
  // Check if base has negative values — need complex path for non-integer exponents
  let hasNeg = false;
  if (isRuntimeNumber(a) && a < 0) hasNeg = true;
  else if (isRuntimeTensor(a)) {
    for (let i = 0; i < a.data.length; i++) {
      if (a.data[i] < 0) {
        hasNeg = true;
        break;
      }
    }
  }
  if (hasNeg) {
    // For integer exponents, use real Math.pow per-element (always produces real results)
    let allInt = false;
    if (isRuntimeNumber(b)) {
      allInt = Number.isInteger(b as number);
    } else if (isRuntimeLogical(b)) {
      allInt = true;
    } else if (isRuntimeTensor(b)) {
      allInt = true;
      for (let i = 0; i < b.data.length; i++) {
        if (!Number.isInteger(b.data[i])) {
          allInt = false;
          break;
        }
      }
    }
    if (allInt) {
      return binaryOp(a, b, (x, y) => Math.pow(x, y));
    }
    return complexBinaryOp(a, b, complexPow);
  }
  return binaryOp(a, b, (x, y) => Math.pow(x, y));
}

/** Negation */
export function mNeg(v: RuntimeValue): RuntimeValue {
  // Fast path: real tensor
  if (isRuntimeTensor(v)) {
    if (v.imag !== undefined) {
      const resultRe = new FloatXArray(v.data.length);
      const resultIm = new FloatXArray(v.imag.length);
      for (let i = 0; i < v.data.length; i++) {
        resultRe[i] = -v.data[i];
        resultIm[i] = -v.imag[i];
      }
      return RTV.tensor(resultRe, v.shape, resultIm);
    }
    const resultRe = new FloatXArray(v.data.length);
    for (let i = 0; i < v.data.length; i++) resultRe[i] = -v.data[i];
    return RTV.tensor(resultRe, v.shape);
  }
  if (isRuntimeSparseMatrix(v)) return sparseNeg(v);
  if (isRuntimeComplexNumber(v)) return RTV.complex(-v.re, -v.im);
  return unaryOp(v, x => -x);
}

/** Transpose a 2-D cell array — same index arithmetic as numeric transpose. */
function transposeCellArray(v: RuntimeCell): RuntimeValue {
  const s = v.shape;
  const rows = s.length === 0 ? 1 : s.length === 1 ? 1 : s[0];
  const cols = s.length === 0 ? 1 : s.length === 1 ? s[0] : s[1];
  const result: RuntimeValue[] = new Array(v.data.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      result[r * cols + c] = v.data[colMajorIndex(r, c, rows)];
    }
  }
  return RTV.cell(result, [cols, rows]);
}

/** Core transpose logic for tensors, with optional conjugation of imaginary part */
function transposeCore(v: RuntimeTensor, conjugate: boolean): RuntimeValue {
  const [rows, cols] = tensorSize2D(v);
  const resultRe = new FloatXArray(v.data.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      resultRe[r * cols + c] = v.data[colMajorIndex(r, c, rows)];
    }
  }
  if (v.imag !== undefined) {
    const resultIm = new FloatXArray(v.imag.length);
    const sign = conjugate ? -1 : 1;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        resultIm[r * cols + c] = sign * v.imag[colMajorIndex(r, c, rows)];
      }
    }
    return RTV.tensor(resultRe, [cols, rows], resultIm);
  }
  const t = RTV.tensor(resultRe, [cols, rows]);
  if (v._isLogical) t._isLogical = true;
  return t;
}

/** Transpose (non-conjugate for complex scalars and tensors) */
export function mTranspose(v: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(v)) return sparseTranspose(v);
  if (isRuntimeComplexNumber(v)) return v;
  if (isRuntimeNumber(v) || isRuntimeLogical(v)) return v;
  if (isRuntimeCell(v)) return transposeCellArray(v);
  if (isRuntimeChar(v)) return v;
  if (!isRuntimeTensor(v))
    throw new RuntimeError("Cannot transpose non-numeric value");
  return transposeCore(v, false);
}

/** Conjugate transpose (Hermitian transpose) - transposes and conjugates */
export function mConjugateTranspose(v: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(v)) return sparseConjugateTranspose(v);
  if (isRuntimeComplexNumber(v)) return RTV.complex(v.re, -v.im);
  if (isRuntimeNumber(v) || isRuntimeLogical(v)) return v;
  if (isRuntimeCell(v)) return transposeCellArray(v);
  if (isRuntimeChar(v)) return v;
  if (!isRuntimeTensor(v))
    throw new RuntimeError("Cannot transpose non-numeric value");
  return transposeCore(v, true);
}

// ── Comparison operators ────────────────────────────────────────────────

/** When one operand is a string and the other is a char, convert the char
 *  to a string so the comparison becomes scalar string comparison (MATLAB
 *  semantics). Returns null if neither operand is a string-char mix. */
function asStringPair(
  a: RuntimeValue,
  b: RuntimeValue
): [string, string] | null {
  if (isRuntimeString(a) && isRuntimeString(b)) return [a, b];
  if (isRuntimeString(a) && isRuntimeChar(b)) return [a, b.value];
  if (isRuntimeChar(a) && isRuntimeString(b)) return [a.value, b];
  return null;
}

function stringComparisonOp(
  a: RuntimeValue,
  b: RuntimeValue,
  op: (x: string, y: string) => boolean
): RuntimeValue | null {
  const pair = asStringPair(a, b);
  if (pair === null) return null;
  return RTV.logical(op(pair[0], pair[1]));
}

/** Convert sparse to dense for fallback paths. */
function densify(v: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(v)) return sparseToDense(v);
  return v;
}

export function mEqual(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mEqual(densify(a), densify(b));
  const sr = stringComparisonOp(a, b, (x, y) => x === y);
  if (sr !== null) return sr;
  if (isComplexOrMixed(a, b)) {
    return complexComparisonOp(
      a,
      b,
      (aRe, aIm, bRe, bIm) => aRe === bRe && aIm === bIm
    );
  }
  return comparisonOp(a, b, (x, y) => x === y);
}

export function mNotEqual(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mNotEqual(densify(a), densify(b));
  const sr = stringComparisonOp(a, b, (x, y) => x !== y);
  if (sr !== null) return sr;
  if (isComplexOrMixed(a, b)) {
    return complexComparisonOp(
      a,
      b,
      (aRe, aIm, bRe, bIm) => aRe !== bRe || aIm !== bIm
    );
  }
  return comparisonOp(a, b, (x, y) => x !== y);
}

export function mLess(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mLess(densify(a), densify(b));
  const sr = stringComparisonOp(a, b, (x, y) => x < y);
  if (sr !== null) return sr;
  return comparisonOp(a, b, (x, y) => x < y);
}

export function mLessEqual(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mLessEqual(densify(a), densify(b));
  const sr = stringComparisonOp(a, b, (x, y) => x <= y);
  if (sr !== null) return sr;
  return comparisonOp(a, b, (x, y) => x <= y);
}

export function mGreater(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mGreater(densify(a), densify(b));
  const sr = stringComparisonOp(a, b, (x, y) => x > y);
  if (sr !== null) return sr;
  return comparisonOp(a, b, (x, y) => x > y);
}

export function mGreaterEqual(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mGreaterEqual(densify(a), densify(b));
  const sr = stringComparisonOp(a, b, (x, y) => x >= y);
  if (sr !== null) return sr;
  return comparisonOp(a, b, (x, y) => x >= y);
}

// ── Internal helpers ────────────────────────────────────────────────────

function asNumeric(
  v: RuntimeValue
):
  | { scalar: true; value: number; isComplex: false }
  | { scalar: true; re: number; im: number; isComplex: true }
  | { scalar: false; tensor: RuntimeTensor } {
  if (isRuntimeNumber(v)) {
    return { scalar: true, value: v, isComplex: false };
  }
  if (isRuntimeLogical(v)) {
    return { scalar: true, value: v ? 1 : 0, isComplex: false };
  }
  if (isRuntimeComplexNumber(v)) {
    return { scalar: true, re: v.re, im: v.im, isComplex: true };
  }
  if (isRuntimeTensor(v)) {
    if (v.data.length === 1) {
      if (v.imag !== undefined) {
        return {
          scalar: true,
          re: v.data[0],
          im: v.imag[0],
          isComplex: true,
        };
      }
      return { scalar: true, value: v.data[0], isComplex: false };
    }
    return { scalar: false, tensor: v };
  }
  if (isRuntimeSparseMatrix(v)) {
    // Densify sparse for generic numeric operations
    return asNumeric(densify(v));
  }
  if (isRuntimeChar(v)) {
    // Char arithmetic: treat each character as its UTF-16 code point
    if (v.value.length === 1) {
      return { scalar: true, value: v.value.charCodeAt(0), isComplex: false };
    }
    // Multi-char: row vector of code points
    const codes = new FloatXArray(
      Array.from(v.value).map(c => c.charCodeAt(0))
    );
    return {
      scalar: false,
      tensor: {
        kind: "tensor",
        data: codes,
        shape: [1, v.value.length],
        _rc: 1,
      },
    };
  }
  throw new RuntimeError(`Cannot use ${kstr(v)} in numeric operation`);
}

/**
 * Check if two shapes are compatible for broadcasting and compute the output shape.
 * Broadcasting: dimensions must be equal or one of them must be 1.
 * Returns null if shapes are incompatible, otherwise returns the broadcast shape.
 */
export function getBroadcastShape(
  shapeA: number[],
  shapeB: number[]
): number[] | null {
  // Pad shorter shape with leading 1s to match lengths
  const ndim = Math.max(shapeA.length, shapeB.length, 2);
  const a = padShape(
    shapeA.length === 0
      ? [1, 1]
      : shapeA.length === 1
        ? [1, shapeA[0]]
        : shapeA,
    ndim
  );
  const b = padShape(
    shapeB.length === 0
      ? [1, 1]
      : shapeB.length === 1
        ? [1, shapeB[0]]
        : shapeB,
    ndim
  );

  const outShape: number[] = [];

  for (let i = 0; i < ndim; i++) {
    const dimA = a[i];
    const dimB = b[i];

    if (dimA === dimB) {
      outShape.push(dimA);
    } else if (dimA === 1) {
      outShape.push(dimB);
    } else if (dimB === 1) {
      outShape.push(dimA);
    } else {
      // Incompatible dimensions
      return null;
    }
  }

  return outShape;
}

/** Pad a shape array to `ndim` dimensions by appending 1s */
function padShape(shape: number[], ndim: number): number[] {
  if (shape.length >= ndim) return shape;
  const padded = [...shape];
  while (padded.length < ndim) padded.push(1);
  return padded;
}

/**
 * Iterate over all elements of a broadcast result in column-major order,
 * calling `visit(aIdx, bIdx, outIdx)` for each output element.
 */
export function broadcastIterate(
  aShape: number[],
  bShape: number[],
  outShape: number[],
  visit: (aIdx: number, bIdx: number, outIdx: number) => void
): void {
  const ndim = outShape.length;
  const totalElems = outShape.reduce((acc, d) => acc * d, 1);
  const aPadded = padShape(aShape, ndim);
  const bPadded = padShape(bShape, ndim);

  // Precompute strides — zero for broadcast (size-1) dimensions
  const aStrides = new Array(ndim);
  const bStrides = new Array(ndim);
  let aStr = 1,
    bStr = 1;
  for (let d = 0; d < ndim; d++) {
    aStrides[d] = aPadded[d] === 1 ? 0 : aStr;
    bStrides[d] = bPadded[d] === 1 ? 0 : bStr;
    aStr *= aPadded[d];
    bStr *= bPadded[d];
  }

  const subs = new Array(ndim).fill(0);
  let aIdx = 0,
    bIdx = 0;
  for (let i = 0; i < totalElems; i++) {
    visit(aIdx, bIdx, i);
    for (let d = 0; d < ndim; d++) {
      subs[d]++;
      aIdx += aStrides[d];
      bIdx += bStrides[d];
      if (subs[d] < outShape[d]) break;
      aIdx -= subs[d] * aStrides[d];
      bIdx -= subs[d] * bStrides[d];
      subs[d] = 0;
    }
  }
}

function broadcastBinary(
  a: RuntimeTensor,
  b: RuntimeTensor,
  outShape: number[],
  op: (x: number, y: number) => number
): RuntimeTensor {
  const result = new FloatXArray(outShape.reduce((acc, d) => acc * d, 1));
  broadcastIterate(a.shape, b.shape, outShape, (aIdx, bIdx, i) => {
    result[i] = op(a.data[aIdx], b.data[bIdx]);
  });
  return RTV.tensor(result, outShape) as RuntimeTensor;
}

function broadcastBinaryComplex(
  a: { re: FloatXArrayType; im: FloatXArrayType; shape: number[] },
  b: { re: FloatXArrayType; im: FloatXArrayType; shape: number[] },
  outShape: number[],
  op: (
    aRe: number,
    aIm: number,
    bRe: number,
    bIm: number
  ) => { re: number; im: number }
): RuntimeTensor {
  const totalElems = outShape.reduce((acc, d) => acc * d, 1);
  const resultRe = new FloatXArray(totalElems);
  const resultIm = new FloatXArray(totalElems);
  broadcastIterate(a.shape, b.shape, outShape, (aIdx, bIdx, i) => {
    const r = op(a.re[aIdx], a.im[aIdx], b.re[bIdx], b.im[bIdx]);
    resultRe[i] = r.re;
    resultIm[i] = r.im;
  });
  const isReal = resultIm.every(x => x === 0);
  return RTV.tensor(
    resultRe,
    outShape,
    isReal ? undefined : resultIm
  ) as RuntimeTensor;
}

function broadcastComparisonComplex(
  a: { re: FloatXArrayType; im: FloatXArrayType; shape: number[] },
  b: { re: FloatXArrayType; im: FloatXArrayType; shape: number[] },
  outShape: number[],
  op: (aRe: number, aIm: number, bRe: number, bIm: number) => boolean
): RuntimeTensor {
  const result = new FloatXArray(outShape.reduce((acc, d) => acc * d, 1));
  broadcastIterate(a.shape, b.shape, outShape, (aIdx, bIdx, i) => {
    result[i] = op(a.re[aIdx], a.im[aIdx], b.re[bIdx], b.im[bIdx]) ? 1 : 0;
  });
  const t = RTV.tensor(result, outShape) as RuntimeTensor;
  t._isLogical = true;
  return t;
}

function broadcastComparison(
  a: RuntimeTensor,
  b: RuntimeTensor,
  outShape: number[],
  op: (x: number, y: number) => boolean
): RuntimeTensor {
  const result = new FloatXArray(outShape.reduce((acc, d) => acc * d, 1));
  broadcastIterate(a.shape, b.shape, outShape, (aIdx, bIdx, i) => {
    result[i] = op(a.data[aIdx], b.data[bIdx]) ? 1 : 0;
  });
  const t = RTV.tensor(result, outShape) as RuntimeTensor;
  t._isLogical = true;
  return t;
}

function binaryOp(
  a: RuntimeValue,
  b: RuntimeValue,
  op: (x: number, y: number) => number
): RuntimeValue {
  // Fast path: real tensor op real tensor (most common in vectorized code)
  if (isRuntimeTensor(a) && a.data.length > 1) {
    if (isRuntimeTensor(b) && b.data.length > 1) {
      // tensor op tensor — same shape fast path
      if (
        a.data.length === b.data.length &&
        a.shape.length === b.shape.length &&
        a.shape.every((d, i) => d === b.shape[i])
      ) {
        const result = new FloatXArray(a.data.length);
        for (let i = 0; i < result.length; i++) {
          result[i] = op(a.data[i], b.data[i]);
        }
        return RTV.tensor(result, a.shape);
      }
      // Try broadcasting
      const broadcastShape = getBroadcastShape(a.shape, b.shape);
      if (broadcastShape !== null) {
        return broadcastBinary(a, b, broadcastShape, op);
      }
      throw new RuntimeError(
        `Matrix dimensions must agree: [${a.shape.join(",")}] vs [${b.shape.join(",")}]`
      );
    }
    if (isRuntimeNumber(b)) {
      // tensor op scalar
      const result = new FloatXArray(a.data.length);
      const sv = b as number;
      for (let i = 0; i < result.length; i++) {
        result[i] = op(a.data[i], sv);
      }
      return RTV.tensor(result, a.shape);
    }
  }
  if (isRuntimeTensor(b) && b.data.length > 1 && isRuntimeNumber(a)) {
    // scalar op tensor
    const result = new FloatXArray(b.data.length);
    const sv = a as number;
    for (let i = 0; i < result.length; i++) {
      result[i] = op(sv, b.data[i]);
    }
    return RTV.tensor(result, b.shape);
  }

  // General path via asNumeric
  const an = asNumeric(a);
  const bn = asNumeric(b);

  // scalar op scalar
  if (an.scalar && bn.scalar) {
    const aVal = an.isComplex ? an.re : an.value;
    const bVal = bn.isComplex ? bn.re : bn.value;
    return RTV.num(op(aVal, bVal));
  }

  // scalar op tensor
  if (an.scalar && !bn.scalar) {
    const t = bn.tensor;
    const result = new FloatXArray(t.data.length);
    const scalarVal = an.isComplex ? an.re : an.value;
    for (let i = 0; i < result.length; i++) {
      result[i] = op(scalarVal, t.data[i]);
    }
    return RTV.tensor(result, t.shape);
  }

  // tensor op scalar
  if (!an.scalar && bn.scalar) {
    const t = an.tensor;
    const result = new FloatXArray(t.data.length);
    const scalarVal = bn.isComplex ? bn.re : bn.value;
    for (let i = 0; i < result.length; i++) {
      result[i] = op(t.data[i], scalarVal);
    }
    return RTV.tensor(result, t.shape);
  }

  // tensor op tensor
  const at = (an as { scalar: false; tensor: RuntimeTensor }).tensor;
  const bt = (bn as { scalar: false; tensor: RuntimeTensor }).tensor;

  // Check if shapes are identical (fast path)
  if (
    at.data.length === bt.data.length &&
    at.shape.length === bt.shape.length &&
    at.shape.every((d, i) => d === bt.shape[i])
  ) {
    const result = new FloatXArray(at.data.length);
    for (let i = 0; i < result.length; i++) {
      result[i] = op(at.data[i], bt.data[i]);
    }
    return RTV.tensor(result, at.shape);
  }

  // Try broadcasting
  const broadcastShape = getBroadcastShape(at.shape, bt.shape);
  if (broadcastShape !== null) {
    return broadcastBinary(at, bt, broadcastShape, op);
  }

  // Incompatible shapes
  throw new RuntimeError(
    `Matrix dimensions must agree: [${at.shape.join(",")}] vs [${bt.shape.join(",")}]`
  );
}

function unaryOp(v: RuntimeValue, op: (x: number) => number): RuntimeValue {
  // Fast path: real tensor
  if (isRuntimeTensor(v) && v.data.length > 1) {
    const result = new FloatXArray(v.data.length);
    for (let i = 0; i < result.length; i++) {
      result[i] = op(v.data[i]);
    }
    return RTV.tensor(result, v.shape);
  }
  const n = asNumeric(v);
  if (n.scalar) {
    const val = n.isComplex ? n.re : n.value;
    return RTV.num(op(val));
  }
  const result = new FloatXArray(n.tensor.data.length);
  for (let i = 0; i < result.length; i++) {
    result[i] = op(n.tensor.data[i]);
  }
  return RTV.tensor(result, n.tensor.shape);
}

function comparisonOp(
  a: RuntimeValue,
  b: RuntimeValue,
  op: (x: number, y: number) => boolean
): RuntimeValue {
  const an = asNumeric(a);
  const bn = asNumeric(b);

  if (an.scalar && bn.scalar) {
    const aVal = an.isComplex ? an.re : an.value;
    const bVal = bn.isComplex ? bn.re : bn.value;
    return RTV.logical(op(aVal, bVal));
  }

  // For tensor comparisons, return a logical tensor (as num tensor with 0/1)
  if (an.scalar && !bn.scalar) {
    const result = new FloatXArray(bn.tensor.data.length);
    const scalarVal = an.isComplex ? an.re : an.value;
    for (let i = 0; i < result.length; i++) {
      result[i] = op(scalarVal, bn.tensor.data[i]) ? 1 : 0;
    }
    const t = RTV.tensor(result, bn.tensor.shape);
    t._isLogical = true;
    return t;
  }

  if (!an.scalar && bn.scalar) {
    const result = new FloatXArray(an.tensor.data.length);
    const scalarVal = bn.isComplex ? bn.re : bn.value;
    for (let i = 0; i < result.length; i++) {
      result[i] = op(an.tensor.data[i], scalarVal) ? 1 : 0;
    }
    const t = RTV.tensor(result, an.tensor.shape);
    t._isLogical = true;
    return t;
  }

  const at = (an as { scalar: false; tensor: RuntimeTensor }).tensor;
  const bt = (bn as { scalar: false; tensor: RuntimeTensor }).tensor;

  // Check if shapes are identical (fast path)
  if (
    at.data.length === bt.data.length &&
    at.shape.length === bt.shape.length &&
    at.shape.every((d, i) => d === bt.shape[i])
  ) {
    const result = new FloatXArray(at.data.length);
    for (let i = 0; i < result.length; i++) {
      result[i] = op(at.data[i], bt.data[i]) ? 1 : 0;
    }
    const t = RTV.tensor(result, at.shape);
    t._isLogical = true;
    return t;
  }

  // Try broadcasting
  const broadcastShape = getBroadcastShape(at.shape, bt.shape);
  if (broadcastShape !== null) {
    return broadcastComparison(at, bt, broadcastShape, op);
  }

  // Incompatible shapes
  throw new RuntimeError(
    `Matrix dimensions must agree for comparison: [${at.shape.join(",")}] vs [${bt.shape.join(",")}]`
  );
}

/** Unwrap a 1×1 tensor to a plain scalar (number or complex). */
function unwrap1x1(v: RuntimeValue): RuntimeValue {
  if (isRuntimeTensor(v) && v.data.length === 1) {
    if (v.imag && v.imag[0] !== 0) return RTV.complex(v.data[0], v.imag[0]);
    return RTV.num(v.data[0]);
  }
  return v;
}

function matMul(a: RuntimeTensor, b: RuntimeTensor): RuntimeValue {
  const [aRows, aCols] = tensorSize2D(a);
  const [bRows, bCols] = tensorSize2D(b);

  if (aCols !== bRows) {
    throw new RuntimeError(
      `Inner matrix dimensions must agree: ${aCols} vs ${bRows}`
    );
  }

  // Handle empty matrix multiplication (K=0, or zero output dimensions).
  // zeros(m,0)*zeros(0,n) = zeros(m,n). BLAS dgemm requires ldb>=1
  // even when K=0, so we must short-circuit before calling native code.
  if (aRows === 0 || aCols === 0 || bCols === 0) {
    return RTV.tensor(new FloatXArray(aRows * bCols), [aRows, bCols]);
  }

  const isComplex = a.imag !== undefined || b.imag !== undefined;

  if (!isComplex) {
    const bridge = getEffectiveBridge("matmul", "matmul");
    const f64A =
      a.data instanceof Float64Array ? a.data : new Float64Array(a.data);
    const f64B =
      b.data instanceof Float64Array ? b.data : new Float64Array(b.data);
    const raw = bridge.matmul!(f64A, aRows, aCols, f64B, bCols);
    return unwrap1x1(RTV.tensor(new FloatXArray(raw), [aRows, bCols]));
  }

  // Complex matrix multiplication — try native zgemm first
  const aIm = a.imag || new FloatXArray(a.data.length);
  const bIm = b.imag || new FloatXArray(b.data.length);

  const bridge = getEffectiveBridge("matmul", "matmulComplex");
  if (bridge.matmulComplex) {
    const f64ARe =
      a.data instanceof Float64Array ? a.data : new Float64Array(a.data);
    const f64AIm = aIm instanceof Float64Array ? aIm : new Float64Array(aIm);
    const f64BRe =
      b.data instanceof Float64Array ? b.data : new Float64Array(b.data);
    const f64BIm = bIm instanceof Float64Array ? bIm : new Float64Array(bIm);
    const raw = bridge.matmulComplex(
      f64ARe,
      f64AIm,
      aRows,
      aCols,
      f64BRe,
      f64BIm,
      bCols
    );
    return unwrap1x1(
      RTV.tensor(
        new FloatXArray(raw.re),
        [aRows, bCols],
        raw.im ? new FloatXArray(raw.im) : undefined
      )
    );
  }

  // Fallback: pure JavaScript complex matmul
  const resultRe = new FloatXArray(aRows * bCols);
  const resultIm = new FloatXArray(aRows * bCols);

  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bCols; j++) {
      let sumRe = 0;
      let sumIm = 0;
      for (let k = 0; k < aCols; k++) {
        const aIdx = colMajorIndex(i, k, aRows);
        const bIdx = colMajorIndex(k, j, bRows);
        const aRe = a.data[aIdx];
        const aI = aIm[aIdx];
        const bRe = b.data[bIdx];
        const bI = bIm[bIdx];
        sumRe += aRe * bRe - aI * bI;
        sumIm += aRe * bI + aI * bRe;
      }
      const outIdx = colMajorIndex(i, j, aRows);
      resultRe[outIdx] = sumRe;
      resultIm[outIdx] = sumIm;
    }
  }

  const isPurelyReal = resultIm.every(x => x === 0);
  return unwrap1x1(
    RTV.tensor(resultRe, [aRows, bCols], isPurelyReal ? undefined : resultIm)
  );
}
