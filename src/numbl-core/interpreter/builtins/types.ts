/**
 * Shared helpers for interpreter builtins (the registry itself is in
 * runtime/builtinRegistry.ts).
 */

import type { ElementClass, RuntimeValue } from "../../runtime/types.js";
import {
  RuntimeTensor,
  RuntimeComplexNumber,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { tensorOps, OpUnary } from "../../ops/index.js";
import { RTV } from "../../runtime/constructors.js";
import { numericArg } from "./args.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import {
  getBroadcastShape,
  broadcastIterate,
} from "../../helpers/arithmetic.js";

// ── Shared helpers ──────────────────────────────────────────────────────

export function mkc(re: number, im: number): number | RuntimeComplexNumber {
  if (im === 0) return re;
  return new RuntimeComplexNumber(re, im);
}

export function makeTensor(
  data: Float64Array,
  imag: Float64Array | undefined,
  shape: number[],
  cls: ElementClass = "double"
): RuntimeTensor {
  return RTV.tensor(data, shape, imag, cls);
}

// ── Apply helpers ───────────────────────────────────────────────────────

/** Apply a unary element-wise function with complex support */
/** Map Math.* functions to tensor-ops unary op codes. */
const nativeUnaryOpCode = new Map<(x: number) => number, number>([
  [Math.exp, OpUnary.EXP],
  [Math.log, OpUnary.LOG],
  [Math.log2, OpUnary.LOG2],
  [Math.log10, OpUnary.LOG10],
  [Math.sqrt, OpUnary.SQRT],
  [Math.abs, OpUnary.ABS],
  [Math.floor, OpUnary.FLOOR],
  [Math.ceil, OpUnary.CEIL],
  [Math.round, OpUnary.ROUND],
  [Math.trunc, OpUnary.TRUNC],
  [Math.sin, OpUnary.SIN],
  [Math.cos, OpUnary.COS],
  [Math.tan, OpUnary.TAN],
  [Math.asin, OpUnary.ASIN],
  [Math.acos, OpUnary.ACOS],
  [Math.atan, OpUnary.ATAN],
  [Math.sinh, OpUnary.SINH],
  [Math.cosh, OpUnary.COSH],
  [Math.tanh, OpUnary.TANH],
  [Math.sign, OpUnary.SIGN],
]);

export function applyUnaryElemwise(
  v: RuntimeValue,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  name: string
): RuntimeValue {
  if (isRuntimeSparseMatrix(v))
    return applyUnaryElemwise(sparseToDense(v), realFn, complexFn, name);
  if (typeof v === "boolean") return realFn(v ? 1 : 0);
  if (typeof v === "number") return realFn(v);

  if (isRuntimeComplexNumber(v)) {
    const r = complexFn(v.re, v.im);
    return mkc(r.re, r.im);
  }

  if (isRuntimeTensor(v)) {
    const n = v.data.length;
    if (!v.imag) {
      // Tensor-ops fast path (native if available, TS fallback otherwise).
      const opCode = nativeUnaryOpCode.get(realFn);
      if (opCode !== undefined) {
        const out = allocFloat64Array(n);
        tensorOps.realUnaryElemwise(opCode, n, v.data, out);
        return RTV.tensorRaw(out, v.shape.slice());
      }
      const out = allocFloat64Array(n);
      for (let i = 0; i < n; i++) out[i] = realFn(v.data[i]);
      return makeTensor(out, undefined, v.shape.slice());
    }
    // Complex tensor path.
    const opCode = nativeUnaryOpCode.get(realFn);
    // tensorOps.complexUnaryElemwise supports all ops except ABS.
    if (opCode !== undefined && opCode !== OpUnary.ABS) {
      const outRe = allocFloat64Array(n);
      const outIm = allocFloat64Array(n);
      tensorOps.complexUnaryElemwise(opCode, n, v.data, v.imag, outRe, outIm);
      return makeTensor(outRe, outIm, v.shape.slice());
    }
    const outR = allocFloat64Array(n);
    const outI = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      const r = complexFn(v.data[i], v.imag[i]);
      outR[i] = r.re;
      outI[i] = r.im;
    }
    return makeTensor(outR, outI, v.shape.slice());
  }

  throw new Error(`${name}: unsupported argument type`);
}

/** Apply a unary element-wise function that may produce complex for out-of-domain real inputs.
 * When realFn returns NaN, complexFn is used instead (e.g., acos(2), asin(2), log(-1)).
 *
 * A NaN *input* is not out of domain: MATLAB's sqrt(NaN), log(NaN), asin(NaN)
 * are all a real NaN. Only a NaN produced from a non-NaN input means the real
 * branch has run out of domain, so NaN inputs keep the real result. */
function applyUnaryElemwiseMaybeComplex(
  v: RuntimeValue,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  name: string,
  nativeOpCode?: number
): RuntimeValue {
  if (isRuntimeSparseMatrix(v))
    return applyUnaryElemwiseMaybeComplex(
      sparseToDense(v),
      realFn,
      complexFn,
      name,
      nativeOpCode
    );
  if (typeof v === "boolean") v = (v ? 1 : 0) as unknown as RuntimeValue;
  if (typeof v === "number") {
    const r = realFn(v);
    if (!Number.isNaN(r) || Number.isNaN(v)) return r;
    const c = complexFn(v, 0);
    return mkc(c.re, c.im);
  }

  if (isRuntimeComplexNumber(v)) {
    const r = complexFn(v.re, v.im);
    return mkc(r.re, r.im);
  }

  if (isRuntimeTensor(v)) {
    const n = v.data.length;
    if (!v.imag) {
      // Tensor-ops fast path: compute the whole buffer via the ops layer,
      // then scan for NaN (indicates out-of-domain, patch with complex fallback).
      if (nativeOpCode !== undefined) {
        const nativeOut = allocFloat64Array(n);
        tensorOps.realUnaryElemwise(nativeOpCode, n, v.data, nativeOut);
        let firstNaN = -1;
        for (let i = 0; i < n; i++) {
          if (Number.isNaN(nativeOut[i]) && !Number.isNaN(v.data[i])) {
            firstNaN = i;
            break;
          }
        }
        if (firstNaN === -1) {
          return RTV.tensorRaw(nativeOut, v.shape.slice());
        }
        // Patch NaN positions with the complex fallback.
        const outR = allocFloat64Array(n);
        const outI = allocFloat64Array(n); // must zero-init: only NaN slots get im written
        let hasImag = false;
        for (let i = 0; i < n; i++) {
          if (!Number.isNaN(nativeOut[i]) || Number.isNaN(v.data[i])) {
            outR[i] = nativeOut[i];
          } else {
            const c = complexFn(v.data[i], 0);
            outR[i] = c.re;
            outI[i] = c.im;
            if (c.im !== 0) hasImag = true;
          }
        }
        return makeTensor(outR, hasImag ? outI : undefined, v.shape.slice());
      }
      const outR = allocFloat64Array(n);
      const outI = allocFloat64Array(n);
      let hasImag = false;
      for (let i = 0; i < n; i++) {
        const r = realFn(v.data[i]);
        if (!Number.isNaN(r) || Number.isNaN(v.data[i])) {
          outR[i] = r;
          outI[i] = 0;
        } else {
          const c = complexFn(v.data[i], 0);
          outR[i] = c.re;
          outI[i] = c.im;
          if (c.im !== 0) hasImag = true;
        }
      }
      return makeTensor(outR, hasImag ? outI : undefined, v.shape.slice());
    }
    const outR = allocFloat64Array(n);
    const outI = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      const r = complexFn(v.data[i], v.imag[i]);
      outR[i] = r.re;
      outI[i] = r.im;
    }
    return makeTensor(outR, outI, v.shape.slice());
  }

  throw new Error(`${name}: unsupported argument type`);
}

/** Apply a unary function that always returns real (e.g., abs) */
function applyUnaryRealResult(
  v: RuntimeValue,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => number,
  name: string
): RuntimeValue {
  if (isRuntimeSparseMatrix(v))
    return applyUnaryRealResult(sparseToDense(v), realFn, complexFn, name);
  if (typeof v === "boolean") return realFn(v ? 1 : 0);
  if (typeof v === "number") return realFn(v);

  if (isRuntimeComplexNumber(v)) {
    return complexFn(v.re, v.im);
  }

  if (isRuntimeTensor(v)) {
    const n = v.data.length;
    // Fast path for abs (realFn = Math.abs).
    if (realFn === Math.abs) {
      const out = allocFloat64Array(n);
      if (!v.imag) tensorOps.realUnaryElemwise(OpUnary.ABS, n, v.data, out);
      else tensorOps.complexAbs(n, v.data, v.imag, out);
      return RTV.tensorRaw(out, v.shape.slice());
    }
    const out = allocFloat64Array(n);
    if (!v.imag) {
      for (let i = 0; i < n; i++) out[i] = realFn(v.data[i]);
    } else {
      for (let i = 0; i < n; i++) out[i] = complexFn(v.data[i], v.imag[i]);
    }
    return makeTensor(out, undefined, v.shape.slice());
  }

  throw new Error(`${name}: unsupported argument type`);
}

/** Apply a binary element-wise function supporting scalars, tensors, and broadcast. */
export function applyBinaryElemwise(
  args: RuntimeValue[],
  fn: (a: number, b: number) => number,
  name: string
): RuntimeValue {
  const a0 = typeof args[0] === "boolean" ? (args[0] ? 1 : 0) : args[0];
  const a1 = typeof args[1] === "boolean" ? (args[1] ? 1 : 0) : args[1];
  const aIsNum = typeof a0 === "number";
  const bIsNum = typeof a1 === "number";
  const aIsTensor = isRuntimeTensor(a0);
  const bIsTensor = isRuntimeTensor(a1);
  if (aIsNum && bIsNum) return fn(a0 as number, a1 as number);
  if (aIsTensor && bIsNum) {
    const t = a0 as RuntimeTensor;
    const s = a1 as number;
    const out = allocFloat64Array(t.data.length);
    for (let i = 0; i < t.data.length; i++) out[i] = fn(t.data[i], s);
    return makeTensor(out, undefined, t.shape);
  }
  if (aIsNum && bIsTensor) {
    const s = a0 as number;
    const t = a1 as RuntimeTensor;
    const out = allocFloat64Array(t.data.length);
    for (let i = 0; i < t.data.length; i++) out[i] = fn(s, t.data[i]);
    return makeTensor(out, undefined, t.shape);
  }
  if (aIsTensor && bIsTensor) {
    const tA = a0 as RuntimeTensor;
    const tB = a1 as RuntimeTensor;
    const outShape = getBroadcastShape(tA.shape, tB.shape);
    if (outShape === null) {
      throw new Error(`${name}: Matrix dimensions must agree`);
    }
    const n = outShape.reduce((acc, d) => acc * d, 1);
    const out = allocFloat64Array(n);
    broadcastIterate(tA.shape, tB.shape, outShape, (ai, bi, oi) => {
      out[oi] = fn(tA.data[ai], tB.data[bi]);
    });
    return makeTensor(out, undefined, outShape);
  }
  throw new Error(`${name}: unsupported argument types`);
}

export interface UnaryElemwiseFns {
  realFn: (x: number) => number;
  complexFn: (re: number, im: number) => { re: number; im: number };
  /** Use applyUnaryElemwiseMaybeComplex instead of applyUnaryElemwise */
  maybeComplex?: boolean;
  /** OpUnary code for the tensor-ops dispatch. Used by the maybeComplex path
   * so that e.g. sqrt/log can run in native C on the common all-in-domain
   * case, then fall back for NaN entries only. */
  nativeOpCode?: number;
}

// ── Apply-only element-wise helpers (for builtin()) ─────────────────────
//
// Each returns an `apply` for `builtin({ nargin: 1, apply })`. Input goes
// through `numericArg`, so logical and char input are handled the way MATLAB
// handles them, sparse input is densified, and any other type raises
// MATLAB's invalid-data-type error.

type UnaryApply = (args: RuntimeValue[]) => RuntimeValue;

/** Unary element-wise math (sin, exp, sqrt, ...). With `maybeComplex`, a
 *  real input outside the real domain (realFn returns NaN) is recomputed
 *  with complexFn. */
export function unaryElemwise(
  spec: UnaryElemwiseFns,
  name: string
): UnaryApply {
  if (spec.maybeComplex)
    return args =>
      applyUnaryElemwiseMaybeComplex(
        numericArg(args[0]),
        spec.realFn,
        spec.complexFn,
        name,
        spec.nativeOpCode
      );
  return args =>
    applyUnaryElemwise(numericArg(args[0]), spec.realFn, spec.complexFn, name);
}

/** Unary element-wise function whose result is always real (abs, real,
 *  imag, angle). */
export function unaryRealResult(
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => number,
  name: string
): UnaryApply {
  return args =>
    applyUnaryRealResult(numericArg(args[0]), realFn, complexFn, name);
}

/** Element-wise numeric predicate (isnan, isinf, isfinite): a logical
 *  scalar for scalar input, else a logical tensor of the input's shape. */
export function unaryPredicate(
  realTest: (x: number) => boolean,
  complexTest: (re: number, im: number) => boolean
): UnaryApply {
  return args => {
    let v = numericArg(args[0]);
    if (isRuntimeSparseMatrix(v)) v = sparseToDense(v);
    if (typeof v === "number") return realTest(v);
    if (isRuntimeComplexNumber(v)) return complexTest(v.re, v.im);
    const n = v.data.length;
    const out = allocFloat64Array(n);
    if (!v.imag) {
      for (let i = 0; i < n; i++) out[i] = realTest(v.data[i]) ? 1 : 0;
    } else {
      for (let i = 0; i < n; i++)
        out[i] = complexTest(v.data[i], v.imag[i]) ? 1 : 0;
    }
    return makeTensor(out, undefined, v.shape.slice(), "logical");
  };
}
