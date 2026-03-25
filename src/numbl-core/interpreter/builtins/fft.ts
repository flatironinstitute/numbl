/**
 * Interpreter IBuiltins for FFT functions:
 * fft, ifft, fftshift, ifftshift.
 *
 * Reuses the same algorithmic implementations as the legacy builtins.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import {
  FloatXArray,
  type FloatXArrayType,
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import type { JitType } from "../jit/jitTypes.js";
import { defineBuiltin } from "./types.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";

// ── Type helpers ──────────────────────────────────────────────────────────

function isNumericJitType(t: JitType): boolean {
  return (
    t.kind === "number" ||
    t.kind === "boolean" ||
    t.kind === "complex_or_number" ||
    t.kind === "tensor"
  );
}

// ── FFT algorithm ─────────────────────────────────────────────────────────

function isPowerOf2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

function fftInPlace(
  re: Float64Array,
  im: Float64Array,
  inverse: boolean
): void {
  const N = re.length;
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const angBase = ((inverse ? 2 : -2) * Math.PI) / len;
    const half = len >> 1;
    for (let i = 0; i < N; i += len) {
      for (let k = 0; k < half; k++) {
        const ang = angBase * k;
        const wRe = Math.cos(ang),
          wIm = Math.sin(ang);
        const uRe = re[i + k],
          uIm = im[i + k];
        const vRe = re[i + k + half] * wRe - im[i + k + half] * wIm;
        const vIm = re[i + k + half] * wIm + im[i + k + half] * wRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + half] = uRe - vRe;
        im[i + k + half] = uIm - vIm;
      }
    }
  }
}

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function bluesteinFFT(
  inRe: Float64Array,
  inIm: Float64Array,
  inverse: boolean
): [Float64Array, Float64Array] {
  const N = inRe.length;
  const sign = inverse ? 1 : -1;
  const chirpRe = new Float64Array(N),
    chirpIm = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    const angle = (sign * Math.PI * ((k * k) % (2 * N))) / N;
    chirpRe[k] = Math.cos(angle);
    chirpIm[k] = Math.sin(angle);
  }
  const M = nextPow2(2 * N - 1);
  const aRe = new Float64Array(M),
    aIm = new Float64Array(M);
  for (let n = 0; n < N; n++) {
    aRe[n] = inRe[n] * chirpRe[n] - inIm[n] * chirpIm[n];
    aIm[n] = inRe[n] * chirpIm[n] + inIm[n] * chirpRe[n];
  }
  const bRe = new Float64Array(M),
    bIm = new Float64Array(M);
  bRe[0] = chirpRe[0];
  bIm[0] = -chirpIm[0];
  for (let k = 1; k < N; k++) {
    bRe[k] = chirpRe[k];
    bIm[k] = -chirpIm[k];
    bRe[M - k] = chirpRe[k];
    bIm[M - k] = -chirpIm[k];
  }
  fftInPlace(aRe, aIm, false);
  fftInPlace(bRe, bIm, false);
  for (let i = 0; i < M; i++) {
    const re = aRe[i] * bRe[i] - aIm[i] * bIm[i];
    const im = aRe[i] * bIm[i] + aIm[i] * bRe[i];
    aRe[i] = re;
    aIm[i] = im;
  }
  fftInPlace(aRe, aIm, true);
  const invM = 1 / M;
  for (let i = 0; i < M; i++) {
    aRe[i] *= invM;
    aIm[i] *= invM;
  }
  const outRe = new Float64Array(N),
    outIm = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    outRe[k] = aRe[k] * chirpRe[k] - aIm[k] * chirpIm[k];
    outIm[k] = aRe[k] * chirpIm[k] + aIm[k] * chirpRe[k];
  }
  return [outRe, outIm];
}

function computeFFT1D(
  inRe: Float64Array,
  inIm: Float64Array,
  inverse: boolean
): [Float64Array, Float64Array] {
  const bridge = getLapackBridge();
  if (bridge?.fft1dComplex && inRe.length > 128) {
    const result = bridge.fft1dComplex(inRe, inIm, inRe.length, inverse);
    return [result.re, result.im];
  }
  if (isPowerOf2(inRe.length)) {
    const re = new Float64Array(inRe),
      im = new Float64Array(inIm);
    fftInPlace(re, im, inverse);
    return [re, im];
  }
  return bluesteinFFT(inRe, inIm, inverse);
}

function enforceConjugateSymmetry(re: Float64Array, im: Float64Array): void {
  const N = re.length;
  im[0] = 0;
  if (N % 2 === 0) im[N / 2] = 0;
  for (let k = 1; k <= Math.floor((N - 1) / 2); k++) {
    const j = N - k;
    const avgRe = (re[k] + re[j]) / 2;
    const avgIm = (im[k] - im[j]) / 2;
    re[k] = avgRe;
    re[j] = avgRe;
    im[k] = avgIm;
    im[j] = -avgIm;
  }
}

function isFiberConjugateSymmetric(
  re: Float64Array,
  im: Float64Array
): boolean {
  const N = re.length;
  if (N <= 1) return Math.abs(im[0]) < 1e-14;
  let maxMag = 0;
  for (let i = 0; i < N; i++) {
    const mag = Math.abs(re[i]) + Math.abs(im[i]);
    if (mag > maxMag) maxMag = mag;
  }
  const tol = Math.max(1e-14, maxMag * 1e-12);
  if (Math.abs(im[0]) > tol) return false;
  if (N % 2 === 0 && Math.abs(im[N / 2]) > tol) return false;
  for (let k = 1; k <= Math.floor((N - 1) / 2); k++) {
    const j = N - k;
    if (Math.abs(re[k] - re[j]) > tol) return false;
    if (Math.abs(im[k] + im[j]) > tol) return false;
  }
  return true;
}

function parseScalarArg(v: RuntimeValue | undefined): number | undefined {
  if (v === undefined) return undefined;
  if (isRuntimeNumber(v)) return v;
  if (isRuntimeTensor(v) && v.data.length === 0) return undefined;
  if (isRuntimeTensor(v) && v.data.length === 1) return v.data[0];
  throw new RuntimeError("fft: n and dim arguments must be scalars");
}

function applyFFTAlongDim(
  inData: Float32Array | Float64Array,
  inImag: Float32Array | Float64Array | undefined,
  shape: number[],
  dim: number,
  n: number,
  inverse: boolean,
  inputIsReal: boolean
): {
  outData: FloatXArrayType;
  outImag: FloatXArrayType | undefined;
  outShape: number[];
} {
  const ndim = shape.length;
  const dimSize = shape[dim];
  let strideDim = 1;
  for (let d = 0; d < dim; d++) strideDim *= shape[d];
  let numAbove = 1;
  for (let d = dim + 1; d < ndim; d++) numAbove *= shape[d];
  const strideAboveIn = strideDim * dimSize;
  const strideAboveOut = strideDim * n;
  const outShape = [...shape];
  outShape[dim] = n;
  const totalOut = strideAboveOut * numAbove;

  // Try native batch FFT
  const bridge = getLapackBridge();
  if (bridge?.fftAlongDim) {
    const inRe64 =
      inData instanceof Float64Array ? inData : new Float64Array(inData);
    const inIm64: Float64Array | null = inImag
      ? inImag instanceof Float64Array
        ? inImag
        : new Float64Array(inImag)
      : null;
    const result = bridge.fftAlongDim(inRe64, inIm64, shape, dim, n, inverse);
    if (result) {
      const outRe = result.re,
        outIm = result.im;
      if (!inverse && inputIsReal) {
        for (let outer = 0; outer < numAbove; outer++) {
          for (let inner = 0; inner < strideDim; inner++) {
            const base = inner + outer * strideAboveOut;
            outIm[base] = 0;
            if (n % 2 === 0) outIm[base + (n / 2) * strideDim] = 0;
            for (let k = 1; k <= Math.floor((n - 1) / 2); k++) {
              const j = n - k;
              const idxK = base + k * strideDim,
                idxJ = base + j * strideDim;
              const avgRe = (outRe[idxK] + outRe[idxJ]) / 2;
              const avgIm = (outIm[idxK] - outIm[idxJ]) / 2;
              outRe[idxK] = avgRe;
              outRe[idxJ] = avgRe;
              outIm[idxK] = avgIm;
              outIm[idxJ] = -avgIm;
            }
          }
        }
      }
      if (inverse) {
        const inv = 1 / n;
        for (let i = 0; i < outRe.length; i++) {
          outRe[i] *= inv;
          outIm[i] *= inv;
        }
      }
      let anyImag = false;
      for (let i = 0; i < outIm.length; i++)
        if (Math.abs(outIm[i]) > 1e-15) {
          anyImag = true;
          break;
        }
      return {
        outData: new FloatXArray(outRe) as FloatXArrayType,
        outImag: anyImag
          ? (new FloatXArray(outIm) as FloatXArrayType)
          : undefined,
        outShape,
      };
    }
  }

  // JS fallback
  const outData = new FloatXArray(totalOut);
  const outImag = new FloatXArray(totalOut);
  let anyImag = false;
  const fiberIn = new Float64Array(n),
    fiberInI = new Float64Array(n);
  const copyLen = Math.min(dimSize, n);
  for (let outer = 0; outer < numAbove; outer++) {
    for (let inner = 0; inner < strideDim; inner++) {
      const baseIn = inner + outer * strideAboveIn;
      const baseOut = inner + outer * strideAboveOut;
      fiberIn.fill(0);
      fiberInI.fill(0);
      for (let i = 0; i < copyLen; i++) {
        fiberIn[i] = inData[baseIn + i * strideDim];
        if (inImag) fiberInI[i] = inImag[baseIn + i * strideDim];
      }
      const [outRe, outIm] = computeFFT1D(fiberIn, fiberInI, inverse);
      if (!inverse && inputIsReal) enforceConjugateSymmetry(outRe, outIm);
      if (inverse) {
        const inv = 1 / n;
        for (let i = 0; i < n; i++) {
          outRe[i] *= inv;
          outIm[i] *= inv;
        }
      }
      if (inverse && isFiberConjugateSymmetric(fiberIn, fiberInI)) {
        for (let i = 0; i < n; i++) outIm[i] = 0;
      }
      for (let i = 0; i < n; i++) {
        outData[baseOut + i * strideDim] = outRe[i];
        outImag[baseOut + i * strideDim] = outIm[i];
        if (Math.abs(outIm[i]) > 1e-15) anyImag = true;
      }
    }
  }
  return { outData, outImag: anyImag ? outImag : undefined, outShape };
}

function applyFFT(args: RuntimeValue[], inverse: boolean): RuntimeValue {
  if (args.length < 1)
    throw new RuntimeError("fft: requires at least 1 argument");
  const x = args[0];
  const nArg = parseScalarArg(args[1]);
  const dimArg = parseScalarArg(args[2]);

  if (isRuntimeNumber(x) || isRuntimeComplexNumber(x)) {
    const xre = isRuntimeNumber(x) ? x : x.re;
    const xim = isRuntimeComplexNumber(x) ? x.im : 0;
    const n = nArg !== undefined ? Math.round(nArg) : 1;
    if (n <= 0) throw new RuntimeError("fft: n must be a positive integer");
    if (n === 1) return xim === 0 ? RTV.num(xre) : RTV.complex(xre, xim);
    const re = new Float64Array(n),
      im = new Float64Array(n);
    re[0] = xre;
    im[0] = xim;
    const [outRe, outIm] = computeFFT1D(re, im, inverse);
    if (!inverse && xim === 0) enforceConjugateSymmetry(outRe, outIm);
    if (inverse) {
      const inv = 1 / n;
      for (let i = 0; i < n; i++) {
        outRe[i] *= inv;
        outIm[i] *= inv;
      }
    }
    if (inverse && isFiberConjugateSymmetric(re, im)) {
      for (let i = 0; i < n; i++) outIm[i] = 0;
    }
    const allReal = outIm.every(v => Math.abs(v) < 1e-15);
    const outD = new FloatXArray(n);
    const outI = allReal ? undefined : new FloatXArray(n);
    for (let i = 0; i < n; i++) {
      outD[i] = outRe[i];
      if (outI) outI[i] = outIm[i];
    }
    return RTV.tensor(outD, [n, 1], outI);
  }

  if (!isRuntimeTensor(x))
    throw new RuntimeError("fft: input must be a numeric array");
  const shape = x.shape;
  const ndim = shape.length;
  let dim: number;
  if (dimArg !== undefined) {
    dim = Math.round(dimArg) - 1;
    if (dim < 0) throw new RuntimeError("fft: dim out of range");
    if (dim >= ndim) return x;
  } else {
    dim = shape.findIndex((s: number) => s > 1);
    if (dim === -1) dim = 0;
  }
  const dimSize = shape[dim];
  const n = nArg !== undefined ? Math.round(nArg) : dimSize;
  if (n <= 0) throw new RuntimeError("fft: n must be a positive integer");
  const inputIsReal = x.imag === undefined;
  const { outData, outImag, outShape } = applyFFTAlongDim(
    x.data,
    x.imag,
    shape,
    dim,
    n,
    inverse,
    inputIsReal
  );
  return RTV.tensor(outData, outShape, outImag);
}

// ── fftshift / ifftshift ──────────────────────────────────────────────────

function circshiftAlongDim(
  data: FloatXArrayType,
  imag: FloatXArrayType | undefined,
  shape: number[],
  dim: number,
  shift: number
): { outData: FloatXArrayType; outImag: FloatXArrayType | undefined } {
  const n = shape[dim];
  shift = ((shift % n) + n) % n;
  if (shift === 0) {
    return {
      outData: new FloatXArray(data) as FloatXArrayType,
      outImag: imag ? (new FloatXArray(imag) as FloatXArrayType) : undefined,
    };
  }
  const total = data.length;
  const outData = new FloatXArray(total) as FloatXArrayType;
  const outImag = imag
    ? (new FloatXArray(total) as FloatXArrayType)
    : undefined;
  let strideDim = 1;
  for (let d = 0; d < dim; d++) strideDim *= shape[d];
  let numAbove = 1;
  for (let d = dim + 1; d < shape.length; d++) numAbove *= shape[d];
  const strideAbove = strideDim * n;
  for (let outer = 0; outer < numAbove; outer++) {
    for (let inner = 0; inner < strideDim; inner++) {
      const base = inner + outer * strideAbove;
      for (let i = 0; i < n; i++) {
        const srcI = (i - shift + n) % n;
        outData[base + i * strideDim] = data[base + srcI * strideDim];
        if (imag && outImag)
          outImag[base + i * strideDim] = imag[base + srcI * strideDim];
      }
    }
  }
  return { outData, outImag };
}

function applyFFTShift(args: RuntimeValue[], inverse: boolean): RuntimeValue {
  if (args.length < 1)
    throw new RuntimeError("fftshift: requires at least 1 argument");
  const x = args[0];
  const dimArg = parseScalarArg(args[1]);
  if (isRuntimeNumber(x) || isRuntimeComplexNumber(x)) return x;
  if (!isRuntimeTensor(x))
    throw new RuntimeError("fftshift: input must be numeric");
  const shape = x.shape;
  const dims: number[] =
    dimArg !== undefined
      ? [Math.round(dimArg) - 1]
      : shape.map((_: number, i: number) => i);
  const sign = inverse ? -1 : 1;
  let curData: FloatXArrayType = x.data;
  let curImag: FloatXArrayType | undefined = x.imag;
  for (const dim of dims) {
    if (dim < 0) throw new RuntimeError("fftshift: dim out of range");
    if (dim >= shape.length) continue;
    const shift = sign * Math.floor(shape[dim] / 2);
    const result = circshiftAlongDim(curData, curImag, shape, dim, shift);
    curData = result.outData;
    curImag = result.outImag;
  }
  return RTV.tensor(curData, [...shape], curImag);
}

// ── Registration ──────────────────────────────────────────────────────────

defineBuiltin({
  name: "fft",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 3) return null;
        if (!isNumericJitType(argTypes[0])) return null;
        return [{ kind: "tensor", isComplex: true }];
      },
      apply: args => applyFFT(args, false),
    },
  ],
});

defineBuiltin({
  name: "ifft",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 3) return null;
        if (!isNumericJitType(argTypes[0])) return null;
        return [{ kind: "tensor", isComplex: true }];
      },
      apply: args => applyFFT(args, true),
    },
  ],
});

defineBuiltin({
  name: "fftshift",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        if (!isNumericJitType(argTypes[0])) return null;
        const a = argTypes[0];
        if (a.kind === "tensor")
          return [{ kind: "tensor", isComplex: a.isComplex }];
        if (a.kind === "number" || a.kind === "boolean")
          return [{ kind: "number" }];
        if (a.kind === "complex_or_number")
          return [{ kind: "complex_or_number" }];
        return null;
      },
      apply: args => applyFFTShift(args, false),
    },
  ],
});

defineBuiltin({
  name: "ifftshift",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        if (!isNumericJitType(argTypes[0])) return null;
        const a = argTypes[0];
        if (a.kind === "tensor")
          return [{ kind: "tensor", isComplex: a.isComplex }];
        if (a.kind === "number" || a.kind === "boolean")
          return [{ kind: "number" }];
        if (a.kind === "complex_or_number")
          return [{ kind: "complex_or_number" }];
        return null;
      },
      apply: args => applyFFTShift(args, true),
    },
  ],
});
