/**
 * FFT and IFFT builtin functions
 *
 * fft(X)         - 1D FFT of vector; column-wise for matrices; along first
 *                  non-singleton dim for N-d arrays.
 * fft(X, n)      - n-point FFT (zero-pad or truncate along operating dim)
 * fft(X, n, dim) - FFT along dimension dim (1-based)
 * ifft(...)      - inverse FFT, same argument conventions
 */

import { getLapackBridge } from "../../native/lapack-bridge.js";
import { RTV, RuntimeValue, RuntimeError } from "../../runtime/index.js";
import { register, builtinSingle } from "../registry.js";
import {
  FloatXArray,
  type FloatXArrayType,
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";

// ── FFT algorithm ─────────────────────────────────────────────────────────

function isPowerOf2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/**
 * In-place Cooley-Tukey radix-2 iterative FFT (requires power-of-2 length).
 */
function fftInPlace(
  re: Float64Array,
  im: Float64Array,
  inverse: boolean
): void {
  const N = re.length;

  // Bit-reverse permutation
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

  // Butterfly stages
  for (let len = 2; len <= N; len <<= 1) {
    const angBase = ((inverse ? 2 : -2) * Math.PI) / len;
    const half = len >> 1;
    for (let i = 0; i < N; i += len) {
      for (let k = 0; k < half; k++) {
        const ang = angBase * k;
        const wRe = Math.cos(ang);
        const wIm = Math.sin(ang);
        const uRe = re[i + k];
        const uIm = im[i + k];
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

/** Next power of 2 >= n. */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Bluestein's FFT algorithm for arbitrary-length transforms.
 * Converts an N-point DFT into a circular convolution computed via
 * power-of-2 Cooley-Tukey FFTs, giving O(N log N) performance and
 * the same numerical accuracy as the radix-2 path.
 */
function bluesteinFFT(
  inRe: Float64Array,
  inIm: Float64Array,
  inverse: boolean
): [Float64Array, Float64Array] {
  const N = inRe.length;
  const sign = inverse ? 1 : -1;

  // Chirp sequence: w[k] = exp(sign * i * pi * k^2 / N)
  const chirpRe = new Float64Array(N);
  const chirpIm = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    const angle = (sign * Math.PI * ((k * k) % (2 * N))) / N;
    chirpRe[k] = Math.cos(angle);
    chirpIm[k] = Math.sin(angle);
  }

  // Padded length M (power of 2, >= 2N-1)
  const M = nextPow2(2 * N - 1);

  // Build sequence a[n] = x[n] * chirp[n]  (zero-padded to M)
  const aRe = new Float64Array(M);
  const aIm = new Float64Array(M);
  for (let n = 0; n < N; n++) {
    aRe[n] = inRe[n] * chirpRe[n] - inIm[n] * chirpIm[n];
    aIm[n] = inRe[n] * chirpIm[n] + inIm[n] * chirpRe[n];
  }

  // Build sequence b: conj(chirp) values wrapped circularly into M
  // b[k] = conj(chirp[k]) for k=0..N-1, b[M-k] = conj(chirp[k]) for k=1..N-1
  const bRe = new Float64Array(M);
  const bIm = new Float64Array(M);
  bRe[0] = chirpRe[0];
  bIm[0] = -chirpIm[0];
  for (let k = 1; k < N; k++) {
    bRe[k] = chirpRe[k];
    bIm[k] = -chirpIm[k];
    bRe[M - k] = chirpRe[k];
    bIm[M - k] = -chirpIm[k];
  }

  // Forward FFT of both (power-of-2, uses Cooley-Tukey)
  fftInPlace(aRe, aIm, false);
  fftInPlace(bRe, bIm, false);

  // Pointwise multiply A * B
  for (let i = 0; i < M; i++) {
    const re = aRe[i] * bRe[i] - aIm[i] * bIm[i];
    const im = aRe[i] * bIm[i] + aIm[i] * bRe[i];
    aRe[i] = re;
    aIm[i] = im;
  }

  // Inverse FFT (in-place, then normalize by M)
  fftInPlace(aRe, aIm, true);
  const invM = 1 / M;
  for (let i = 0; i < M; i++) {
    aRe[i] *= invM;
    aIm[i] *= invM;
  }

  // Extract result: X[k] = chirp[k] * conv[k]
  const outRe = new Float64Array(N);
  const outIm = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    outRe[k] = aRe[k] * chirpRe[k] - aIm[k] * chirpIm[k];
    outIm[k] = aRe[k] * chirpIm[k] + aIm[k] * chirpRe[k];
  }

  return [outRe, outIm];
}

/** Compute FFT of a 1D sequence (returns [re, im], not normalized). */
function computeFFT1D(
  inRe: Float64Array,
  inIm: Float64Array,
  inverse: boolean
): [Float64Array, Float64Array] {
  // Try native FFTW bridge
  const bridge = getLapackBridge();
  if (bridge?.fft1dComplex && inRe.length > 128) {
    // only use for large sizes due to overhead
    const result = bridge.fft1dComplex(inRe, inIm, inRe.length, inverse);
    return [result.re, result.im];
  }
  // Fallback to pure JS
  if (isPowerOf2(inRe.length)) {
    const re = new Float64Array(inRe);
    const im = new Float64Array(inIm);
    fftInPlace(re, im, inverse);
    return [re, im];
  }
  return bluesteinFFT(inRe, inIm, inverse);
}

/** Enforce exact conjugate symmetry on a 1D FFT result (for real input). */
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

/** Check if a 1D spectrum has conjugate symmetry (within tolerance). */
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

// ── Argument parsing ──────────────────────────────────────────────────────

/** Extract a scalar number argument, or undefined if absent/empty tensor. */
function parseScalarArg(v: RuntimeValue | undefined): number | undefined {
  if (v === undefined) return undefined;
  if (isRuntimeNumber(v)) return v;
  // Empty matrix [] → treat as "use default"
  if (isRuntimeTensor(v) && v.data.length === 0) return undefined;
  if (isRuntimeTensor(v) && v.data.length === 1) return v.data[0];
  throw new RuntimeError("fft: n and dim arguments must be scalars");
}

// ── Core N-d FFT logic ────────────────────────────────────────────────────

/**
 * Apply FFT/IFFT along a single dimension of an N-d tensor.
 *
 * Column-major layout: element at [i0, i1, ..., iK-1] is at flat index
 *   i0 + i1*s0 + i2*s0*s1 + ...
 * For dimension d with stride = s0*...*s_{d-1}:
 *   fiber base = (inner index, encoding dims 0..d-1) + outer * stride * shape[d]
 *   fiber elements at: base, base+stride, base+2*stride, ..., base+(shape[d]-1)*stride
 */
function applyFFTAlongDim(
  inData: Float32Array | Float64Array,
  inImag: Float32Array | Float64Array | undefined,
  shape: number[],
  dim: number, // 0-based dimension index
  n: number, // output length along dim
  inverse: boolean,
  inputIsReal: boolean
): {
  outData: FloatXArrayType;
  outImag: FloatXArrayType | undefined;
  outShape: number[];
} {
  const ndim = shape.length;
  const dimSize = shape[dim];

  // stride for dimension dim in input
  let strideDim = 1;
  for (let d = 0; d < dim; d++) strideDim *= shape[d];

  // Number of "above" slabs (product of dims > dim)
  let numAbove = 1;
  for (let d = dim + 1; d < ndim; d++) numAbove *= shape[d];

  const strideAboveIn = strideDim * dimSize;
  const strideAboveOut = strideDim * n;

  // Output shape: identical to input except shape[dim] → n
  const outShape = [...shape];
  outShape[dim] = n;
  const totalOut = strideAboveOut * numAbove;

  // ── Try native batch FFT via FFTW guru interface ──────────────────────
  const bridge = getLapackBridge();
  if (bridge?.fftAlongDim) {
    // Ensure Float64Array for native call
    const inRe64 =
      inData instanceof Float64Array ? inData : new Float64Array(inData);
    const inIm64: Float64Array | null = inImag
      ? inImag instanceof Float64Array
        ? inImag
        : new Float64Array(inImag)
      : null;

    const result = bridge.fftAlongDim(inRe64, inIm64, shape, dim, n, inverse);

    if (result) {
      const outRe = result.re;
      const outIm = result.im;

      // For forward FFT of real input, enforce exact conjugate symmetry
      // per fiber (FFTW is accurate but not bitwise exact)
      if (!inverse && inputIsReal) {
        for (let outer = 0; outer < numAbove; outer++) {
          for (let inner = 0; inner < strideDim; inner++) {
            const base = inner + outer * strideAboveOut;
            // DC component: imag = 0
            outIm[base] = 0;
            // Nyquist for even n: imag = 0
            if (n % 2 === 0) outIm[base + (n / 2) * strideDim] = 0;
            // Conjugate pairs
            for (let k = 1; k <= Math.floor((n - 1) / 2); k++) {
              const j = n - k;
              const idxK = base + k * strideDim;
              const idxJ = base + j * strideDim;
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

      // Normalize for inverse FFT
      if (inverse) {
        const inv = 1 / n;
        for (let i = 0; i < outRe.length; i++) {
          outRe[i] *= inv;
          outIm[i] *= inv;
        }
      }

      // Check if output has any significant imaginary component
      let anyImag = false;
      for (let i = 0; i < outIm.length; i++) {
        if (Math.abs(outIm[i]) > 1e-15) {
          anyImag = true;
          break;
        }
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

  // ── Fallback: per-fiber JS FFT ───────────────────────────────────────────
  const outData = new FloatXArray(totalOut);
  const outImag = new FloatXArray(totalOut);
  let anyImag = false;

  const fiberIn = new Float64Array(n); // padded/truncated input fiber (re)
  const fiberInI = new Float64Array(n); // padded/truncated input fiber (im)
  const copyLen = Math.min(dimSize, n);

  for (let outer = 0; outer < numAbove; outer++) {
    for (let inner = 0; inner < strideDim; inner++) {
      const baseIn = inner + outer * strideAboveIn;
      const baseOut = inner + outer * strideAboveOut;

      // Extract fiber (with zero-padding / truncation to length n)
      fiberIn.fill(0);
      fiberInI.fill(0);
      for (let i = 0; i < copyLen; i++) {
        fiberIn[i] = inData[baseIn + i * strideDim];
        if (inImag) fiberInI[i] = inImag[baseIn + i * strideDim];
      }

      const [outRe, outIm] = computeFFT1D(fiberIn, fiberInI, inverse);

      // For forward FFT of real input, enforce exact conjugate symmetry
      if (!inverse && inputIsReal) {
        enforceConjugateSymmetry(outRe, outIm);
      }

      // Normalize for inverse FFT
      if (inverse) {
        const inv = 1 / n;
        for (let i = 0; i < n; i++) {
          outRe[i] *= inv;
          outIm[i] *= inv;
        }
      }

      // For inverse FFT of conjugate-symmetric input, force real output
      if (inverse && isFiberConjugateSymmetric(fiberIn, fiberInI)) {
        for (let i = 0; i < n; i++) outIm[i] = 0;
      }

      // Store fiber
      for (let i = 0; i < n; i++) {
        outData[baseOut + i * strideDim] = outRe[i];
        outImag[baseOut + i * strideDim] = outIm[i];
        if (Math.abs(outIm[i]) > 1e-15) anyImag = true;
      }
    }
  }
  return { outData, outImag: anyImag ? outImag : undefined, outShape };
}

// ── Main dispatch ─────────────────────────────────────────────────────────

function applyFFT(args: RuntimeValue[], inverse: boolean): RuntimeValue {
  if (args.length < 1)
    throw new RuntimeError("fft: requires at least 1 argument");

  const x = args[0];
  const nArg = parseScalarArg(args[1]);
  const dimArg = parseScalarArg(args[2]);

  // Handle scalar input (treat as 1-element tensor)
  if (isRuntimeNumber(x) || isRuntimeComplexNumber(x)) {
    const xre = isRuntimeNumber(x) ? x : x.re;
    const xim = isRuntimeComplexNumber(x) ? x.im : 0;
    const n = nArg !== undefined ? Math.round(nArg) : 1;
    if (n <= 0) throw new RuntimeError("fft: n must be a positive integer");
    if (n === 1) {
      if (inverse) return xim === 0 ? RTV.num(xre) : RTV.complex(xre, xim);
      return xim === 0 ? RTV.num(xre) : RTV.complex(xre, xim);
    }
    // Zero-pad scalar to length n: result is a column vector
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    re[0] = xre;
    im[0] = xim;
    const [outRe, outIm] = computeFFT1D(re, im, inverse);
    // For forward FFT of real scalar, enforce conjugate symmetry
    if (!inverse && xim === 0) {
      enforceConjugateSymmetry(outRe, outIm);
    }
    if (inverse) {
      const inv = 1 / n;
      for (let i = 0; i < n; i++) {
        outRe[i] *= inv;
        outIm[i] *= inv;
      }
    }
    // For inverse FFT of conjugate-symmetric input, force real output
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

  if (!isRuntimeTensor(x)) {
    throw new RuntimeError("fft: input must be a numeric array");
  }

  const shape = x.shape;
  const ndim = shape.length;

  // Determine operating dimension (0-based)
  let dim: number;
  if (dimArg !== undefined) {
    dim = Math.round(dimArg) - 1; // convert 1-based to 0-based
    if (dim < 0) throw new RuntimeError("fft: dim out of range");
    // dim beyond ndims: size is implicitly 1, FFT is identity → return input
    if (dim >= ndim) return x;
  } else {
    // First dimension whose size > 1
    dim = shape.findIndex((s: number) => s > 1);
    if (dim === -1) dim = 0; // all singleton: use first
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

/**
 * Apply a circular shift by `shift` positions along dimension `dim` (0-based).
 * Returns new data/imag arrays with the same total size.
 */
function circshiftAlongDim(
  data: FloatXArrayType,
  imag: FloatXArrayType | undefined,
  shape: number[],
  dim: number,
  shift: number
): { outData: FloatXArrayType; outImag: FloatXArrayType | undefined } {
  const n = shape[dim];
  shift = ((shift % n) + n) % n; // normalize to [0, n)
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
        // output[i] = input[(i - shift + n) % n]
        const srcI = (i - shift + n) % n;
        outData[base + i * strideDim] = data[base + srcI * strideDim];
        if (imag && outImag) {
          outImag[base + i * strideDim] = imag[base + srcI * strideDim];
        }
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

  // Scalar input: no-op
  if (isRuntimeNumber(x) || isRuntimeComplexNumber(x)) return x;

  if (!isRuntimeTensor(x))
    throw new RuntimeError("fftshift: input must be numeric");

  const shape = x.shape;
  // Determine which dimensions to shift
  const dims: number[] =
    dimArg !== undefined
      ? [Math.round(dimArg) - 1] // 1-based → 0-based
      : shape.map((_: number, i: number) => i); // all dimensions

  // sign: fftshift = +floor(n/2); ifftshift = -floor(n/2)
  const sign = inverse ? -1 : 1;

  let curData: FloatXArrayType = x.data;
  let curImag: FloatXArrayType | undefined = x.imag;

  for (const dim of dims) {
    if (dim < 0) throw new RuntimeError("fftshift: dim out of range");
    // dim beyond ndims: size is implicitly 1, shift is 0 → no-op
    if (dim >= shape.length) continue;
    const shift = sign * Math.floor(shape[dim] / 2);
    const result = circshiftAlongDim(curData, curImag, shape, dim, shift);
    curData = result.outData;
    curImag = result.outImag;
  }

  return RTV.tensor(curData, [...shape], curImag);
}

// ── Registration ──────────────────────────────────────────────────────────

export function registerFft(): void {
  register(
    "fft",
    builtinSingle(args => applyFFT(args, false), {
      outputType: {
        kind: "Tensor",
        isComplex: true,
      },
    })
  );

  register(
    "ifft",
    builtinSingle(args => applyFFT(args, true), {
      outputType: {
        kind: "Tensor",
        isComplex: true,
      },
    })
  );

  register(
    "fftshift",
    builtinSingle(args => applyFFTShift(args, false), {
      outputType: {
        kind: "Tensor",
        isComplex: true,
      },
    })
  );

  register(
    "ifftshift",
    builtinSingle(args => applyFFTShift(args, true), {
      outputType: {
        kind: "Tensor",
        isComplex: true,
      },
    })
  );
}
