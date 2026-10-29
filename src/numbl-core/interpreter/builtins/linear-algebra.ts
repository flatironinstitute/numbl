/**
 * Linear algebra builtins: norm, vecnorm, dot, det, trace, cross, inv,
 * expm, svd, null, bandwidth, istriu, istril, isbanded, qr, qrupdate, lu,
 * eig, chol, linsolve, cond, rcond, rank, pinv, kron, blkdiag, pagemtimes,
 * pagetranspose, qz.
 */

import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeStructArray,
} from "../../runtime/types.js";
import {
  RTV,
  RuntimeError,
  tensorSize2D,
  colMajorIndex,
} from "../../runtime/index.js";
import { builtin } from "./builtin.js";
import { boolToNumber, builtinError, realScalarArg } from "./args.js";
import {
  errorFn,
  scalarTensor,
  floatMatrixArg,
  denseFloatMatrixArg,
  numericMatrixArg,
  denseNumericMatrixArg,
  realScalarOr,
  dimArgOr,
  outputs,
} from "./linalg-args.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import {
  toF64,
  parseStringArgLower,
  buildDiagMatrix,
  buildEigenvectorMatrix,
  maybeComplexTensor,
} from "../../helpers/check-helpers.js";
import { forEachSlice, copyTensor } from "../../helpers/reduction-helpers.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import { isRuntimeSparseMatrix } from "../../runtime/types.js";
import {
  linsolveLapack,
  linsolveComplexLapack,
} from "../../helpers/linsolve.js";
import { mAdd, mSub, mMul, mLeftDiv } from "../../helpers/arithmetic.js";
import { allocFloat64Array, withScratch } from "../../runtime/alloc.js";
import type { RuntimeSparseMatrix } from "../../runtime/types.js";
import { sparseTranspose } from "../../helpers/sparse-arithmetic.js";
import { emitWarning } from "../../helpers/warn.js";
import {
  sparseKron,
  sparseChol,
  sparseLu,
  sparseIsUpper,
  sparseIsLower,
  sparseInverse,
  sparsePermute,
  permutationMatrix,
  permVectorTensor,
  invertPerm,
  identityPerm,
  minimumDegreeOrdering,
  columnOrdering,
  denseToSparse,
} from "../../helpers/sparse-direct.js";

function speyeSparse(n: number): RuntimeSparseMatrix {
  const ir = new Int32Array(n);
  const jc = new Int32Array(n + 1);
  const pr = allocFloat64Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    ir[i] = i;
    jc[i + 1] = i + 1;
  }
  return RTV.sparseMatrix(n, n, ir, jc, pr);
}

// ── MATLAB errors ─────────────────────────────────────────────────────────

const squareError = errorFn("MATLAB:square", "Matrix must be square.");
const mustBeFloatError = errorFn(
  "MATLAB:math:mustBeFloat",
  "Invalid data type. Argument must be double or single."
);
const svdTypeError = errorFn(
  "MATLAB:svd:inputType",
  "First input must be single or double."
);
const svdSparseError = errorFn(
  "MATLAB:svd:noSVDsparse",
  "SVD does not support sparse matrices. Use SVDS to compute a subset of the singular values and vectors of a sparse matrix."
);
const floatOrLogicalError = errorFn(
  "MATLAB:math:mustBeFloatOrLogical",
  "Invalid data type. Argument must be double, single, or logical."
);

/** True for a real numeric scalar equal to 0 (the legacy economy flag of
 *  qr and svd). */
function isRealZero(v: RuntimeValue): boolean {
  if (isText(v)) return false;
  try {
    return realScalarArg(v) === 0;
  } catch {
    return false;
  }
}

/** True for a string scalar or char value. */
function isText(v: RuntimeValue): boolean {
  return isRuntimeString(v) || isRuntimeChar(v);
}

// ── norm ──────────────────────────────────────────────────────────────────

const normTypeError = errorFn(
  "MATLAB:norm:inputType",
  "First argument must be single or double."
);
const unknownNormError = errorFn(
  "MATLAB:norm:unknownNorm",
  "When the first input is a matrix, the second input must be 1, 2, inf, or 'fro'.  When the first input is a vector, the second input must be a positive real-valued scalar, inf, -inf, or 'fro'."
);

builtin({
  name: "norm",
  nargin: [1, 2],
  // normImplTensor may allocate scratch (e.g. the SVD for the 2-norm); the
  // result is a scalar, so release it on return.
  apply: args => withScratch(() => normApply(args)),
});

function normApply(args: RuntimeValue[]): RuntimeValue {
  const v = boolToNumber(args[0]);
  if (isRuntimeNumber(v)) return RTV.num(Math.abs(v));
  if (isRuntimeComplexNumber(v)) return RTV.num(Math.hypot(v.re, v.im));
  const A = floatMatrixArg(v, normTypeError);
  if (isRuntimeSparseMatrix(A)) {
    const fast = sparseNormFast(A, args[1]);
    if (fast !== null) return RTV.num(fast);
    return normImplTensor(sparseToDense(A), args);
  }
  return normImplTensor(A, args);
}

/** O(nnz) matrix norms for sparse input (1, Inf, 'fro'); null otherwise. */
function sparseNormFast(
  S: RuntimeSparseMatrix,
  p: RuntimeValue | undefined
): number | null {
  if (S.m === 1 || S.n === 1) return null; // vector norms: dense path
  let kind: "one" | "inf" | "fro" | null = null;
  if (p === undefined) return null;
  if (isRuntimeNumber(p)) {
    if (p === 1) kind = "one";
    else if (p === Infinity) kind = "inf";
  } else if (isRuntimeChar(p) || isRuntimeString(p)) {
    const s = parseStringArgLower(p);
    if (s === "fro") kind = "fro";
    else if (s === "inf") kind = "inf";
  }
  if (kind === null) return null;
  const mag = (k: number) =>
    S.pi ? Math.hypot(S.pr[k], S.pi[k]) : Math.abs(S.pr[k]);
  const nnz = S.jc[S.n];
  if (kind === "fro") {
    let sum = 0;
    for (let k = 0; k < nnz; k++) {
      const a = mag(k);
      sum += a * a;
    }
    return Math.sqrt(sum);
  }
  if (kind === "one") {
    let best = 0;
    for (let j = 0; j < S.n; j++) {
      let sum = 0;
      for (let k = S.jc[j]; k < S.jc[j + 1]; k++) sum += mag(k);
      if (sum > best) best = sum;
    }
    return best;
  }
  const rowSums = allocFloat64Array(S.m);
  for (let k = 0; k < nnz; k++) rowSums[S.ir[k]] += mag(k);
  let best = 0;
  for (let i = 0; i < S.m; i++) if (rowSums[i] > best) best = rowSums[i];
  return best;
}

// LAPACK dlassq-style Euclidean norm: finite for any inputs in double range.
function scaledEuclid(magnitudes: ArrayLike<number>, count: number): number {
  let scale = 0;
  let ssq = 1;
  for (let i = 0; i < count; i++) {
    const ax = Math.abs(magnitudes[i]);
    if (ax === 0) continue;
    if (!isFinite(ax)) return isNaN(ax) ? NaN : Infinity;
    if (scale < ax) {
      const r = scale / ax;
      ssq = 1 + ssq * r * r;
      scale = ax;
    } else {
      const r = ax / scale;
      ssq += r * r;
    }
  }
  return scale === 0 ? 0 : scale * Math.sqrt(ssq);
}

function elemMag(
  re: ArrayLike<number>,
  im: ArrayLike<number> | undefined,
  i: number
): number {
  return im ? Math.hypot(re[i], im[i]) : Math.abs(re[i]);
}

function normImplTensor(v: RuntimeTensor, args: RuntimeValue[]): RuntimeValue {
  const shape = v.shape;
  const rows = shape[0] || 1;
  const cols = shape.length >= 2 ? shape[1] : 1;
  const isVec =
    shape.length <= 2
      ? rows === 1 || cols === 1
      : v.data.length === Math.max(...shape);

  let p: number | "fro" = 2;
  if (args.length >= 2) {
    const arg1 = args[1];
    if (isText(arg1)) {
      const s = parseStringArgLower(arg1);
      if (s === "inf") p = Infinity;
      else if (s === "fro") p = "fro";
      else throw unknownNormError();
    } else {
      p = realScalarOr(arg1, unknownNormError);
    }
  }

  const imag = v.imag;
  if (isVec) {
    const vp = p === "fro" ? 2 : p;
    if (vp === Infinity) {
      let m = 0;
      for (let i = 0; i < v.data.length; i++) {
        const a = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
        m = Math.max(m, a);
      }
      return RTV.num(m);
    }
    if (vp === -Infinity) {
      let m = Infinity;
      for (let i = 0; i < v.data.length; i++) {
        const a = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
        m = Math.min(m, a);
      }
      return RTV.num(m);
    }
    if (vp === 2) {
      const mags = allocFloat64Array(v.data.length);
      for (let i = 0; i < v.data.length; i++)
        mags[i] = elemMag(v.data, imag, i);
      return RTV.num(scaledEuclid(mags, mags.length));
    }
    let s = 0;
    for (let i = 0; i < v.data.length; i++) {
      const a = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
      s += Math.pow(a, vp);
    }
    return RTV.num(Math.pow(s, 1 / vp));
  }

  // Matrix norms
  if (p === "fro" || (typeof p === "number" && isNaN(p))) {
    const mags = allocFloat64Array(v.data.length);
    for (let i = 0; i < v.data.length; i++) mags[i] = elemMag(v.data, imag, i);
    return RTV.num(scaledEuclid(mags, mags.length));
  }
  if (p === 1) {
    let maxColSum = 0;
    for (let j = 0; j < cols; j++) {
      let colSum = 0;
      for (let i = 0; i < rows; i++) {
        const idx = j * rows + i;
        colSum += imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
      }
      maxColSum = Math.max(maxColSum, colSum);
    }
    return RTV.num(maxColSum);
  }
  if (p === Infinity) {
    let maxRowSum = 0;
    for (let i = 0; i < rows; i++) {
      let rowSum = 0;
      for (let j = 0; j < cols; j++) {
        const idx = j * rows + i;
        rowSum += imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
      }
      maxRowSum = Math.max(maxRowSum, rowSum);
    }
    return RTV.num(maxRowSum);
  }
  if (p === 2) {
    // Match MATLAB: a matrix with any NaN has 2-norm NaN; otherwise any Inf
    // gives 2-norm Inf. (svd of non-finite input returns NaN, so handle here.)
    let anyNaN = false;
    let anyInf = false;
    for (let i = 0; i < v.data.length; i++) {
      const re = v.data[i];
      const im = imag ? imag[i] : 0;
      if (Number.isNaN(re) || Number.isNaN(im)) anyNaN = true;
      else if (!isFinite(re) || !isFinite(im)) anyInf = true;
    }
    if (anyNaN) return RTV.num(NaN);
    if (anyInf) return RTV.num(Infinity);

    // The matrix 2-norm is the largest singular value. For complex A = X + iY,
    // use the real embedding R = [X -Y; Y X] (2m×2n): its singular values are
    // those of A (each doubled), so max(svd(R)) = ‖A‖₂. This lets the complex
    // 2-norm use the real SVD (which has no TypeScript gap); using only the
    // real part would be wrong.
    const bridge = getLapackBridge();
    if (imag) {
      const R = allocFloat64Array(4 * rows * cols);
      const tm = 2 * rows;
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
          const a = v.data[j * rows + i];
          const b = imag[j * rows + i];
          R[j * tm + i] = a; // top-left   X
          R[j * tm + (rows + i)] = b; // bottom-left  Y
          R[(cols + j) * tm + i] = -b; // top-right   -Y
          R[(cols + j) * tm + (rows + i)] = a; // bottom-right X
        }
      }
      const result = bridge.svd(R, tm, 2 * cols, false, false);
      return RTV.num(result.S[0]);
    }
    const result = bridge.svd(v.data, rows, cols, false, false);
    return RTV.num(result.S[0]);
  }
  throw unknownNormError();
}

// ── vecnorm ──────────────────────────────────────────────────────────────

const vecnormTypeError = errorFn(
  "MATLAB:vecnorm:inputType",
  "Invalid data type. First argument must be single or double."
);

builtin({
  name: "vecnorm",
  nargin: [1, 3],
  apply: args => {
    const v = boolToNumber(args[0]);
    let p = 2;
    if (args.length >= 2)
      p = realScalarOr(
        args[1],
        errorFn(
          "MATLAB:vecnorm:unknownNorm",
          "Second argument must be a positive real value or Inf."
        )
      );
    const dimArg =
      args.length >= 3
        ? dimArgOr(
            args[2],
            errorFn(
              "MATLAB:getdimarg:dimensionMustBePositiveInteger",
              "Dimension argument must be a positive integer scalar within indexing range."
            )
          )
        : undefined;
    if (isRuntimeNumber(v)) return RTV.num(Math.abs(v));
    if (isRuntimeComplexNumber(v)) return RTV.num(Math.hypot(v.re, v.im));
    const A = floatMatrixArg(v, vecnormTypeError);
    // Sparse input gives a sparse result, as in MATLAB.
    const T = isRuntimeSparseMatrix(A) ? sparseToDense(A) : A;
    let dim: number;
    if (dimArg !== undefined) dim = dimArg;
    else {
      const idx = T.shape.findIndex((d: number) => d > 1);
      dim = idx === -1 ? 1 : idx + 1;
    }
    const r = vecnormAlongDim(T, p, dim);
    return isRuntimeSparseMatrix(A) && isRuntimeTensor(r)
      ? denseToSparse(r)
      : r;
  },
});

function vecnormAlongDim(
  v: RuntimeTensor,
  p: number,
  dim: number
): RuntimeValue {
  const dimIdx = dim - 1;
  if (dimIdx >= v.shape.length) {
    const result = allocFloat64Array(v.data.length);
    const imag = v.imag;
    for (let i = 0; i < v.data.length; i++) {
      result[i] = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
    }
    return RTV.tensor(result, [...v.shape]);
  }
  const info = forEachSlice(v.shape, dim, () => {});
  if (!info) return copyTensor(v);
  const result = allocFloat64Array(info.totalElems);
  const imag = v.imag;
  forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
    if (p === Infinity) {
      let m = 0;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const a = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        if (a > m) m = a;
      }
      result[outIdx] = m;
    } else if (p === -Infinity) {
      let m = Infinity;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const a = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        if (a < m) m = a;
      }
      result[outIdx] = m;
    } else if (p === 2) {
      let scale = 0;
      let ssq = 1;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const ax = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        if (ax === 0) continue;
        if (!isFinite(ax)) {
          scale = isNaN(ax) ? NaN : Infinity;
          ssq = 1;
          break;
        }
        if (scale < ax) {
          const r = scale / ax;
          ssq = 1 + ssq * r * r;
          scale = ax;
        } else {
          const r = ax / scale;
          ssq += r * r;
        }
      }
      result[outIdx] = scale === 0 ? 0 : scale * Math.sqrt(ssq);
    } else {
      let s = 0;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const a = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        s += Math.pow(a, p);
      }
      result[outIdx] = Math.pow(s, 1 / p);
    }
  });
  return RTV.tensor(result, info.resultShape);
}

// ── dot ──────────────────────────────────────────────────────────────────

builtin({
  name: "dot",
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    const A = denseNumericMatrixArg(args[0]);
    const B = denseNumericMatrixArg(args[1]);
    const r =
      args.length === 3
        ? dotAlongDim(
            A,
            B,
            dimArgOr(
              args[2],
              errorFn(
                "MATLAB:getdimarg:invalidDim",
                "Dimension argument must be a positive integer scalar, a vector of unique positive integers, or 'all'."
              )
            )
          )
        : dotDefault(A, B);
    // dot of two sparse arrays is sparse, as in MATLAB.
    if (isRuntimeSparseMatrix(args[0]) && isRuntimeSparseMatrix(args[1]))
      return denseToSparse(
        isRuntimeTensor(r)
          ? r
          : isRuntimeComplexNumber(r)
            ? scalarTensor(r.re, r.im)
            : scalarTensor(r as number)
      );
    return r;
  },
});

/** dot(A, B, dim): sum(conj(A) .* B, dim). */
function dotAlongDim(
  A: RuntimeTensor,
  B: RuntimeTensor,
  dim: number
): RuntimeValue {
  if (A.data.length !== B.data.length)
    throw new RuntimeError("dot: A and B must be the same size");
  const aRe = A.data,
    aIm = A.imag ?? null,
    bRe = B.data,
    bIm = B.imag ?? null;
  const hasComplex = aIm !== null || bIm !== null;
  const n = aRe.length;

  // dim beyond the tensor rank: the reduction is over a singleton
  // dimension, so the result is the element-wise product conj(A).*B.
  const collapse = forEachSlice(A.shape, dim, () => {});
  if (collapse === null) {
    const outRe = allocFloat64Array(n);
    const outIm = hasComplex ? allocFloat64Array(n) : null;
    for (let i = 0; i < n; i++) {
      const aRei = aRe[i],
        aImi = aIm ? aIm[i] : 0;
      const bRei = bRe[i],
        bImi = bIm ? bIm[i] : 0;
      outRe[i] = aRei * bRei + aImi * bImi;
      if (outIm) outIm[i] = aRei * bImi - aImi * bRei;
    }
    return RTV.tensor(outRe, [...A.shape], outIm ?? undefined);
  }

  const outRe = allocFloat64Array(collapse.totalElems);
  const outIm = hasComplex ? allocFloat64Array(collapse.totalElems) : null;
  forEachSlice(A.shape, dim, (outIdx, srcIndices) => {
    let sRe = 0,
      sIm = 0;
    for (let k = 0; k < srcIndices.length; k++) {
      const idx = srcIndices[k];
      const aRei = aRe[idx],
        aImi = aIm ? aIm[idx] : 0;
      const bRei = bRe[idx],
        bImi = bIm ? bIm[idx] : 0;
      sRe += aRei * bRei + aImi * bImi;
      sIm += aRei * bImi - aImi * bRei;
    }
    outRe[outIdx] = sRe;
    if (outIm) outIm[outIdx] = sIm;
  });
  return RTV.tensor(outRe, collapse.resultShape, outIm ?? undefined);
}

/** dot(A, B): along the first dimension for matrices, else over all
 *  elements. */
function dotDefault(A: RuntimeTensor, B: RuntimeTensor): RuntimeValue {
  const aRe = A.data,
    aIm = A.imag ?? null,
    bRe = B.data,
    bIm = B.imag ?? null;
  if (aRe.length !== bRe.length)
    throw new RuntimeError("dot: vectors must be same length");

  const hasComplex = aIm !== null || bIm !== null;
  const rows = A.shape[0];
  const cols = A.shape.length >= 2 ? A.shape[1] : 1;
  const isMatrix = rows > 1 && cols > 1;

  if (isMatrix) {
    const resultRe = allocFloat64Array(cols);
    const resultIm = hasComplex ? allocFloat64Array(cols) : null;
    for (let c = 0; c < cols; c++) {
      let sRe = 0,
        sIm = 0;
      for (let r = 0; r < rows; r++) {
        const idx = c * rows + r;
        const aRei = aRe[idx],
          aImi = aIm ? aIm[idx] : 0;
        const bRei = bRe[idx],
          bImi = bIm ? bIm[idx] : 0;
        sRe += aRei * bRei + aImi * bImi;
        sIm += aRei * bImi - aImi * bRei;
      }
      resultRe[c] = sRe;
      if (resultIm) resultIm[c] = sIm;
    }
    return RTV.tensor(resultRe, [1, cols], resultIm ?? undefined);
  }

  if (!hasComplex) {
    let s = 0;
    for (let i = 0; i < aRe.length; i++) s += aRe[i] * bRe[i];
    return RTV.num(s);
  }
  let sRe = 0,
    sIm = 0;
  for (let i = 0; i < aRe.length; i++) {
    const aRei = aRe[i],
      aImi = aIm ? aIm[i] : 0;
    const bRei = bRe[i],
      bImi = bIm ? bIm[i] : 0;
    sRe += aRei * bRei + aImi * bImi;
    sIm += aRei * bImi - aImi * bRei;
  }
  if (sIm === 0) return RTV.num(sRe);
  return RTV.complex(sRe, sIm);
}

// ── det ──────────────────────────────────────────────────────────────────

builtin({
  name: "det",
  nargin: 1,
  apply: args => {
    const v = boolToNumber(args[0]);
    if (isRuntimeNumber(v) || isRuntimeComplexNumber(v)) return v;
    const A = denseFloatMatrixArg(v, mustBeFloatError);
    const [m, n] = tensorSize2D(A);
    if (m !== n || A.shape.length > 2) throw squareError();
    // detJS / detComplexJS allocate working buffers but return a scalar
    // — wrap so those scratch allocations release on return.
    return withScratch(() => {
      if (A.imag) {
        const [detRe, detIm] = detComplexJS(toF64(A.data), toF64(A.imag), n);
        if (Math.abs(detIm) < 1e-15) return RTV.num(detRe);
        return RTV.complex(detRe, detIm);
      }
      return RTV.num(detJS(toF64(A.data), n));
    });
  },
});

function detJS(data: Float32Array | Float64Array, n: number): number {
  const a = allocFloat64Array(n * n);
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) a[r * n + c] = data[r + c * n];
  let det = 1;
  for (let col = 0; col < n; col++) {
    let maxRow = col,
      maxVal = Math.abs(a[col * n + col]);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(a[row * n + col]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }
    if (maxVal === 0) return 0;
    if (maxRow !== col) {
      for (let k = 0; k < n; k++) {
        const tmp = a[col * n + k];
        a[col * n + k] = a[maxRow * n + k];
        a[maxRow * n + k] = tmp;
      }
      det *= -1;
    }
    const pivot = a[col * n + col];
    det *= pivot;
    for (let row = col + 1; row < n; row++) {
      const factor = a[row * n + col] / pivot;
      for (let k = col; k < n; k++) a[row * n + k] -= factor * a[col * n + k];
    }
  }
  return det;
}

function detComplexJS(
  dataRe: Float32Array | Float64Array,
  dataIm: Float32Array | Float64Array,
  n: number
): [number, number] {
  const re = allocFloat64Array(n * n);
  const im = allocFloat64Array(n * n);
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      re[r * n + c] = dataRe[r + c * n];
      im[r * n + c] = dataIm[r + c * n];
    }
  let detRe = 1,
    detIm = 0;
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal =
      re[col * n + col] * re[col * n + col] +
      im[col * n + col] * im[col * n + col];
    for (let row = col + 1; row < n; row++) {
      const v =
        re[row * n + col] * re[row * n + col] +
        im[row * n + col] * im[row * n + col];
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }
    if (maxVal === 0) return [0, 0];
    if (maxRow !== col) {
      for (let k = 0; k < n; k++) {
        let tmp = re[col * n + k];
        re[col * n + k] = re[maxRow * n + k];
        re[maxRow * n + k] = tmp;
        tmp = im[col * n + k];
        im[col * n + k] = im[maxRow * n + k];
        im[maxRow * n + k] = tmp;
      }
      detRe = -detRe;
      detIm = -detIm;
    }
    const pivRe = re[col * n + col],
      pivIm = im[col * n + col];
    const newDetRe = detRe * pivRe - detIm * pivIm;
    const newDetIm = detRe * pivIm + detIm * pivRe;
    detRe = newDetRe;
    detIm = newDetIm;
    const pivMag2 = pivRe * pivRe + pivIm * pivIm;
    for (let row = col + 1; row < n; row++) {
      const rRe = re[row * n + col],
        rIm = im[row * n + col];
      const fRe = (rRe * pivRe + rIm * pivIm) / pivMag2;
      const fIm = (rIm * pivRe - rRe * pivIm) / pivMag2;
      for (let k = col; k < n; k++) {
        re[row * n + k] -= fRe * re[col * n + k] - fIm * im[col * n + k];
        im[row * n + k] -= fRe * im[col * n + k] + fIm * re[col * n + k];
      }
    }
  }
  return [detRe, detIm];
}

// ── trace ────────────────────────────────────────────────────────────────

builtin({
  name: "trace",
  nargin: 1,
  mfile: true,
  apply: args => {
    const v = boolToNumber(args[0]);
    if (isRuntimeNumber(v) || isRuntimeComplexNumber(v)) return v;
    if (
      isRuntimeCell(v) ||
      isRuntimeStruct(v) ||
      isRuntimeStructArray(v) ||
      isRuntimeString(v)
    )
      throw builtinError(
        "MATLAB:diag:inputType",
        "Inputs must be numeric, char, or logical."
      );
    const A = denseNumericMatrixArg(v);
    const [rows, cols] = tensorSize2D(A);
    if (rows !== cols || A.shape.length > 2)
      throw builtinError("MATLAB:trace:square", "Matrix must be square.");
    let sumRe = 0;
    let sumIm = 0;
    for (let i = 0; i < rows; i++) {
      sumRe += A.data[i + i * rows];
      if (A.imag) sumIm += A.imag[i + i * rows];
    }
    if (A.imag && Math.abs(sumIm) >= 1e-15) return RTV.complex(sumRe, sumIm);
    return RTV.num(sumRe);
  },
});

// ── cross ────────────────────────────────────────────────────────────────

builtin({
  name: "cross",
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    const a = denseNumericMatrixArg(args[0]),
      b = denseNumericMatrixArg(args[1]);
    const shape = a.shape;
    if (
      shape.length !== b.shape.length ||
      shape.some((s, i) => s !== b.shape[i])
    )
      throw new RuntimeError("cross: A and B must have the same size");
    let dim: number;
    if (args.length === 3) {
      dim = dimArgOr(
        args[2],
        errorFn(
          "MATLAB:size:invalidDim",
          "Dimension argument must be a positive integer scalar or a vector of positive integers. When an empty dimension argument is specified, it must be a 0-by-0, 1-by-0, or 0-by-1 array."
        )
      );
    } else {
      dim = shape.indexOf(3) + 1;
      if (dim === 0)
        throw new RuntimeError(
          "cross: A and B must have at least one dimension of length 3"
        );
    }
    const dimIdx = dim - 1;
    if (dimIdx >= shape.length || shape[dimIdx] !== 3)
      throw new RuntimeError(
        `cross: size(A,${dim}) and size(B,${dim}) must be 3`
      );
    const totalLen = a.data.length;
    const result = allocFloat64Array(totalLen);
    const aIm = a.imag,
      bIm = b.imag;
    const resultIm = aIm || bIm ? allocFloat64Array(totalLen) : undefined;
    const strides = new Array(shape.length);
    strides[0] = 1;
    for (let d = 1; d < shape.length; d++)
      strides[d] = strides[d - 1] * shape[d - 1];
    const dimStride = strides[dimIdx];
    const outerStride =
      dimIdx + 1 < shape.length ? strides[dimIdx + 1] : totalLen;
    const innerSize = dimStride;
    const numOuter = totalLen / outerStride;
    for (let outer = 0; outer < numOuter; outer++) {
      const blockBase = outer * outerStride;
      for (let inner = 0; inner < innerSize; inner++) {
        const base = blockBase + inner;
        const i0 = base,
          i1 = base + dimStride,
          i2 = base + 2 * dimStride;
        const ax = a.data[i0],
          ay = a.data[i1],
          az = a.data[i2];
        const bx = b.data[i0],
          by = b.data[i1],
          bz = b.data[i2];
        if (!resultIm) {
          result[i0] = ay * bz - az * by;
          result[i1] = az * bx - ax * bz;
          result[i2] = ax * by - ay * bx;
          continue;
        }
        const axi = aIm ? aIm[i0] : 0,
          ayi = aIm ? aIm[i1] : 0,
          azi = aIm ? aIm[i2] : 0;
        const bxi = bIm ? bIm[i0] : 0,
          byi = bIm ? bIm[i1] : 0,
          bzi = bIm ? bIm[i2] : 0;
        // (p*q - r*s) for complex p, q, r, s (no conjugation, as MATLAB).
        result[i0] = ay * bz - ayi * bzi - (az * by - azi * byi);
        resultIm[i0] = ay * bzi + ayi * bz - (az * byi + azi * by);
        result[i1] = az * bx - azi * bxi - (ax * bz - axi * bzi);
        resultIm[i1] = az * bxi + azi * bx - (ax * bzi + axi * bz);
        result[i2] = ax * by - axi * byi - (ay * bx - ayi * bxi);
        resultIm[i2] = ax * byi + axi * by - (ay * bxi + ayi * bx);
      }
    }
    const r = maybeComplexTensor(result, [...shape], resultIm);
    // A sparse operand gives a sparse result, as in MATLAB.
    return isRuntimeSparseMatrix(args[0]) || isRuntimeSparseMatrix(args[1])
      ? denseToSparse(r)
      : r;
  },
});

// ── inv ──────────────────────────────────────────────────────────────────

const invTypeError = errorFn(
  "MATLAB:inv:inputType",
  "Invalid data type. Input matrix must be double or single."
);

builtin({
  name: "inv",
  nargin: 1,
  apply: args => invValue(args[0]),
});

function invValue(v: RuntimeValue): RuntimeValue {
  const A = boolToNumber(v);
  if (isRuntimeSparseMatrix(A)) {
    // Sparse inverse stays sparse (MATLAB). Real: solve against the
    // identity with the sparse LU; complex: dense path, then sparsify.
    if (!A.pi) return sparseInverse(A);
    const dense = invValue(sparseToDense(A));
    return isRuntimeTensor(dense) ? denseToSparse(dense) : dense;
  }
  if (isRuntimeComplexNumber(A)) {
    const { re, im } = A;
    if (re === 0 && im === 0)
      throw new RuntimeError("inv: argument is singular");
    // Smith's algorithm for 1/(re + im*i) to avoid spurious over/underflow.
    if (Math.abs(re) >= Math.abs(im)) {
      const r = im / re;
      const d = re + im * r;
      return RTV.complex(1 / d, -r / d);
    }
    const r = re / im;
    const d = im + re * r;
    return RTV.complex(r / d, -1 / d);
  }
  if (isRuntimeNumber(A)) {
    if (A === 0) throw new RuntimeError("inv: argument is singular");
    return RTV.num(1 / A);
  }
  const T = denseFloatMatrixArg(A, invTypeError);
  const [m, n] = tensorSize2D(T);
  if (m !== n || T.shape.length > 2) throw squareError();
  const bridge = getLapackBridge();
  if (T.imag !== undefined) {
    const result = bridge.invComplex(toF64(T.data), toF64(T.imag), n);
    return RTV.tensor(result.re, [n, n], result.im);
  }
  return RTV.tensor(bridge.inv(toF64(T.data), n), [n, n]);
}

// ── expm ───────────────────────────────────────────────────────────────────

builtin({
  name: "expm",
  nargin: 1,
  mfile: true,
  apply: args => expmApply(args[0]),
});

/**
 * Matrix exponential via scaling-and-squaring with a degree-6 Padé
 * approximant (Moler & Van Loan; the algorithm MATLAB's expm uses). It
 * reuses the runtime matrix-multiply (mMul) and linear-solve (mLeftDiv)
 * helpers, so it handles both real and complex matrices in every
 * environment — unlike the textbook V*diag(exp(λ))/V form, which would
 * need a complex eigendecomposition (Node + native LAPACK only).
 */
function expmApply(v: RuntimeValue): RuntimeValue {
  const a = boolToNumber(v);
  // Scalar fast paths: expm of a scalar is just exp of that scalar.
  if (isRuntimeNumber(a)) return RTV.num(Math.exp(a));
  if (isRuntimeComplexNumber(a)) {
    const ex = Math.exp(a.re);
    return RTV.complex(ex * Math.cos(a.im), ex * Math.sin(a.im));
  }
  const A = denseFloatMatrixArg(
    a,
    errorFn("MATLAB:expm:inputType", "Input must be single or double matrix.")
  );
  const [m, n] = tensorSize2D(A);
  if (m !== n || A.shape.length > 2)
    throw builtinError(
      "MATLAB:expm:inputMustBeSquare",
      "Input must be square."
    );
  if (n === 0) return A; // expm([]) = []
  if (n === 1) {
    const ex = Math.exp(A.data[0]);
    if (A.imag)
      return RTV.tensor(
        Float64Array.of(ex * Math.cos(A.imag[0])),
        [1, 1],
        Float64Array.of(ex * Math.sin(A.imag[0]))
      );
    return RTV.tensor(Float64Array.of(ex), [1, 1]);
  }

  // Scale A so that ||A / 2^s||_1 <= 1/2, which keeps the Padé approximant
  // accurate; the squaring loop at the end undoes the scaling.
  let s = 0;
  for (let scaled = matrixOneNorm(A, n); scaled > 0.5; scaled /= 2) s++;
  const As = mMul(A, RTV.num(Math.pow(2, -s))); // scalar * matrix (element-wise)

  // Degree-6 Padé approximant of exp(As): R = D \ E.
  const q = 6;
  const I = identityTensor(n);
  let c = 0.5;
  let E = mAdd(I, mMul(As, RTV.num(c))); // numerator   N = I + c*As
  let D = mSub(I, mMul(As, RTV.num(c))); // denominator D = I - c*As
  let X = As;
  let plus = true;
  for (let k = 2; k <= q; k++) {
    c = (c * (q - k + 1)) / (k * (2 * q - k + 1));
    X = mMul(As, X); // As^k
    const cX = mMul(X, RTV.num(c));
    E = mAdd(E, cX);
    D = plus ? mAdd(D, cX) : mSub(D, cX);
    plus = !plus;
  }
  let R = mLeftDiv(D, E); // solve D * R = E

  // Undo the scaling: square the result s times.
  for (let i = 0; i < s; i++) R = mMul(R, R);
  return R;
}

/** 1-norm (max absolute column sum) of a square (possibly complex) matrix. */
function matrixOneNorm(A: RuntimeTensor, n: number): number {
  const re = A.data;
  const im = A.imag;
  let maxSum = 0;
  for (let j = 0; j < n; j++) {
    let colSum = 0;
    for (let i = 0; i < n; i++) {
      const idx = i + j * n;
      colSum += im ? Math.hypot(re[idx], im[idx]) : Math.abs(re[idx]);
    }
    if (colSum > maxSum) maxSum = colSum;
  }
  return maxSum;
}

/** n×n identity matrix as a runtime tensor. */
function identityTensor(n: number): RuntimeValue {
  const data = allocFloat64Array(n * n);
  for (let i = 0; i < n; i++) data[i + i * n] = 1;
  return RTV.tensor(data, [n, n]);
}

// ── svd ──────────────────────────────────────────────────────────────────

builtin({
  name: "svd",
  nargin: [1, 3],
  nargout: 3,
  apply: (args, nargout) => {
    let econ = false;
    let form: "vector" | "matrix" | undefined;
    const optError = errorFn(
      "MATLAB:svd:invalidOption",
      "Invalid option. Option must be: 'econ', 'matrix', or 'vector'."
    );
    for (let i = 1; i < args.length; i++) {
      const o = args[i];
      if (isText(o)) {
        const s = parseStringArgLower(o);
        if (s === "econ") econ = true;
        else if (s === "vector" || s === "matrix") form = s;
        else throw optError();
      } else if (i === 1 && isRealZero(o)) econ = true;
      else throw optError();
    }
    return svdApply(args[0], nargout, econ, form);
  },
});

/** svd(A): the singular values (a column vector by default for one output,
 *  a diagonal matrix for three), or [U, S, V]. */
function svdApply(
  v: RuntimeValue,
  nargout: number,
  econ = false,
  form?: "vector" | "matrix"
): RuntimeValue | RuntimeValue[] {
  const a = boolToNumber(v);
  const outForm = form ?? (nargout <= 1 ? "vector" : "matrix");
  if (isRuntimeNumber(a)) {
    const val = Math.abs(a);
    const S = RTV.tensor(allocFloat64Array([val]), [1, 1]);
    if (nargout <= 1) return S;
    return outputs(
      [
        RTV.tensor(allocFloat64Array([a >= 0 ? 1 : -1]), [1, 1]),
        S,
        RTV.tensor(allocFloat64Array([1]), [1, 1]),
      ],
      nargout
    );
  }
  const A = denseFloatMatrixArg(a, svdTypeError, svdSparseError);
  const [m, n] = tensorSize2D(A);
  const k = Math.min(m, n);
  const bridge = getLapackBridge();
  const uCols = econ ? k : m;
  const vCols = econ ? k : n;
  const wantVectors = nargout >= 2;
  const Sout = (S: Float64Array) =>
    outForm === "vector"
      ? RTV.tensor(S, [k, 1])
      : buildDiagMatrix(S, undefined, econ ? k : [m, n]);
  if (A.imag) {
    const result = bridge.svdComplex(
      toF64(A.data),
      toF64(A.imag),
      m,
      n,
      econ,
      wantVectors
    );
    if (nargout <= 1) return Sout(result.S);
    return outputs(
      [
        RTV.tensor(result.URe!, [m, uCols], result.UIm!),
        Sout(result.S),
        RTV.tensor(result.VRe!, [n, vCols], result.VIm!),
      ],
      nargout
    );
  }
  const result = bridge.svd(toF64(A.data), m, n, econ, wantVectors);
  if (nargout <= 1) return Sout(result.S);
  return outputs(
    [
      RTV.tensor(result.U!, [m, uCols]),
      Sout(result.S),
      RTV.tensor(result.V!, [n, vCols]),
    ],
    nargout
  );
}

// ── null ───────────────────────────────────────────────────────────────────

/** null(A): orthonormal basis (columns) for the null space of A, via SVD.
 *  null(A, tol) overrides the singular-value tolerance. */
function nullApply(args: RuntimeValue[]): RuntimeValue {
  const A = denseFloatMatrixArg(args[0], svdTypeError, svdSparseError);
  const [m, n] = tensorSize2D(A);
  let tolArg: number | undefined;
  if (args.length > 1) {
    const err = errorFn(
      "MATLAB:null:invalidSecondArgument",
      'Second input must be the option "rational" or a real numeric scalar tolerance.'
    );
    if (isText(args[1])) {
      if (parseStringArgLower(args[1]) !== "rational") throw err();
      throw new RuntimeError("null: the 'rational' option is not supported");
    }
    tolArg = realScalarOr(args[1], err);
  }

  // Full SVD: V is n×n; the trailing columns span the null space.
  const [, Sdiag, V] = svdApply(A, 3) as RuntimeTensor[];
  const k = Math.min(m, n);
  const s: number[] = [];
  for (let i = 0; i < k; i++) s.push(Sdiag.data[colMajorIndex(i, i, m)]);
  const maxS = s.length > 0 ? Math.max(...s) : 0;

  // Tolerance: explicit second arg, else max(m,n)*eps(max singular value).
  const tol = tolArg ?? Math.max(m, n) * epsOf(maxS);
  let r = 0;
  for (const sv of s) if (sv > tol) r++;

  // Extract columns r .. n-1 of V (n×n) into an n×(n-r) matrix.
  const cols = n - r;
  const out = allocFloat64Array(n * cols);
  const outImag = V.imag ? allocFloat64Array(n * cols) : undefined;
  for (let c = 0; c < cols; c++) {
    for (let i = 0; i < n; i++) {
      const src = colMajorIndex(i, r + c, n);
      out[colMajorIndex(i, c, n)] = V.data[src];
      if (outImag) outImag[colMajorIndex(i, c, n)] = V.imag![src];
    }
  }
  return RTV.tensor(out, [n, cols], outImag);
}

builtin({
  name: "null",
  nargin: [1, 2],
  mfile: true,
  apply: args => nullApply(args),
});

// ── bandwidth ───────────────────────────────────────────────────────────────

/** Lower/upper bandwidth of a matrix: the max distance below/above the
 *  diagonal at which a nonzero entry appears. */
function computeBandwidth(
  v: RuntimeValue,
  err: () => RuntimeError
): [number, number] {
  const A = boolToNumber(v);
  let lower = 0;
  let upper = 0;
  const note = (i: number, j: number) => {
    if (i > j) lower = Math.max(lower, i - j);
    else if (j > i) upper = Math.max(upper, j - i);
  };
  if (isRuntimeNumber(A) || isRuntimeComplexNumber(A)) return [0, 0];
  if (isRuntimeSparseMatrix(A)) {
    for (let j = 0; j < A.n; j++) {
      for (let k = A.jc[j]; k < A.jc[j + 1]; k++) {
        if (A.pr[k] === 0 && (!A.pi || A.pi[k] === 0)) continue;
        note(A.ir[k], j);
      }
    }
    return [lower, upper];
  }
  if (isRuntimeTensor(A)) {
    const [m, n] = tensorSize2D(A);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) {
        const idx = colMajorIndex(i, j, m);
        if (A.data[idx] === 0 && (!A.imag || A.imag[idx] === 0)) continue;
        note(i, j);
      }
    }
    return [lower, upper];
  }
  throw err();
}

const bandwidthTypeError = errorFn(
  "MATLAB:bandwidth:inputType",
  "Invalid data type. Arguments must be double, single or logical matrices."
);

function bandwidthApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const [lower, upper] = computeBandwidth(args[0], bandwidthTypeError);
  if (args.length >= 2 && args[1] !== undefined) {
    const type = parseStringArgLower(args[1]);
    if (type === "lower") return RTV.num(lower);
    if (type === "upper") return RTV.num(upper);
    throw new RuntimeError("bandwidth: TYPE must be 'lower' or 'upper'");
  }
  // No TYPE: `[lower,upper] = bandwidth(A)`, or `B = bandwidth(A)` -> lower.
  if (nargout >= 2) return [RTV.num(lower), RTV.num(upper)];
  return RTV.num(lower);
}

builtin({
  name: "bandwidth",
  nargin: [1, 2],
  nargout: 2,
  apply: (args, nargout) => bandwidthApply(args, nargout),
});

// ── istriu / istril / isbanded ──────────────────────────────────────────────

/** True when every nonzero lies on or above (upper) / below (lower) the
 *  diagonal. Numeric check: explicitly stored zeros do not count. */
function isTriangularValue(v: RuntimeValue, upper: boolean): boolean {
  const A = boolToNumber(v);
  if (isRuntimeNumber(A) || isRuntimeComplexNumber(A)) return true;
  if (isRuntimeSparseMatrix(A))
    return upper ? sparseIsUpper(A) : sparseIsLower(A);
  if (isRuntimeTensor(A)) {
    if (A.shape.length > 2) return false;
    const [m, n] = tensorSize2D(A);
    for (let j = 0; j < n; j++)
      for (let i = 0; i < m; i++) {
        if (upper ? i <= j : i >= j) continue;
        const idx = colMajorIndex(i, j, m);
        if (A.data[idx] !== 0 || (A.imag && A.imag[idx] !== 0)) return false;
      }
    return true;
  }
  throw floatOrLogicalError();
}

for (const [name, upper] of [
  ["istriu", true],
  ["istril", false],
] as const) {
  builtin({
    name,
    help: {
      signatures: [`tf = ${name}(A)`],
      description: `True if A is ${upper ? "upper" : "lower"} triangular (all entries ${upper ? "below" : "above"} the main diagonal are zero). Works for dense and sparse matrices.`,
    },
    nargin: 1,
    apply: args => RTV.logical(isTriangularValue(args[0], upper)),
  });
}

builtin({
  name: "isbanded",
  help: {
    signatures: ["tf = isbanded(A, lower, upper)"],
    description:
      "True if A has lower bandwidth at most `lower` and upper bandwidth at most `upper`. Works for dense and sparse matrices.",
  },
  nargin: 3,
  apply: args => {
    const typeError = errorFn(
      "MATLAB:isbanded:inputType",
      "Inputs must be double, single, or logical."
    );
    const boundError = errorFn(
      "MATLAB:isbanded:nonNegIntScalar",
      "Second and third inputs must be nonnegative integer scalars."
    );
    const bound = (v: RuntimeValue) => {
      if (isRuntimeChar(v) || isRuntimeString(v)) throw typeError();
      const b = realScalarOr(v, boundError);
      if (!(Number.isInteger(b) && b >= 0)) throw boundError();
      return b;
    };
    const lo = bound(args[1]);
    const up = bound(args[2]);
    const [l, u] = computeBandwidth(args[0], typeError);
    return RTV.logical(l <= lo && u <= up);
  },
});

// ── qr ──────────────────────────────────────────────────────────────────

builtin({
  name: "qr",
  nargin: [1, 3],
  nargout: 3,
  apply: (args, nargout) => {
    // Options: 0 (economy size, permutation vector), "econ" (economy
    // size), "vector" / "matrix" (permutation form).
    let econ = false;
    let permVector = false;
    for (let i = 1; i < args.length; i++) {
      const o = args[i];
      if (isText(o)) {
        const s = parseStringArgLower(o);
        if (s === "econ") econ = true;
        else if (s === "vector") permVector = true;
        else if (s === "matrix") permVector = false;
        else
          throw builtinError(
            "MATLAB:qr:unknownOption",
            "Option must be 'econ', 'matrix', or 'vector'."
          );
      } else if (i === 1 && isRealZero(o)) {
        econ = true;
        permVector = true;
      } else if (isRuntimeSparseMatrix(args[0])) {
        throw new RuntimeError("qr: sparse input is not supported");
      } else {
        throw builtinError(
          "MATLAB:qr:notSupportFill",
          "Right-hand side matrix is only supported if the first input matrix is sparse."
        );
      }
    }
    if (isRuntimeSparseMatrix(args[0]))
      throw new RuntimeError("qr: sparse input is not supported");
    const A = denseFloatMatrixArg(
      args[0],
      errorFn(
        "MATLAB:qr:inputType",
        "Invalid data type. First argument must be double or single."
      )
    );
    return qrApply(A, nargout, econ, permVector);
  },
});

function qrApply(
  A: RuntimeTensor,
  nargout: number,
  econ: boolean,
  permVector: boolean
): RuntimeValue | RuntimeValue[] {
  const [m, n] = tensorSize2D(A);
  const k = Math.min(m, n);

  // 3-output: column-pivoted QR → [Q, R, E]
  if (nargout === 3) {
    return qrPivotApply(A, m, n, k, econ, permVector);
  }

  const bridge = getLapackBridge();
  const rRows = econ ? k : m;
  const qCols = econ ? k : m;
  if (A.imag) {
    const result = bridge.qrComplex(
      toF64(A.data),
      toF64(A.imag),
      m,
      n,
      econ,
      nargout === 2
    );
    if (nargout <= 1) return RTV.tensor(result.RRe, [rRows, n], result.RIm);
    return [
      RTV.tensor(result.QRe!, [m, qCols], result.QIm!),
      RTV.tensor(result.RRe, [rRows, n], result.RIm),
    ];
  }

  const result = bridge.qr(toF64(A.data), m, n, econ, nargout === 2);
  if (nargout <= 1) return RTV.tensor(result.R, [rRows, n]);
  return [RTV.tensor(result.Q, [m, qCols]), RTV.tensor(result.R, [rRows, n])];
}

/** Convert 1-based permutation vector to permutation matrix or keep as vector. */
function permResult(
  jpvt: Float64Array | Int32Array,
  n: number,
  asVector: boolean
): RuntimeValue {
  const perm = allocFloat64Array(n);
  for (let i = 0; i < n; i++) perm[i] = jpvt[i];
  if (asVector) return RTV.tensor(perm, [1, n]);
  // Non-economy: return n×n permutation matrix
  const mat = allocFloat64Array(n * n);
  for (let j = 0; j < n; j++) mat[colMajorIndex(jpvt[j] - 1, j, n)] = 1;
  return RTV.tensor(mat, [n, n]);
}

function qrPivotApply(
  A: RuntimeTensor,
  m: number,
  n: number,
  k: number,
  econ: boolean,
  permVector: boolean
): RuntimeValue[] {
  const bridge = getLapackBridge();
  const rRows = econ ? k : m;
  const qCols = econ ? k : m;
  if (A.imag) {
    const result = bridge.qrPivotComplex(
      toF64(A.data),
      toF64(A.imag),
      m,
      n,
      econ
    );
    return [
      RTV.tensor(result.QRe, [m, qCols], result.QIm),
      RTV.tensor(result.RRe, [rRows, n], result.RIm),
      permResult(result.jpvt, n, permVector),
    ];
  }
  const result = bridge.qrPivot(toF64(A.data), m, n, econ);
  return [
    RTV.tensor(result.Q, [m, qCols]),
    RTV.tensor(result.R, [rRows, n]),
    permResult(result.jpvt, n, permVector),
  ];
}

// ── qrupdate ────────────────────────────────────────────────────────────

builtin({
  name: "qrupdate",
  help: {
    signatures: ["[Q1, R1] = qrupdate(Q, R, u, v)"],
    description:
      "Rank-1 update of a full QR factorization: if A = Q*R (Q m-by-m), returns Q1, R1 with Q1*R1 = A + u*v'. Uses Givens rotations, O(m^2 + m*n).",
  },
  nargin: 4,
  nargout: 2,
  apply: (args, nargout) => outputs(qrupdateApply(args), nargout),
});

/** Dense split re/im copy of a numeric argument (im null if real). */
function qrupdateArg(arg: RuntimeValue): {
  re: Float64Array;
  im: Float64Array | null;
  rows: number;
  cols: number;
} {
  const v = boolToNumber(arg);
  if (isRuntimeNumber(v))
    return { re: allocFloat64Array([v]), im: null, rows: 1, cols: 1 };
  if (isRuntimeComplexNumber(v))
    return {
      re: allocFloat64Array([v.re]),
      im: allocFloat64Array([v.im]),
      rows: 1,
      cols: 1,
    };
  if (!isRuntimeTensor(v) || v.shape.length > 2)
    throw new RuntimeError(
      "qrupdate: inputs must be full matrices of class single or double"
    );
  const [rows, cols] = tensorSize2D(v);
  return {
    re: allocFloat64Array(v.data),
    im: v.imag ? allocFloat64Array(v.imag) : null,
    rows,
    cols,
  };
}

function qrupdateApply(args: RuntimeValue[]): [RuntimeValue, RuntimeValue] {
  const [Qa, Ra, ua, va] = args.map(qrupdateArg);
  const m = Qa.rows;
  const n = Ra.cols;
  if (Qa.cols !== m || Ra.rows !== m)
    throw new RuntimeError("qrupdate: matrix dimensions must agree");
  if (ua.cols !== 1 || ua.rows !== m)
    throw new RuntimeError(
      `qrupdate: third argument must be a column vector of length ${m}`
    );
  if (va.cols !== 1 || va.rows !== n)
    throw new RuntimeError(
      `qrupdate: fourth argument must be a column vector of length ${n}`
    );
  const isComplex = !!(Qa.im || Ra.im || ua.im || va.im);
  const Qr = Qa.re,
    Qi = Qa.im ?? allocFloat64Array(m * m);
  const Rr = Ra.re,
    Ri = Ra.im ?? allocFloat64Array(m * n);
  const ur = ua.re,
    ui = ua.im ?? allocFloat64Array(m);
  const vr = va.re,
    vi = va.im ?? allocFloat64Array(n);

  // Givens rotation G = [c s; -conj(s) c] (c real >= 0) with
  // G*[f; g] = [r; 0], following the LAPACK 3.10 zlartg convention.
  let c = 0,
    sr = 0,
    si = 0,
    rr = 0,
    ri = 0;
  const rot = (fr: number, fi: number, gr: number, gi: number) => {
    const nf = Math.hypot(fr, fi),
      ng = Math.hypot(gr, gi);
    if (ng === 0) {
      c = 1;
      sr = si = 0;
      rr = fr;
      ri = fi;
    } else if (nf === 0) {
      c = 0;
      sr = gr / ng;
      si = -gi / ng;
      rr = ng;
      ri = 0;
    } else {
      const nrm = Math.hypot(nf, ng);
      const pr = fr / nf,
        pi = fi / nf; // f/|f|
      c = nf / nrm;
      // s = (f/|f|) * conj(g) / nrm
      sr = (pr * gr + pi * gi) / nrm;
      si = (pi * gr - pr * gi) / nrm;
      rr = pr * nrm;
      ri = pi * nrm;
    }
  };
  // rows i, i+1 of R (columns j0..n-1) <- G * rows
  const rotR = (i: number, j0: number) => {
    for (let j = j0; j < n; j++) {
      const a = i + j * m,
        b = a + 1;
      const xr = Rr[a],
        xi = Ri[a],
        yr = Rr[b],
        yi = Ri[b];
      Rr[a] = c * xr + sr * yr - si * yi;
      Ri[a] = c * xi + sr * yi + si * yr;
      Rr[b] = c * yr - (sr * xr + si * xi);
      Ri[b] = c * yi - (sr * xi - si * xr);
    }
  };
  // columns i, i+1 of Q <- [Qi Qi1] * G'
  const rotQ = (i: number) => {
    for (let r = 0; r < m; r++) {
      const a = r + i * m,
        b = a + m;
      const xr = Qr[a],
        xi = Qi[a],
        yr = Qr[b],
        yi = Qi[b];
      Qr[a] = c * xr + sr * yr + si * yi;
      Qi[a] = c * xi + sr * yi - si * yr;
      Qr[b] = c * yr - (sr * xr - si * xi);
      Qi[b] = c * yi - (sr * xi + si * xr);
    }
  };

  // w = Q' * u
  const wr = allocFloat64Array(m),
    wi = allocFloat64Array(m);
  for (let j = 0; j < m; j++) {
    let sumR = 0,
      sumI = 0;
    for (let i = 0; i < m; i++) {
      const qr = Qr[i + j * m],
        qi = Qi[i + j * m];
      sumR += qr * ur[i] + qi * ui[i];
      sumI += qr * ui[i] - qi * ur[i];
    }
    wr[j] = sumR;
    wi[j] = sumI;
  }
  // Reduce w to a multiple of e1 from the bottom; R becomes upper Hessenberg.
  for (let i = m - 2; i >= 0; i--) {
    rot(wr[i], wi[i], wr[i + 1], wi[i + 1]);
    wr[i] = rr;
    wi[i] = ri;
    rotR(i, i);
    rotQ(i);
  }
  // R(1,:) += w(1) * v'
  for (let j = 0; j < n; j++) {
    Rr[j * m] += wr[0] * vr[j] + wi[0] * vi[j];
    Ri[j * m] += wi[0] * vr[j] - wr[0] * vi[j];
  }
  // Retriangularize the Hessenberg R.
  for (let j = 0; j < Math.min(m - 1, n); j++) {
    const a = j + j * m;
    rot(Rr[a], Ri[a], Rr[a + 1], Ri[a + 1]);
    Rr[a] = rr;
    Ri[a] = ri;
    Rr[a + 1] = Ri[a + 1] = 0;
    rotR(j, j + 1);
    rotQ(j);
  }
  if (!isComplex) return [RTV.tensor(Qr, [m, m]), RTV.tensor(Rr, [m, n])];
  return [RTV.tensor(Qr, [m, m], Qi), RTV.tensor(Rr, [m, n], Ri)];
}

// ── lu ──────────────────────────────────────────────────────────────────

const invalidPermFlagError = errorFn(
  "MATLAB:factorFlags:invalidPermFlag",
  "Option must be 'matrix' or 'vector'."
);

builtin({
  name: "lu",
  nargin: [1, 3],
  nargout: 5,
  apply: (args, nargout) => withScratch(() => luApply(args, nargout)),
});

/** Sparse LU: [L,U,P] = lu(S), [L,U,P,Q] = lu(S), with optional pivot
 *  threshold and 'vector' output form. */
function sparseLuApply(
  S: RuntimeSparseMatrix,
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  let outputForm: "matrix" | "vector" = "matrix";
  let tol = nargout >= 4 ? 0.1 : 1.0;
  for (const opt of args.slice(1)) {
    if (isRuntimeNumber(opt)) {
      if (!(opt >= 0 && opt <= 1))
        throw new RuntimeError("lu: pivot threshold must be in [0, 1]");
      tol = opt;
    } else {
      const form = isText(opt) ? parseLuOutputForm(opt) : null;
      if (form === null) throw invalidPermFlagError();
      outputForm = form;
    }
  }
  const q = nargout >= 4 ? columnOrdering(S) : null;
  const res = sparseLu(S, q, tol);
  const { L, U, p } = res;
  if (nargout <= 1) {
    // Y = L + U - I (permutation information is lost, as in MATLAB).
    return mSub(mAdd(L, U), speyeSparse(Math.min(S.m, S.n)));
  }
  if (nargout === 2) {
    // A = (P'*L) * U: unpermute the rows of L.
    const Lperm = sparsePermute(L, invertPerm(p), identityPerm(L.n));
    return [Lperm, U];
  }
  const permOut = (perm: Int32Array) =>
    outputForm === "vector" ? permVectorTensor(perm) : permutationMatrix(perm);
  if (nargout === 3) return [L, U, permOut(p)];
  // P*A*Q = L*U with Q(q[k], k) = 1, i.e. Q = permutationMatrix(q)'.
  const Q =
    outputForm === "vector"
      ? permVectorTensor(res.q)
      : sparseTranspose(permutationMatrix(res.q));
  return [L, U, permOut(p), Q];
}

function luApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const S = args[0];
  if (isRuntimeSparseMatrix(S)) {
    if (nargout >= 5)
      throw new RuntimeError(
        "lu: the row-scaling output [L,U,P,Q,R] = lu(S) is not supported"
      );
    if (!S.pi) return sparseLuApply(S, args, nargout);
    if (nargout >= 4)
      throw new RuntimeError(
        "lu: [L,U,P,Q] = lu(S) is not supported for complex sparse matrices"
      );
    // Complex sparse: dense factorization, sparse factors. Pivot
    // thresholds only apply to the sparse factorization.
    const dense = luDense(
      sparseToDense(S),
      luDenseForm(args.slice(1).filter(o => !isRuntimeNumber(o))),
      nargout
    );
    const toSp = (v: RuntimeValue) =>
      isRuntimeTensor(v) ? denseToSparse(v) : v;
    return Array.isArray(dense) ? dense.map(toSp) : toSp(dense);
  }
  if (nargout > 3)
    throw builtinError(
      "MATLAB:lu:DenseThreeOutputOnly",
      "Fourth output only available for sparse input matrices."
    );
  return luDense(args[0], luDenseForm(args.slice(1)), nargout);
}

/** The output form of a dense lu: options must be 'matrix' or 'vector'. */
function luDenseForm(opts: RuntimeValue[]): "matrix" | "vector" {
  let form: "matrix" | "vector" = "matrix";
  for (const opt of opts) {
    const f = isText(opt) ? parseLuOutputForm(opt) : null;
    if (f === null) throw invalidPermFlagError();
    form = f;
  }
  return form;
}

function luDense(
  v: RuntimeValue,
  outputForm: "matrix" | "vector",
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const a = boolToNumber(v);
  if (isRuntimeNumber(a)) {
    const L = RTV.tensor(allocFloat64Array([1]), [1, 1]);
    const U = RTV.tensor(allocFloat64Array([a]), [1, 1]);
    // Y = L + U - I = U for a scalar.
    if (nargout <= 1) return U;
    if (nargout === 2) return [L, U];
    if (outputForm === "vector") return [L, U, RTV.num(1)];
    return [L, U, RTV.tensor(allocFloat64Array([1]), [1, 1])];
  }
  const A = denseFloatMatrixArg(
    a,
    errorFn(
      "MATLAB:lu:inputType",
      "Invalid data type. First argument must be double or single."
    )
  );
  const [m, n] = tensorSize2D(A);
  const k = Math.min(m, n);
  const isComplex = A.imag !== undefined;
  let LU_re: Float64Array, LU_im: Float64Array | undefined, ipiv: Int32Array;
  const bridge = getLapackBridge();
  if (isComplex) {
    const result = bridge.luComplex(toF64(A.data), toF64(A.imag!), m, n);
    LU_re = result.LURe;
    LU_im = result.LUIm;
    ipiv = result.ipiv;
  } else {
    const result = bridge.lu(toF64(A.data), m, n);
    LU_re = result.LU;
    ipiv = result.ipiv;
  }
  const U_re = allocFloat64Array(k * n);
  const U_im = isComplex ? allocFloat64Array(k * n) : undefined;
  for (let j = 0; j < n; j++) {
    const imax = Math.min(j, k - 1);
    for (let i = 0; i <= imax; i++) {
      U_re[i + j * k] = LU_re[i + j * m];
      if (U_im && LU_im) U_im[i + j * k] = LU_im[i + j * m];
    }
  }
  if (nargout <= 1) return RTV.tensor(LU_re, [m, n], LU_im);
  const perm = ipivToPermVector(ipiv, m);
  if (nargout === 2) {
    const L_unit_re = allocFloat64Array(m * k);
    const L_unit_im = isComplex ? allocFloat64Array(m * k) : undefined;
    for (let j = 0; j < k; j++) {
      L_unit_re[j + j * m] = 1;
      for (let i = j + 1; i < m; i++) {
        L_unit_re[i + j * m] = LU_re[i + j * m];
        if (L_unit_im && LU_im) L_unit_im[i + j * m] = LU_im[i + j * m];
      }
    }
    const L_re = allocFloat64Array(m * k);
    const L_im = isComplex ? allocFloat64Array(m * k) : undefined;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < k; j++) {
        L_re[perm[i] + j * m] = L_unit_re[i + j * m];
        if (L_im && L_unit_im) L_im[perm[i] + j * m] = L_unit_im[i + j * m];
      }
    }
    return [RTV.tensor(L_re, [m, k], L_im), RTV.tensor(U_re, [k, n], U_im)];
  }
  // nargout === 3
  const L_re = allocFloat64Array(m * k);
  const L_im = isComplex ? allocFloat64Array(m * k) : undefined;
  for (let j = 0; j < k; j++) {
    L_re[j + j * m] = 1;
    for (let i = j + 1; i < m; i++) {
      L_re[i + j * m] = LU_re[i + j * m];
      if (L_im && LU_im) L_im[i + j * m] = LU_im[i + j * m];
    }
  }
  if (outputForm === "vector") {
    const P_data = allocFloat64Array(m);
    for (let i = 0; i < m; i++) P_data[i] = perm[i] + 1;
    return [
      RTV.tensor(L_re, [m, k], L_im),
      RTV.tensor(U_re, [k, n], U_im),
      RTV.tensor(P_data, [1, m]),
    ];
  }
  const P_data = allocFloat64Array(m * m);
  for (let i = 0; i < m; i++) P_data[i + perm[i] * m] = 1;
  return [
    RTV.tensor(L_re, [m, k], L_im),
    RTV.tensor(U_re, [k, n], U_im),
    RTV.tensor(P_data, [m, m]),
  ];
}

function parseLuOutputForm(arg: RuntimeValue): "matrix" | "vector" | null {
  if (arg === undefined) return "matrix";
  const s = parseStringArgLower(arg);
  if (s === "vector") return "vector";
  if (s === "matrix") return "matrix";
  return null;
}

function ipivToPermVector(ipiv: Int32Array, m: number): Int32Array {
  const perm = new Int32Array(m);
  for (let i = 0; i < m; i++) perm[i] = i;
  for (let i = 0; i < ipiv.length; i++) {
    const j = ipiv[i] - 1;
    if (j !== i) {
      const tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
  }
  return perm;
}

// ── eig ──────────────────────────────────────────────────────────────────

const eigTypeError = errorFn(
  "MATLAB:eig:inputType",
  "Invalid data type. Input matrix must be double or single."
);

builtin({
  name: "eig",
  nargin: [1, 4],
  nargout: 3,
  apply: (args, nargout) => eigApply(args, nargout),
});

function eigApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const a = boolToNumber(args[0]);
  const { B, balance, outputForm, algorithm } = parseEigOptionsRuntime(args);
  if (B !== undefined)
    return eigGeneralizedApply(a, B, nargout, outputForm, algorithm);
  if (isRuntimeNumber(a)) {
    const val = a;
    if (nargout <= 1) return RTV.num(val);
    const V = RTV.tensor(allocFloat64Array([1]), [1, 1]);
    if (nargout === 2) {
      if (outputForm === "vector") return [V, RTV.num(val)];
      return [V, RTV.tensor(allocFloat64Array([val]), [1, 1])];
    }
    const D =
      outputForm === "vector"
        ? RTV.num(val)
        : RTV.tensor(allocFloat64Array([val]), [1, 1]);
    return [V, D, RTV.tensor(allocFloat64Array([1]), [1, 1])];
  }
  if (isRuntimeSparseMatrix(a) && nargout >= 2)
    throw builtinError(
      "MATLAB:eig:useEigsForSparseStandardVector",
      "Too many output arguments. Only one output argument is supported when the input matrix is sparse. Use eigs to compute a subset of eigenvectors and eigenvalues of a sparse matrix."
    );
  const A = denseFloatMatrixArg(a, eigTypeError);
  const [m, n] = tensorSize2D(A);
  if (m !== n || A.shape.length > 2)
    throw builtinError(
      "MATLAB:eig:inputMustBeSquareStandard",
      "Input matrix must be square."
    );
  const computeVL = nargout >= 3,
    computeVR = nargout >= 2;
  // Symmetric (Hermitian) input goes to the symmetric solver, which — unlike
  // dgeev — guarantees real eigenvalues in ascending order and an orthonormal
  // eigenvector basis across repeated eigenvalues. The test is exact equality,
  // as in MATLAB's isequal(A, A'); a tolerance would make the choice of
  // algorithm depend on how close to symmetric the input happens to be.
  if (isExactlyHermitian(A, n))
    return eigSymmetricApply(A, n, nargout, outputForm);
  const bridge = getLapackBridge();
  if (A.imag) {
    const result = bridge.eigComplex(
      toF64(A.data),
      toF64(A.imag),
      n,
      computeVL,
      computeVR
    );
    const { wRe, wIm, VLRe, VLIm, VRRe, VRIm } = result;
    if (nargout <= 1) return maybeComplexTensor(wRe, [n, 1], wIm);
    const Vout =
      computeVR && VRRe && VRIm
        ? maybeComplexTensor(VRRe, [n, n], VRIm)
        : RTV.tensor(allocFloat64Array(n * n), [n, n]);
    const Dout =
      outputForm === "vector"
        ? maybeComplexTensor(wRe, [n, 1], wIm)
        : buildDiagMatrix(wRe, wIm, n);
    if (nargout === 2) return [Vout, Dout];
    const Wout =
      computeVL && VLRe && VLIm
        ? maybeComplexTensor(VLRe, [n, n], VLIm)
        : RTV.tensor(allocFloat64Array(n * n), [n, n]);
    return [Vout, Dout, Wout];
  }
  const result = bridge.eig(toF64(A.data), n, computeVL, computeVR, balance);
  const { wr, wi, VL, VR } = result;
  const hasComplex = wi.some((v: number) => v !== 0);
  if (nargout <= 1) return maybeComplexTensor(wr, [n, 1], wi);
  const Vout =
    computeVR && VR
      ? buildEigenvectorMatrix(VR, wi, n, hasComplex)
      : RTV.tensor(allocFloat64Array(n * n), [n, n]);
  const Dout =
    outputForm === "vector"
      ? maybeComplexTensor(wr, [n, 1], wi)
      : buildDiagMatrix(wr, wi, n);
  if (nargout === 2) return [Vout, Dout];
  const Wout =
    computeVL && VL
      ? buildEigenvectorMatrix(VL, wi, n, hasComplex)
      : RTV.tensor(allocFloat64Array(n * n), [n, n]);
  return [Vout, Dout, Wout];
}

/**
 * Exact symmetry (Hermitian for complex) test, matching MATLAB's
 * isequal(A, A'). NaN entries make it false, as they do in MATLAB.
 */
function isExactlyHermitian(A: RuntimeTensor, n: number): boolean {
  const re = A.data;
  const im = A.imag;
  for (let j = 0; j < n; j++) {
    if (im && im[j + j * n] !== 0) return false;
    const dj = re[j + j * n];
    if (Number.isNaN(dj)) return false;
    for (let i = j + 1; i < n; i++) {
      if (re[i + j * n] !== re[j + i * n]) return false;
      if (im && im[i + j * n] !== -im[j + i * n]) return false;
    }
  }
  return true;
}

/**
 * Symmetric / Hermitian path: dsyev (real) or zheev (complex).
 *
 * For a Hermitian matrix the left and right eigenvectors coincide, so the
 * third output is a copy of the second.
 */
function eigSymmetricApply(
  A: RuntimeTensor,
  n: number,
  nargout: number,
  outputForm: "vector" | "matrix"
): RuntimeValue | RuntimeValue[] {
  const computeV = nargout >= 2;
  const bridge = getLapackBridge();
  let w: Float64Array;
  let VRe: Float64Array | undefined;
  let VIm: Float64Array | undefined;
  if (A.imag) {
    const r = bridge.eigHerm(toF64(A.data), toF64(A.imag), n, computeV);
    w = r.w;
    VRe = r.VRe;
    VIm = r.VIm;
  } else {
    const r = bridge.eigSym(toF64(A.data), n, computeV);
    w = r.w;
    VRe = r.V;
  }
  if (nargout <= 1) return RTV.tensor(w, [n, 1]);
  const makeV = () =>
    VRe
      ? maybeComplexTensor(
          allocFloat64Array(VRe),
          [n, n],
          VIm ? allocFloat64Array(VIm) : undefined
        )
      : RTV.tensor(allocFloat64Array(n * n), [n, n]);
  const Vout = makeV();
  const Dout =
    outputForm === "vector"
      ? RTV.tensor(allocFloat64Array(w), [n, 1])
      : buildDiagMatrix(w, undefined, n);
  if (nargout === 2) return [Vout, Dout];
  return [Vout, Dout, makeV()];
}

function parseEigOptionsRuntime(args: RuntimeValue[]): {
  B?: RuntimeValue;
  balance: boolean;
  outputForm: "vector" | "matrix";
  algorithm: "auto" | "chol" | "qz";
} {
  let balance = true,
    outputForm: "vector" | "matrix" = "matrix",
    algorithm: "auto" | "chol" | "qz" = "auto";
  // A numeric second argument is the B of the generalized problem A*v =
  // lambda*B*v, not an option. Before this was recognized, eig(A, B) silently
  // returned eig(A).
  let B: RuntimeValue | undefined;
  let first = 1;
  if (args.length >= 2 && !isStringLikeArg(args[1])) {
    B = args[1];
    first = 2;
  }
  for (let i = first; i < args.length; i++) {
    const arg = args[i];
    if (!isStringLikeArg(arg))
      throw new RuntimeError(
        "eig: option arguments must be character vectors or strings"
      );
    const val = parseStringArgLower(arg);
    if (val === "nobalance") balance = false;
    else if (val === "balance") balance = true;
    else if (val === "vector") outputForm = "vector";
    else if (val === "matrix") outputForm = "matrix";
    else if (val === "chol" || val === "qz") algorithm = val;
    else throw new RuntimeError(`eig: unknown option '${val}'`);
  }
  if (algorithm !== "auto" && B === undefined)
    throw new RuntimeError(
      `eig: the '${algorithm}' algorithm applies only to eig(A, B)`
    );
  return { B, balance, outputForm, algorithm };
}

function isStringLikeArg(arg: RuntimeValue): boolean {
  return isRuntimeString(arg) || isRuntimeChar(arg);
}

// ── eig(A, B): the generalized eigenvalue problem ────────────────────────

/** A dense n×n matrix as split real/imaginary parts (imag null when real). */
type SplitMatrix = { re: Float64Array; im: Float64Array | null };

function toSplitSquare(
  v: RuntimeValue,
  name: string
): { M: SplitMatrix; n: number } {
  const t = denseFloatMatrixArg(v, eigTypeError);
  const [rows, cols] = tensorSize2D(t);
  if (rows !== cols)
    throw new RuntimeError(`eig: ${name} must be a square matrix`);
  return {
    M: { re: toF64(t.data), im: t.imag ? toF64(t.imag) : null },
    n: rows,
  };
}

/** Exact symmetry / Hermitian test on a SplitMatrix. */
function splitIsHermitian(M: SplitMatrix, n: number): boolean {
  for (let j = 0; j < n; j++) {
    if (M.im && M.im[j + j * n] !== 0) return false;
    if (Number.isNaN(M.re[j + j * n])) return false;
    for (let i = j + 1; i < n; i++) {
      if (M.re[i + j * n] !== M.re[j + i * n]) return false;
      if (M.im && M.im[i + j * n] !== -M.im[j + i * n]) return false;
    }
  }
  return true;
}

function eigGeneralizedApply(
  Aval: RuntimeValue,
  Bval: RuntimeValue,
  nargout: number,
  outputForm: "vector" | "matrix",
  algorithm: "auto" | "chol" | "qz"
): RuntimeValue | RuntimeValue[] {
  const { M: A, n } = toSplitSquare(Aval, "A");
  const { M: B, n: nB } = toSplitSquare(Bval, "B");
  if (n !== nB) throw new RuntimeError("eig: A and B must be the same size");

  if (algorithm !== "qz" && splitIsHermitian(A, n) && splitIsHermitian(B, n)) {
    const R = choleskyFactor(B, n);
    if (R) return eigSymmetricDefinite(A, R, n, nargout, outputForm);
    if (algorithm === "chol")
      throw new RuntimeError(
        "eig: the 'chol' algorithm requires B to be symmetric positive definite"
      );
  }
  return eigGeneralizedQz(A, B, n, nargout, outputForm);
}

/**
 * Upper-triangular Cholesky factor R with B = R'*R, or null when B is not
 * positive definite.
 */
function choleskyFactor(B: SplitMatrix, n: number): SplitMatrix | null {
  const bridge = getLapackBridge();
  if (B.im) {
    const r = bridge.cholComplex(B.re, B.im, n, true);
    if (r.info !== 0) return null;
    return { re: r.RRe, im: r.RIm };
  }
  const r = bridge.chol(B.re, n, true);
  if (r.info !== 0) return null;
  return { re: r.R, im: null };
}

/**
 * Symmetric-definite generalized problem A*v = lambda*B*v with B = R'*R.
 * Reduce to the standard symmetric problem on C = R^-H * A * R^-1, solve it
 * with the symmetric eigensolver, and map the vectors back with V = R^-1 * S.
 * That normalization gives V'*B*V = I, as MATLAB does.
 */
function eigSymmetricDefinite(
  A: SplitMatrix,
  R: SplitMatrix,
  n: number,
  nargout: number,
  outputForm: "vector" | "matrix"
): RuntimeValue | RuntimeValue[] {
  const complex = !!(A.im || R.im);
  // Y = R^-H * A, then C = Y * R^-1
  const Y = solveLeftConjTransposeUpper(R, A, n, complex);
  const C = solveRightUpper(Y, R, n, complex);
  // Re-symmetrize: the reduction is symmetric in exact arithmetic, and the
  // symmetric solver reads only one triangle, so this only removes rounding
  // asymmetry that would otherwise be silently ignored.
  symmetrizeInPlace(C, n);

  const computeV = nargout >= 2;
  let w: Float64Array;
  let SRe: Float64Array | undefined;
  let SIm: Float64Array | undefined;
  const bridge = getLapackBridge();
  if (complex) {
    const r = bridge.eigHerm(
      C.re,
      C.im ?? allocFloat64Array(n * n),
      n,
      computeV
    );
    w = r.w;
    SRe = r.VRe;
    SIm = r.VIm;
  } else {
    const r = bridge.eigSym(C.re, n, computeV);
    w = r.w;
    SRe = r.V;
  }

  if (nargout <= 1) return RTV.tensor(w, [n, 1]);
  const S: SplitMatrix = {
    re: SRe ?? allocFloat64Array(n * n),
    im: SIm ?? null,
  };
  const V = SRe ? solveUpper(R, S, n, complex) : S;
  const makeV = () =>
    maybeComplexTensor(
      allocFloat64Array(V.re),
      [n, n],
      V.im ? allocFloat64Array(V.im) : undefined
    );
  const Vout = makeV();
  const Dout =
    outputForm === "vector"
      ? RTV.tensor(allocFloat64Array(w), [n, 1])
      : buildDiagMatrix(w, undefined, n);
  if (nargout === 2) return [Vout, Dout];
  // A and B are Hermitian, so the left eigenvectors coincide with the right
  // ones: W'*A = D*W'*B holds for W = V.
  return [Vout, Dout, makeV()];
}

/** In-place Hermitian averaging of a square matrix. */
function symmetrizeInPlace(M: SplitMatrix, n: number): void {
  for (let j = 0; j < n; j++) {
    if (M.im) M.im[j + j * n] = 0;
    for (let i = j + 1; i < n; i++) {
      const a = (M.re[i + j * n] + M.re[j + i * n]) / 2;
      M.re[i + j * n] = a;
      M.re[j + i * n] = a;
      if (M.im) {
        const b = (M.im[i + j * n] - M.im[j + i * n]) / 2;
        M.im[i + j * n] = b;
        M.im[j + i * n] = -b;
      }
    }
  }
}

/** Solve R^H * X = Bm for X, R upper triangular (forward substitution). */
function solveLeftConjTransposeUpper(
  R: SplitMatrix,
  Bm: SplitMatrix,
  n: number,
  complex: boolean
): SplitMatrix {
  const X: SplitMatrix = {
    re: allocFloat64Array(n * n),
    im: complex ? allocFloat64Array(n * n) : null,
  };
  for (let c = 0; c < n; c++) {
    for (let i = 0; i < n; i++) {
      let sr = Bm.re[i + c * n];
      let si = Bm.im ? Bm.im[i + c * n] : 0;
      for (let k = 0; k < i; k++) {
        // L(i,k) = conj(R(k,i))
        const lr = R.re[k + i * n];
        const li = R.im ? -R.im[k + i * n] : 0;
        const xr = X.re[k + c * n];
        const xi = X.im ? X.im[k + c * n] : 0;
        sr -= lr * xr - li * xi;
        si -= lr * xi + li * xr;
      }
      const dr = R.re[i + i * n];
      const di = R.im ? -R.im[i + i * n] : 0;
      const [qr, qi] = complexDivide(sr, si, dr, di);
      X.re[i + c * n] = qr;
      if (X.im) X.im[i + c * n] = qi;
    }
  }
  return X;
}

/** Solve C * R = Y for C, R upper triangular (column-forward substitution). */
function solveRightUpper(
  Y: SplitMatrix,
  R: SplitMatrix,
  n: number,
  complex: boolean
): SplitMatrix {
  const C: SplitMatrix = {
    re: allocFloat64Array(n * n),
    im: complex ? allocFloat64Array(n * n) : null,
  };
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let sr = Y.re[i + j * n];
      let si = Y.im ? Y.im[i + j * n] : 0;
      for (let k = 0; k < j; k++) {
        const cr = C.re[i + k * n];
        const ci = C.im ? C.im[i + k * n] : 0;
        const rr = R.re[k + j * n];
        const ri = R.im ? R.im[k + j * n] : 0;
        sr -= cr * rr - ci * ri;
        si -= cr * ri + ci * rr;
      }
      const dr = R.re[j + j * n];
      const di = R.im ? R.im[j + j * n] : 0;
      const [qr, qi] = complexDivide(sr, si, dr, di);
      C.re[i + j * n] = qr;
      if (C.im) C.im[i + j * n] = qi;
    }
  }
  return C;
}

/** Solve R * V = S for V, R upper triangular (back substitution). */
function solveUpper(
  R: SplitMatrix,
  S: SplitMatrix,
  n: number,
  complex: boolean
): SplitMatrix {
  const V: SplitMatrix = {
    re: allocFloat64Array(n * n),
    im: complex ? allocFloat64Array(n * n) : null,
  };
  for (let c = 0; c < n; c++) {
    for (let i = n - 1; i >= 0; i--) {
      let sr = S.re[i + c * n];
      let si = S.im ? S.im[i + c * n] : 0;
      for (let k = i + 1; k < n; k++) {
        const rr = R.re[i + k * n];
        const ri = R.im ? R.im[i + k * n] : 0;
        const vr = V.re[k + c * n];
        const vi = V.im ? V.im[k + c * n] : 0;
        sr -= rr * vr - ri * vi;
        si -= rr * vi + ri * vr;
      }
      const dr = R.re[i + i * n];
      const di = R.im ? R.im[i + i * n] : 0;
      const [qr, qi] = complexDivide(sr, si, dr, di);
      V.re[i + c * n] = qr;
      if (V.im) V.im[i + c * n] = qi;
    }
  }
  return V;
}

function complexDivide(
  ar: number,
  ai: number,
  br: number,
  bi: number
): [number, number] {
  if (bi === 0) return [ar / br, ai / br];
  if (Math.abs(br) >= Math.abs(bi)) {
    const r = bi / br;
    const t = br + bi * r;
    return [(ar + ai * r) / t, (ai - ar * r) / t];
  }
  const r = br / bi;
  const t = br * r + bi;
  return [(ar * r + ai) / t, (ai * r - ar) / t];
}

/**
 * General generalized problem via the QZ (generalized Schur) factorization.
 * ts-lapack has no dgges/dggev, so without the native addon the bridge throws
 * here rather than answering a different problem.
 */
function eigGeneralizedQz(
  A: SplitMatrix,
  B: SplitMatrix,
  n: number,
  nargout: number,
  outputForm: "vector" | "matrix"
): RuntimeValue | RuntimeValue[] {
  const computeEigvecs = nargout >= 2;
  const wr = allocFloat64Array(n);
  const wi = allocFloat64Array(n);
  let V: RuntimeValue | null = null;
  let W: RuntimeValue | null = null;
  const bridge = getLapackBridge();

  if (A.im || B.im) {
    const r = bridge.qzComplex(
      A.re,
      A.im ?? allocFloat64Array(n * n),
      B.re,
      B.im ?? allocFloat64Array(n * n),
      n,
      computeEigvecs
    );
    for (let i = 0; i < n; i++) {
      const [qr, qi] = complexDivide(
        r.alphaRe[i],
        r.alphaIm[i],
        r.betaRe[i],
        r.betaIm[i]
      );
      wr[i] = qr;
      wi[i] = qi;
    }
    if (computeEigvecs) {
      if (!r.VRe || !r.VIm || !r.WRe || !r.WIm)
        throw new RuntimeError(
          "eig: failed to compute generalized eigenvectors"
        );
      V = maybeComplexTensor(
        allocFloat64Array(r.VRe),
        [n, n],
        allocFloat64Array(r.VIm)
      );
      W = maybeComplexTensor(
        allocFloat64Array(r.WRe),
        [n, n],
        allocFloat64Array(r.WIm)
      );
    }
  } else {
    const r = bridge.qz(A.re, B.re, n, computeEigvecs);
    let hasComplex = false;
    for (let i = 0; i < n; i++) {
      const beta = r.beta[i];
      wr[i] = r.alphar[i] / beta;
      wi[i] = r.alphai[i] === 0 && beta === 0 ? 0 : r.alphai[i] / beta;
      if (r.alphai[i] !== 0) hasComplex = true;
    }
    if (computeEigvecs) {
      if (!r.V || !r.W)
        throw new RuntimeError(
          "eig: failed to compute generalized eigenvectors"
        );
      V = buildEigenvectorMatrix(r.V, r.alphai, n, hasComplex);
      W = buildEigenvectorMatrix(r.W, r.alphai, n, hasComplex);
    }
  }

  if (nargout <= 1) return maybeComplexTensor(wr, [n, 1], wi);
  const Dout =
    outputForm === "vector"
      ? maybeComplexTensor(wr, [n, 1], wi)
      : buildDiagMatrix(wr, wi, n);
  if (nargout === 2) return [V as RuntimeValue, Dout];
  return [V as RuntimeValue, Dout, W as RuntimeValue];
}

// ── chol ────────────────────────────────────────────────────────────────

const posdefError = errorFn(
  "MATLAB:posdef",
  "Matrix must be positive definite."
);

builtin({
  name: "chol",
  nargin: [1, 3],
  nargout: 3,
  apply: (args, nargout) => withScratch(() => cholApply(args, nargout)),
});

/** Sparse Cholesky. Reads the upper triangle only (no symmetry check, as in
 *  MATLAB). The three-output form applies a fill-reducing ordering. */
function sparseCholApply(
  S: RuntimeSparseMatrix,
  upper: boolean,
  outputForm: "matrix" | "vector",
  nargout: number
): RuntimeValue | RuntimeValue[] {
  if (S.m !== S.n) throw squareError();
  const n = S.n;
  const perm = nargout >= 3 ? minimumDegreeOrdering(S) : identityPerm(n);
  const A = nargout >= 3 ? sparsePermute(S, perm, perm) : S;
  const res = sparseChol(A);
  const R = upper ? sparseTranspose(res.L) : res.L;
  const permOut = () =>
    outputForm === "vector"
      ? permVectorTensor(perm)
      : permutationMatrix(invertPerm(perm));
  if (!res.ok) {
    if (nargout < 2) throw posdefError();
    if (nargout >= 3) return [R, RTV.num(res.p), permOut()];
    return [R, RTV.num(res.p)];
  }
  if (nargout >= 3) return [R, RTV.num(0), permOut()];
  if (nargout === 2) return [R, RTV.num(0)];
  return R;
}

function cholApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const a = boolToNumber(args[0]);
  const inputIsSparse = isRuntimeSparseMatrix(a);
  if (nargout >= 3 && !inputIsSparse)
    throw builtinError(
      "MATLAB:chol:DenseTwoOutputOnly",
      "Third output only available for sparse matrices."
    );
  let triangle: "upper" | "lower" = "upper";
  let outputForm: "matrix" | "vector" = "matrix";
  if (args.length === 2) {
    const s = parseStringArgLower(args[1]);
    if (s === "upper" || s === "lower") triangle = s;
    else if (s === "matrix" || s === "vector") outputForm = s;
    else throw new RuntimeError("chol: invalid option");
  } else if (args.length === 3) {
    const tri = parseStringArgLower(args[1]);
    if (tri !== "upper" && tri !== "lower")
      throw new RuntimeError("chol: triangle must be 'upper' or 'lower'");
    triangle = tri;
    const form = parseStringArgLower(args[2]);
    if (form !== "matrix" && form !== "vector")
      throw new RuntimeError("chol: outputForm must be 'matrix' or 'vector'");
    outputForm = form as "matrix" | "vector";
  }
  const upper = triangle === "upper";
  if (isRuntimeSparseMatrix(a) && !a.pi)
    return sparseCholApply(a, upper, outputForm, nargout);
  if (isRuntimeNumber(a)) {
    const val = a;
    if (val <= 0) {
      if (nargout >= 2)
        return [RTV.tensor(allocFloat64Array([0]), [1, 1]), RTV.num(1)];
      throw posdefError();
    }
    const r = Math.sqrt(val);
    if (nargout >= 2)
      return [RTV.tensor(allocFloat64Array([r]), [1, 1]), RTV.num(0)];
    return RTV.tensor(allocFloat64Array([r]), [1, 1]);
  }
  // Complex sparse input takes the dense path.
  const A = denseFloatMatrixArg(
    a,
    errorFn(
      "MATLAB:chol:inputType",
      "Invalid data type. First argument must be a double or single matrix."
    )
  );
  const [m, n] = tensorSize2D(A);
  if (m !== n || A.shape.length > 2) throw squareError();
  // LAPACK reads only the real part of the diagonal; MATLAB requires it to
  // be real.
  if (A.imag)
    for (let i = 0; i < n; i++)
      if (A.imag[i + i * n] !== 0)
        throw builtinError(
          "MATLAB:chol:matrixMustBePosDefWithRealDiag",
          "Matrix must be positive definite with real diagonal."
        );
  const isComplex = A.imag !== undefined;
  let R_re: Float64Array, R_im: Float64Array | undefined, info_val: number;
  const bridge = getLapackBridge();
  if (isComplex) {
    const result = bridge.cholComplex(toF64(A.data), toF64(A.imag!), n, upper);
    R_re = result.RRe;
    R_im = result.RIm;
    info_val = result.info;
  } else {
    const result = bridge.chol(toF64(A.data), n, upper);
    R_re = result.R;
    info_val = result.info;
  }
  if (nargout >= 2 && info_val > 0) {
    const k = info_val - 1;
    if (nargout >= 3) {
      // 3-output partial result
      let partialR: RuntimeValue;
      if (upper) {
        const pr = allocFloat64Array(k * n);
        const pi = isComplex ? allocFloat64Array(k * n) : undefined;
        for (let j = 0; j < n; j++)
          for (let i = 0; i <= Math.min(j, k - 1); i++) {
            pr[i + j * k] = (R_re as Float64Array)[i + j * n];
            if (pi && R_im) pi[i + j * k] = (R_im as Float64Array)[i + j * n];
          }
        partialR = RTV.tensor(pr, [k, n], pi);
      } else {
        const pr = allocFloat64Array(n * k);
        const pi = isComplex ? allocFloat64Array(n * k) : undefined;
        for (let j = 0; j < k; j++)
          for (let i = j; i < n; i++) {
            pr[i + j * n] = (R_re as Float64Array)[i + j * n];
            if (pi && R_im) pi[i + j * n] = (R_im as Float64Array)[i + j * n];
          }
        partialR = RTV.tensor(pr, [n, k], pi);
      }
      const permOut =
        outputForm === "vector"
          ? RTV.tensor(
              allocFloat64Array(Array.from({ length: n }, (_, i) => i + 1)),
              [n, 1]
            )
          : (() => {
              const P = allocFloat64Array(n * n);
              for (let i = 0; i < n; i++) P[i + i * n] = 1;
              return RTV.tensor(P, [n, n]);
            })();
      return [partialR, RTV.num(info_val), permOut];
    }
    // 2-output partial result
    let partialR: RuntimeValue;
    if (upper) {
      const pr = allocFloat64Array(k * k);
      const pi = isComplex ? allocFloat64Array(k * k) : undefined;
      for (let j = 0; j < k; j++)
        for (let i = 0; i <= j; i++) {
          pr[i + j * k] = (R_re as Float64Array)[i + j * n];
          if (pi && R_im) pi[i + j * k] = (R_im as Float64Array)[i + j * n];
        }
      partialR = RTV.tensor(pr, [k, k], pi);
    } else {
      const pr = allocFloat64Array(k * k);
      const pi = isComplex ? allocFloat64Array(k * k) : undefined;
      for (let j = 0; j < k; j++)
        for (let i = j; i < k; i++) {
          pr[i + j * k] = (R_re as Float64Array)[i + j * n];
          if (pi && R_im) pi[i + j * k] = (R_im as Float64Array)[i + j * n];
        }
      partialR = RTV.tensor(pr, [k, k], pi);
    }
    return [partialR, RTV.num(info_val)];
  }
  if (nargout >= 2) {
    const R = RTV.tensor(
      allocFloat64Array(R_re),
      [n, n],
      R_im ? allocFloat64Array(R_im) : undefined
    );
    if (nargout >= 3) {
      const permOut =
        outputForm === "vector"
          ? RTV.tensor(
              allocFloat64Array(Array.from({ length: n }, (_, i) => i + 1)),
              [n, 1]
            )
          : (() => {
              const P = allocFloat64Array(n * n);
              for (let i = 0; i < n; i++) P[i + i * n] = 1;
              return RTV.tensor(P, [n, n]);
            })();
      return [R, RTV.num(0), permOut];
    }
    return [R, RTV.num(0)];
  }
  if (info_val > 0) throw posdefError();
  return RTV.tensor(
    allocFloat64Array(R_re),
    [n, n],
    R_im ? allocFloat64Array(R_im) : undefined
  );
}

// ── linsolve ────────────────────────────────────────────────────────────

builtin({
  name: "linsolve",
  nargin: [2, 3],
  nargout: 2,
  apply: (args, nargout) => {
    if (args.length > 2)
      throw new RuntimeError(
        "linsolve: the options structure (third argument) is not supported"
      );
    if (nargout > 1)
      throw new RuntimeError(
        "linsolve: the second output [X, r] = linsolve(...) is not supported"
      );
    const typeError = errorFn(
      "MATLAB:linsolve:inputType",
      "First and second arguments must be single or double."
    );
    const sparseError = errorFn(
      "MATLAB:linsolve:sparseNotSupported",
      "Linsolve is not supported for sparse inputs."
    );
    const A = denseFloatMatrixArg(args[0], typeError, sparseError);
    const B = denseFloatMatrixArg(args[1], typeError, sparseError);
    const [m, n] = tensorSize2D(A);
    const [Bm, p] = tensorSize2D(B);
    if (Bm !== m)
      throw builtinError("MATLAB:dimagree", "Matrix dimensions must agree.");
    if (A.imag || B.imag) {
      const ARe = A.data,
        AIm = A.imag ?? allocFloat64Array(A.data.length);
      const BRe = B.data,
        BIm = B.imag ?? allocFloat64Array(B.data.length);
      const X = linsolveComplexLapack(ARe, AIm, m, n, BRe, BIm, p);
      return RTV.tensor(X.re, [n, p], X.im);
    }
    const X = linsolveLapack(A.data, m, n, B.data, p);
    return RTV.tensor(X, [n, p]);
  },
});

// ── cond ────────────────────────────────────────────────────────────────

builtin({
  name: "cond",
  nargin: [1, 2],
  mfile: true,
  apply: args => {
    const v = boolToNumber(args[0]);
    let p: number | "fro" = 2;
    if (args.length >= 2) {
      const pArg = args[1];
      if (isText(pArg)) {
        const pStr = parseStringArgLower(pArg);
        if (pStr === "fro") p = "fro";
        else if (pStr === "inf") p = Infinity;
        else throw unknownNormError();
      } else p = realScalarOr(pArg, unknownNormError);
    }
    if (isRuntimeNumber(v)) return RTV.num(v === 0 ? Infinity : 1);
    if (isRuntimeComplexNumber(v))
      return RTV.num(v.re === 0 && v.im === 0 ? Infinity : 1);
    const M = floatMatrixArg(v, svdTypeError);
    if (isRuntimeSparseMatrix(M)) {
      // MATLAB estimates the 1-norm condition number of a sparse matrix
      // (condest); numbl computes it exactly.
      emitWarning("Using CONDEST instead of COND for sparse matrix.");
      return condNorm(sparseToDense(M), 1);
    }
    if (p === 2) {
      const s = (svdApply(M, 1) as RuntimeTensor).data;
      let sMax = -Infinity,
        sMin = Infinity;
      for (let i = 0; i < s.length; i++) {
        if (s[i] > sMax) sMax = s[i];
        if (s[i] < sMin) sMin = s[i];
      }
      if (sMin === 0) return RTV.num(Infinity);
      return RTV.num(sMax / sMin);
    }
    const [rows, cols] = tensorSize2D(M);
    if (rows !== cols)
      throw builtinError(
        "MATLAB:cond:normMismatchSizeA",
        "A is rectangular.  Use the 2 norm."
      );
    return condNorm(M, p);
  },
});

/** norm(A, p) * norm(inv(A), p) for a square matrix. */
function condNorm(A: RuntimeTensor, p: number | "fro"): RuntimeValue {
  const normArg: RuntimeValue = p === "fro" ? RTV.char("fro") : RTV.num(p);
  const normA = normApply([A, normArg]) as number;
  const normInvA = normApply([invValue(A), normArg]) as number;
  return RTV.num(normA * normInvA);
}

// ── rcond ───────────────────────────────────────────────────────────────

builtin({
  name: "rcond",
  nargin: 1,
  // inv() allocates the full inverse; wrap so that scratch releases on
  // return (the result is a scalar, so it survives).
  apply: args => withScratch(() => rcondApply(args[0])),
});

/**
 * Estimate of the reciprocal 1-norm condition number,
 * 1 / (norm(A,1) * norm(inv(A),1)). Near 1 for well-conditioned matrices,
 * near 0 for badly conditioned ones. Unlike cond, a singular matrix yields 0
 * rather than Inf. (MATLAB's rcond uses an LU-based estimator; forming the
 * inverse here gives a value that agrees to several significant figures.)
 */
function rcondApply(v: RuntimeValue): RuntimeValue {
  const a = boolToNumber(v);
  // Scalars: well-conditioned (1) unless zero, which is singular (0).
  if (isRuntimeNumber(a)) return RTV.num(a === 0 ? 0 : 1);
  if (isRuntimeComplexNumber(a))
    return RTV.num(a.re === 0 && a.im === 0 ? 0 : 1);
  const A = denseFloatMatrixArg(a, mustBeFloatError);
  const [rows, cols] = tensorSize2D(A);
  if (rows !== cols || A.shape.length > 2)
    throw builtinError(
      "MATLAB:rcond:notSquareMatrix",
      "Input must be a square matrix."
    );
  if (rows === 0) return RTV.num(Infinity); // rcond([]) = Inf
  const normA = normApply([A, RTV.num(1)]) as number;
  if (!(normA > 0)) return RTV.num(0); // zero matrix (or NaN) → singular
  let invA: RuntimeValue;
  try {
    invA = invValue(A);
  } catch (e) {
    // A singular matrix has rcond 0 (MATLAB does not error here).
    const msg = e instanceof Error ? e.message : String(e);
    if (/singular/i.test(msg)) return RTV.num(0);
    throw e;
  }
  const normInvA = normApply([invA, RTV.num(1)]) as number;
  if (!isFinite(normInvA) || normInvA === 0) return RTV.num(0);
  const rc = 1 / (normA * normInvA);
  return RTV.num(isFinite(rc) ? rc : 0);
}

// ── rank ────────────────────────────────────────────────────────────────

builtin({
  name: "rank",
  nargin: [1, 2],
  mfile: true,
  apply: args => {
    const tolArg =
      args.length >= 2
        ? realScalarOr(
            args[1],
            errorFn(
              "MATLAB:rank:invalidSecondArgument",
              "Second input must be a real numeric scalar tolerance."
            )
          )
        : undefined;
    const v = boolToNumber(args[0]);
    if (isRuntimeNumber(v)) return RTV.num(v === 0 ? 0 : 1);
    if (isRuntimeComplexNumber(v))
      return RTV.num(v.re === 0 && v.im === 0 ? 0 : 1);
    const A = denseFloatMatrixArg(v, svdTypeError, svdSparseError);
    const [rows, cols] = tensorSize2D(A);
    const s = (svdApply(A, 1) as RuntimeTensor).data;
    let tol: number;
    if (tolArg !== undefined) tol = tolArg;
    else {
      let sMax = 0;
      for (let i = 0; i < s.length; i++) if (s[i] > sMax) sMax = s[i];
      tol = Math.max(rows, cols) * epsOf(sMax);
    }
    let k = 0;
    for (let i = 0; i < s.length; i++) if (s[i] > tol) k++;
    return RTV.num(k);
  },
});

function epsOf(x: number): number {
  if (!isFinite(x) || x === 0) return Number.EPSILON;
  const ax = Math.abs(x);
  return Math.pow(2, Math.floor(Math.log2(ax)) - 52);
}

// ── pinv ────────────────────────────────────────────────────────────────

builtin({
  name: "pinv",
  nargin: [1, 2],
  mfile: true,
  apply: args =>
    withScratch(() => {
      const tolArg =
        args.length >= 2
          ? realScalarOr(
              args[1],
              errorFn(
                "MATLAB:pinv:invalidSecondArgument",
                "Second input must be a real numeric scalar tolerance."
              )
            )
          : 0;
      const v = boolToNumber(args[0]);
      if (isRuntimeNumber(v)) return RTV.num(v === 0 ? 0 : 1 / v);
      if (isRuntimeComplexNumber(v)) {
        const d = v.re * v.re + v.im * v.im;
        return d === 0 ? RTV.num(0) : RTV.complex(v.re / d, -v.im / d);
      }
      const A = denseFloatMatrixArg(v, svdTypeError, svdSparseError);
      const [m, n] = tensorSize2D(A);
      const k = Math.min(m, n);
      if (A.imag) {
        // pinv(A) = V * diag(1./S) * U'  (U' is the conjugate transpose)
        const { S, URe, UIm, VRe, VIm } = getLapackBridge().svdComplex(
          toF64(A.data),
          toF64(A.imag),
          m,
          n,
          true,
          true
        );
        const tol =
          args.length >= 2
            ? tolArg
            : Math.max(m, n) * S[0] * 2.220446049250313e-16;
        const re = allocFloat64Array(n * m);
        const im = allocFloat64Array(n * m);
        for (let i = 0; i < n; i++)
          for (let j = 0; j < m; j++) {
            let sr = 0,
              si = 0;
            for (let l = 0; l < k; l++) {
              if (!(S[l] > tol)) continue;
              const vr = VRe![l * n + i] / S[l],
                vi = VIm![l * n + i] / S[l];
              const ur = URe![l * m + j],
                ui = -UIm![l * m + j];
              sr += vr * ur - vi * ui;
              si += vr * ui + vi * ur;
            }
            re[j * n + i] = sr;
            im[j * n + i] = si;
          }
        return maybeComplexTensor(re, [n, m], im);
      }
      const { U, S, V } = getLapackBridge().svd(
        toF64(A.data),
        m,
        n,
        true,
        true
      ) as { U: Float64Array; S: Float64Array; V: Float64Array };
      const tol =
        args.length >= 2
          ? tolArg
          : Math.max(m, n) * S[0] * 2.220446049250313e-16;
      const result = allocFloat64Array(n * m);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
          let sum = 0;
          for (let l = 0; l < k; l++)
            if (S[l] > tol) sum += V[l * n + i] * (1 / S[l]) * U[l * m + j];
          result[j * n + i] = sum;
        }
      return RTV.tensor(result, [n, m]);
    }),
});

// ── kron ────────────────────────────────────────────────────────────────

builtin({
  name: "kron",
  nargin: 2,
  mfile: true,
  apply: args => {
    const A = numericMatrixArg(args[0]);
    const B = numericMatrixArg(args[1]);
    // Sparse operand: the result is sparse with nnz(A)*nnz(B) entries.
    if (isRuntimeSparseMatrix(A) || isRuntimeSparseMatrix(B))
      return sparseKron(
        isRuntimeSparseMatrix(A) ? A : denseToSparse(A),
        isRuntimeSparseMatrix(B) ? B : denseToSparse(B)
      );
    const [m, n] = tensorSize2D(A);
    const [p, q] = tensorSize2D(B);
    const rows = m * p,
      cols = n * q;
    const aIm = A.imag,
      bIm = B.imag;
    const result = allocFloat64Array(rows * cols);
    const resultIm = aIm || bIm ? allocFloat64Array(rows * cols) : undefined;
    for (let ja = 0; ja < n; ja++)
      for (let ia = 0; ia < m; ia++) {
        const aVal = A.data[ia + ja * m];
        const aIv = aIm ? aIm[ia + ja * m] : 0;
        for (let jb = 0; jb < q; jb++)
          for (let ib = 0; ib < p; ib++) {
            const bVal = B.data[ib + jb * p];
            const dst = ia * p + ib + (ja * q + jb) * rows;
            if (!resultIm) {
              result[dst] = aVal * bVal;
              continue;
            }
            const bIv = bIm ? bIm[ib + jb * p] : 0;
            result[dst] = aVal * bVal - aIv * bIv;
            resultIm[dst] = aVal * bIv + aIv * bVal;
          }
      }
    return maybeComplexTensor(result, [rows, cols], resultIm);
  },
});

// ── blkdiag ─────────────────────────────────────────────────────────────

builtin({
  name: "blkdiag",
  mfile: true,
  apply: args => {
    if (args.some(a => isRuntimeChar(a)))
      throw new RuntimeError("blkdiag: char input is not supported");
    const blocks = args.map(a => denseNumericMatrixArg(a));
    const anyComplex = blocks.some(b => b.imag);
    let totalRows = 0,
      totalCols = 0;
    const dims: [number, number][] = [];
    for (const block of blocks) {
      const [m, n] = tensorSize2D(block);
      dims.push([m, n]);
      totalRows += m;
      totalCols += n;
    }
    const result = allocFloat64Array(totalRows * totalCols);
    const resultIm = anyComplex
      ? allocFloat64Array(totalRows * totalCols)
      : undefined;
    let rowOffset = 0,
      colOffset = 0;
    for (let k = 0; k < blocks.length; k++) {
      const [m, n] = dims[k];
      const data = blocks[k].data;
      const imag = blocks[k].imag;
      for (let j = 0; j < n; j++)
        for (let i = 0; i < m; i++) {
          const dst = rowOffset + i + (colOffset + j) * totalRows;
          result[dst] = data[i + j * m];
          if (resultIm && imag) resultIm[dst] = imag[i + j * m];
        }
      rowOffset += m;
      colOffset += n;
    }
    const r = maybeComplexTensor(result, [totalRows, totalCols], resultIm);
    // A sparse block gives a sparse result, as in MATLAB.
    return args.some(a => isRuntimeSparseMatrix(a)) ? denseToSparse(r) : r;
  },
});

// ── pagemtimes ──────────────────────────────────────────────────────────

builtin({
  name: "pagemtimes",
  nargin: [2, 4],
  apply: args => {
    let X: RuntimeValue, Y: RuntimeValue;
    let transpX = "none",
      transpY = "none";
    if (args.length === 2) {
      X = args[0];
      Y = args[1];
    } else if (args.length === 4) {
      X = args[0];
      Y = args[2];
      const opt = (v: RuntimeValue) => {
        const s = isText(v) ? parseStringArgLower(v) : "";
        if (s !== "none" && s !== "transpose" && s !== "ctranspose")
          throw new RuntimeError(
            "pagemtimes: transpose options must be 'none', 'transpose', or 'ctranspose'"
          );
        return s;
      };
      transpX = opt(args[1]);
      transpY = opt(args[3]);
    } else
      throw builtinError(
        "MATLAB:pagemtimes:threeInputNotSupported",
        "Incorrect number of input arguments. Specify either two or four input arguments."
      );

    const typeError = errorFn(
      "MATLAB:pagemtimes:invalidArray",
      "Invalid data types. Input arrays must be double or single."
    );
    const sparseError = errorFn(
      "MATLAB:pagemtimes:sparseNotSupported",
      "Sparse inputs are not supported."
    );
    const xT = denseFloatMatrixArg(X, typeError, sparseError);
    const yT = denseFloatMatrixArg(Y, typeError, sparseError);

    const xShape = xT.shape.length < 2 ? [1, xT.shape[0] || 1] : xT.shape;
    const yShape = yT.shape.length < 2 ? [1, yT.shape[0] || 1] : yT.shape;
    let xRows = xShape[0],
      xCols = xShape[1];
    let yRows = yShape[0],
      yCols = yShape[1];
    if (transpX === "transpose" || transpX === "ctranspose")
      [xRows, xCols] = [xCols, xRows];
    if (transpY === "transpose" || transpY === "ctranspose")
      [yRows, yCols] = [yCols, yRows];
    if (xCols !== yRows)
      throw new RuntimeError(
        `pagemtimes: inner matrix dimensions must agree: ${xCols} vs ${yRows}`
      );

    const xExtra = xShape.slice(2),
      yExtra = yShape.slice(2);
    const maxExtraDims = Math.max(xExtra.length, yExtra.length);
    const outExtra: number[] = [];
    for (let d = 0; d < maxExtraDims; d++) {
      const xd = d < xExtra.length ? xExtra[d] : 1;
      const yd = d < yExtra.length ? yExtra[d] : 1;
      if (xd !== yd && xd !== 1 && yd !== 1)
        throw new RuntimeError(
          "pagemtimes: dimensions beyond first two must be compatible"
        );
      outExtra.push(Math.max(xd, yd));
    }
    const outShape = [xRows, yCols, ...outExtra];
    const pageSize = xRows * yCols;
    const xPageSize = xShape[0] * xShape[1],
      yPageSize = yShape[0] * yShape[1];
    const totalPages = outExtra.reduce((a, b) => a * b, 1);
    const isComplex = !!(xT.imag || yT.imag);
    const result = allocFloat64Array(pageSize * totalPages);
    const resultIm = isComplex
      ? allocFloat64Array(pageSize * totalPages)
      : undefined;
    const xExtraStrides: number[] = [],
      yExtraStrides: number[] = [];
    let xStride = 1,
      yStride = 1;
    for (let d = 0; d < maxExtraDims; d++) {
      xExtraStrides.push(xStride);
      yExtraStrides.push(yStride);
      xStride *= d < xExtra.length ? xExtra[d] : 1;
      yStride *= d < yExtra.length ? yExtra[d] : 1;
    }
    // Copy one page into (re, im) buffers in op(X) layout, applying the
    // requested transpose / conjugate transpose.
    const loadPage = (
      T: RuntimeTensor,
      shp: number[],
      off: number,
      mode: string,
      re: Float64Array,
      im: Float64Array | undefined
    ) => {
      const r = shp[0],
        c = shp[1];
      const tr = mode === "transpose" || mode === "ctranspose";
      const sgn = mode === "ctranspose" ? -1 : 1;
      for (let j = 0; j < c; j++)
        for (let i = 0; i < r; i++) {
          const src = off + i + j * r;
          const dst = tr ? j + i * c : i + j * r;
          re[dst] = T.data[src];
          if (im) im[dst] = T.imag ? sgn * T.imag[src] : 0;
        }
    };
    const aRe = allocFloat64Array(xPageSize),
      bRe = allocFloat64Array(yPageSize);
    const aIm = isComplex ? allocFloat64Array(xPageSize) : undefined,
      bIm = isComplex ? allocFloat64Array(yPageSize) : undefined;
    for (let p = 0; p < totalPages; p++) {
      let xPageIdx = 0,
        yPageIdx = 0,
        rem = p;
      for (let d = maxExtraDims - 1; d >= 0; d--) {
        const outD = outExtra[d],
          idx = rem % outD;
        rem = (rem - idx) / outD;
        const xd = d < xExtra.length ? xExtra[d] : 1;
        const yd = d < yExtra.length ? yExtra[d] : 1;
        xPageIdx += (xd === 1 ? 0 : idx) * xExtraStrides[d];
        yPageIdx += (yd === 1 ? 0 : idx) * yExtraStrides[d];
      }
      loadPage(xT, xShape, xPageIdx * xPageSize, transpX, aRe, aIm);
      loadPage(yT, yShape, yPageIdx * yPageSize, transpY, bRe, bIm);
      const outOff = p * pageSize;
      const aR = xRows,
        aC = xCols,
        bR = yRows;
      for (let j = 0; j < yCols; j++)
        for (let i = 0; i < aR; i++) {
          let sr = 0,
            si = 0;
          for (let kk = 0; kk < aC; kk++) {
            const ar = aRe[i + kk * aR],
              br = bRe[kk + j * bR];
            if (aIm && bIm) {
              const ai = aIm[i + kk * aR],
                bi = bIm[kk + j * bR];
              sr += ar * br - ai * bi;
              si += ar * bi + ai * br;
            } else {
              sr += ar * br;
            }
          }
          result[outOff + i + j * aR] = sr;
          if (resultIm) resultIm[outOff + i + j * aR] = si;
        }
    }
    while (outShape.length > 2 && outShape[outShape.length - 1] === 1)
      outShape.pop();
    return maybeComplexTensor(result, outShape, resultIm);
  },
});

// ── pagetranspose ───────────────────────────────────────────────────────

builtin({
  name: "pagetranspose",
  nargin: 1,
  mfile: true,
  apply: (args, _nargout, ctx) => {
    const X = args[0];
    if (
      !(
        typeof X === "number" ||
        typeof X === "boolean" ||
        isRuntimeComplexNumber(X) ||
        isRuntimeTensor(X) ||
        isRuntimeSparseMatrix(X)
      )
    )
      throw new RuntimeError(
        "pagetranspose: input must be a numeric or logical array"
      );
    // Pages of a 2-D array: the array itself.
    if (!isRuntimeTensor(X) || X.shape.length <= 2)
      return ctx.rt.transpose(X) as RuntimeValue;
    const xShape = X.shape;
    const rows = xShape[0],
      cols = xShape[1];
    const extraDims = xShape.slice(2);
    const totalPages = extraDims.reduce((a: number, b: number) => a * b, 1);
    const pageSize = rows * cols;
    const outData = allocFloat64Array(X.data.length);
    const xIm = X.imag;
    const outIm = xIm ? allocFloat64Array(X.data.length) : undefined;
    const outShape = [cols, rows, ...extraDims];
    for (let p = 0; p < totalPages; p++) {
      const inOff = p * pageSize,
        outOff = p * pageSize;
      for (let j = 0; j < cols; j++)
        for (let i = 0; i < rows; i++) {
          outData[outOff + j + i * cols] = X.data[inOff + i + j * rows];
          if (outIm) outIm[outOff + j + i * cols] = xIm![inOff + i + j * rows];
        }
    }
    while (outShape.length > 2 && outShape[outShape.length - 1] === 1)
      outShape.pop();
    const out = RTV.tensor(outData, outShape, outIm, X.cls);
    return out;
  },
});

// ── qz ──────────────────────────────────────────────────────────────────

builtin({
  name: "qz",
  nargin: [2, 3],
  nargout: 6,
  apply: (args, nargout) => outputs(qzApply(args, nargout), nargout),
});

function qzApply(args: RuntimeValue[], nargout: number): RuntimeValue[] {
  let mode: "real" | "complex" = "complex";
  if (args.length >= 3) {
    const modeStr = parseStringArgLower(args[2]);
    if (modeStr === "complex") mode = "complex";
    else if (modeStr === "real") mode = "real";
    else throw new RuntimeError(`qz: unknown mode '${modeStr}'`);
  }
  const sparseError = errorFn(
    "MATLAB:qz:noQZsparse",
    "Sparse matrices not supported."
  );
  const A = denseFloatMatrixArg(args[0], mustBeFloatError, sparseError);
  const B = denseFloatMatrixArg(args[1], mustBeFloatError, sparseError);
  const [mA, nA] = tensorSize2D(A);
  const [mB, nB] = tensorSize2D(B);
  if (mA !== nA) throw new RuntimeError("qz: A must be a square matrix");
  if (mB !== nB) throw new RuntimeError("qz: B must be a square matrix");
  if (mA !== mB) throw new RuntimeError("qz: A and B must be the same size");
  const n = mA;
  const computeEigvecs = nargout >= 5;
  const isComplex = !!(A.imag || B.imag) || mode === "complex";
  const bridge = getLapackBridge();
  if (isComplex) {
    const nn = n * n;
    const aRe = toF64(A.data),
      aIm = A.imag ? toF64(A.imag) : allocFloat64Array(nn);
    const bRe = toF64(B.data),
      bIm = B.imag ? toF64(B.imag) : allocFloat64Array(nn);
    const result = bridge.qzComplex(aRe, aIm, bRe, bIm, n, computeEigvecs);
    const AAout = RTV.tensor(result.AARe, [n, n], result.AAIm);
    const BBout = RTV.tensor(result.BBRe, [n, n], result.BBIm);
    const Qout = RTV.tensor(result.QRe, [n, n], result.QIm);
    const Zout = RTV.tensor(result.ZRe, [n, n], result.ZIm);
    if (!computeEigvecs) return [AAout, BBout, Qout, Zout];
    const { VRe, VIm, WRe, WIm } = result;
    if (!VRe || !VIm || !WRe || !WIm)
      throw new RuntimeError("qz: failed to compute generalized eigenvectors");
    return [
      AAout,
      BBout,
      Qout,
      Zout,
      RTV.tensor(allocFloat64Array(VRe), [n, n], allocFloat64Array(VIm)),
      RTV.tensor(allocFloat64Array(WRe), [n, n], allocFloat64Array(WIm)),
    ];
  }
  const result = bridge.qz(toF64(A.data), toF64(B.data), n, computeEigvecs);
  const AAout = RTV.tensor(result.AA, [n, n]);
  const BBout = RTV.tensor(result.BB, [n, n]);
  const Qout = RTV.tensor(result.Q, [n, n]);
  const Zout = RTV.tensor(result.Z, [n, n]);
  if (!computeEigvecs) return [AAout, BBout, Qout, Zout];
  const { alphai, V, W } = result;
  if (!V || !W)
    throw new RuntimeError("qz: failed to compute generalized eigenvectors");
  let hasComplex = false;
  for (let i = 0; i < n; i++)
    if (Math.abs(alphai[i]) > 0) {
      hasComplex = true;
      break;
    }
  return [
    AAout,
    BBout,
    Qout,
    Zout,
    buildEigenvectorMatrix(V, alphai, n, hasComplex),
    buildEigenvectorMatrix(W, alphai, n, hasComplex),
  ];
}
