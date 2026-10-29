/**
 * Arithmetic, comparison, and related internal helpers.
 *
 * The element-wise operators classify their operands (strings, sparse
 * matrices, char, logical, complex scalars) and then make one call into the
 * element-wise kernel layer (`ops/elementwise.ts`), which owns broadcasting,
 * the real and complex formulas and the native dispatch.
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  type RuntimeCell,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeFunction,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  RuntimeClassInstanceArray,
  isRuntimeStringArray,
  RuntimeStringArray,
  isRuntimeStruct,
  isRuntimeStructArray,
  RuntimeStructArray,
  type RuntimeStruct,
  stringArrayValue,
} from "../runtime/types.js";
import { valueNumel } from "../runtime/kinds.js";
import { RuntimeError, invalidDataTypeError } from "../runtime/error.js";
import { RTV } from "../runtime/constructors.js";
import { tensorSize2D, colMajorIndex } from "../runtime/utils.js";
import { toNumber } from "../runtime/convert.js";
import { charToNumericTensor } from "../runtime/indexing.js";
import { getLapackBridge } from "../native/lapack-bridge.js";
import {
  OpBin,
  OpCmp,
  elementwiseBinary,
  elementwiseCompare,
  binaryScalar,
  compareScalar,
  type Operand,
  type ElemResult,
} from "../ops/index.js";
import { linsolveLapack, linsolveComplexLapack } from "./linsolve.js";
import { getIBuiltin } from "../runtime/builtinRegistry.js";
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
import { sparseSolve, denseDataToSparse } from "./sparse-direct.js";
import type { RuntimeSparseMatrix } from "../runtime/types.js";
import { emitWarning } from "./warn.js";
import { allocFloat64Array } from "../runtime/alloc.js";

// ── Operand classification ──────────────────────────────────────────────

const SCALAR_SHAPE: readonly number[] = [1, 1];

/** Whether `v` is a scalar value (not an array): a number, logical,
 *  complex number or single char. Element-wise results on two scalars are
 *  scalars; any array operand (including a 1×1 tensor) gives a tensor. */
function isScalarValue(v: RuntimeValue): boolean {
  return (
    typeof v === "number" ||
    typeof v === "boolean" ||
    isRuntimeComplexNumber(v) ||
    (isRuntimeChar(v) && v.value.length === 1 && isCharScalarShape(v))
  );
}

function isCharScalarShape(v: { shape?: number[] }): boolean {
  return v.shape === undefined || (v.shape[0] === 1 && v.shape[1] === 1);
}

/** Real part of a scalar value (see `isScalarValue`). */
function scalarRe(v: RuntimeValue): number {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (isRuntimeComplexNumber(v)) return v.re;
  return (v as { value: string }).value.charCodeAt(0);
}

/** A value as a kernel operand: a tensor as it is, a scalar as a number
 *  (complex as a 1-element array), char as its character codes, sparse
 *  densified. Raises the invalid-data-type error for other values. */
function operandOf(v: RuntimeValue): Operand {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (isRuntimeTensor(v)) return v;
  if (isRuntimeComplexNumber(v))
    return {
      data: Float64Array.of(v.re),
      imag: Float64Array.of(v.im),
      shape: SCALAR_SHAPE,
    };
  if (isRuntimeChar(v))
    return isScalarValue(v) ? scalarRe(v) : charToNumericTensor(v);
  if (isRuntimeSparseMatrix(v)) return sparseToDense(v);
  throw invalidDataTypeError();
}

function tensorOf(r: ElemResult): RuntimeValue {
  return RTV.tensor(r.data, r.shape, r.imag);
}

/** Element-wise `a op b` for an `OpBin` code on dense operands. */
function elemwise(op: number, a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isScalarValue(a) && isScalarValue(b)) {
    const ac = isRuntimeComplexNumber(a);
    const bc = isRuntimeComplexNumber(b);
    const r = binaryScalar(
      op,
      scalarRe(a),
      ac ? a.im : 0,
      ac,
      scalarRe(b),
      bc ? b.im : 0,
      bc
    );
    return r.im === 0 ? r.re : RTV.complex(r.re, r.im);
  }
  return tensorOf(elementwiseBinary(op, operandOf(a), operandOf(b)));
}

/** Element-wise `a op b` for an `OpCmp` code; the result is logical. */
function elemCompare(
  op: number,
  a: RuntimeValue,
  b: RuntimeValue
): RuntimeValue {
  if (isScalarValue(a) && isScalarValue(b)) {
    return RTV.logical(
      compareScalar(
        op,
        scalarRe(a),
        isRuntimeComplexNumber(a) ? a.im : 0,
        scalarRe(b),
        isRuntimeComplexNumber(b) ? b.im : 0
      )
    );
  }
  const r = elementwiseCompare(op, operandOf(a), operandOf(b));
  return RTV.logicalTensor(r.data, r.shape);
}

/** Element-wise logical `a & b`, `a | b` or `xor(a, b)` (an `OpCmp` code:
 *  AND, OR, XOR) on numeric, logical or char operands. */
export function mLogical(
  op: number,
  a: RuntimeValue,
  b: RuntimeValue
): RuntimeValue {
  return elemCompare(op, a, b);
}

/** A dense complex value whose imaginary part is all zero, as real (what
 *  `.*` and `./` treat it as). */
function realIfZeroImag(v: RuntimeValue): RuntimeValue {
  if (isRuntimeComplexNumber(v)) return v.im === 0 ? v.re : v;
  if (isRuntimeTensor(v) && v.imag) {
    const im = v.imag;
    for (let i = 0; i < im.length; i++) if (im[i] !== 0) return v;
    return RTV.tensorRaw(v.data, v.shape, v.cls);
  }
  return v;
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
    if (v.cls === "logical") return x ? "true" : "false";
    return num2strScalar(x);
  }
  return null;
}

/** Add two RuntimeValues */
export function mAdd(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  // String-array `+` appends elementwise (broadcast scalar against array).
  if (isRuntimeStringArray(a) || isRuntimeStringArray(b)) {
    const sa = isRuntimeStringArray(a) ? a : null;
    const sb = isRuntimeStringArray(b) ? b : null;
    if (sa && sb) {
      if (sa.shape[0] !== sb.shape[0] || sa.shape[1] !== sb.shape[1]) {
        throw new RuntimeError("Array dimensions must match for string plus");
      }
      return stringArrayValue(
        sa.data.map((s, i) => s + sb.data[i]),
        sa.shape
      );
    }
    const arr = (sa ?? sb)!;
    const other = coerceToConcatString(sa ? b : a);
    if (other === null) {
      throw new RuntimeError("Cannot concatenate this value with a string");
    }
    return stringArrayValue(
      arr.data.map(s => (sa ? s + other : other + s)),
      arr.shape
    );
  }
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
  if (isRuntimeSparseMatrix(a) && isRuntimeSparseMatrix(b))
    return mAddSparse(a, b);
  return elemwise(OpBin.ADD, a, b);
}

/** Subtract two RuntimeValues */
export function mSub(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) && isRuntimeSparseMatrix(b))
    return mSubSparse(a, b);
  return elemwise(OpBin.SUB, a, b);
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
  return elemwise(OpBin.MUL, a, b);
}

/** Element-wise multiply */
export function mElemMul(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b))
    return mElemMulSparse(realIfZeroImag(a), realIfZeroImag(b));
  return elemwise(OpBin.MUL, a, b);
}

/** Divide */
export function mDiv(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  // Matrix right division: A / B = (B' \ A')' (uses mldivide)
  // When B is a matrix, always use matrix division (scalar A is promoted);
  // mLeftDiv will error on dimension mismatch, matching MATLAB behaviour.
  if (isRuntimeSparseMatrix(a) || isRuntimeSparseMatrix(b)) {
    // S / scalar stays sparse (element-wise scaling).
    if (
      isRuntimeSparseMatrix(a) &&
      (isRuntimeNumber(b) ||
        isRuntimeLogical(b) ||
        isRuntimeComplexNumber(b) ||
        (isRuntimeTensor(b) && b.data.length === 1))
    ) {
      const r = mElemDivSparse(a, b);
      if (r !== null) return r;
    }
    // A / S with a sparse divisor: (S' \ A')' keeps the solve sparse.
    if (isRuntimeSparseMatrix(b) && !(b.m === 1 && b.n === 1)) {
      const result = mLeftDiv(mConjugateTranspose(b), mConjugateTranspose(a));
      return mConjugateTranspose(result);
    }
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
  // Scalar divisor: element-wise division
  return elemwise(OpBin.DIV, a, b);
}

/** Element-wise divide */
export function mElemDiv(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a)) {
    const r = mElemDivSparse(a, realIfZeroImag(b));
    if (r !== null) return r;
  }
  return elemwise(OpBin.DIV, a, b);
}

/** Sparse A \ B via the sparse direct solvers. Returns null when the dense
 *  path should be used instead (complex operands, non-square or nearly full
 *  A, or a 1×1 sparse scalar). */
function sparseLeftDiv(
  A: RuntimeSparseMatrix,
  b: RuntimeValue
): RuntimeValue | null {
  if (A.pi) return null;
  if (A.m === 1 && A.n === 1) return null;
  let B: Float64Array;
  let rows: number, nrhs: number;
  const bSparse = isRuntimeSparseMatrix(b);
  if (isRuntimeSparseMatrix(b)) {
    if (b.pi) return null;
    const d = sparseToDense(b);
    B = d.data;
    rows = b.m;
    nrhs = b.n;
  } else if (isRuntimeTensor(b)) {
    if (b.imag) return null;
    [rows, nrhs] = tensorSize2D(b);
    B = b.data;
  } else if (isRuntimeNumber(b) || isRuntimeLogical(b)) {
    B = allocFloat64Array([toNumber(b)]);
    rows = 1;
    nrhs = 1;
  } else {
    return null;
  }
  if (rows !== A.m)
    throw new RuntimeError(
      `mldivide (\\): A is ${A.m}×${A.n} but B has ${rows} rows`
    );
  if (A.m !== A.n) {
    emitWarning(
      "mldivide: sparse least-squares uses a dense factorization; " +
        "the sparse matrix is converted to full."
    );
    return null;
  }
  const X = sparseSolve(A, B, nrhs);
  if (X === null) return null;
  if (bSparse) return denseDataToSparse(X, A.n, nrhs);
  return unwrap1x1(RTV.tensor(X, [A.n, nrhs]));
}

/** Left division (mldivide): a \ b — for scalars b/a, for matrices solve a*x = b */
export function mLeftDiv(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  // Sparse A: direct sparse solve (triangular / Cholesky / LU). Falls back
  // to the dense path for complex, non-square, or nearly full A.
  if (isRuntimeSparseMatrix(a)) {
    const r = sparseLeftDiv(a, b);
    if (r !== null) return r;
    return mLeftDiv(densify(a), densify(b));
  }
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
      ? RTV.tensor(allocFloat64Array([toNumber(b)]), [1, 1])
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
    return RTV.tensor(allocFloat64Array(nA * nB), [nA, nB]);
  }
  if (tensorA.imag || tensorB.imag) {
    const ARe = tensorA.data;
    const AIm = tensorA.imag ?? allocFloat64Array(tensorA.data.length);
    const BRe = tensorB.data;
    const BIm = tensorB.imag ?? allocFloat64Array(tensorB.data.length);
    const X = linsolveComplexLapack(ARe, AIm, mA, nA, BRe, BIm, nB);
    return unwrap1x1(RTV.tensor(X.re, [nA, nB], X.im));
  }
  const X = linsolveLapack(tensorA.data, mA, nA, tensorB.data, nB);
  return unwrap1x1(RTV.tensor(X, [nA, nB]));
}

/** Element-wise left division (ldivide): a .\ b = b ./ a */
export function mElemLeftDiv(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  return mElemDiv(b, a);
}

/** Matrix power `a ^ b` (mpower): element-wise for two scalars, repeated
 *  multiplication for a square matrix to an integer power, and a matrix
 *  function through the eigendecomposition for a square matrix to a
 *  non-integer or complex power and for a scalar to a square matrix power.
 *  Anything else is MATLAB's dimension error. */
export function mPow(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isScalarValue(a) && isScalarValue(b)) return elemwise(OpBin.POW, a, b);
  if (isRuntimeSparseMatrix(a)) a = densify(a);
  if (isRuntimeSparseMatrix(b)) b = densify(b);
  const ta = powOperand(a);
  const tb = powOperand(b);
  if (ta.shape.length > 2 || tb.shape.length > 2)
    throw new RuntimeError(
      "Arguments must be 2-D. To operate on each element of the matrix individually, use POWER (.^) for elementwise power.",
      null,
      { identifier: "MATLAB:mpower:inputsMustBe2D" }
    );
  const aScalar = ta.data.length === 1;
  const bScalar = tb.data.length === 1;
  const square = (t: RuntimeTensor) => t.shape[0] === t.shape[1];
  if (aScalar && bScalar)
    return elemwise(OpBin.POW, unwrap1x1(ta), unwrap1x1(tb));
  if (bScalar && square(ta)) return matrixToPower(ta, tb);
  if (aScalar && square(tb)) return scalarToMatrixPower(ta, tb);
  throw new RuntimeError(
    "Incorrect dimensions for raising a matrix to a power. Check that the matrix is square and the power is a scalar. To operate on each element of the matrix individually, use POWER (.^) for elementwise power.",
    null,
    { identifier: "MATLAB:mpower:notScalarAndSquareMatrix" }
  );
}

/** An mpower operand as a dense tensor. */
function powOperand(v: RuntimeValue): RuntimeTensor {
  if (isRuntimeChar(v)) return charToNumericTensor(v);
  if (
    isRuntimeTensor(v) ||
    isRuntimeNumber(v) ||
    isRuntimeLogical(v) ||
    isRuntimeComplexNumber(v)
  )
    return coerceToTensor(v, "mpower");
  throw invalidDataTypeError();
}

/** `A ^ p` for a square matrix `A` and a 1x1 `p`. */
function matrixToPower(A: RuntimeTensor, p: RuntimeTensor): RuntimeValue {
  const n = A.shape[0];
  const pRe = p.data[0];
  const pIm = p.imag ? p.imag[0] : 0;
  if (n === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
  if (pRe !== pRe || pIm !== pIm)
    return RTV.tensor(allocFloat64Array(n * n).fill(NaN), [n, n]);
  if (!Number.isFinite(pRe) || !Number.isFinite(pIm))
    throw new RuntimeError(
      "mpower: an infinite power of a matrix is not supported"
    );
  if (pIm === 0 && Number.isInteger(pRe)) {
    if (pRe === 0) {
      const data = allocFloat64Array(n * n);
      for (let i = 0; i < n; i++) data[colMajorIndex(i, i, n)] = 1;
      return RTV.tensor(data, [n, n]);
    }
    let base: RuntimeTensor = A;
    let k = pRe;
    if (k < 0) {
      // A^(-k) = inv(A)^k
      const invA = getIBuiltin("inv")!.call([A], 1) as RuntimeValue;
      if (!isRuntimeTensor(invA))
        throw new RuntimeError("mpower: inv returned non-tensor");
      base = invA;
      k = -k;
    }
    // Binary powering: k = 1e300 takes ~1000 products, not 1e300.
    let result: RuntimeTensor | null = null;
    let sq = base;
    for (;;) {
      if (k % 2 === 1)
        result = result ? (matMul(result, sq) as RuntimeTensor) : sq;
      k = Math.floor(k / 2);
      if (k === 0) break;
      sq = matMul(sq, sq) as RuntimeTensor;
    }
    return result!;
  }
  // A real matrix to a real power is real unless an eigenvalue lies on the
  // branch cut (the negative real axis).
  const realInputs = !A.imag && pIm === 0;
  return eigMatrixFunction(
    A,
    (re, im) => binaryScalar(OpBin.POW, re, im, true, pRe, pIm, true),
    lambdas => realInputs && !lambdas.some(l => l.im === 0 && l.re < 0)
  );
}

/** `s ^ B` for a 1x1 `s` and a square matrix `B`. */
function scalarToMatrixPower(s: RuntimeTensor, B: RuntimeTensor): RuntimeValue {
  if (B.shape[0] === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
  const sRe = s.data[0];
  const sIm = s.imag ? s.imag[0] : 0;
  const realResult = !B.imag && sIm === 0 && sRe > 0;
  return eigMatrixFunction(
    B,
    (re, im) => binaryScalar(OpBin.POW, sRe, sIm, true, re, im, true),
    () => realResult
  );
}

/** f(A) = V * diag(f(lambda)) / V from the eigendecomposition of a square
 *  matrix. Raises a not-supported error when `A` is not diagonalizable to
 *  working accuracy (MATLAB uses a Schur method there). `isReal` says,
 *  given the eigenvalues, that the exact result is real, so roundoff in
 *  its imaginary part is dropped. */
function eigMatrixFunction(
  A: RuntimeTensor,
  f: (re: number, im: number) => { re: number; im: number },
  isReal: (lambdas: { re: number; im: number }[]) => boolean
): RuntimeValue {
  const n = A.shape[0];
  const [V, D] = getIBuiltin("eig")!.call([A], 2) as RuntimeValue[];
  const Vt = coerceToTensor(V, "mpower");
  const Dt = coerceToTensor(D, "mpower");
  const rc = toNumber(getIBuiltin("rcond")!.call([Vt], 1) as RuntimeValue);
  if (!(rc >= Math.sqrt(Number.EPSILON)))
    throw new RuntimeError(
      "mpower: a non-integer or matrix power of a matrix that is not diagonalizable is not supported"
    );
  const lambdas: { re: number; im: number }[] = [];
  for (let k = 0; k < n; k++) {
    const j = colMajorIndex(k, k, n);
    lambdas.push({ re: Dt.data[j], im: Dt.imag ? Dt.imag[j] : 0 });
  }
  // W = V * diag(f(lambda)), column by column.
  const wRe = allocFloat64Array(n * n);
  const wIm = allocFloat64Array(n * n);
  for (let k = 0; k < n; k++) {
    const fk = f(lambdas[k].re, lambdas[k].im);
    const fRe = fk.re;
    const fIm = fk.im;
    for (let i = 0; i < n; i++) {
      const j = colMajorIndex(i, k, n);
      const vRe = Vt.data[j];
      const vIm = Vt.imag ? Vt.imag[j] : 0;
      wRe[j] = vRe * fRe - vIm * fIm;
      wIm[j] = vRe * fIm + vIm * fRe;
    }
  }
  const F = coerceToTensor(mDiv(RTV.tensor(wRe, [n, n], wIm), Vt), "mpower");
  if (isReal(lambdas) || !F.imag) return RTV.tensor(F.data, [n, n]);
  return realIfZeroImag(F);
}

/** Element-wise power */
export function mElemPow(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  return elemwise(OpBin.POW, a, b);
}

/** Negation */
export function mNeg(v: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(v)) return sparseNeg(v);
  if (isRuntimeComplexNumber(v)) return RTV.complex(-v.re, -v.im);
  const x = operandOf(v);
  if (typeof x === "number") return -x;
  const n = x.data.length;
  const re = allocFloat64Array(n);
  for (let i = 0; i < n; i++) re[i] = -x.data[i];
  let im: Float64Array | undefined;
  if (x.imag) {
    im = allocFloat64Array(n);
    for (let i = 0; i < n; i++) im[i] = -x.imag[i];
  }
  return RTV.tensor(re, [...x.shape], im);
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
  const resultRe = allocFloat64Array(v.data.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      resultRe[r * cols + c] = v.data[colMajorIndex(r, c, rows)];
    }
  }
  if (v.imag !== undefined) {
    const resultIm = allocFloat64Array(v.imag.length);
    const sign = conjugate ? -1 : 1;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        resultIm[r * cols + c] = sign * v.imag[colMajorIndex(r, c, rows)];
      }
    }
    return RTV.tensor(resultRe, [cols, rows], resultIm);
  }
  const t = RTV.tensor(resultRe, [cols, rows], undefined, v.cls);
  return t;
}

/** Transpose an object array: swap the 2-D shape and permute the
 *  column-major elements so out(j,i) = in(i,j). For a scalar or vector this
 *  leaves the element order unchanged and only flips the orientation. */
function transposeClassInstanceArray(
  v: RuntimeClassInstanceArray
): RuntimeValue {
  const [r, c] = v.shape;
  const out = new Array<(typeof v.elements)[number]>(v.elements.length);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      out[i * c + j] = v.elements[j * r + i];
    }
  }
  return new RuntimeClassInstanceArray(v.className, out, [c, r]);
}

/** Transpose a 2-D struct array by permuting its elements. */
function transposeStructArray(v: RuntimeStructArray): RuntimeValue {
  if (v.shape.length > 2)
    throw new RuntimeError("Transpose on ND array is not defined.");
  const [r, c] = v.shape;
  const out = new Array<RuntimeStruct>(v.elements.length);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      out[i * c + j] = v.elements[j * r + i];
    }
  }
  return new RuntimeStructArray(v.fieldNames.slice(), out, [c, r]);
}

/** Transpose a string array by permuting elements (strings have no
 *  elementwise conjugate). */
function transposeStringArray(v: RuntimeStringArray): RuntimeValue {
  const [r, c] = v.shape;
  const out: string[] = new Array(r * c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      out[i * c + j] = v.data[j * r + i];
    }
  }
  return stringArrayValue(out, [c, r]);
}

/** Transpose of a scalar object of another kind (a dictionary or graphics
 *  handle is an identity); a function handle cannot be transposed. */
function transposeOther(v: RuntimeValue): RuntimeValue {
  if (isRuntimeFunction(v) || valueNumel(v) !== 1)
    throw new RuntimeError("Data type not supported.", null, {
      identifier: "MATLAB:transpose:unsupportedType",
    });
  return v;
}

/** Transpose (non-conjugate for complex scalars and tensors) */
export function mTranspose(v: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(v)) return sparseTranspose(v);
  if (isRuntimeComplexNumber(v)) return v;
  if (isRuntimeNumber(v) || isRuntimeLogical(v)) return v;
  if (isRuntimeCell(v)) return transposeCellArray(v);
  if (isRuntimeChar(v)) return v;
  if (isRuntimeString(v)) return v;
  if (isRuntimeStringArray(v)) return transposeStringArray(v);
  // Object values without a class `transpose` method: MATLAB's built-in
  // array transpose permutes the elements. Reached only as the fallback
  // after method dispatch declines. A scalar instance is an identity.
  if (isRuntimeClassInstance(v)) return v;
  if (isRuntimeClassInstanceArray(v)) return transposeClassInstanceArray(v);
  if (isRuntimeStruct(v)) return v;
  if (isRuntimeStructArray(v)) return transposeStructArray(v);
  if (!isRuntimeTensor(v)) return transposeOther(v);
  return transposeCore(v, false);
}

/** Conjugate transpose (Hermitian transpose) - transposes and conjugates */
export function mConjugateTranspose(v: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(v)) return sparseConjugateTranspose(v);
  if (isRuntimeComplexNumber(v)) return RTV.complex(v.re, -v.im);
  if (isRuntimeNumber(v) || isRuntimeLogical(v)) return v;
  if (isRuntimeCell(v)) return transposeCellArray(v);
  if (isRuntimeChar(v)) return v;
  if (isRuntimeString(v)) return v;
  if (isRuntimeStringArray(v)) return transposeStringArray(v);
  // Object values without a class `ctranspose` method: the default array
  // operation rearranges elements (objects are not element-wise conjugated),
  // identical to mTranspose. A scalar instance is an identity.
  if (isRuntimeClassInstance(v)) return v;
  if (isRuntimeClassInstanceArray(v)) return transposeClassInstanceArray(v);
  if (isRuntimeStruct(v)) return v;
  if (isRuntimeStructArray(v)) return transposeStructArray(v);
  if (!isRuntimeTensor(v)) return transposeOther(v);
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

/** View a text operand as elements + shape for elementwise comparison
 *  (string scalar / single-row char / string array). */
function textOperandElems(
  v: RuntimeValue
): { data: string[]; shape: [number, number] } | null {
  if (isRuntimeString(v)) return { data: [v], shape: [1, 1] };
  if (isRuntimeChar(v)) {
    const rows = v.shape ? v.shape[0] : 1;
    if (rows > 1) return null;
    return { data: [v.value], shape: [1, 1] };
  }
  if (isRuntimeStringArray(v)) return { data: v.data, shape: v.shape };
  return null;
}

function stringComparisonOp(
  a: RuntimeValue,
  b: RuntimeValue,
  op: (x: string, y: string) => boolean
): RuntimeValue | null {
  // Elementwise comparison when either side is a string array.
  if (isRuntimeStringArray(a) || isRuntimeStringArray(b)) {
    const ta = textOperandElems(a);
    const tb = textOperandElems(b);
    if (ta === null || tb === null) return null;
    const aScalar = ta.data.length === 1 && ta.shape[0] * ta.shape[1] === 1;
    const bScalar = tb.data.length === 1 && tb.shape[0] * tb.shape[1] === 1;
    if (
      !aScalar &&
      !bScalar &&
      (ta.shape[0] !== tb.shape[0] || ta.shape[1] !== tb.shape[1])
    ) {
      throw new RuntimeError("Array dimensions must match for comparison");
    }
    const shape = aScalar ? tb.shape : ta.shape;
    const n = aScalar ? tb.data.length : ta.data.length;
    const out = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      const x = aScalar ? ta.data[0] : ta.data[i];
      const y = bScalar ? tb.data[0] : tb.data[i];
      out[i] = op(x, y) ? 1 : 0;
    }
    const t = RTV.logicalTensor(out, [shape[0], shape[1]]);
    return t;
  }
  const pair = asStringPair(a, b);
  if (pair === null) return null;
  return RTV.logical(op(pair[0], pair[1]));
}

/** Convert sparse to dense for fallback paths. */
function densify(v: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(v)) return sparseToDense(v);
  return v;
}

/** A comparison operator: string operands compare as text, everything else
 *  numerically (sparse densified). */
function compareOp(
  op: number,
  a: RuntimeValue,
  b: RuntimeValue,
  textOp: (x: string, y: string) => boolean
): RuntimeValue {
  a = densify(a);
  b = densify(b);
  const sr = stringComparisonOp(a, b, textOp);
  if (sr !== null) return sr;
  return elemCompare(op, a, b);
}

export function mEqual(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  return compareOp(OpCmp.EQ, a, b, (x, y) => x === y);
}

export function mNotEqual(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  return compareOp(OpCmp.NE, a, b, (x, y) => x !== y);
}

export function mLess(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  return compareOp(OpCmp.LT, a, b, (x, y) => x < y);
}

export function mLessEqual(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  return compareOp(OpCmp.LE, a, b, (x, y) => x <= y);
}

export function mGreater(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  return compareOp(OpCmp.GT, a, b, (x, y) => x > y);
}

export function mGreaterEqual(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  return compareOp(OpCmp.GE, a, b, (x, y) => x >= y);
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
    return RTV.tensor(allocFloat64Array(aRows * bCols), [aRows, bCols]);
  }

  const isComplex = a.imag !== undefined || b.imag !== undefined;

  const bridge = getLapackBridge();
  if (!isComplex) {
    const raw = bridge.matmul(a.data, aRows, aCols, b.data, bCols);
    return unwrap1x1(RTV.tensor(raw, [aRows, bCols]));
  }

  const aIm = a.imag || allocFloat64Array(a.data.length);
  const bIm = b.imag || allocFloat64Array(b.data.length);
  const raw = bridge.matmulComplex(
    a.data,
    aIm,
    aRows,
    aCols,
    b.data,
    bIm,
    bCols
  );
  return unwrap1x1(RTV.tensor(raw.re, [aRows, bCols], raw.im));
}
