/**
 * Sparse matrix arithmetic operations (with complex support).
 */

import {
  type RuntimeValue,
  type RuntimeSparseMatrix,
  type RuntimeTensor,
  FloatXArray,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeSparseMatrix,
  isRuntimeComplexNumber,
} from "../runtime/types.js";
import { RuntimeError } from "../runtime/error.js";
import { RTV } from "../runtime/constructors.js";
import { tensorSize2D } from "../runtime/utils.js";

// ── Helpers ─────────────────────────────────────────────────────────────

/** Is this sparse matrix complex? */
function isComplexSparse(S: RuntimeSparseMatrix): boolean {
  return S.pi !== undefined;
}

/** Get pr[k] for a sparse matrix (always valid). */
function sRe(S: RuntimeSparseMatrix, k: number): number {
  return S.pr[k];
}
/** Get pi[k] for a sparse matrix (0 if real). */
function sIm(S: RuntimeSparseMatrix, k: number): number {
  return S.pi ? S.pi[k] : 0;
}

/** Check if a complex value is zero. */
function isZero(re: number, im: number): boolean {
  return re === 0 && im === 0;
}

/** Build optional pi array: return undefined if all zeros. */
function maybePi(piList: number[]): Float64Array | undefined {
  for (let i = 0; i < piList.length; i++) {
    if (piList[i] !== 0) return new Float64Array(piList);
  }
  return undefined;
}

/** Convert a sparse matrix to dense. */
function sparseToDense(S: RuntimeSparseMatrix): RuntimeTensor {
  const data = new FloatXArray(S.m * S.n);
  const imag = S.pi ? new FloatXArray(S.m * S.n) : undefined;
  for (let col = 0; col < S.n; col++) {
    for (let k = S.jc[col]; k < S.jc[col + 1]; k++) {
      const idx = col * S.m + S.ir[k];
      data[idx] = S.pr[k];
      if (imag && S.pi) imag[idx] = S.pi[k];
    }
  }
  return { kind: "tensor", data, imag, shape: [S.m, S.n], _rc: 1 };
}

/** Get scalar value (real) from a RuntimeValue, or NaN if not scalar. */
function scalarVal(v: RuntimeValue): number {
  if (isRuntimeNumber(v)) return v;
  if (isRuntimeLogical(v)) return v ? 1 : 0;
  if (isRuntimeTensor(v) && v.data.length === 1 && !v.imag) return v.data[0];
  return NaN;
}

/** Get complex scalar from a RuntimeValue. Returns null if not scalar. */
function complexScalarVal(v: RuntimeValue): { re: number; im: number } | null {
  if (isRuntimeNumber(v)) return { re: v, im: 0 };
  if (isRuntimeLogical(v)) return { re: v ? 1 : 0, im: 0 };
  if (isRuntimeComplexNumber(v)) return { re: v.re, im: v.im };
  if (isRuntimeTensor(v) && v.data.length === 1)
    return { re: v.data[0], im: v.imag ? v.imag[0] : 0 };
  return null;
}

// ── Sparse + Sparse ─────────────────────────────────────────────────────

export function sparseAdd(
  a: RuntimeSparseMatrix,
  b: RuntimeSparseMatrix
): RuntimeSparseMatrix {
  if (a.m !== b.m || a.n !== b.n)
    throw new RuntimeError(
      `Matrix dimensions must agree: [${a.m},${a.n}] vs [${b.m},${b.n}]`
    );
  const hasImag = isComplexSparse(a) || isComplexSparse(b);
  const irList: number[] = [];
  const prList: number[] = [];
  const piList: number[] = [];
  const jc = new Int32Array(a.n + 1);

  for (let col = 0; col < a.n; col++) {
    jc[col] = irList.length;
    let ai = a.jc[col],
      bi = b.jc[col];
    const aEnd = a.jc[col + 1],
      bEnd = b.jc[col + 1];
    while (ai < aEnd && bi < bEnd) {
      if (a.ir[ai] < b.ir[bi]) {
        irList.push(a.ir[ai]);
        prList.push(sRe(a, ai));
        piList.push(sIm(a, ai));
        ai++;
      } else if (a.ir[ai] > b.ir[bi]) {
        irList.push(b.ir[bi]);
        prList.push(sRe(b, bi));
        piList.push(sIm(b, bi));
        bi++;
      } else {
        const re = sRe(a, ai) + sRe(b, bi);
        const im = sIm(a, ai) + sIm(b, bi);
        if (!isZero(re, im)) {
          irList.push(a.ir[ai]);
          prList.push(re);
          piList.push(im);
        }
        ai++;
        bi++;
      }
    }
    while (ai < aEnd) {
      irList.push(a.ir[ai]);
      prList.push(sRe(a, ai));
      piList.push(sIm(a, ai));
      ai++;
    }
    while (bi < bEnd) {
      irList.push(b.ir[bi]);
      prList.push(sRe(b, bi));
      piList.push(sIm(b, bi));
      bi++;
    }
  }
  jc[a.n] = irList.length;
  return RTV.sparseMatrix(
    a.m,
    a.n,
    new Int32Array(irList),
    jc,
    new Float64Array(prList),
    hasImag ? maybePi(piList) : undefined
  );
}

// ── Sparse - Sparse ─────────────────────────────────────────────────────

export function sparseSub(
  a: RuntimeSparseMatrix,
  b: RuntimeSparseMatrix
): RuntimeSparseMatrix {
  if (a.m !== b.m || a.n !== b.n)
    throw new RuntimeError(
      `Matrix dimensions must agree: [${a.m},${a.n}] vs [${b.m},${b.n}]`
    );
  const hasImag = isComplexSparse(a) || isComplexSparse(b);
  const irList: number[] = [];
  const prList: number[] = [];
  const piList: number[] = [];
  const jc = new Int32Array(a.n + 1);

  for (let col = 0; col < a.n; col++) {
    jc[col] = irList.length;
    let ai = a.jc[col],
      bi = b.jc[col];
    const aEnd = a.jc[col + 1],
      bEnd = b.jc[col + 1];
    while (ai < aEnd && bi < bEnd) {
      if (a.ir[ai] < b.ir[bi]) {
        irList.push(a.ir[ai]);
        prList.push(sRe(a, ai));
        piList.push(sIm(a, ai));
        ai++;
      } else if (a.ir[ai] > b.ir[bi]) {
        irList.push(b.ir[bi]);
        prList.push(-sRe(b, bi));
        piList.push(-sIm(b, bi));
        bi++;
      } else {
        const re = sRe(a, ai) - sRe(b, bi);
        const im = sIm(a, ai) - sIm(b, bi);
        if (!isZero(re, im)) {
          irList.push(a.ir[ai]);
          prList.push(re);
          piList.push(im);
        }
        ai++;
        bi++;
      }
    }
    while (ai < aEnd) {
      irList.push(a.ir[ai]);
      prList.push(sRe(a, ai));
      piList.push(sIm(a, ai));
      ai++;
    }
    while (bi < bEnd) {
      irList.push(b.ir[bi]);
      prList.push(-sRe(b, bi));
      piList.push(-sIm(b, bi));
      bi++;
    }
  }
  jc[a.n] = irList.length;
  return RTV.sparseMatrix(
    a.m,
    a.n,
    new Int32Array(irList),
    jc,
    new Float64Array(prList),
    hasImag ? maybePi(piList) : undefined
  );
}

// ── Sparse * complex scalar ─────────────────────────────────────────────

function sparseScaleComplex(
  S: RuntimeSparseMatrix,
  sRe_: number,
  sIm_: number
): RuntimeSparseMatrix {
  if (sRe_ === 0 && sIm_ === 0) {
    return RTV.sparseMatrix(
      S.m,
      S.n,
      new Int32Array(0),
      new Int32Array(S.n + 1),
      new Float64Array(0)
    );
  }
  const nnz = S.pr.length;
  const pr = new Float64Array(nnz);
  const pi = new Float64Array(nnz);
  for (let i = 0; i < nnz; i++) {
    const aRe = S.pr[i];
    const aIm = S.pi ? S.pi[i] : 0;
    pr[i] = aRe * sRe_ - aIm * sIm_;
    pi[i] = aRe * sIm_ + aIm * sRe_;
  }
  return RTV.sparseMatrix(
    S.m,
    S.n,
    new Int32Array(S.ir),
    new Int32Array(S.jc),
    pr,
    maybePi(Array.from(pi))
  );
}

// ── Sparse * real scalar ────────────────────────────────────────────────

export function sparseScale(
  S: RuntimeSparseMatrix,
  scalar: number
): RuntimeSparseMatrix {
  if (scalar === 0) {
    return RTV.sparseMatrix(
      S.m,
      S.n,
      new Int32Array(0),
      new Int32Array(S.n + 1),
      new Float64Array(0)
    );
  }
  const pr = new Float64Array(S.pr.length);
  for (let i = 0; i < pr.length; i++) pr[i] = S.pr[i] * scalar;
  const pi = S.pi ? new Float64Array(S.pi.length) : undefined;
  if (pi && S.pi) {
    for (let i = 0; i < pi.length; i++) pi[i] = S.pi[i] * scalar;
  }
  return RTV.sparseMatrix(
    S.m,
    S.n,
    new Int32Array(S.ir),
    new Int32Array(S.jc),
    pr,
    pi
  );
}

// ── -S (negation) ───────────────────────────────────────────────────────

export function sparseNeg(S: RuntimeSparseMatrix): RuntimeSparseMatrix {
  const pr = new Float64Array(S.pr.length);
  for (let i = 0; i < pr.length; i++) pr[i] = -S.pr[i];
  const pi = S.pi ? new Float64Array(S.pi.length) : undefined;
  if (pi && S.pi) {
    for (let i = 0; i < pi.length; i++) pi[i] = -S.pi[i];
  }
  return RTV.sparseMatrix(
    S.m,
    S.n,
    new Int32Array(S.ir),
    new Int32Array(S.jc),
    pr,
    pi
  );
}

// ── Transpose (non-conjugate) ───────────────────────────────────────────

export function sparseTranspose(S: RuntimeSparseMatrix): RuntimeSparseMatrix {
  const nnz = S.jc[S.n];
  const tIr = new Int32Array(nnz);
  const tJc = new Int32Array(S.m + 1);
  const tPr = new Float64Array(nnz);
  const tPi = S.pi ? new Float64Array(nnz) : undefined;

  // Count entries per row of S (= per column of S')
  for (let k = 0; k < nnz; k++) tJc[S.ir[k] + 1]++;
  // Cumulative sum to get column pointers
  for (let i = 0; i < S.m; i++) tJc[i + 1] += tJc[i];
  // Fill values
  const pos = new Int32Array(S.m);
  for (let i = 0; i < S.m; i++) pos[i] = tJc[i];
  for (let col = 0; col < S.n; col++) {
    for (let k = S.jc[col]; k < S.jc[col + 1]; k++) {
      const row = S.ir[k];
      const dest = pos[row]++;
      tIr[dest] = col;
      tPr[dest] = S.pr[k];
      if (tPi && S.pi) tPi[dest] = S.pi[k];
    }
  }
  return RTV.sparseMatrix(S.n, S.m, tIr, tJc, tPr, tPi);
}

// ── Conjugate transpose ─────────────────────────────────────────────────

export function sparseConjugateTranspose(
  S: RuntimeSparseMatrix
): RuntimeSparseMatrix {
  if (!S.pi) return sparseTranspose(S); // real: same as transpose
  const nnz = S.jc[S.n];
  const tIr = new Int32Array(nnz);
  const tJc = new Int32Array(S.m + 1);
  const tPr = new Float64Array(nnz);
  const tPi = new Float64Array(nnz);

  for (let k = 0; k < nnz; k++) tJc[S.ir[k] + 1]++;
  for (let i = 0; i < S.m; i++) tJc[i + 1] += tJc[i];
  const pos = new Int32Array(S.m);
  for (let i = 0; i < S.m; i++) pos[i] = tJc[i];
  for (let col = 0; col < S.n; col++) {
    for (let k = S.jc[col]; k < S.jc[col + 1]; k++) {
      const row = S.ir[k];
      const dest = pos[row]++;
      tIr[dest] = col;
      tPr[dest] = S.pr[k];
      tPi[dest] = -S.pi![k]; // conjugate
    }
  }
  return RTV.sparseMatrix(S.n, S.m, tIr, tJc, tPr, maybePi(Array.from(tPi)));
}

// ── Sparse * Sparse (matrix multiply) ───────────────────────────────────

export function sparseMatMul(
  A: RuntimeSparseMatrix,
  B: RuntimeSparseMatrix
): RuntimeSparseMatrix {
  if (A.n !== B.m)
    throw new RuntimeError(
      `Inner matrix dimensions must agree: ${A.n} vs ${B.m}`
    );
  const m = A.m;
  const n = B.n;
  const hasImag = isComplexSparse(A) || isComplexSparse(B);
  const irList: number[] = [];
  const prList: number[] = [];
  const piList: number[] = [];
  const jc = new Int32Array(n + 1);
  const accRe = new Float64Array(m);
  const accIm = hasImag ? new Float64Array(m) : null;
  const marker = new Int32Array(m).fill(-1);

  for (let col = 0; col < n; col++) {
    jc[col] = irList.length;
    const rowList: number[] = [];

    for (let kb = B.jc[col]; kb < B.jc[col + 1]; kb++) {
      const kB = B.ir[kb];
      const vBRe = sRe(B, kb);
      const vBIm = hasImag ? sIm(B, kb) : 0;
      for (let ka = A.jc[kB]; ka < A.jc[kB + 1]; ka++) {
        const row = A.ir[ka];
        const aRe = sRe(A, ka);
        const aIm = hasImag ? sIm(A, ka) : 0;
        // (aRe + aIm*i) * (vBRe + vBIm*i)
        const prodRe = aRe * vBRe - aIm * vBIm;
        const prodIm = aRe * vBIm + aIm * vBRe;
        if (marker[row] !== col) {
          marker[row] = col;
          accRe[row] = prodRe;
          if (accIm) accIm[row] = prodIm;
          rowList.push(row);
        } else {
          accRe[row] += prodRe;
          if (accIm) accIm[row] += prodIm;
        }
      }
    }
    rowList.sort((a, b) => a - b);
    for (const row of rowList) {
      const re = accRe[row];
      const im = accIm ? accIm[row] : 0;
      if (!isZero(re, im)) {
        irList.push(row);
        prList.push(re);
        piList.push(im);
      }
    }
  }
  jc[n] = irList.length;
  return RTV.sparseMatrix(
    m,
    n,
    new Int32Array(irList),
    jc,
    new Float64Array(prList),
    hasImag ? maybePi(piList) : undefined
  );
}

// ── Element-wise multiply (Sparse .* Sparse) ───────────────────────────

export function sparseElemMul(
  a: RuntimeSparseMatrix,
  b: RuntimeSparseMatrix
): RuntimeSparseMatrix {
  if (a.m !== b.m || a.n !== b.n)
    throw new RuntimeError(
      `Matrix dimensions must agree: [${a.m},${a.n}] vs [${b.m},${b.n}]`
    );
  const hasImag = isComplexSparse(a) || isComplexSparse(b);
  const irList: number[] = [];
  const prList: number[] = [];
  const piList: number[] = [];
  const jc = new Int32Array(a.n + 1);

  for (let col = 0; col < a.n; col++) {
    jc[col] = irList.length;
    let ai = a.jc[col],
      bi = b.jc[col];
    const aEnd = a.jc[col + 1],
      bEnd = b.jc[col + 1];
    while (ai < aEnd && bi < bEnd) {
      if (a.ir[ai] < b.ir[bi]) {
        ai++;
      } else if (a.ir[ai] > b.ir[bi]) {
        bi++;
      } else {
        const aRe = sRe(a, ai),
          aIm = sIm(a, ai);
        const bRe = sRe(b, bi),
          bIm = sIm(b, bi);
        const re = aRe * bRe - aIm * bIm;
        const im = aRe * bIm + aIm * bRe;
        if (!isZero(re, im)) {
          irList.push(a.ir[ai]);
          prList.push(re);
          piList.push(im);
        }
        ai++;
        bi++;
      }
    }
  }
  jc[a.n] = irList.length;
  return RTV.sparseMatrix(
    a.m,
    a.n,
    new Int32Array(irList),
    jc,
    new Float64Array(prList),
    hasImag ? maybePi(piList) : undefined
  );
}

// ── Sparse .* Dense (result is sparse) ──────────────────────────────────

export function sparseElemMulDense(
  S: RuntimeSparseMatrix,
  D: RuntimeTensor
): RuntimeSparseMatrix {
  const [dRows, dCols] = tensorSize2D(D);
  if (S.m !== dRows || S.n !== dCols)
    throw new RuntimeError(
      `Matrix dimensions must agree: [${S.m},${S.n}] vs [${dRows},${dCols}]`
    );
  const hasImag = isComplexSparse(S) || D.imag !== undefined;
  const irList: number[] = [];
  const prList: number[] = [];
  const piList: number[] = [];
  const jc = new Int32Array(S.n + 1);

  for (let col = 0; col < S.n; col++) {
    jc[col] = irList.length;
    for (let k = S.jc[col]; k < S.jc[col + 1]; k++) {
      const row = S.ir[k];
      const idx = col * S.m + row;
      const aRe = S.pr[k],
        aIm = S.pi ? S.pi[k] : 0;
      const bRe = D.data[idx],
        bIm = D.imag ? D.imag[idx] : 0;
      const re = aRe * bRe - aIm * bIm;
      const im = aRe * bIm + aIm * bRe;
      if (!isZero(re, im)) {
        irList.push(row);
        prList.push(re);
        piList.push(im);
      }
    }
  }
  jc[S.n] = irList.length;
  return RTV.sparseMatrix(
    S.m,
    S.n,
    new Int32Array(irList),
    jc,
    new Float64Array(prList),
    hasImag ? maybePi(piList) : undefined
  );
}

// ── Dispatch helpers for arithmetic.ts ──────────────────────────────────

export function mAddSparse(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) && isRuntimeSparseMatrix(b)) {
    return sparseAdd(a, b);
  }
  // sparse + scalar (real or complex)
  if (isRuntimeSparseMatrix(a)) {
    const cs = complexScalarVal(b);
    if (cs) {
      const dense = sparseToDense(a);
      const data = new FloatXArray(dense.data.length);
      for (let i = 0; i < data.length; i++) data[i] = dense.data[i] + cs.re;
      const imag =
        cs.im !== 0 || dense.imag
          ? (() => {
              const r = new FloatXArray(data.length);
              for (let i = 0; i < r.length; i++)
                r[i] = (dense.imag ? dense.imag[i] : 0) + cs.im;
              return r;
            })()
          : undefined;
      return RTV.tensor(data, dense.shape, imag);
    }
  }
  if (isRuntimeSparseMatrix(b)) {
    const cs = complexScalarVal(a);
    if (cs) {
      const dense = sparseToDense(b);
      const data = new FloatXArray(dense.data.length);
      for (let i = 0; i < data.length; i++) data[i] = cs.re + dense.data[i];
      const imag =
        cs.im !== 0 || dense.imag
          ? (() => {
              const r = new FloatXArray(data.length);
              for (let i = 0; i < r.length; i++)
                r[i] = cs.im + (dense.imag ? dense.imag[i] : 0);
              return r;
            })()
          : undefined;
      return RTV.tensor(data, dense.shape, imag);
    }
  }
  // sparse + dense tensor → dense
  if (isRuntimeSparseMatrix(a) && isRuntimeTensor(b)) {
    const dense = sparseToDense(a);
    const data = new FloatXArray(dense.data.length);
    for (let i = 0; i < data.length; i++) data[i] = dense.data[i] + b.data[i];
    const imag = mergeImag(dense.imag, b.imag, data.length, (x, y) => x + y);
    return RTV.tensor(data, dense.shape, imag);
  }
  if (isRuntimeTensor(a) && isRuntimeSparseMatrix(b)) {
    const dense = sparseToDense(b);
    const data = new FloatXArray(dense.data.length);
    for (let i = 0; i < data.length; i++) data[i] = a.data[i] + dense.data[i];
    const imag = mergeImag(a.imag, dense.imag, data.length, (x, y) => x + y);
    return RTV.tensor(data, a.shape, imag);
  }
  throw new RuntimeError("mAddSparse: unexpected operand types");
}

export function mSubSparse(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) && isRuntimeSparseMatrix(b)) {
    return sparseSub(a, b);
  }
  if (isRuntimeSparseMatrix(a)) {
    const cs = complexScalarVal(b);
    if (cs) {
      const dense = sparseToDense(a);
      const data = new FloatXArray(dense.data.length);
      for (let i = 0; i < data.length; i++) data[i] = dense.data[i] - cs.re;
      const imag =
        cs.im !== 0 || dense.imag
          ? (() => {
              const r = new FloatXArray(data.length);
              for (let i = 0; i < r.length; i++)
                r[i] = (dense.imag ? dense.imag[i] : 0) - cs.im;
              return r;
            })()
          : undefined;
      return RTV.tensor(data, dense.shape, imag);
    }
  }
  if (isRuntimeSparseMatrix(b)) {
    const cs = complexScalarVal(a);
    if (cs) {
      const dense = sparseToDense(b);
      const data = new FloatXArray(dense.data.length);
      for (let i = 0; i < data.length; i++) data[i] = cs.re - dense.data[i];
      const imag =
        cs.im !== 0 || dense.imag
          ? (() => {
              const r = new FloatXArray(data.length);
              for (let i = 0; i < r.length; i++)
                r[i] = cs.im - (dense.imag ? dense.imag[i] : 0);
              return r;
            })()
          : undefined;
      return RTV.tensor(data, dense.shape, imag);
    }
  }
  if (isRuntimeSparseMatrix(a) && isRuntimeTensor(b)) {
    const dense = sparseToDense(a);
    const data = new FloatXArray(dense.data.length);
    for (let i = 0; i < data.length; i++) data[i] = dense.data[i] - b.data[i];
    const imag = mergeImag(dense.imag, b.imag, data.length, (x, y) => x - y);
    return RTV.tensor(data, dense.shape, imag);
  }
  if (isRuntimeTensor(a) && isRuntimeSparseMatrix(b)) {
    const dense = sparseToDense(b);
    const data = new FloatXArray(dense.data.length);
    for (let i = 0; i < data.length; i++) data[i] = a.data[i] - dense.data[i];
    const imag = mergeImag(a.imag, dense.imag, data.length, (x, y) => x - y);
    return RTV.tensor(data, a.shape, imag);
  }
  throw new RuntimeError("mSubSparse: unexpected operand types");
}

/** Merge two optional imag arrays with the given op. Returns undefined if both are absent. */
function mergeImag(
  a: import("../runtime/types.js").FloatXArrayType | undefined,
  b: import("../runtime/types.js").FloatXArrayType | undefined,
  len: number,
  op: (x: number, y: number) => number
): import("../runtime/types.js").FloatXArrayType | undefined {
  if (!a && !b) return undefined;
  const result = new FloatXArray(len);
  for (let i = 0; i < len; i++) {
    result[i] = op(a ? a[i] : 0, b ? b[i] : 0);
  }
  return result;
}

export function mMulSparse(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  // Sparse * Sparse matrix multiply
  if (isRuntimeSparseMatrix(a) && isRuntimeSparseMatrix(b)) {
    return sparseMatMul(a, b);
  }
  // Sparse * scalar / scalar * Sparse (handle complex scalars)
  if (isRuntimeSparseMatrix(a)) {
    const cs = complexScalarVal(b);
    if (cs) {
      if (cs.im === 0) return sparseScale(a, cs.re);
      return sparseScaleComplex(a, cs.re, cs.im);
    }
  }
  if (isRuntimeSparseMatrix(b)) {
    const cs = complexScalarVal(a);
    if (cs) {
      if (cs.im === 0) return sparseScale(b, cs.re);
      return sparseScaleComplex(b, cs.re, cs.im);
    }
  }
  // Sparse * Dense matrix multiply → dense
  if (isRuntimeSparseMatrix(a) && isRuntimeTensor(b)) {
    const [bRows, bCols] = tensorSize2D(b);
    if (a.n !== bRows)
      throw new RuntimeError(
        `Inner matrix dimensions must agree: ${a.n} vs ${bRows}`
      );
    const m = a.m;
    const hasImag = isComplexSparse(a) || b.imag !== undefined;
    const result = new FloatXArray(m * bCols);
    const resultImag = hasImag ? new FloatXArray(m * bCols) : undefined;
    for (let col = 0; col < bCols; col++) {
      for (let k = 0; k < a.n; k++) {
        const bRe = b.data[col * bRows + k];
        const bIm = b.imag ? b.imag[col * bRows + k] : 0;
        if (bRe === 0 && bIm === 0) continue;
        for (let ka = a.jc[k]; ka < a.jc[k + 1]; ka++) {
          const idx = col * m + a.ir[ka];
          const aRe = a.pr[ka];
          const aIm = a.pi ? a.pi[ka] : 0;
          result[idx] += aRe * bRe - aIm * bIm;
          if (resultImag) resultImag[idx] += aRe * bIm + aIm * bRe;
        }
      }
    }
    return RTV.tensor(result, [m, bCols], resultImag);
  }
  // Dense * Sparse → dense
  if (isRuntimeTensor(a) && isRuntimeSparseMatrix(b)) {
    const [aRows, aCols] = tensorSize2D(a);
    if (aCols !== b.m)
      throw new RuntimeError(
        `Inner matrix dimensions must agree: ${aCols} vs ${b.m}`
      );
    const hasImag = a.imag !== undefined || isComplexSparse(b);
    const result = new FloatXArray(aRows * b.n);
    const resultImag = hasImag ? new FloatXArray(aRows * b.n) : undefined;
    for (let col = 0; col < b.n; col++) {
      for (let kb = b.jc[col]; kb < b.jc[col + 1]; kb++) {
        const k = b.ir[kb];
        const bRe = b.pr[kb];
        const bIm = b.pi ? b.pi[kb] : 0;
        for (let row = 0; row < aRows; row++) {
          const aIdx = k * aRows + row;
          const aRe = a.data[aIdx];
          const aIm = a.imag ? a.imag[aIdx] : 0;
          const oIdx = col * aRows + row;
          result[oIdx] += aRe * bRe - aIm * bIm;
          if (resultImag) resultImag[oIdx] += aRe * bIm + aIm * bRe;
        }
      }
    }
    return RTV.tensor(result, [aRows, b.n], resultImag);
  }
  throw new RuntimeError("mMulSparse: unexpected operand types");
}

export function mElemMulSparse(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  if (isRuntimeSparseMatrix(a) && isRuntimeSparseMatrix(b)) {
    return sparseElemMul(a, b);
  }
  // Sparse .* scalar (handle complex)
  if (isRuntimeSparseMatrix(a)) {
    const cs = complexScalarVal(b);
    if (cs) {
      if (cs.im === 0) return sparseScale(a, cs.re);
      return sparseScaleComplex(a, cs.re, cs.im);
    }
  }
  if (isRuntimeSparseMatrix(b)) {
    const cs = complexScalarVal(a);
    if (cs) {
      if (cs.im === 0) return sparseScale(b, cs.re);
      return sparseScaleComplex(b, cs.re, cs.im);
    }
  }
  // Sparse .* Dense → sparse
  if (isRuntimeSparseMatrix(a) && isRuntimeTensor(b)) {
    return sparseElemMulDense(a, b);
  }
  if (isRuntimeTensor(a) && isRuntimeSparseMatrix(b)) {
    return sparseElemMulDense(b, a);
  }
  throw new RuntimeError("mElemMulSparse: unexpected operand types");
}

export function mElemDivSparse(a: RuntimeValue, b: RuntimeValue): RuntimeValue {
  // Sparse ./ real scalar → sparse
  if (isRuntimeSparseMatrix(a)) {
    const s = scalarVal(b);
    if (!isNaN(s)) return sparseScale(a, 1 / s);
    // Sparse ./ complex scalar
    const cs = complexScalarVal(b);
    if (cs) {
      const denom = cs.re * cs.re + cs.im * cs.im;
      return sparseScaleComplex(a, cs.re / denom, -cs.im / denom);
    }
  }
  // scalar ./ sparse → densify sparse, broadcast scalar
  if (isRuntimeSparseMatrix(b)) {
    const cs = complexScalarVal(a);
    if (cs) {
      const bDense = sparseToDense(b);
      const len = bDense.data.length;
      const hasImag = cs.im !== 0 || bDense.imag !== undefined;
      const data = new FloatXArray(len);
      const imag = hasImag ? new FloatXArray(len) : undefined;
      for (let i = 0; i < len; i++) {
        const bRe = bDense.data[i];
        const bIm = bDense.imag ? bDense.imag[i] : 0;
        if (!hasImag) {
          data[i] = cs.re / bRe;
        } else {
          const denom = bRe * bRe + bIm * bIm;
          data[i] = (cs.re * bRe + cs.im * bIm) / denom;
          if (imag) imag[i] = (cs.im * bRe - cs.re * bIm) / denom;
        }
      }
      return RTV.tensor(data, bDense.shape, imag);
    }
  }
  // All other combinations → convert to dense, return dense
  const aDense = isRuntimeSparseMatrix(a) ? sparseToDense(a) : a;
  const bDense = isRuntimeSparseMatrix(b) ? sparseToDense(b) : b;
  if (!isRuntimeTensor(aDense) || !isRuntimeTensor(bDense))
    throw new RuntimeError("mElemDivSparse: unexpected operand types");
  const hasImag = aDense.imag !== undefined || bDense.imag !== undefined;
  const data = new FloatXArray(aDense.data.length);
  const imag = hasImag ? new FloatXArray(aDense.data.length) : undefined;
  for (let i = 0; i < data.length; i++) {
    const bRe = bDense.data[i];
    const bIm = bDense.imag ? bDense.imag[i] : 0;
    if (!hasImag) {
      data[i] = aDense.data[i] / bRe;
    } else {
      const aRe = aDense.data[i];
      const aIm = aDense.imag ? aDense.imag[i] : 0;
      const denom = bRe * bRe + bIm * bIm;
      data[i] = (aRe * bRe + aIm * bIm) / denom;
      if (imag) imag[i] = (aIm * bRe - aRe * bIm) / denom;
    }
  }
  return RTV.tensor(data, aDense.shape, imag);
}

/** Convert sparse to dense RuntimeTensor (exported for use in arithmetic dispatch). */
export { sparseToDense };
