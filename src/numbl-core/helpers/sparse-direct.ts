/**
 * Sparse direct solvers on the CSC representation (real arithmetic):
 * Kronecker product, structural triangularity checks, triangular solves,
 * up-looking sparse Cholesky, left-looking sparse LU with partial pivoting
 * (Gilbert-Peierls), permutations, fill-reducing orderings (minimum degree,
 * reverse Cuthill-McKee), and the `A \ b` driver used by mldivide.
 *
 * Complex sparse matrices are not handled here; callers fall back to the
 * dense path for them.
 */

import {
  type RuntimeSparseMatrix,
  type RuntimeTensor,
  type SparseTriInfo,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { RuntimeError } from "../runtime/error.js";
import { allocFloat64Array } from "../runtime/alloc.js";
import { sparseTranspose } from "./sparse-arithmetic.js";
import { emitWarning } from "./warn.js";

type Sp = RuntimeSparseMatrix;

export const SINGULAR_WARNING = "Matrix is singular to working precision.";

// ── Construction helpers ──────────────────────────────────────────────────

/** Convert a dense tensor to CSC (dropping zeros). */
export function denseToSparse(t: RuntimeTensor): Sp {
  const m = t.shape[0] ?? 1;
  const n = t.shape.length >= 2 ? t.shape[1] : 1;
  const irList: number[] = [];
  const prList: number[] = [];
  const piList: number[] | undefined = t.imag ? [] : undefined;
  const jc = new Int32Array(n + 1);
  for (let c = 0; c < n; c++) {
    jc[c] = irList.length;
    for (let r = 0; r < m; r++) {
      const idx = c * m + r;
      const re = t.data[idx];
      const im = t.imag ? t.imag[idx] : 0;
      if (re !== 0 || im !== 0) {
        irList.push(r);
        prList.push(re);
        if (piList) piList.push(im);
      }
    }
  }
  jc[n] = irList.length;
  return RTV.sparseMatrix(
    m,
    n,
    new Int32Array(irList),
    jc,
    allocFloat64Array(prList),
    piList ? allocFloat64Array(piList) : undefined
  );
}

/** Build CSC from column-major dense data, dropping zeros. */
export function denseDataToSparse(
  data: Float64Array,
  m: number,
  n: number
): Sp {
  return denseToSparse(RTV.tensor(data, [m, n]));
}

/** Build CSC from unsorted (row, col, val) lists with no duplicates. */
function fromUnsortedColumns(
  m: number,
  n: number,
  ir: Int32Array,
  jc: Int32Array,
  pr: Float64Array
): Sp {
  // Transposing twice yields sorted row indices in O(nnz).
  return sparseTranspose(sparseTranspose(RTV.sparseMatrix(m, n, ir, jc, pr)));
}

/** Sparse permutation matrix P with P(k, p[k]) = 1, so (P*A)(k,:) = A(p[k],:). */
export function permutationMatrix(p: Int32Array | number[]): Sp {
  const n = p.length;
  const ir = new Int32Array(n);
  const jc = new Int32Array(n + 1);
  const pr = allocFloat64Array(n);
  // Column p[k] holds row k.
  for (let k = 0; k < n; k++) jc[p[k] + 1]++;
  for (let c = 0; c < n; c++) jc[c + 1] += jc[c];
  for (let k = 0; k < n; k++) {
    ir[jc[p[k]]] = k;
    pr[jc[p[k]]] = 1;
  }
  return RTV.sparseMatrix(n, n, ir, jc, pr);
}

/** 1-based row vector from a 0-based permutation. */
export function permVectorTensor(p: Int32Array): RuntimeTensor {
  const d = allocFloat64Array(p.length);
  for (let i = 0; i < p.length; i++) d[i] = p[i] + 1;
  return RTV.tensor(d, [1, p.length]);
}

export function invertPerm(p: Int32Array): Int32Array {
  const inv = new Int32Array(p.length);
  for (let i = 0; i < p.length; i++) inv[p[i]] = i;
  return inv;
}

/** C = A(p, q) for 0-based permutations p (rows) and q (columns). */
export function sparsePermute(A: Sp, p: Int32Array, q: Int32Array): Sp {
  const pinv = invertPerm(p);
  const nnz = A.jc[A.n];
  const ir = new Int32Array(nnz);
  const jc = new Int32Array(A.n + 1);
  const pr = allocFloat64Array(nnz);
  let pos = 0;
  for (let k = 0; k < A.n; k++) {
    jc[k] = pos;
    const j = q[k];
    for (let t = A.jc[j]; t < A.jc[j + 1]; t++) {
      ir[pos] = pinv[A.ir[t]];
      pr[pos] = A.pr[t];
      pos++;
    }
  }
  jc[A.n] = pos;
  return fromUnsortedColumns(A.m, A.n, ir, jc, pr);
}

// ── Kronecker product ─────────────────────────────────────────────────────

export function sparseKron(A: Sp, B: Sp): Sp {
  const m = A.m,
    n = A.n,
    p = B.m,
    q = B.n;
  const nnz = A.jc[n] * B.jc[q];
  const ir = new Int32Array(nnz);
  const jc = new Int32Array(n * q + 1);
  const pr = allocFloat64Array(nnz);
  const cplx = A.pi !== undefined || B.pi !== undefined;
  const pi = cplx ? allocFloat64Array(nnz) : undefined;
  let pos = 0;
  for (let ja = 0; ja < n; ja++) {
    for (let jb = 0; jb < q; jb++) {
      jc[ja * q + jb] = pos;
      for (let ka = A.jc[ja]; ka < A.jc[ja + 1]; ka++) {
        const rowBase = A.ir[ka] * p;
        const aRe = A.pr[ka];
        const aIm = A.pi ? A.pi[ka] : 0;
        for (let kb = B.jc[jb]; kb < B.jc[jb + 1]; kb++) {
          ir[pos] = rowBase + B.ir[kb];
          const bRe = B.pr[kb];
          const bIm = B.pi ? B.pi[kb] : 0;
          pr[pos] = aRe * bRe - aIm * bIm;
          if (pi) pi[pos] = aRe * bIm + aIm * bRe;
          pos++;
        }
      }
    }
  }
  jc[n * q] = pos;
  return RTV.sparseMatrix(m * p, n * q, ir, jc, pr, pi);
}

// ── Structural checks ─────────────────────────────────────────────────────

/**
 * One O(nnz) pass over the pattern: is the matrix upper / lower triangular
 * (explicitly stored zeros ignored), and where is each diagonal entry. The
 * result is cached on the matrix object.
 */
export function triangularInfo(S: Sp): SparseTriInfo {
  if (S._triInfo) return S._triInfo;
  const n = S.n;
  const ir = S.ir,
    jc = S.jc,
    pr = S.pr,
    pi = S.pi;
  const diagPos = new Int32Array(n).fill(-1);
  const diag = allocFloat64Array(n);
  let upper = true;
  let lower = true;
  for (let j = 0; j < n; j++) {
    const end = jc[j + 1];
    for (let k = jc[j]; k < end; k++) {
      const i = ir[k];
      if (i === j) {
        diagPos[j] = k;
        diag[j] = pr[k];
        continue;
      }
      if (pr[k] === 0 && (!pi || pi[k] === 0)) continue;
      if (i > j) upper = false;
      else lower = false;
    }
  }
  const info: SparseTriInfo = { upper, lower, diagPos, diag };
  S._triInfo = info;
  return info;
}

/** True if every stored nonzero lies on or above the diagonal. */
export function sparseIsUpper(S: Sp): boolean {
  return triangularInfo(S).upper;
}

/** True if every stored nonzero lies on or below the diagonal. */
export function sparseIsLower(S: Sp): boolean {
  return triangularInfo(S).lower;
}

/** Exact numerical symmetry (A == A'), real sparse only. */
export function sparseIsSymmetric(S: Sp): boolean {
  if (S.m !== S.n || S.pi) return false;
  const T = sparseTranspose(S);
  const nnz = S.jc[S.n];
  if (T.jc[T.n] !== nnz) return false;
  for (let j = 0; j <= S.n; j++) if (S.jc[j] !== T.jc[j]) return false;
  for (let k = 0; k < nnz; k++)
    if (S.ir[k] !== T.ir[k] || S.pr[k] !== T.pr[k]) return false;
  return true;
}

/** Value at (i, j), or 0. Binary search within the column. */
function sparseAt(S: Sp, i: number, j: number): number {
  let lo = S.jc[j],
    hi = S.jc[j + 1] - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const r = S.ir[mid];
    if (r === i) return S.pr[mid];
    if (r < i) lo = mid + 1;
    else hi = mid - 1;
  }
  return 0;
}

function allDiagPositive(S: Sp): boolean {
  for (let j = 0; j < S.n; j++) if (!(sparseAt(S, j, j) > 0)) return false;
  return true;
}

// ── Triangular solves (real, dense right-hand sides) ──────────────────────

/** Solve R * X = B for upper-triangular R (n×n), B column-major n×nrhs.
 *  Returns the solution in a fresh array and whether a zero pivot was hit. */
export function sparseSolveUpper(
  R: Sp,
  B: Float64Array,
  nrhs: number
): { x: Float64Array; singular: boolean } {
  const n = R.n;
  const { diagPos, diag } = triangularInfo(R);
  const ir = R.ir,
    jc = R.jc,
    pr = R.pr;
  const x = allocFloat64Array(B);
  let singular = false;
  for (let j = 0; j < n; j++) if (diag[j] === 0) singular = true;
  for (let c = 0; c < nrhs; c++) {
    const off = c * n;
    for (let j = n - 1; j >= 0; j--) {
      const xj = x[off + j] / diag[j];
      x[off + j] = xj;
      if (xj === 0) continue;
      // Rows are sorted, so the strictly-upper entries precede the diagonal;
      // anything after it is an explicitly stored zero.
      const dp = diagPos[j];
      const end = dp >= 0 ? dp : jc[j + 1];
      for (let k = jc[j]; k < end; k++) x[off + ir[k]] -= pr[k] * xj;
    }
  }
  return { x, singular };
}

/** Solve L * X = B for lower-triangular L (n×n). */
export function sparseSolveLower(
  L: Sp,
  B: Float64Array,
  nrhs: number
): { x: Float64Array; singular: boolean } {
  const n = L.n;
  const { diagPos, diag } = triangularInfo(L);
  const ir = L.ir,
    jc = L.jc,
    pr = L.pr;
  const x = allocFloat64Array(B);
  let singular = false;
  for (let j = 0; j < n; j++) if (diag[j] === 0) singular = true;
  for (let c = 0; c < nrhs; c++) {
    const off = c * n;
    for (let j = 0; j < n; j++) {
      const xj = x[off + j] / diag[j];
      x[off + j] = xj;
      if (xj === 0) continue;
      const dp = diagPos[j];
      const end = jc[j + 1];
      for (let k = dp >= 0 ? dp + 1 : jc[j]; k < end; k++)
        x[off + ir[k]] -= pr[k] * xj;
    }
  }
  return { x, singular };
}

/** Apply a row permutation: Y(k, :) = X(p[k], :). */
function permuteRows(
  X: Float64Array,
  n: number,
  nrhs: number,
  p: Int32Array
): Float64Array {
  const Y = allocFloat64Array(X.length);
  for (let c = 0; c < nrhs; c++) {
    const off = c * n;
    for (let k = 0; k < n; k++) Y[off + k] = X[off + p[k]];
  }
  return Y;
}

/** Inverse row permutation: Y(p[k], :) = X(k, :). */
function unpermuteRows(
  X: Float64Array,
  n: number,
  nrhs: number,
  p: Int32Array
): Float64Array {
  const Y = allocFloat64Array(X.length);
  for (let c = 0; c < nrhs; c++) {
    const off = c * n;
    for (let k = 0; k < n; k++) Y[off + p[k]] = X[off + k];
  }
  return Y;
}

// ── Sparse Cholesky (up-looking) ──────────────────────────────────────────

/** Elimination tree of a symmetric matrix, reading its upper triangle. */
function eliminationTree(A: Sp): Int32Array {
  const n = A.n;
  const parent = new Int32Array(n).fill(-1);
  const ancestor = new Int32Array(n).fill(-1);
  for (let k = 0; k < n; k++) {
    for (let p = A.jc[k]; p < A.jc[k + 1]; p++) {
      let i = A.ir[p];
      while (i !== -1 && i < k) {
        const inext = ancestor[i];
        ancestor[i] = k;
        if (inext === -1) parent[i] = k;
        i = inext;
      }
    }
  }
  return parent;
}

/** Pattern of row k of L (topological order) into s[top..n-1]; returns top.
 *  `mark` must satisfy mark[i] !== k on entry for all i. */
function etreeReach(
  A: Sp,
  k: number,
  parent: Int32Array,
  s: Int32Array,
  mark: Int32Array
): number {
  const n = A.n;
  let top = n;
  mark[k] = k;
  for (let p = A.jc[k]; p < A.jc[k + 1]; p++) {
    let i = A.ir[p];
    if (i > k) continue;
    let len = 0;
    for (; mark[i] !== k; i = parent[i]) {
      s[len++] = i;
      mark[i] = k;
    }
    while (len > 0) s[--top] = s[--len];
  }
  return top;
}

export interface CholResult {
  /** Lower factor with L * L' = A (only when ok). */
  L: Sp;
  ok: boolean;
  /** 0 on success, otherwise the 1-based column at which the pivot failed. */
  p: number;
}

/**
 * Cholesky factorization of a symmetric positive definite sparse matrix.
 * Only the upper triangle of A is read; symmetry is not checked. On failure
 * returns the leading (p-1)×(p-1) factor computed so far.
 */
export function sparseChol(A: Sp): CholResult {
  const n = A.n;
  if (A.m !== n) throw new RuntimeError("chol: Matrix must be square.");
  const parent = eliminationTree(A);
  const s = new Int32Array(n);
  const mark = new Int32Array(n).fill(-1);
  // Symbolic: count entries per column of L (row patterns give column counts).
  const colCount = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    const top = etreeReach(A, k, parent, s, mark);
    for (let t = top; t < n; t++) colCount[s[t]]++;
    colCount[k]++; // diagonal
  }
  const Lp = new Int32Array(n + 1);
  for (let j = 0; j < n; j++) Lp[j + 1] = Lp[j] + colCount[j];
  const nnzL = Lp[n];
  const Li = new Int32Array(nnzL);
  const Lx = allocFloat64Array(nnzL);
  const c = new Int32Array(n);
  for (let j = 0; j < n; j++) c[j] = Lp[j];
  const x = allocFloat64Array(n);
  mark.fill(-1);
  for (let k = 0; k < n; k++) {
    const top = etreeReach(A, k, parent, s, mark);
    x[k] = 0;
    for (let p = A.jc[k]; p < A.jc[k + 1]; p++) {
      if (A.ir[p] <= k) x[A.ir[p]] = A.pr[p];
    }
    let d = x[k];
    x[k] = 0;
    for (let t = top; t < n; t++) {
      const i = s[t];
      const lki = x[i] / Lx[Lp[i]];
      x[i] = 0;
      for (let p = Lp[i] + 1; p < c[i]; p++) x[Li[p]] -= Lx[p] * lki;
      d -= lki * lki;
      const pos = c[i]++;
      Li[pos] = k;
      Lx[pos] = lki;
    }
    if (!(d > 0)) {
      // Not positive definite at column k. MATLAB returns the first k
      // columns of L in full (R(1:k, :) for the upper form), so finish the
      // remaining rows restricted to columns < k.
      for (let i = k + 1; i < n; i++) {
        const top2 = etreeReach(A, i, parent, s, mark);
        for (let p = A.jc[i]; p < A.jc[i + 1]; p++)
          if (A.ir[p] < k) x[A.ir[p]] = A.pr[p];
        for (let t = top2; t < n; t++) {
          const j = s[t];
          if (j >= k) {
            x[j] = 0;
            continue;
          }
          const lij = x[j] / Lx[Lp[j]];
          x[j] = 0;
          for (let p = Lp[j] + 1; p < c[j]; p++) x[Li[p]] -= Lx[p] * lij;
          const pos = c[j]++;
          Li[pos] = i;
          Lx[pos] = lij;
        }
      }
      return { L: leadingColumns(n, k, Lp, Li, Lx, c), ok: false, p: k + 1 };
    }
    const pos = c[k]++;
    Li[pos] = k;
    Lx[pos] = Math.sqrt(d);
  }
  return { L: RTV.sparseMatrix(n, n, Li, Lp, Lx), ok: true, p: 0 };
}

/** The first k columns (all n rows) of a partially built L. */
function leadingColumns(
  n: number,
  k: number,
  Lp: Int32Array,
  Li: Int32Array,
  Lx: Float64Array,
  c: Int32Array
): Sp {
  let nnz = 0;
  for (let j = 0; j < k; j++) nnz += c[j] - Lp[j];
  const ir = new Int32Array(nnz);
  const pr = allocFloat64Array(nnz);
  const jc = new Int32Array(k + 1);
  let pos = 0;
  for (let j = 0; j < k; j++) {
    jc[j] = pos;
    for (let p = Lp[j]; p < c[j]; p++) {
      ir[pos] = Li[p];
      pr[pos] = Lx[p];
      pos++;
    }
  }
  jc[k] = pos;
  return RTV.sparseMatrix(n, k, ir, jc, pr);
}

// ── Sparse LU (left-looking, partial pivoting) ────────────────────────────

export interface LuResult {
  /** Unit lower triangular, m × min(m,n), rows in pivot order. */
  L: Sp;
  /** Upper triangular, min(m,n) × n. */
  U: Sp;
  /** Row permutation: row k of P*A*Q is row p[k] of A. */
  p: Int32Array;
  /** Column permutation: column k of P*A*Q is column q[k] of A. */
  q: Int32Array;
  singular: boolean;
}

/**
 * LU factorization P*A*Q = L*U of a real sparse matrix with threshold
 * partial pivoting (tol = 1 is plain partial pivoting; smaller values prefer
 * the diagonal entry when it is at least tol times the largest candidate).
 */
export function sparseLu(A: Sp, q: Int32Array | null, tol: number): LuResult {
  const m = A.m,
    n = A.n;
  const kmin = Math.min(m, n);
  const Lp: number[] = [0];
  const Li: number[] = [];
  const Lx: number[] = [];
  const Up: number[] = [0];
  const Ui: number[] = [];
  const Ux: number[] = [];
  const pinv = new Int32Array(m).fill(-1);
  const x = allocFloat64Array(m);
  const xi = new Int32Array(m);
  const stack = new Int32Array(m);
  const pstack = new Int32Array(m);
  const mark = new Int32Array(m).fill(-1);
  let singular = false;

  for (let k = 0; k < n; k++) {
    const col = q ? q[k] : k;
    // Reach: rows reachable from the pattern of A(:,col) through L.
    let top = m;
    for (let t = A.jc[col]; t < A.jc[col + 1]; t++) {
      const i0 = A.ir[t];
      if (mark[i0] === k) continue;
      let head = 0;
      stack[0] = i0;
      mark[i0] = k;
      pstack[i0] = pinv[i0] >= 0 ? Lp[pinv[i0]] : -1;
      while (head >= 0) {
        const j = stack[head];
        const J = pinv[j];
        let descended = false;
        if (J >= 0) {
          const end = Lp[J + 1];
          let p = pstack[j];
          for (; p < end; p++) {
            const i = Li[p];
            if (mark[i] !== k) {
              mark[i] = k;
              pstack[j] = p + 1;
              stack[++head] = i;
              pstack[i] = pinv[i] >= 0 ? Lp[pinv[i]] : -1;
              descended = true;
              break;
            }
          }
          if (descended) continue;
        }
        head--;
        xi[--top] = j;
      }
    }
    // Scatter A(:,col) and solve L * x = A(:,col) for the pivoted part.
    for (let t = A.jc[col]; t < A.jc[col + 1]; t++) x[A.ir[t]] = A.pr[t];
    for (let t = top; t < m; t++) {
      const j = xi[t];
      const J = pinv[j];
      if (J < 0) continue;
      const xj = x[j];
      if (xj === 0) continue;
      for (let p = Lp[J] + 1; p < Lp[J + 1]; p++) x[Li[p]] -= Lx[p] * xj;
    }
    // Partition into U entries (pivoted rows) and L candidates.
    let ipiv = -1;
    let amax = -1;
    for (let t = top; t < m; t++) {
      const i = xi[t];
      if (pinv[i] >= 0) {
        Ui.push(pinv[i]);
        Ux.push(x[i]);
      } else {
        const a = Math.abs(x[i]);
        if (a > amax) {
          amax = a;
          ipiv = i;
        }
      }
    }
    if (k < kmin) {
      if (ipiv < 0 || amax === 0) {
        // No usable pivot in this column: take the first unpivoted row.
        singular = true;
        if (ipiv < 0) {
          for (let i = 0; i < m; i++)
            if (pinv[i] < 0) {
              ipiv = i;
              break;
            }
        }
      } else if (
        col < m &&
        pinv[col] < 0 &&
        Math.abs(x[col]) >= tol * amax &&
        x[col] !== 0
      ) {
        ipiv = col;
      }
      const pivot = x[ipiv];
      if (!Number.isFinite(pivot)) singular = true;
      pinv[ipiv] = k;
      Ui.push(k);
      Ux.push(pivot);
      Li.push(ipiv);
      Lx.push(1);
      if (pivot !== 0 && Number.isFinite(pivot)) {
        for (let t = top; t < m; t++) {
          const i = xi[t];
          if (pinv[i] >= 0) continue;
          const v = x[i] / pivot;
          if (v !== 0) {
            Li.push(i);
            Lx.push(v);
          }
        }
      }
      Lp.push(Li.length);
    }
    Up.push(Ui.length);
    for (let t = top; t < m; t++) x[xi[t]] = 0;
  }
  // Rows never chosen as pivots (m > n) keep their relative order at the end.
  let next = kmin;
  for (let i = 0; i < m; i++) if (pinv[i] < 0) pinv[i] = next++;
  // L rows into pivot order.
  const LiArr = new Int32Array(Li.length);
  for (let p = 0; p < Li.length; p++) LiArr[p] = pinv[Li[p]];
  const L = fromUnsortedColumns(
    m,
    kmin,
    LiArr,
    new Int32Array(Lp),
    allocFloat64Array(Lx)
  );
  const U = fromUnsortedColumns(
    kmin,
    n,
    new Int32Array(Ui),
    new Int32Array(Up),
    allocFloat64Array(Ux)
  );
  const p = invertPerm(pinv);
  const qOut = q ? Int32Array.from(q) : identityPerm(n);
  return { L, U, p, q: qOut, singular };
}

export function identityPerm(n: number): Int32Array {
  const p = new Int32Array(n);
  for (let i = 0; i < n; i++) p[i] = i;
  return p;
}

// ── Orderings ─────────────────────────────────────────────────────────────

/** Adjacency sets of the undirected pattern of A + A' (no self loops). */
function symmetricAdjacency(A: Sp): Set<number>[] {
  const n = Math.max(A.m, A.n);
  const adj: Set<number>[] = new Array(n);
  for (let i = 0; i < n; i++) adj[i] = new Set();
  for (let j = 0; j < A.n; j++)
    for (let k = A.jc[j]; k < A.jc[j + 1]; k++) {
      const i = A.ir[k];
      if (i === j) continue;
      adj[i].add(j);
      adj[j].add(i);
    }
  return adj;
}

/**
 * Minimum degree ordering on the pattern of A + A', computed on an explicit
 * elimination graph. Returns a 0-based permutation p such that A(p,p) tends
 * to have a sparser Cholesky factor. This is a plain minimum-degree
 * ordering, not MATLAB's approximate minimum degree, so the permutation
 * differs from MATLAB's `amd` while serving the same purpose.
 */
export function minimumDegreeOrdering(A: Sp): Int32Array {
  const adj = symmetricAdjacency(A);
  const n = adj.length;
  const perm = new Int32Array(n);
  if (n === 0) return perm;
  // Degree buckets as doubly linked lists.
  const head = new Int32Array(n + 1).fill(-1);
  const next = new Int32Array(n).fill(-1);
  const prev = new Int32Array(n).fill(-1);
  const deg = new Int32Array(n);
  const insert = (v: number) => {
    const d = deg[v];
    next[v] = head[d];
    prev[v] = -1;
    if (head[d] >= 0) prev[head[d]] = v;
    head[d] = v;
  };
  const remove = (v: number) => {
    const d = deg[v];
    if (prev[v] >= 0) next[prev[v]] = next[v];
    else head[d] = next[v];
    if (next[v] >= 0) prev[next[v]] = prev[v];
  };
  for (let v = 0; v < n; v++) {
    deg[v] = adj[v].size;
    insert(v);
  }
  let mind = 0;
  for (let step = 0; step < n; step++) {
    while (head[mind] < 0) mind++;
    const v = head[mind];
    remove(v);
    perm[step] = v;
    const nbrs = Array.from(adj[v]);
    for (const u of nbrs) adj[u].delete(v);
    for (let a = 0; a < nbrs.length; a++) {
      const u = nbrs[a];
      const su = adj[u];
      for (let b = a + 1; b < nbrs.length; b++) {
        const w = nbrs[b];
        if (!su.has(w)) {
          su.add(w);
          adj[w].add(u);
        }
      }
    }
    for (const u of nbrs) {
      remove(u);
      deg[u] = adj[u].size;
      insert(u);
      if (deg[u] < mind) mind = deg[u];
    }
    adj[v].clear();
  }
  return perm;
}

/**
 * Reverse Cuthill-McKee ordering on the pattern of A + A'. Returns a 0-based
 * permutation that tends to reduce the bandwidth of A(p,p).
 */
export function reverseCuthillMcKee(A: Sp): Int32Array {
  const adj = symmetricAdjacency(A);
  const n = adj.length;
  const nb: number[][] = adj.map(s =>
    Array.from(s).sort((a, b) => adj[a].size - adj[b].size || a - b)
  );
  const visited = new Uint8Array(n);
  const order: number[] = [];
  const bfsLevels = (start: number): { last: number; levels: number } => {
    const seen = new Uint8Array(n);
    let frontier = [start];
    seen[start] = 1;
    let levels = 0;
    let last = start;
    while (frontier.length > 0) {
      const nextF: number[] = [];
      for (const u of frontier)
        for (const w of nb[u])
          if (!seen[w]) {
            seen[w] = 1;
            nextF.push(w);
          }
      if (nextF.length === 0) {
        // Pick the smallest-degree node of the final level.
        last = frontier.reduce((a, b) => (adj[b].size < adj[a].size ? b : a));
        break;
      }
      frontier = nextF;
      levels++;
    }
    return { last, levels };
  };
  for (let s0 = 0; s0 < n; s0++) {
    if (visited[s0]) continue;
    // Pseudo-peripheral start: iterate BFS to a far node of low degree.
    let start = s0;
    let info = bfsLevels(start);
    for (let iter = 0; iter < 4; iter++) {
      const cand = bfsLevels(info.last);
      if (cand.levels > info.levels) {
        start = info.last;
        info = cand;
      } else {
        start = info.last;
        break;
      }
    }
    const queue: number[] = [start];
    visited[start] = 1;
    let qh = 0;
    while (qh < queue.length) {
      const u = queue[qh++];
      order.push(u);
      for (const w of nb[u])
        if (!visited[w]) {
          visited[w] = 1;
          queue.push(w);
        }
    }
  }
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) perm[i] = order[n - 1 - i];
  return perm;
}

/** Pattern of A' * A as a sparse matrix of ones (for column orderings). */
function ataPattern(A: Sp): Sp {
  const At = sparseTranspose(A);
  const n = A.n;
  const marker = new Int32Array(n).fill(-1);
  const ir: number[] = [];
  const jc = new Int32Array(n + 1);
  for (let j = 0; j < n; j++) {
    jc[j] = ir.length;
    for (let k = A.jc[j]; k < A.jc[j + 1]; k++) {
      const row = A.ir[k];
      for (let t = At.jc[row]; t < At.jc[row + 1]; t++) {
        const i = At.ir[t];
        if (marker[i] !== j) {
          marker[i] = j;
          ir.push(i);
        }
      }
    }
  }
  jc[n] = ir.length;
  const pr = allocFloat64Array(ir.length).fill(1);
  return fromUnsortedColumns(n, n, new Int32Array(ir), jc, pr);
}

/** Column ordering for LU: minimum degree on the pattern of A'*A. */
export function columnOrdering(A: Sp): Int32Array {
  return minimumDegreeOrdering(ataPattern(A));
}

// ── Solve driver ──────────────────────────────────────────────────────────

/** Densify when the matrix is at least this full (dense LAPACK is faster). */
const DENSE_FRACTION = 0.5;

/**
 * Solve A * X = B for square real sparse A and dense B (n × nrhs,
 * column-major). Returns null when the caller should use the dense path
 * (non-square, complex, or nearly full A).
 */
export function sparseSolve(
  A: Sp,
  B: Float64Array,
  nrhs: number
): Float64Array | null {
  const n = A.n;
  if (A.m !== n || A.pi) return null;
  if (n === 0) return allocFloat64Array(0);
  const nnz = A.jc[n];
  if (nnz > DENSE_FRACTION * n * n && n > 8) return null;
  if (sparseIsUpper(A)) {
    const r = sparseSolveUpper(A, B, nrhs);
    if (r.singular) emitWarning(SINGULAR_WARNING);
    return r.x;
  }
  if (sparseIsLower(A)) {
    const r = sparseSolveLower(A, B, nrhs);
    if (r.singular) emitWarning(SINGULAR_WARNING);
    return r.x;
  }
  // One-shot solve: reverse Cuthill-McKee is O(nnz) to compute and keeps
  // the factor banded, which beats minimum degree (a much sparser factor,
  // but far costlier to find) when the factors are used only once.
  const perm = reverseCuthillMcKee(A);
  if (sparseIsSymmetric(A) && allDiagPositive(A)) {
    const Ap = sparsePermute(A, perm, perm);
    const ch = sparseChol(Ap);
    if (ch.ok) {
      const Lt = sparseTranspose(ch.L);
      const bp = permuteRows(B, n, nrhs, perm);
      const y = sparseSolveLower(ch.L, bp, nrhs).x;
      const z = sparseSolveUpper(Lt, y, nrhs).x;
      return unpermuteRows(z, n, nrhs, perm);
    }
  }
  const lu = sparseLu(A, perm, 1.0);
  return luSolve(lu, B, nrhs);
}

/** Solve with precomputed LU factors: X = Q * U \ (L \ (P * B)). */
export function luSolve(
  lu: LuResult,
  B: Float64Array,
  nrhs: number
): Float64Array {
  const n = lu.U.n;
  const pb = permuteRows(B, n, nrhs, lu.p);
  const y = sparseSolveLower(lu.L, pb, nrhs);
  const z = sparseSolveUpper(lu.U, y.x, nrhs);
  if (lu.singular || z.singular) emitWarning(SINGULAR_WARNING);
  return unpermuteRows(z.x, n, nrhs, lu.q);
}

/** Solve with LU factors of A for the transposed system A' * X = B. */
export function luSolveTransposed(
  lu: LuResult,
  B: Float64Array,
  nrhs: number
): Float64Array {
  // A = P' L U Q'  =>  A' = Q U' L' P  =>  X = P' (L' \ (U' \ (Q' B))).
  const n = lu.U.n;
  const qb = permuteRows(B, n, nrhs, lu.q);
  const y = sparseSolveLower(sparseTranspose(lu.U), qb, nrhs);
  const z = sparseSolveUpper(sparseTranspose(lu.L), y.x, nrhs);
  if (lu.singular || y.singular) emitWarning(SINGULAR_WARNING);
  return unpermuteRows(z.x, n, nrhs, lu.p);
}

/** Inverse of a square real sparse matrix, returned sparse. */
export function sparseInverse(A: Sp): Sp {
  const n = A.n;
  if (A.m !== n) throw new RuntimeError("inv: matrix must be square");
  const I = allocFloat64Array(n * n);
  for (let i = 0; i < n; i++) I[i + i * n] = 1;
  const X = sparseSolve(A, I, n);
  if (X === null) throw new RuntimeError("inv: unsupported sparse matrix");
  return denseDataToSparse(X, n, n);
}
