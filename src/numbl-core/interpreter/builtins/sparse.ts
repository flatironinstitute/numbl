/**
 * Sparse matrix builtins: sparse, speye, spdiags, spconvert.
 */

import {
  FloatXArray,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type {
  RuntimeValue,
  RuntimeSparseMatrix,
  FloatXArrayType,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toNumber } from "../../runtime/convert.js";
import { registerIBuiltin } from "./types.js";

// ── Helpers ──────────────────────────────────────────────────────────────

function toNumericArray(v: RuntimeValue, name: string): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  if (isRuntimeComplexNumber(v)) return [v.re];
  throw new RuntimeError(`${name}: arguments must be numeric`);
}

function toImagArray(v: RuntimeValue): number[] | undefined {
  if (isRuntimeComplexNumber(v)) return [v.im];
  if (isRuntimeTensor(v) && v.imag) return Array.from(v.imag);
  return undefined;
}

function buildSparseFromTriplets(
  iArr: number[],
  jArr: number[],
  vArr: number[],
  m: number,
  n: number,
  vImag?: number[]
): RuntimeSparseMatrix {
  const nnz = iArr.length;
  const isComplex = vImag !== undefined;
  const triplets: { col: number; row: number; re: number; im: number }[] = [];
  for (let k = 0; k < nnz; k++) {
    triplets.push({
      col: jArr[k] - 1,
      row: iArr[k] - 1,
      re: vArr[k],
      im: isComplex ? vImag[k] : 0,
    });
  }
  triplets.sort((a, b) => a.col - b.col || a.row - b.row);
  const mergedIr: number[] = [];
  const mergedPr: number[] = [];
  const mergedPi: number[] = [];
  const mergedCols: number[] = [];
  let prevCol = -1;
  let prevRow = -1;
  for (const t of triplets) {
    if (t.col === prevCol && t.row === prevRow) {
      mergedPr[mergedPr.length - 1] += t.re;
      if (isComplex) mergedPi[mergedPi.length - 1] += t.im;
    } else {
      mergedIr.push(t.row);
      mergedPr.push(t.re);
      if (isComplex) mergedPi.push(t.im);
      mergedCols.push(t.col);
      prevCol = t.col;
      prevRow = t.row;
    }
  }
  const jc = new Int32Array(n + 1);
  let ci = 0;
  for (let c = 0; c < n; c++) {
    jc[c] = ci;
    while (ci < mergedCols.length && mergedCols[ci] === c) ci++;
  }
  jc[n] = ci;
  return RTV.sparseMatrix(
    m,
    n,
    new Int32Array(mergedIr),
    jc,
    new Float64Array(mergedPr),
    isComplex ? new Float64Array(mergedPi) : undefined
  );
}

// ── sparse ───────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "sparse",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: args => {
      if (args.length === 1) {
        const v = args[0];
        if (isRuntimeSparseMatrix(v)) return v;
        if (isRuntimeNumber(v)) {
          if (v === 0)
            return RTV.sparseMatrix(
              1,
              1,
              new Int32Array(0),
              new Int32Array(2),
              new Float64Array(0)
            );
          return RTV.sparseMatrix(
            1,
            1,
            new Int32Array([0]),
            new Int32Array([0, 1]),
            new Float64Array([v])
          );
        }
        if (isRuntimeLogical(v)) {
          const nv = v ? 1 : 0;
          if (nv === 0)
            return RTV.sparseMatrix(
              1,
              1,
              new Int32Array(0),
              new Int32Array(2),
              new Float64Array(0)
            );
          return RTV.sparseMatrix(
            1,
            1,
            new Int32Array([0]),
            new Int32Array([0, 1]),
            new Float64Array([nv])
          );
        }
        if (isRuntimeComplexNumber(v)) {
          if (v.re === 0 && v.im === 0)
            return RTV.sparseMatrix(
              1,
              1,
              new Int32Array(0),
              new Int32Array(2),
              new Float64Array(0)
            );
          return RTV.sparseMatrix(
            1,
            1,
            new Int32Array([0]),
            new Int32Array([0, 1]),
            new Float64Array([v.re]),
            v.im !== 0 ? new Float64Array([v.im]) : undefined
          );
        }
        if (!isRuntimeTensor(v))
          throw new RuntimeError("sparse: argument must be numeric");
        const rows = v.shape[0] || 1;
        const cols = v.shape.length >= 2 ? v.shape[1] : 1;
        const hasImag = v.imag !== undefined;
        const irList: number[] = [];
        const prList: number[] = [];
        const piList: number[] | undefined = hasImag ? [] : undefined;
        const jcArr = new Int32Array(cols + 1);
        for (let c = 0; c < cols; c++) {
          jcArr[c] = irList.length;
          for (let r = 0; r < rows; r++) {
            const idx = c * rows + r;
            const re = v.data[idx];
            const im = hasImag ? v.imag![idx] : 0;
            if (re !== 0 || im !== 0) {
              irList.push(r);
              prList.push(re);
              if (piList) piList.push(im);
            }
          }
        }
        jcArr[cols] = irList.length;
        return RTV.sparseMatrix(
          rows,
          cols,
          new Int32Array(irList),
          jcArr,
          new Float64Array(prList),
          piList ? new Float64Array(piList) : undefined
        );
      }
      if (args.length === 2) {
        const m = Math.round(toNumber(args[0]));
        const n = Math.round(toNumber(args[1]));
        return RTV.sparseMatrix(
          m,
          n,
          new Int32Array(0),
          new Int32Array(n + 1),
          new Float64Array(0)
        );
      }
      if (args.length >= 3) {
        const iArr = toNumericArray(args[0], "sparse");
        const jArr = toNumericArray(args[1], "sparse");
        const len = Math.max(iArr.length, jArr.length);
        let vArr: number[];
        let vImag: number[] | undefined;
        const vArg = args[2];
        if (isRuntimeNumber(vArg)) {
          vArr = new Array(len).fill(vArg);
        } else if (isRuntimeComplexNumber(vArg)) {
          vArr = new Array(len).fill(vArg.re);
          vImag = new Array(len).fill(vArg.im);
        } else if (isRuntimeLogical(vArg)) {
          vArr = new Array(len).fill(vArg ? 1 : 0);
        } else {
          vArr = toNumericArray(vArg, "sparse");
          vImag = toImagArray(vArg);
        }
        if (iArr.length !== jArr.length || iArr.length !== vArr.length) {
          if (iArr.length === 1 && jArr.length === vArr.length) {
            const iv = iArr[0];
            iArr.length = 0;
            for (let k = 0; k < vArr.length; k++) iArr.push(iv);
          } else if (jArr.length === 1 && iArr.length === vArr.length) {
            const jv = jArr[0];
            jArr.length = 0;
            for (let k = 0; k < vArr.length; k++) jArr.push(jv);
          } else {
            throw new RuntimeError("sparse: i, j, v must have the same length");
          }
        }
        let m: number, n: number;
        if (args.length >= 5) {
          m = Math.round(toNumber(args[3]));
          n = Math.round(toNumber(args[4]));
        } else {
          m = 0;
          n = 0;
          for (let k = 0; k < iArr.length; k++) {
            if (iArr[k] > m) m = iArr[k];
            if (jArr[k] > n) n = jArr[k];
          }
        }
        return buildSparseFromTriplets(iArr, jArr, vArr, m, n, vImag);
      }
      throw new RuntimeError("sparse: unsupported call signature");
    },
  }),
});

// ── speye ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "speye",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: args => {
      let rows: number, cols: number;
      if (args.length === 0) {
        rows = cols = 1;
      } else if (args.length === 1) {
        if (isRuntimeTensor(args[0])) {
          rows = Math.round(args[0].data[0]);
          cols = args[0].data.length >= 2 ? Math.round(args[0].data[1]) : rows;
        } else {
          rows = cols = Math.round(toNumber(args[0]));
        }
      } else {
        rows = Math.round(toNumber(args[0]));
        cols = Math.round(toNumber(args[1]));
      }
      const k = Math.min(rows, cols);
      const ir = new Int32Array(k);
      const pr = new Float64Array(k);
      const jc = new Int32Array(cols + 1);
      for (let i = 0; i < k; i++) {
        ir[i] = i;
        pr[i] = 1;
      }
      for (let c = 0; c <= cols; c++) {
        jc[c] = Math.min(c, k);
      }
      return RTV.sparseMatrix(rows, cols, ir, jc, pr);
    },
  }),
});

// ── spconvert ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "spconvert",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: args => {
      if (args.length !== 1)
        throw new RuntimeError("spconvert requires 1 argument");
      const S = args[0];
      if (!isRuntimeTensor(S))
        throw new RuntimeError("spconvert: argument must be a matrix");
      const nrows = S.shape[0];
      const ncols = S.shape.length >= 2 ? S.shape[1] : 1;
      if (ncols < 3)
        throw new RuntimeError("spconvert: input must have at least 3 columns");

      const iArr: number[] = [];
      const jArr: number[] = [];
      const vArr: number[] = [];
      let m = 0;
      let n = 0;

      for (let k = 0; k < nrows; k++) {
        const i = S.data[k];
        const j = S.data[k + nrows];
        const v = S.data[k + 2 * nrows];
        if (i > m) m = i;
        if (j > n) n = j;
        if (v !== 0) {
          iArr.push(i);
          jArr.push(j);
          vArr.push(v);
        }
      }
      return buildSparseFromTriplets(iArr, jArr, vArr, m, n);
    },
  }),
});

// ── spdiags ──────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "spdiags",
  resolve: (_argTypes, nargout) => ({
    outputTypes:
      nargout >= 2
        ? [{ kind: "unknown" }, { kind: "unknown" }]
        : [{ kind: "unknown" }],
    apply: (args, nargout) => {
      function getDiags(v: RuntimeValue): number[] {
        if (isRuntimeNumber(v)) return [Math.round(v)];
        if (isRuntimeTensor(v)) {
          const res: number[] = [];
          for (let i = 0; i < v.data.length; i++)
            res.push(Math.round(v.data[i]));
          return res;
        }
        throw new RuntimeError("spdiags: diagonal indices must be numeric");
      }

      function matInfo(v: RuntimeValue) {
        if (isRuntimeNumber(v))
          return {
            re: new FloatXArray([v]),
            im: undefined as FloatXArrayType | undefined,
            rows: 1,
            cols: 1,
          };
        if (isRuntimeComplexNumber(v))
          return {
            re: new FloatXArray([v.re]),
            im: new FloatXArray([v.im]) as FloatXArrayType | undefined,
            rows: 1,
            cols: 1,
          };
        if (isRuntimeTensor(v))
          return {
            re: v.data,
            im: v.imag as FloatXArrayType | undefined,
            rows: v.shape[0] ?? 1,
            cols: v.shape.length >= 2 ? v.shape[1] : 1,
          };
        if (isRuntimeSparseMatrix(v)) {
          const data = new FloatXArray(v.m * v.n);
          const imag = v.pi ? new FloatXArray(v.m * v.n) : undefined;
          for (let c = 0; c < v.n; c++) {
            for (let k = v.jc[c]; k < v.jc[c + 1]; k++) {
              data[c * v.m + v.ir[k]] = v.pr[k];
              if (imag && v.pi) imag[c * v.m + v.ir[k]] = v.pi[k];
            }
          }
          return {
            re: data,
            im: imag as FloatXArrayType | undefined,
            rows: v.m,
            cols: v.n,
          };
        }
        throw new RuntimeError("spdiags: argument must be numeric");
      }

      function elemRow(
        m: number,
        n: number,
        dk: number,
        diagLen: number,
        j: number
      ): number {
        const minMN = Math.min(m, n);
        if (m >= n) return dk <= 0 ? j : minMN - diagLen + j;
        return dk >= 0 ? j : minMN - diagLen + j;
      }

      // S = spdiags(Bin, d, m, n)
      if (args.length === 4) {
        const {
          re: binRe,
          im: binIm,
          rows: binRows,
          cols: binCols,
        } = matInfo(args[0]);
        const diags = getDiags(args[1]);
        const m = Math.round(toNumber(args[2]));
        const n = Math.round(toNumber(args[3]));
        const isComplex = binIm !== undefined;

        const triplets: { row: number; col: number; re: number; im: number }[] =
          [];
        for (let k = 0; k < diags.length; k++) {
          const dk = diags[k];
          const iStart = Math.max(0, -dk);
          const jStart = Math.max(0, dk);
          const diagLen = Math.min(m - iStart, n - jStart);
          if (diagLen <= 0) continue;
          const bCol = Math.min(k, binCols - 1);
          for (let j = 0; j < diagLen; j++) {
            let br = elemRow(m, n, dk, diagLen, j);
            if (binRows === 1) br = 0;
            const binIdx = br + bCol * binRows;
            const re = binRe[binIdx];
            const im = isComplex && binIm ? binIm[binIdx] : 0;
            if (re !== 0 || im !== 0) {
              triplets.push({ row: iStart + j, col: jStart + j, re, im });
            }
          }
        }
        triplets.sort((a, b) => a.col - b.col || a.row - b.row);
        const nnz = triplets.length;
        const ir = new Int32Array(nnz);
        const pr = new Float64Array(nnz);
        const pi = isComplex ? new Float64Array(nnz) : undefined;
        const jc = new Int32Array(n + 1);
        let ti = 0;
        for (let c = 0; c < n; c++) {
          jc[c] = ti;
          while (ti < nnz && triplets[ti].col === c) {
            ir[ti] = triplets[ti].row;
            pr[ti] = triplets[ti].re;
            if (pi) pi[ti] = triplets[ti].im;
            ti++;
          }
        }
        jc[n] = ti;
        return RTV.sparseMatrix(m, n, ir, jc, pr, pi);
      }

      // S = spdiags(Bin, d, A)
      if (args.length === 3) {
        const {
          re: binRe,
          im: binIm,
          rows: binRows,
          cols: binCols,
        } = matInfo(args[0]);
        const diags = getDiags(args[1]);
        const A = args[2];
        if (!isRuntimeTensor(A))
          throw new RuntimeError("spdiags: third argument must be a matrix");
        const m = A.shape[0];
        const n = A.shape.length >= 2 ? A.shape[1] : 1;
        const data = new FloatXArray(A.data);
        const isComplex = binIm !== undefined || A.imag !== undefined;
        const idata = isComplex
          ? A.imag
            ? new FloatXArray(A.imag)
            : new FloatXArray(m * n)
          : undefined;

        for (let k = 0; k < diags.length; k++) {
          const dk = diags[k];
          const iStart = Math.max(0, -dk);
          const jStart = Math.max(0, dk);
          const diagLen = Math.min(m - iStart, n - jStart);
          if (diagLen <= 0) continue;
          const col = Math.min(k, binCols - 1);
          for (let j = 0; j < diagLen; j++) {
            let br = elemRow(m, n, dk, diagLen, j);
            if (binRows === 1) br = 0;
            const binIdx = br + col * binRows;
            const idx = iStart + j + (jStart + j) * m;
            data[idx] = binRe[binIdx];
            if (idata) idata[idx] = binIm ? binIm[binIdx] : 0;
          }
        }
        return RTV.tensor(data, [m, n], idata);
      }

      // Bout = spdiags(A, d)
      if (args.length === 2) {
        const { re: aRe, im: aIm, rows: m, cols: n } = matInfo(args[0]);
        const diags = getDiags(args[1]);
        const minMN = Math.min(m, n);
        const p = diags.length;
        const data = new FloatXArray(minMN * p);
        const isComplex = aIm !== undefined;
        const idata = isComplex ? new FloatXArray(minMN * p) : undefined;

        for (let k = 0; k < p; k++) {
          const dk = diags[k];
          const iStart = Math.max(0, -dk);
          const jStart = Math.max(0, dk);
          const diagLen = Math.min(m - iStart, n - jStart);
          for (let j = 0; j < diagLen; j++) {
            const br = elemRow(m, n, dk, diagLen, j);
            const aIdx = iStart + j + (jStart + j) * m;
            const boutIdx = br + k * minMN;
            data[boutIdx] = aRe[aIdx];
            if (idata && aIm) idata[boutIdx] = aIm[aIdx];
          }
        }
        return RTV.tensor(data, [minMN, p], idata);
      }

      // [Bout, id] = spdiags(A)
      if (args.length === 1) {
        const { re: aRe, im: aIm, rows: m, cols: n } = matInfo(args[0]);
        const minMN = Math.min(m, n);
        const nonzeroDiags: number[] = [];
        for (let dk = -(m - 1); dk <= n - 1; dk++) {
          const iStart = Math.max(0, -dk);
          const jStart = Math.max(0, dk);
          const diagLen = Math.min(m - iStart, n - jStart);
          let hasNonzero = false;
          for (let j = 0; j < diagLen; j++) {
            const idx = iStart + j + (jStart + j) * m;
            if (aRe[idx] !== 0 || (aIm && aIm[idx] !== 0)) {
              hasNonzero = true;
              break;
            }
          }
          if (hasNonzero) nonzeroDiags.push(dk);
        }

        const p = nonzeroDiags.length;
        if (p === 0) {
          const empty = RTV.tensor(new FloatXArray(0), [minMN, 0]);
          if (nargout <= 1) return empty;
          return [empty, RTV.tensor(new FloatXArray(0), [0, 1])];
        }

        const data = new FloatXArray(minMN * p);
        const isComplex = aIm !== undefined;
        const idata = isComplex ? new FloatXArray(minMN * p) : undefined;

        for (let k = 0; k < p; k++) {
          const dk = nonzeroDiags[k];
          const iStart = Math.max(0, -dk);
          const jStart = Math.max(0, dk);
          const diagLen = Math.min(m - iStart, n - jStart);
          for (let j = 0; j < diagLen; j++) {
            const br = elemRow(m, n, dk, diagLen, j);
            const aIdx = iStart + j + (jStart + j) * m;
            const boutIdx = br + k * minMN;
            data[boutIdx] = aRe[aIdx];
            if (idata && aIm) idata[boutIdx] = aIm[aIdx];
          }
        }

        const Bout = RTV.tensor(data, [minMN, p], idata);
        if (nargout <= 1) return Bout;
        const idData = new FloatXArray(p);
        for (let i = 0; i < p; i++) idData[i] = nonzeroDiags[i];
        return [Bout, RTV.tensor(idData, [p, 1])];
      }

      throw new RuntimeError("spdiags requires 1 to 4 arguments");
    },
  }),
});
