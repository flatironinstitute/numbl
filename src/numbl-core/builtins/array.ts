/**
 * Array construction builtin functions
 */

import {
  RuntimeValue,
  RTV,
  RuntimeError,
  toNumber,
  numel,
  mRange,
  colMajorIndex,
} from "../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeNumber,
  isRuntimeTensor,
  isRuntimeComplexNumber,
} from "../runtime/types.js";
import {
  register,
  builtinSingle,
  realArrayConstructorCheck,
} from "./registry.js";
import { parseShapeArgs } from "./shape-utils.js";

export function registerArrayFunctions(): void {
  register("zeros", [
    {
      check: realArrayConstructorCheck,
      apply: args => {
        if (args.length === 0) return RTV.num(0);
        const shape = parseShapeArgs(args);
        if (shape.length === 1) shape.push(shape[0]); // zeros(n) = zeros(n,n)
        const n = numel(shape);
        return RTV.tensor(new FloatXArray(n), shape);
      },
    },
  ]);

  register("ones", [
    {
      check: realArrayConstructorCheck,
      apply: args => {
        if (args.length === 0) return RTV.num(1);
        const shape = parseShapeArgs(args);
        if (shape.length === 1) shape.push(shape[0]);
        const n = numel(shape);
        const data = new FloatXArray(n);
        data.fill(1);
        return RTV.tensor(data, shape);
      },
    },
  ]);

  const nanApply = (args: RuntimeValue[]) => {
    if (args.length === 0) return RTV.num(NaN);
    const shape = parseShapeArgs(args);
    if (shape.length === 1) shape.push(shape[0]); // NaN(n) = NaN(n,n)
    const n = numel(shape);
    const data = new FloatXArray(n);
    data.fill(NaN);
    return RTV.tensor(data, shape);
  };
  register("NaN", [{ check: realArrayConstructorCheck, apply: nanApply }]);
  register("nan", [{ check: realArrayConstructorCheck, apply: nanApply }]);

  const eyeApply = (args: RuntimeValue[]): RuntimeValue => {
    let rows: number, cols: number;
    if (args.length === 0) {
      rows = 1;
      cols = 1;
    } else if (args.length === 1) {
      const shape = parseShapeArgs(args);
      if (shape.length >= 2) {
        rows = shape[0];
        cols = shape[1];
      } else {
        rows = shape[0];
        cols = rows;
      }
    } else {
      rows = Math.round(toNumber(args[0]));
      cols = Math.round(toNumber(args[1]));
    }
    const data = new FloatXArray(rows * cols);
    const minDim = Math.min(rows, cols);
    for (let i = 0; i < minDim; i++) {
      data[colMajorIndex(i, i, rows)] = 1;
    }
    return RTV.tensor(data, [rows, cols]);
  };

  register("eye", [{ check: realArrayConstructorCheck, apply: eyeApply }]);
  register("speye", [{ check: realArrayConstructorCheck, apply: eyeApply }]);

  register(
    "linspace",
    builtinSingle(args => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("linspace requires 2 or 3 arguments");
      const start = toNumber(args[0]);
      const end = toNumber(args[1]);
      const n = args.length === 3 ? Math.round(toNumber(args[2])) : 100;
      if (n <= 0) return RTV.tensor(new FloatXArray(0), [1, 0]);
      if (n === 1) return RTV.tensor(new FloatXArray([end]), [1, 1]);
      const data = new FloatXArray(n);
      for (let i = 0; i < n; i++) {
        data[i] = start + ((end - start) * i) / (n - 1);
      }
      return RTV.tensor(data, [1, n]);
    })
  );

  register(
    "colon",
    builtinSingle(args => {
      if (args.length === 2)
        return mRange(toNumber(args[0]), 1, toNumber(args[1]));
      if (args.length === 3)
        return mRange(toNumber(args[0]), toNumber(args[1]), toNumber(args[2]));
      throw new RuntimeError("colon requires 2 or 3 arguments");
    })
  );

  register(
    "toeplitz",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("toeplitz requires 1 or 2 arguments");

      // Extract real and imaginary data from a numeric scalar or vector argument.
      function vecData(v: RuntimeValue): {
        re: FloatXArrayType;
        im: FloatXArrayType | undefined;
        len: number;
      } {
        if (isRuntimeNumber(v))
          return { re: new FloatXArray([v]), im: undefined, len: 1 };
        if (isRuntimeComplexNumber(v))
          return {
            re: new FloatXArray([v.re]),
            im: new FloatXArray([v.im]),
            len: 1,
          };
        if (isRuntimeTensor(v))
          return { re: v.data, im: v.imag, len: v.data.length };
        throw new RuntimeError("toeplitz: arguments must be numeric vectors");
      }

      // col = first column, row = first row.
      let colRe: FloatXArrayType,
        colIm: FloatXArrayType | undefined,
        colLen: number,
        rowRe: FloatXArrayType,
        rowIm: FloatXArrayType | undefined,
        rowLen: number;

      if (args.length === 1) {
        // Symmetric/Hermitian case: row = r, col = conj(r) with col(1) = r(1).
        const { re, im, len } = vecData(args[0]);
        rowRe = re;
        rowIm = im;
        rowLen = len;
        colRe = new FloatXArray(re); // copy
        colLen = len;
        if (im) {
          const ci = new FloatXArray(im); // copy then negate off-diagonal
          for (let k = 1; k < len; k++) ci[k] = -im[k];
          colIm = ci;
        }
      } else {
        // Nonsymmetric case: col = args[0], row = args[1]. Column wins diagonal.
        const c = vecData(args[0]);
        const r = vecData(args[1]);
        colRe = c.re;
        colIm = c.im;
        colLen = c.len;
        rowRe = r.re;
        rowIm = r.im;
        rowLen = r.len;
        // Diagonal comes from col (row[0] is unused in the construction below).
      }

      const m = colLen; // number of rows
      const n = rowLen; // number of cols
      const isComplex = colIm !== undefined || rowIm !== undefined;
      const data = new FloatXArray(m * n);
      const idata = isComplex ? new FloatXArray(m * n) : undefined;

      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          const idx = i + j * m; // column-major
          const diag = i - j; // >=0: on/below diagonal (use col); <0: above (use row)
          if (diag >= 0) {
            data[idx] = colRe[diag];
            if (idata) idata[idx] = colIm ? colIm[diag] : 0;
          } else {
            const k = -diag;
            data[idx] = rowRe[k];
            if (idata) idata[idx] = rowIm ? rowIm[k] : 0;
          }
        }
      }

      return RTV.tensor(data, [m, n], idata);
    })
  );

  register(
    "spdiags",
    builtinSingle((args, nargout) => {
      // Helper: extract diagonal index vector from a runtime value.
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

      // Helper: get matrix info from a runtime value.
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
        throw new RuntimeError("spdiags: argument must be numeric");
      }

      // Helper: compute the Bin row index for the j-th element on diagonal dk.
      // When m >= n: sub-diags take from top, super-diags from bottom.
      // When m < n: super-diags take from top, sub-diags from bottom.
      function elemRow(
        m: number,
        n: number,
        dk: number,
        diagLen: number,
        j: number
      ): number {
        const minMN = Math.min(m, n);
        if (m >= n) {
          return dk <= 0 ? j : minMN - diagLen + j;
        } else {
          return dk >= 0 ? j : minMN - diagLen + j;
        }
      }

      // --- S = spdiags(Bin, d, m, n) --- create matrix from diagonals
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

        const data = new FloatXArray(m * n);
        const isComplex = binIm !== undefined;
        const idata = isComplex ? new FloatXArray(m * n) : undefined;

        for (let k = 0; k < diags.length; k++) {
          const dk = diags[k];
          const iStart = Math.max(0, -dk);
          const jStart = Math.max(0, dk);
          const diagLen = Math.min(m - iStart, n - jStart);
          if (diagLen <= 0) continue;

          const col = Math.min(k, binCols - 1);
          for (let j = 0; j < diagLen; j++) {
            let br = elemRow(m, n, dk, diagLen, j);
            if (binRows === 1) br = 0; // broadcast row vectors
            const binIdx = br + col * binRows;
            const idx = iStart + j + (jStart + j) * m;
            data[idx] = binRe[binIdx];
            if (idata && binIm) idata[idx] = binIm[binIdx];
          }
        }

        return RTV.tensor(data, [m, n], idata);
      }

      // --- S = spdiags(Bin, d, A) --- replace diagonals in A
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

      // --- Bout = spdiags(A, d) --- extract specific diagonals
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

      // --- [Bout, id] = spdiags(A) --- extract all nonzero diagonals
      if (args.length === 1) {
        const { re: aRe, im: aIm, rows: m, cols: n } = matInfo(args[0]);
        const minMN = Math.min(m, n);

        // Find all nonzero diagonals
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
    })
  );

  // Helper: extract triangular part of a matrix.
  // keepFn(i, j, k) returns true for (0-based) row i, col j with diagonal offset k.
  function triPart(
    args: RuntimeValue[],
    keepFn: (i: number, j: number, k: number) => boolean
  ): RuntimeValue {
    if (args.length < 1 || args.length > 2)
      throw new RuntimeError("triu/tril requires 1 or 2 arguments");
    const k = args.length === 2 ? Math.round(toNumber(args[1])) : 0;
    const v = args[0];
    if (isRuntimeNumber(v)) return keepFn(0, 0, k) ? v : RTV.num(0);
    if (!isRuntimeTensor(v))
      throw new RuntimeError("triu/tril: argument must be a matrix");
    const nrows = v.shape[0] ?? 1;
    const ncols = v.shape.length >= 2 ? v.shape[1] : 1;
    const data = new FloatXArray(nrows * ncols);
    const idata = v.imag ? new FloatXArray(nrows * ncols) : undefined;
    for (let j = 0; j < ncols; j++) {
      for (let i = 0; i < nrows; i++) {
        if (keepFn(i, j, k)) {
          const idx = i + j * nrows;
          data[idx] = v.data[idx];
          if (idata && v.imag) idata[idx] = v.imag[idx];
        }
      }
    }
    return RTV.tensor(data, [nrows, ncols], idata);
  }

  register(
    "triu",
    builtinSingle(args => triPart(args, (i, j, k) => j - i >= k))
  );

  register(
    "tril",
    builtinSingle(args => triPart(args, (i, j, k) => i - j >= -k))
  );

  register(
    "logspace",
    builtinSingle(args => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("logspace requires 2 or 3 arguments");
      const a = toNumber(args[0]);
      const b = toNumber(args[1]);
      const n = args.length === 3 ? Math.round(toNumber(args[2])) : 50;
      if (n <= 0) return RTV.tensor(new FloatXArray(0), [1, 0]);
      // Special case: when b == pi, endpoint is pi instead of 10^pi
      const isPi = b === Math.PI;
      const endVal = isPi ? Math.PI : Math.pow(10, b);
      const startVal = Math.pow(10, a);
      if (n === 1) return RTV.tensor(new FloatXArray([endVal]), [1, 1]);
      const data = new FloatXArray(n);
      if (isPi) {
        // Logarithmically spaced between 10^a and pi
        const logStart = Math.log10(startVal);
        const logEnd = Math.log10(Math.PI);
        for (let i = 0; i < n; i++) {
          const t = logStart + ((logEnd - logStart) * i) / (n - 1);
          data[i] = Math.pow(10, t);
        }
      } else {
        for (let i = 0; i < n; i++) {
          const t = a + ((b - a) * i) / (n - 1);
          data[i] = Math.pow(10, t);
        }
      }
      return RTV.tensor(data, [1, n]);
    })
  );

  register(
    "magic",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("magic requires exactly 1 argument");
      const n = Math.round(toNumber(args[0]));
      if (n < 3)
        throw new RuntimeError("Order N must be greater than or equal to 3");

      const M = new Array<number>(n * n).fill(0);
      // Helper to set M(row, col) in column-major order
      const set = (r: number, c: number, v: number) => {
        M[c * n + r] = v;
      };
      const get = (r: number, c: number) => M[c * n + r];

      if (n % 2 === 1) {
        // Odd order: Siamese method
        let i = 0;
        let j = Math.floor(n / 2);
        for (let k = 1; k <= n * n; k++) {
          set(i, j, k);
          const ni = (i - 1 + n) % n;
          const nj = (j + 1) % n;
          if (get(ni, nj) !== 0) {
            i = (i + 1) % n;
          } else {
            i = ni;
            j = nj;
          }
        }
      } else if (n % 4 === 0) {
        // Doubly even order
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            set(i, j, i * n + j + 1);
          }
        }
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            const ii = i % 4;
            const jj = j % 4;
            if (
              ((ii === 0 || ii === 3) && (jj === 0 || jj === 3)) ||
              ((ii === 1 || ii === 2) && (jj === 1 || jj === 2))
            ) {
              set(i, j, n * n + 1 - get(i, j));
            }
          }
        }
      } else {
        // Singly even order (n = 4k+2)
        const p = n / 2;
        // Build magic square of order p (odd) in each quadrant
        const sub = new Array<number>(p * p).fill(0);
        const sset = (r: number, c: number, v: number) => {
          sub[c * p + r] = v;
        };
        const sget = (r: number, c: number) => sub[c * p + r];

        // Siamese for sub
        let si = 0;
        let sj = Math.floor(p / 2);
        for (let k = 1; k <= p * p; k++) {
          sset(si, sj, k);
          const ni = (si - 1 + p) % p;
          const nj = (sj + 1) % p;
          if (sget(ni, nj) !== 0) {
            si = (si + 1) % p;
          } else {
            si = ni;
            sj = nj;
          }
        }

        // Place sub into quadrants with offsets
        // A (top-left) = sub, B (top-right) = sub + 2*p^2,
        // C (bottom-left) = sub + 3*p^2, D (bottom-right) = sub + p^2
        for (let i = 0; i < p; i++) {
          for (let j = 0; j < p; j++) {
            const v = sget(i, j);
            set(i, j, v); // A
            set(i, j + p, v + 2 * p * p); // B
            set(i + p, j, v + 3 * p * p); // C
            set(i + p, j + p, v + p * p); // D
          }
        }

        // Swap columns to fix magic property
        const k = Math.floor((n - 2) / 4);
        for (let i = 0; i < p; i++) {
          for (let j = 0; j < k; j++) {
            if (j === 0) {
              // Swap A[i,0] with C[i,0], except middle row
              if (i === Math.floor(p / 2)) {
                // Middle row: swap column 1 instead of 0
                const tmp = get(i, 1);
                set(i, 1, get(i + p, 1));
                set(i + p, 1, tmp);
              } else {
                const tmp = get(i, 0);
                set(i, 0, get(i + p, 0));
                set(i + p, 0, tmp);
              }
            } else {
              const tmp = get(i, j);
              set(i, j, get(i + p, j));
              set(i + p, j, tmp);
            }
          }
          // Swap rightmost columns
          for (let j = n - k + 2; j < n; j++) {
            const tmp = get(i, j);
            set(i, j, get(i + p, j));
            set(i + p, j, tmp);
          }
        }
      }

      const data = new FloatXArray(M);
      return RTV.tensor(data, [n, n]);
    })
  );
}
