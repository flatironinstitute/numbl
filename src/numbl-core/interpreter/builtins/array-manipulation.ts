/**
 * Array manipulation builtins for the interpreter system:
 * reshape, diag, cat, horzcat, vertcat, flip, fliplr, flipud, rot90,
 * repmat, repelem, squeeze, circshift, permute, ipermute,
 * ndgrid, meshgrid, sub2ind, ind2sub.
 */

import {
  RTV,
  toNumber,
  RuntimeError,
  tensorSize2D,
  colMajorIndex,
  numel,
  horzcat,
  vertcat,
} from "../../runtime/index.js";
import {
  RuntimeTensor,
  RuntimeChar,
  RuntimeClassInstance,
  type RuntimeValue,
  type RuntimeClassInstanceArray,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeChar,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeCell,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
} from "../../runtime/types.js";
import { defineBuiltin } from "./types.js";
import type { JitType } from "../../jitTypes.js";
import { coerceToTensor } from "../../helpers/shape-utils.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import { mTranspose, mConjugateTranspose } from "../../helpers/arithmetic.js";
import { allocFloat64Array, releaseFloat64Array } from "../../runtime/alloc.js";

// ── Helpers ──────────────────────────────────────────────────────────

/** Flip a tensor along a specific dimension (0-based dimIdx). N-D safe. */
function flipAlongDim(v: RuntimeTensor, dimIdx: number): RuntimeTensor {
  const shape = v.shape;
  const totalElems = numel(shape);
  const result = allocFloat64Array(totalElems);
  const resultImag = v.imag ? allocFloat64Array(totalElems) : undefined;
  const dimSize = dimIdx < shape.length ? shape[dimIdx] : 1;

  let strideDim = 1;
  for (let d = 0; d < dimIdx; d++) strideDim *= shape[d];
  const slabSize = strideDim * dimSize;
  const numOuter = totalElems / slabSize;

  for (let outer = 0; outer < numOuter; outer++) {
    const base = outer * slabSize;
    for (let k = 0; k < dimSize; k++) {
      const srcOff = base + k * strideDim;
      const dstOff = base + (dimSize - 1 - k) * strideDim;
      result.set(v.data.subarray(srcOff, srcOff + strideDim), dstOff);
      if (resultImag) {
        resultImag.set(v.imag!.subarray(srcOff, srcOff + strideDim), dstOff);
      }
    }
  }

  return RTV.tensor(result, [...shape], resultImag) as RuntimeTensor;
}

/** Flip a sparse matrix along dimIdx: 0=rows (flipud), 1=cols (fliplr). */
function flipSparse(
  v: import("../../runtime/types.js").RuntimeSparseMatrix,
  dimIdx: number
): import("../../runtime/types.js").RuntimeSparseMatrix {
  const isComplex = v.pi !== undefined;
  const nnz = v.jc[v.n];

  if (dimIdx === 1) {
    const ir = new Int32Array(nnz);
    const pr = allocFloat64Array(nnz);
    const pi = isComplex ? allocFloat64Array(nnz) : undefined;
    const jc = new Int32Array(v.n + 1);
    let dst = 0;
    for (let c = 0; c < v.n; c++) {
      const origCol = v.n - 1 - c;
      jc[c] = dst;
      for (let k = v.jc[origCol]; k < v.jc[origCol + 1]; k++) {
        ir[dst] = v.ir[k];
        pr[dst] = v.pr[k];
        if (pi && v.pi) pi[dst] = v.pi[k];
        dst++;
      }
    }
    jc[v.n] = dst;
    return RTV.sparseMatrix(v.m, v.n, ir, jc, pr, pi);
  }
  const ir = new Int32Array(nnz);
  const pr = allocFloat64Array(nnz);
  const pi = isComplex ? allocFloat64Array(nnz) : undefined;
  const jc = new Int32Array(v.n + 1);
  let dst = 0;
  for (let c = 0; c < v.n; c++) {
    jc[c] = dst;
    const start = v.jc[c];
    const end = v.jc[c + 1];
    for (let k = end - 1; k >= start; k--) {
      ir[dst] = v.m - 1 - v.ir[k];
      pr[dst] = v.pr[k];
      if (pi && v.pi) pi[dst] = v.pi[k];
      dst++;
    }
  }
  jc[v.n] = dst;
  return RTV.sparseMatrix(v.m, v.n, ir, jc, pr, pi);
}

// Match helper: accepts any number of args, returns unknown output type
function varargMatch(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length < 1) return null;
  return [{ kind: "unknown" }];
}

function varargMatch2(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length < 2) return null;
  return [{ kind: "unknown" }];
}

// ── reshape ──────────────────────────────────────────────────────────

// Size / replication args must be integers — MATLAB errors on a
// non-integer (reshape(x,2.5,[]), repmat(x,1,2.5)), matching numbl's
// array-construction validateDim and the JIT's mtoc2_check_dim. A
// negative clamps to 0 (empty axis), as validateDim does.
function validateSizeArg(x: number): number {
  if (!Number.isFinite(x) || !Number.isInteger(x)) {
    throw new RuntimeError("Size inputs must be nonnegative integers.");
  }
  return x < 0 ? 0 : x;
}

/** Resolve reshape's target shape from its dimension arguments (args[1..])
 *  against a known element count. Handles the [] auto-dimension and the
 *  reshape(x, [d1 d2 ...]) vector form, and validates the element count. */
function parseReshapeDims(args: RuntimeValue[], total: number): number[] {
  let rawDims: (number | null)[];
  if (
    args.length === 2 &&
    isRuntimeTensor(args[1]) &&
    args[1].data.length > 1
  ) {
    rawDims = Array.from(args[1].data).map(x => validateSizeArg(x));
  } else {
    rawDims = args.slice(1).map(a => {
      if (isRuntimeTensor(a) && a.data.length === 0) return null;
      return validateSizeArg(toNumber(a));
    });
  }
  const autoCount = rawDims.filter(d => d === null).length;
  if (autoCount > 1)
    throw new RuntimeError("reshape: only one dimension size can be []");
  let shape: number[];
  if (autoCount === 1) {
    const known = rawDims.filter(d => d !== null) as number[];
    const knownProduct = known.reduce((a, b) => a * b, 1);
    if (knownProduct === 0 || total % knownProduct !== 0)
      throw new RuntimeError("reshape: number of elements must not change");
    shape = rawDims.map(d => (d === null ? total / knownProduct : d));
  } else {
    shape = rawDims as number[];
  }
  if (numel(shape) !== total)
    throw new RuntimeError("reshape: number of elements must not change");
  return shape;
}

defineBuiltin({
  name: "reshape",
  cases: [
    {
      match: varargMatch2,
      apply: args => {
        if (args.length < 2)
          throw new RuntimeError("reshape requires at least 2 arguments");
        const v = args[0];

        // Sparse reshape
        if (isRuntimeSparseMatrix(v)) {
          const totalEl = v.m * v.n;
          let rawDims: (number | null)[];
          if (
            args.length === 2 &&
            isRuntimeTensor(args[1]) &&
            args[1].data.length > 1
          ) {
            rawDims = Array.from(args[1].data).map(x => Math.round(x));
          } else {
            rawDims = args.slice(1).map(a => {
              if (isRuntimeTensor(a) && a.data.length === 0) return null;
              return Math.round(toNumber(a));
            });
          }
          const autoCount = rawDims.filter(d => d === null).length;
          if (autoCount > 1)
            throw new RuntimeError(
              "reshape: only one dimension size can be []"
            );
          let shape: number[];
          if (autoCount === 1) {
            const known = rawDims.filter(d => d !== null) as number[];
            const knownProduct = known.reduce((a, b) => a * b, 1);
            if (totalEl % knownProduct !== 0)
              throw new RuntimeError(
                "reshape: number of elements must not change"
              );
            shape = rawDims.map(d => (d === null ? totalEl / knownProduct : d));
          } else {
            shape = rawDims as number[];
          }
          if (shape.length !== 2)
            throw new RuntimeError("reshape: sparse matrices must be 2-D");
          const newM = shape[0];
          const newN = shape[1];
          if (newM * newN !== totalEl)
            throw new RuntimeError(
              "reshape: number of elements must not change"
            );
          const nnz = v.jc[v.n];
          const triplets: { row: number; col: number; idx: number }[] = [];
          for (let c = 0; c < v.n; c++) {
            for (let k = v.jc[c]; k < v.jc[c + 1]; k++) {
              const lin = c * v.m + v.ir[k];
              const newCol = Math.floor(lin / newM);
              const newRow = lin % newM;
              triplets.push({ row: newRow, col: newCol, idx: k });
            }
          }
          triplets.sort((a, b) => a.col - b.col || a.row - b.row);
          const ir = new Int32Array(nnz);
          const pr = allocFloat64Array(nnz);
          const pi = v.pi ? allocFloat64Array(nnz) : undefined;
          const jc = new Int32Array(newN + 1);
          let ti = 0;
          for (let c = 0; c < newN; c++) {
            jc[c] = ti;
            while (ti < nnz && triplets[ti].col === c) {
              ir[ti] = triplets[ti].row;
              pr[ti] = v.pr[triplets[ti].idx];
              if (pi && v.pi) pi[ti] = v.pi[triplets[ti].idx];
              ti++;
            }
          }
          jc[newN] = ti;
          return RTV.sparseMatrix(newM, newN, ir, jc, pr, pi);
        }

        // Char arrays reshape too (MATLAB accepts any type). Reorder the
        // characters column-major into the new 2-D shape.
        if (isRuntimeChar(v)) {
          const srcRows = v.shape ? v.shape[0] : 1;
          const srcCols = v.shape ? (v.shape[1] ?? 0) : v.value.length;
          const total = v.value.length;
          const reqShape = parseReshapeDims(args, total);
          const s = [...reqShape];
          while (s.length > 2 && s[s.length - 1] === 1) s.pop();
          if (s.length > 2)
            throw new RuntimeError("reshape: char arrays must be 2-D");
          const nr = s[0];
          const nc = s.length >= 2 ? s[1] : 1;
          // column-major linear order of the source (stored row-major)
          const linear: string[] = new Array(total);
          for (let j = 0; j < srcCols; j++)
            for (let i = 0; i < srcRows; i++)
              linear[j * srcRows + i] = v.value[i * srcCols + j];
          const chars: string[] = new Array(total);
          for (let j2 = 0; j2 < nc; j2++)
            for (let i2 = 0; i2 < nr; i2++)
              chars[i2 * nc + j2] = linear[j2 * nr + i2];
          return new RuntimeChar(chars.join(""), [nr, nc]);
        }

        if (
          !isRuntimeTensor(v) &&
          !isRuntimeNumber(v) &&
          !isRuntimeComplexNumber(v)
        )
          throw new RuntimeError("reshape: first argument must be numeric");
        const data = isRuntimeTensor(v)
          ? v.data
          : isRuntimeComplexNumber(v)
            ? allocFloat64Array([v.re])
            : allocFloat64Array([v as number]);
        const imag = isRuntimeTensor(v)
          ? v.imag
          : isRuntimeComplexNumber(v)
            ? allocFloat64Array([v.im])
            : undefined;

        const shape = parseReshapeDims(args, data.length);
        if (isRuntimeTensor(v)) {
          // Copy the data buffer rather than share it. Sharing buffers
          // across wrappers would require ref-tracking the buffer
          // itself (multiple wrappers can be alive at once), and the
          // current refcount design ties buffer lifetime to a single
          // wrapper. The cost is one extra Float64Array copy on
          // reshape, traded for simpler refcount semantics.
          const dataCopy = allocFloat64Array(data);
          const imagCopy = imag ? allocFloat64Array(imag) : undefined;
          const s = [...shape];
          while (s.length > 2 && s[s.length - 1] === 1) s.pop();
          return new RuntimeTensor(dataCopy, s, imagCopy, v._isLogical);
        }
        return RTV.tensor(
          allocFloat64Array(data),
          shape,
          imag ? allocFloat64Array(imag) : undefined
        );
      },
    },
  ],
});

// ── transpose / ctranspose ───────────────────────────────────────────

defineBuiltin({
  name: "transpose",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => mTranspose(args[0]),
    },
  ],
});

defineBuiltin({
  name: "ctranspose",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => mConjugateTranspose(args[0]),
    },
  ],
});

// ── diag ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "diag",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const v = args[0];
        const k = args.length >= 2 ? Math.round(toNumber(args[1])) : 0;
        const absK = Math.abs(k);
        if (isRuntimeNumber(v)) {
          const m = 1 + absK;
          const data = allocFloat64Array(m * m);
          const r = k < 0 ? -k : 0;
          const c = k > 0 ? k : 0;
          data[colMajorIndex(r, c, m)] = v as number;
          return RTV.tensor(data, [m, m]);
        }
        if (isRuntimeComplexNumber(v)) {
          const m = 1 + absK;
          const data = allocFloat64Array(m * m);
          const imag = allocFloat64Array(m * m);
          const r = k < 0 ? -k : 0;
          const c = k > 0 ? k : 0;
          data[colMajorIndex(r, c, m)] = v.re;
          imag[colMajorIndex(r, c, m)] = v.im;
          return RTV.tensor(data, [m, m], imag);
        }
        if (isRuntimeSparseMatrix(v)) {
          const m = v.m;
          const n = v.n;
          const isVec = m === 1 || n === 1;
          if (isVec) {
            const vecLen = Math.max(m, n);
            const sz = vecLen + absK;
            const irArr: number[] = [];
            const prArr: number[] = [];
            const piArr: number[] = [];
            const isComplex = v.pi !== undefined;
            for (let c2 = 0; c2 < v.n; c2++) {
              for (let kk = v.jc[c2]; kk < v.jc[c2 + 1]; kk++) {
                const vecIdx = m === 1 ? c2 : v.ir[kk];
                irArr.push(vecIdx + (k < 0 ? -k : 0));
                prArr.push(v.pr[kk]);
                if (isComplex) piArr.push(v.pi![kk]);
              }
            }
            const entries: {
              row: number;
              col: number;
              re: number;
              im: number;
            }[] = [];
            for (let i = 0; i < irArr.length; i++) {
              const r = irArr[i];
              if (k < 0) {
                const vecIdx = r - absK;
                entries.push({
                  row: r,
                  col: vecIdx,
                  re: prArr[i],
                  im: isComplex ? piArr[i] : 0,
                });
              } else {
                entries.push({
                  row: r,
                  col: r + k,
                  re: prArr[i],
                  im: isComplex ? piArr[i] : 0,
                });
              }
            }
            entries.sort((a2, b) => a2.col - b.col || a2.row - b.row);
            const newIr = new Int32Array(entries.length);
            const newPr = allocFloat64Array(entries.length);
            const newPi = isComplex
              ? allocFloat64Array(entries.length)
              : undefined;
            const jc = new Int32Array(sz + 1);
            let ti = 0;
            for (let c2 = 0; c2 < sz; c2++) {
              jc[c2] = ti;
              while (ti < entries.length && entries[ti].col === c2) {
                newIr[ti] = entries[ti].row;
                newPr[ti] = entries[ti].re;
                if (newPi) newPi[ti] = entries[ti].im;
                ti++;
              }
            }
            jc[sz] = ti;
            return RTV.sparseMatrix(sz, sz, newIr, jc, newPr, newPi);
          }
          // Sparse matrix → extract k-th diagonal
          const iStart = Math.max(0, -k);
          const jStart = Math.max(0, k);
          const diagLen = Math.min(m - iStart, n - jStart);
          if (diagLen <= 0) {
            return RTV.sparseMatrix(
              0,
              1,
              new Int32Array(0),
              new Int32Array([0]),
              allocFloat64Array(0)
            );
          }
          const isComplex = v.pi !== undefined;
          const dIr: number[] = [];
          const dPr: number[] = [];
          const dPi: number[] = [];
          for (let j = 0; j < diagLen; j++) {
            const row = iStart + j;
            const col = jStart + j;
            for (let kk = v.jc[col]; kk < v.jc[col + 1]; kk++) {
              if (v.ir[kk] === row) {
                dIr.push(j);
                dPr.push(v.pr[kk]);
                if (isComplex) dPi.push(v.pi![kk]);
                break;
              }
            }
          }
          const jcOut = new Int32Array([0, dIr.length]);
          return RTV.sparseMatrix(
            diagLen,
            1,
            new Int32Array(dIr),
            jcOut,
            allocFloat64Array(dPr),
            isComplex ? allocFloat64Array(dPi) : undefined
          );
        }

        if (!isRuntimeTensor(v))
          throw new RuntimeError("diag: argument must be numeric");
        const [rows, cols] = tensorSize2D(v);
        if (rows === 1 || cols === 1) {
          const vecLen = Math.max(rows, cols);
          const m2 = vecLen + absK;
          const data = allocFloat64Array(m2 * m2);
          const imag = v.imag ? allocFloat64Array(m2 * m2) : undefined;
          for (let i = 0; i < vecLen; i++) {
            const r = k < 0 ? i - k : i;
            const c = k > 0 ? i + k : i;
            data[colMajorIndex(r, c, m2)] = v.data[i];
            if (imag) imag[colMajorIndex(r, c, m2)] = v.imag![i];
          }
          return RTV.tensor(data, [m2, m2], imag);
        }
        const diagLen =
          k >= 0
            ? Math.max(0, Math.min(rows, cols - k))
            : Math.max(0, Math.min(rows + k, cols));
        const data = allocFloat64Array(diagLen);
        const imag = v.imag ? allocFloat64Array(diagLen) : undefined;
        for (let i = 0; i < diagLen; i++) {
          const r = k < 0 ? i - k : i;
          const c = k > 0 ? i + k : i;
          data[i] = v.data[colMajorIndex(r, c, rows)];
          if (imag) imag[i] = v.imag![colMajorIndex(r, c, rows)];
        }
        return RTV.tensor(data, [diagLen, 1], imag);
      },
    },
  ],
});

// ── cat ──────────────────────────────────────────────────────────────

defineBuiltin({
  name: "cat",
  cases: [
    {
      match: varargMatch,
      apply: args => {
        if (args.length < 1)
          throw new RuntimeError("cat requires at least 1 argument");
        const dimRaw = toNumber(args[0]);
        if (!Number.isFinite(dimRaw) || !Number.isInteger(dimRaw) || dimRaw < 1)
          throw new RuntimeError("cat: dimension must be a positive integer.");
        const dim = dimRaw;
        const arrays = args.slice(1);
        if (dim === 1) return vertcat(...arrays);
        if (dim === 2) return horzcat(...arrays);
        const dimIdx = dim - 1;
        let tensors = arrays.map(a => {
          if (isRuntimeNumber(a))
            return {
              data: allocFloat64Array([a as number]),
              imag: null as Float64Array | null,
              shape: [1, 1],
            };
          if (!isRuntimeTensor(a))
            throw new RuntimeError("cat: arguments must be numeric");
          return { data: a.data, imag: a.imag ?? null, shape: [...a.shape] };
        });
        // Drop fully-empty operands (all dims 0, e.g. the `[]` literal) before
        // the dimension-equality check. MATLAB ignores empties in cat; the
        // dim 1/2 paths (catAlongDim) already do this via shape.some(d>0), so
        // cat(3, A, []) must too. Done on the ORIGINAL shape, before padding
        // to `dim` adds trailing 1s (which would mask a [0,0]).
        tensors = tensors.filter(t => t.shape.some(d => d > 0));
        if (tensors.length === 0)
          return RTV.tensor(allocFloat64Array(0), [0, 0]);
        const hasComplex = tensors.some(t => t.imag !== null);
        for (const t of tensors) {
          while (t.shape.length < dim) t.shape.push(1);
        }
        const refShape = tensors[0].shape;
        for (let i = 1; i < tensors.length; i++) {
          for (let d = 0; d < refShape.length; d++) {
            if (d === dimIdx) continue;
            if (tensors[i].shape[d] !== refShape[d])
              throw new RuntimeError(
                `cat: dimension mismatch on dimension ${d + 1}`
              );
          }
        }
        const resultShape = [...refShape];
        resultShape[dimIdx] = tensors.reduce((s, t) => s + t.shape[dimIdx], 0);
        const totalElems = numel(resultShape);
        const result = allocFloat64Array(totalElems);
        const resultImag = hasComplex
          ? allocFloat64Array(totalElems)
          : undefined;

        const ndim = resultShape.length;
        let strideDim = 1;
        for (let d = 0; d < dimIdx; d++) strideDim *= resultShape[d];
        let numOuter = 1;
        for (let d = dimIdx + 1; d < ndim; d++) numOuter *= resultShape[d];

        for (let outer = 0; outer < numOuter; outer++) {
          let dstOff = outer * strideDim * resultShape[dimIdx];
          for (let t = 0; t < tensors.length; t++) {
            const srcDimSize = tensors[t].shape[dimIdx];
            const blockSize = strideDim * srcDimSize;
            const srcOff = outer * blockSize;
            for (let j = 0; j < blockSize; j++) {
              result[dstOff + j] = tensors[t].data[srcOff + j];
            }
            if (resultImag) {
              const srcImag = tensors[t].imag;
              if (srcImag) {
                for (let j = 0; j < blockSize; j++) {
                  resultImag[dstOff + j] = srcImag[srcOff + j];
                }
              }
            }
            dstOff += blockSize;
          }
        }
        return RTV.tensor(result, resultShape, resultImag);
      },
    },
  ],
});

// ── horzcat / vertcat ────────────────────────────────────────────────

defineBuiltin({
  name: "horzcat",
  cases: [
    {
      match: varargMatch,
      apply: args => horzcat(...args),
    },
  ],
});

defineBuiltin({
  name: "vertcat",
  cases: [
    {
      match: varargMatch,
      apply: args => vertcat(...args),
    },
  ],
});

// ── fliplr / flipud / flip ───────────────────────────────────────────

defineBuiltin({
  name: "fliplr",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeNumber(v)) return v;
        if (isRuntimeChar(v))
          return RTV.char(v.value.split("").reverse().join(""));
        if (isRuntimeSparseMatrix(v)) return flipSparse(v, 1);
        if (!isRuntimeTensor(v))
          throw new RuntimeError("fliplr: argument must be numeric or char");
        return flipAlongDim(v, 1);
      },
    },
  ],
});

defineBuiltin({
  name: "flipud",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeNumber(v)) return v;
        if (isRuntimeChar(v)) return v;
        if (isRuntimeSparseMatrix(v)) return flipSparse(v, 0);
        if (!isRuntimeTensor(v))
          throw new RuntimeError("flipud: argument must be numeric or char");
        return flipAlongDim(v, 0);
      },
    },
  ],
});

defineBuiltin({
  name: "flip",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeNumber(v)) return v;
        if (!isRuntimeTensor(v))
          throw new RuntimeError("flip: argument must be numeric");
        let dimIdx = 0;
        if (args.length >= 2) {
          dimIdx = Math.round(toNumber(args[1])) - 1;
        } else {
          const shape = v.shape.length >= 2 ? v.shape : [1, ...v.shape];
          dimIdx = shape.findIndex(d => d > 1);
          if (dimIdx === -1) dimIdx = 0;
        }
        return flipAlongDim(v, dimIdx);
      },
    },
  ],
});

// ── rot90 ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "rot90",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeNumber(v)) return v;
        if (!isRuntimeTensor(v))
          throw new RuntimeError("rot90: argument must be numeric");
        let k = args.length >= 2 ? Math.round(toNumber(args[1])) : 1;
        k = ((k % 4) + 4) % 4;
        if (k === 0) {
          const result = RTV.tensor(
            allocFloat64Array(v.data),
            [...v.shape],
            v.imag ? allocFloat64Array(v.imag) : undefined
          );
          if (v._isLogical) result._isLogical = true;
          return result;
        }
        const [rows, cols] = tensorSize2D(v);
        let data = v.data;
        let imag = v.imag;
        let r = rows,
          c = cols;
        for (let iter = 0; iter < k; iter++) {
          const newData = allocFloat64Array(r * c);
          const newImag = imag ? allocFloat64Array(r * c) : undefined;
          for (let i = 0; i < c; i++) {
            for (let j = 0; j < r; j++) {
              const srcIdx = (c - 1 - i) * r + j;
              const dstIdx = j * c + i;
              newData[dstIdx] = data[srcIdx];
              if (newImag) newImag[dstIdx] = imag![srcIdx];
            }
          }
          data = newData;
          imag = newImag;
          const tmp = r;
          r = c;
          c = tmp;
        }
        const result = RTV.tensor(data, [r, c], imag);
        if (v._isLogical) result._isLogical = true;
        return result;
      },
    },
  ],
});

// ── repmat ───────────────────────────────────────────────────────────

defineBuiltin({
  name: "repmat",
  cases: [
    {
      // Result is a tensor of the same complexity as arg[0]. Shape depends
      // on the reps args (runtime-only), so we leave `shape` undefined.
      match: argTypes => {
        if (argTypes.length < 2) return null;
        const a = argTypes[0];
        if (a.kind === "number" || a.kind === "boolean") {
          return [{ kind: "tensor", isComplex: false }];
        }
        if (a.kind === "tensor") {
          return [{ kind: "tensor", isComplex: a.isComplex === true }];
        }
        if (a.kind === "complex_or_number") {
          return [{ kind: "tensor", isComplex: true }];
        }
        return [{ kind: "unknown" }];
      },
      apply: args => {
        if (args.length < 2)
          throw new RuntimeError("repmat requires at least 2 arguments");
        let v = args[0];
        if (isRuntimeSparseMatrix(v)) v = sparseToDense(v);
        let reps: number[];
        if (args.length === 2) {
          const arg1 = args[1];
          if (isRuntimeTensor(arg1)) {
            reps = Array.from(arg1.data).map(x => validateSizeArg(x));
          } else {
            const n = validateSizeArg(toNumber(arg1));
            reps = [n, n];
          }
        } else {
          reps = args.slice(1).map(a => validateSizeArg(toNumber(a)));
        }
        if (isRuntimeNumber(v)) {
          const total = reps.reduce((a, b) => a * b, 1);
          const data = allocFloat64Array(total).fill(v as number);
          return RTV.tensor(data, reps.length >= 2 ? reps : [reps[0], reps[0]]);
        }
        if (isRuntimeLogical(v)) {
          const total = reps.reduce((a, b) => a * b, 1);
          const data = allocFloat64Array(total).fill(v ? 1 : 0);
          const shape = reps.length >= 2 ? reps : [reps[0], reps[0]];
          const t = RTV.tensor(data, shape);
          t._isLogical = true;
          return t;
        }
        if (isRuntimeChar(v)) {
          const rowReps = reps[0] ?? 1;
          const colReps = reps.length >= 2 ? reps[1] : 1;
          if (rowReps === 0 || colReps === 0) return RTV.char("");
          let row = "";
          for (let j = 0; j < colReps; j++) row += v.value;
          let result = "";
          for (let i = 0; i < rowReps; i++) result += row;
          return RTV.char(result);
        }
        if (isRuntimeComplexNumber(v)) {
          const total = reps.reduce((a, b) => a * b, 1);
          const data = allocFloat64Array(total).fill(v.re);
          const imag = allocFloat64Array(total).fill(v.im);
          const shape = reps.length >= 2 ? reps : [reps[0], reps[0]];
          return RTV.tensor(data, shape, imag);
        }
        if (isRuntimeCell(v)) {
          // Tile a cell array (column-major), like any other array type.
          const srcShape =
            v.shape.length >= 2 ? v.shape : [1, v.shape[0] ?? v.data.length];
          const [sm, sn] = [srcShape[0], srcShape[1]];
          const r0 = reps[0] ?? 1;
          const r1 = reps.length >= 2 ? reps[1] : (reps[0] ?? 1);
          const om = sm * r0;
          const on = sn * r1;
          const out: RuntimeValue[] = new Array(om * on);
          for (let J = 0; J < on; J++) {
            for (let I = 0; I < om; I++) {
              out[I + J * om] = v.data[(I % sm) + (J % sn) * sm];
            }
          }
          return RTV.cell(out, [om, on]);
        }
        if (!isRuntimeTensor(v))
          throw new RuntimeError("repmat: first argument must be numeric");
        // Fast path: all reps are 1 → return copy without data duplication
        if (reps.every(r => r === 1)) {
          return RTV.tensor(
            allocFloat64Array(v.data),
            v.shape,
            v.imag ? allocFloat64Array(v.imag) : undefined
          );
        }
        const srcShape = v.shape.length >= 2 ? v.shape : [1, v.shape[0] || 1];
        const ndim = Math.max(srcShape.length, reps.length);
        const padSrc = [...srcShape];
        while (padSrc.length < ndim) padSrc.push(1);
        const padReps = [...reps];
        while (padReps.length < ndim) padReps.push(1);
        const resultShape = padSrc.map((s, i) => s * padReps[i]);

        let curData: Float64Array = allocFloat64Array(v.data) as Float64Array;
        let curImag: Float64Array | undefined = v.imag
          ? (allocFloat64Array(v.imag) as Float64Array)
          : undefined;
        const curShape = [...padSrc];

        for (let d = 0; d < ndim; d++) {
          const rep = padReps[d];
          if (rep === 1) continue;
          const curTotal = curData.length;
          const newTotal = curTotal * rep;
          const newData = allocFloat64Array(newTotal);
          const newImag = curImag ? allocFloat64Array(newTotal) : undefined;

          if (rep === 0) {
            // 0 reps along this axis -> the result is empty (newTotal===0).
            // Skip the copy loops: the general (blockSize>1) branch would do
            // newData.set(subarray, ...) into a zero-length buffer (offset out
            // of bounds). Nothing to copy; just adopt the empty buffer.
            releaseFloat64Array(curData);
            if (curImag) releaseFloat64Array(curImag);
            curData = newData;
            curImag = newImag;
            curShape[d] = 0;
            continue;
          }

          let blockSize = 1;
          for (let i = 0; i <= d; i++) blockSize *= curShape[i];
          const numBlocks = curTotal / blockSize;

          if (blockSize === 1) {
            // Each block is a single scalar: fill `rep` consecutive positions
            // with the same value.  One TypedArray.fill() per source element,
            // which is a tight C loop inside V8.
            for (let b = 0; b < numBlocks; b++) {
              const dstBase = b * rep;
              newData.fill(curData[b], dstBase, dstBase + rep);
              if (newImag && curImag) {
                newImag.fill(curImag[b], dstBase, dstBase + rep);
              }
            }
          } else {
            // General case: copy the block once at the start of each tile,
            // then use copyWithin() to double it in place — O(log rep)
            // copyWithin calls per block instead of O(rep) .set() calls.
            for (let b = 0; b < numBlocks; b++) {
              const srcOff = b * blockSize;
              const dstBase = b * blockSize * rep;
              const totalToWrite = rep * blockSize;
              newData.set(
                curData.subarray(srcOff, srcOff + blockSize),
                dstBase
              );
              if (newImag && curImag) {
                newImag.set(
                  curImag.subarray(srcOff, srcOff + blockSize),
                  dstBase
                );
              }
              let written = blockSize;
              while (written * 2 <= totalToWrite) {
                newData.copyWithin(
                  dstBase + written,
                  dstBase,
                  dstBase + written
                );
                if (newImag) {
                  newImag.copyWithin(
                    dstBase + written,
                    dstBase,
                    dstBase + written
                  );
                }
                written *= 2;
              }
              if (written < totalToWrite) {
                const remaining = totalToWrite - written;
                newData.copyWithin(
                  dstBase + written,
                  dstBase,
                  dstBase + remaining
                );
                if (newImag) {
                  newImag.copyWithin(
                    dstBase + written,
                    dstBase,
                    dstBase + remaining
                  );
                }
              }
            }
          }
          // The prior curData/curImag are intermediate scratch buffers —
          // not held by any wrapper. Return them to the pool so they can
          // be reused (often by the next iteration's allocFloat64Array).
          releaseFloat64Array(curData);
          if (curImag) releaseFloat64Array(curImag);
          curData = newData;
          curImag = newImag;
          curShape[d] *= rep;
        }

        const out = RTV.tensor(curData, resultShape, curImag);
        if (v._isLogical) out._isLogical = true;
        return out;
      },
    },
  ],
});

// ── repelem ──────────────────────────────────────────────────────────

/** Copy a class instance for value-class replication (handle instances are
 *  shared by reference, matching MATLAB's repelem on object arrays). */
function copyClassInstance(inst: RuntimeClassInstance): RuntimeClassInstance {
  if (inst.isHandleClass) return inst;
  return new RuntimeClassInstance(
    inst.className,
    new Map(inst.fields),
    inst.isHandleClass,
    inst._builtinData
  );
}

/** repelem for class instances / object arrays. Mirrors the numeric cases but
 *  operates on the column-major element list and produces an object array. */
function repelemObjects(
  v: RuntimeClassInstance | RuntimeClassInstanceArray,
  repArgs: RuntimeValue[]
): RuntimeValue {
  const srcElements = isRuntimeClassInstanceArray(v) ? v.elements : [v];
  const [rows, cols]: [number, number] = isRuntimeClassInstanceArray(v)
    ? v.shape
    : [1, 1];
  const className = v.className;

  const wrap = (elements: RuntimeClassInstance[], shape: [number, number]) =>
    elements.length === 1
      ? elements[0]
      : RTV.classInstanceArray(className, elements, shape);

  if (repArgs.length === 1) {
    const repArg = repArgs[0];
    // Per-element count vector: repelem(v, [c1 c2 ...]).
    if (isRuntimeTensor(repArg) && repArg.data.length > 1) {
      const counts = repArg.data;
      if (counts.length !== srcElements.length)
        throw new RuntimeError(
          `repelem: counts vector length (${counts.length}) must match the number of elements (${srcElements.length})`
        );
      const out: RuntimeClassInstance[] = [];
      for (let i = 0; i < srcElements.length; i++) {
        const c = Math.max(0, Math.round(counts[i]));
        for (let j = 0; j < c; j++) out.push(copyClassInstance(srcElements[i]));
      }
      const isCol = cols === 1 && rows !== 1;
      return wrap(out, isCol ? [out.length, 1] : [1, out.length]);
    }
    // Scalar count: repeat each element n times, preserving vector orientation.
    const n = Math.max(0, Math.round(toNumber(repArg)));
    const out: RuntimeClassInstance[] = [];
    for (const el of srcElements)
      for (let j = 0; j < n; j++) out.push(copyClassInstance(el));
    const isCol = cols === 1 && rows !== 1;
    return wrap(out, isCol ? [out.length, 1] : [1, out.length]);
  }

  // Block form: repelem(v, rRep, cRep) — replicate each element rRep×cRep.
  const rRep = Math.max(0, Math.round(toNumber(repArgs[0])));
  const cRep = Math.max(0, Math.round(toNumber(repArgs[1])));
  const newRows = rows * rRep;
  const newCols = cols * cRep;
  const out: RuntimeClassInstance[] = new Array(newRows * newCols);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const src = srcElements[c * rows + r];
      for (let dc = 0; dc < cRep; dc++) {
        for (let dr = 0; dr < rRep; dr++) {
          const dstRow = r * rRep + dr;
          const dstCol = c * cRep + dc;
          out[dstCol * newRows + dstRow] = copyClassInstance(src);
        }
      }
    }
  }
  return wrap(out, [newRows, newCols]);
}

defineBuiltin({
  name: "repelem",
  cases: [
    {
      match: varargMatch2,
      apply: args => {
        if (args.length < 2)
          throw new RuntimeError("repelem requires at least 2 arguments");
        const v = args[0];
        if (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v))
          return repelemObjects(v, args.slice(1));
        if (args.length === 2) {
          const repArg = args[1];
          // Per-element count vector: repelem(v, [c1 c2 ...]) repeats v(i)
          // counts(i) times. (A 1-element count is treated as a scalar.)
          if (isRuntimeTensor(repArg) && repArg.data.length > 1) {
            const counts = repArg.data;
            const vData = isRuntimeNumber(v)
              ? Float64Array.of(v as number)
              : isRuntimeTensor(v)
                ? v.data
                : null;
            if (vData === null)
              throw new RuntimeError("repelem: first argument must be numeric");
            const len = vData.length;
            if (counts.length !== len)
              throw new RuntimeError(
                `repelem: counts vector length (${counts.length}) must match the number of elements (${len})`
              );
            const vImag = isRuntimeTensor(v) ? v.imag : undefined;
            let total = 0;
            for (let i = 0; i < len; i++)
              total += Math.max(0, Math.round(counts[i]));
            const result = allocFloat64Array(total);
            const resultImag = vImag ? allocFloat64Array(total) : undefined;
            let k = 0;
            for (let i = 0; i < len; i++) {
              const c = Math.max(0, Math.round(counts[i]));
              for (let j = 0; j < c; j++) {
                result[k] = vData[i];
                if (resultImag) resultImag[k] = vImag![i];
                k++;
              }
            }
            const isCol =
              isRuntimeTensor(v) && v.shape.length === 2 && v.shape[1] === 1;
            const out = RTV.tensor(
              result,
              isCol ? [total, 1] : [1, total],
              resultImag
            );
            if (isRuntimeTensor(v) && v._isLogical) out._isLogical = true;
            return out;
          }
          const n = Math.round(toNumber(args[1]));
          if (isRuntimeNumber(v)) {
            const data = allocFloat64Array(n).fill(v as number);
            return RTV.tensor(data, [1, n]);
          }
          if (!isRuntimeTensor(v))
            throw new RuntimeError("repelem: first argument must be numeric");
          const len = v.data.length;
          const result = allocFloat64Array(len * n);
          const resultImag = v.imag ? allocFloat64Array(len * n) : undefined;
          const isCol = v.shape.length === 2 && v.shape[1] === 1;
          for (let i = 0; i < len; i++) {
            for (let j = 0; j < n; j++) {
              result[i * n + j] = v.data[i];
              if (resultImag) resultImag[i * n + j] = v.imag![i];
            }
          }
          if (isCol) return RTV.tensor(result, [len * n, 1], resultImag);
          return RTV.tensor(result, [1, len * n], resultImag);
        }
        const rRep = Math.round(toNumber(args[1]));
        const cRep = Math.round(toNumber(args[2]));
        if (isRuntimeNumber(v)) {
          const data = allocFloat64Array(rRep * cRep).fill(v as number);
          return RTV.tensor(data, [rRep, cRep]);
        }
        if (!isRuntimeTensor(v))
          throw new RuntimeError("repelem: first argument must be numeric");
        const [rows, cols] = tensorSize2D(v);
        const newRows = rows * rRep;
        const newCols = cols * cRep;
        const result = allocFloat64Array(newRows * newCols);
        const resultImag = v.imag
          ? allocFloat64Array(newRows * newCols)
          : undefined;
        for (let c = 0; c < cols; c++) {
          for (let r = 0; r < rows; r++) {
            const srcIdx = c * rows + r;
            const val = v.data[srcIdx];
            const valIm = v.imag ? v.imag[srcIdx] : 0;
            for (let dc = 0; dc < cRep; dc++) {
              for (let dr = 0; dr < rRep; dr++) {
                const dstRow = r * rRep + dr;
                const dstCol = c * cRep + dc;
                const dstIdx = dstCol * newRows + dstRow;
                result[dstIdx] = val;
                if (resultImag) resultImag[dstIdx] = valIm;
              }
            }
          }
        }
        return RTV.tensor(result, [newRows, newCols], resultImag);
      },
    },
  ],
});

// ── squeeze ──────────────────────────────────────────────────────────

defineBuiltin({
  name: "squeeze",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeNumber(v) || isRuntimeLogical(v)) return v;
        if (isRuntimeTensor(v)) {
          const effectiveShape = [...v.shape];
          while (
            effectiveShape.length > 2 &&
            effectiveShape[effectiveShape.length - 1] === 1
          ) {
            effectiveShape.pop();
          }
          // Copy buffers — sharing them across wrappers conflicts with
          // refcount-driven pool reclamation (see reshape for the same
          // tradeoff).
          const dataCopy = allocFloat64Array(v.data);
          const imagCopy = v.imag ? allocFloat64Array(v.imag) : undefined;
          if (effectiveShape.length <= 2) {
            return RTV.tensor(dataCopy, effectiveShape, imagCopy);
          }
          const newShape = effectiveShape.filter(d => d !== 1);
          if (newShape.length === 0) {
            if (v.imag && v.imag[0] !== 0)
              return RTV.complex(v.data[0], v.imag[0]);
            return RTV.num(v.data[0]);
          }
          if (newShape.length === 1) {
            return RTV.tensor(dataCopy, [newShape[0], 1], imagCopy);
          }
          return RTV.tensor(dataCopy, newShape, imagCopy);
        }
        // Cell arrays may carry N-D shapes; squeeze the singleton dims.
        // Element references are shared (RuntimeCell increfs them).
        if (isRuntimeCell(v)) {
          const shape = [...v.shape];
          while (shape.length > 2 && shape[shape.length - 1] === 1) shape.pop();
          if (shape.length <= 2) return RTV.cell(v.data, shape);
          const newShape = shape.filter(d => d !== 1);
          while (newShape.length < 2) newShape.push(1);
          return RTV.cell(v.data, newShape);
        }
        // All other array kinds (structs, struct/class arrays, char, string)
        // are at most 2-D in numbl, so squeeze is a no-op — return as-is, like
        // MATLAB, which accepts any type.
        return v;
      },
    },
  ],
});

// ── circshift ────────────────────────────────────────────────────────

defineBuiltin({
  name: "circshift",
  cases: [
    {
      match: varargMatch2,
      apply: args => {
        if (args.length < 2)
          throw new RuntimeError("circshift requires 2 arguments");
        const v = args[0];
        if (isRuntimeNumber(v)) return v;
        if (!isRuntimeTensor(v))
          throw new RuntimeError("circshift: first argument must be numeric");
        const shiftArg = args[1];
        const shape = v.shape.length >= 2 ? v.shape : [1, ...v.shape];

        let shifts: number[];
        if (isRuntimeTensor(shiftArg)) {
          shifts = Array.from(shiftArg.data).map(s => Math.round(s));
        } else {
          const scalarShift = Math.round(toNumber(shiftArg));
          let dimIdx = 0;
          if (shape[0] === 1 && shape.length === 2) dimIdx = 1;
          shifts = new Array(shape.length).fill(0);
          shifts[dimIdx] = scalarShift;
        }

        const totalElems = v.data.length;
        const result = allocFloat64Array(totalElems);
        const resultImag = v.imag ? allocFloat64Array(totalElems) : undefined;

        const ndimCS = shape.length;
        const strides = new Array(ndimCS);
        strides[0] = 1;
        for (let d = 1; d < ndimCS; d++)
          strides[d] = strides[d - 1] * shape[d - 1];

        const dimLookup: number[][] = new Array(ndimCS);
        for (let d = 0; d < ndimCS; d++) {
          const n = shape[d];
          const s = d < shifts.length ? ((-shifts[d] % n) + n) % n : 0;
          const lookup = new Array(n);
          for (let k = 0; k < n; k++) {
            lookup[k] = ((k + s) % n) * strides[d];
          }
          dimLookup[d] = lookup;
        }

        const subs = new Array(ndimCS).fill(0);
        let srcIdx = 0;
        for (let d = 0; d < ndimCS; d++) srcIdx += dimLookup[d][0];
        for (let i = 0; i < totalElems; i++) {
          result[i] = v.data[srcIdx];
          if (resultImag) resultImag[i] = v.imag![srcIdx];
          for (let d = 0; d < ndimCS; d++) {
            const prev = subs[d];
            subs[d]++;
            if (subs[d] < shape[d]) {
              srcIdx += dimLookup[d][subs[d]] - dimLookup[d][prev];
              break;
            }
            srcIdx -= dimLookup[d][prev] - dimLookup[d][0];
            subs[d] = 0;
          }
        }
        return RTV.tensor(result, [...v.shape], resultImag);
      },
    },
  ],
});

// ── permute / ipermute ───────────────────────────────────────────────

defineBuiltin({
  name: "permute",
  cases: [
    {
      match: varargMatch2,
      apply: args => {
        if (args.length !== 2)
          throw new RuntimeError("permute requires 2 arguments");
        const v = coerceToTensor(args[0], "permute");
        const orderArg = args[1];
        let order: number[];
        if (isRuntimeTensor(orderArg)) {
          order = Array.from(orderArg.data).map(x => Math.round(x));
        } else if (isRuntimeNumber(orderArg)) {
          order = [Math.round(orderArg as number)];
        } else {
          throw new RuntimeError("permute: second argument must be numeric");
        }
        const perm = order.map(x => x - 1);
        const srcShape = v.shape;
        const maxDim = Math.max(...perm) + 1;
        const padShape = [...srcShape];
        while (padShape.length < maxDim) padShape.push(1);
        const newShape = perm.map(d => padShape[d]);
        const totalElems = v.data.length;
        const result = allocFloat64Array(totalElems);
        const resultImag = v.imag ? allocFloat64Array(totalElems) : undefined;

        const ndim = perm.length;
        const srcStrides = new Array(padShape.length);
        srcStrides[0] = 1;
        for (let d = 1; d < padShape.length; d++)
          srcStrides[d] = srcStrides[d - 1] * padShape[d - 1];

        const mappedStrides = new Array(ndim);
        for (let d = 0; d < ndim; d++) mappedStrides[d] = srcStrides[perm[d]];

        const subs = new Array(ndim).fill(0);
        let srcIdx = 0;
        for (let i = 0; i < totalElems; i++) {
          result[i] = v.data[srcIdx];
          if (resultImag) resultImag[i] = v.imag![srcIdx];
          for (let d = 0; d < ndim; d++) {
            subs[d]++;
            srcIdx += mappedStrides[d];
            if (subs[d] < newShape[d]) break;
            srcIdx -= subs[d] * mappedStrides[d];
            subs[d] = 0;
          }
        }
        return RTV.tensor(result, newShape, resultImag);
      },
    },
  ],
});

defineBuiltin({
  name: "ipermute",
  cases: [
    {
      match: varargMatch2,
      apply: args => {
        if (args.length !== 2)
          throw new RuntimeError("ipermute requires 2 arguments");
        const v = coerceToTensor(args[0], "ipermute");
        const orderArg = args[1];
        let order: number[];
        if (isRuntimeTensor(orderArg)) {
          order = Array.from(orderArg.data).map(x => Math.round(x));
        } else if (isRuntimeNumber(orderArg)) {
          order = [Math.round(orderArg as number)];
        } else {
          throw new RuntimeError("ipermute: second argument must be numeric");
        }
        const invPerm = new Array(order.length);
        for (let i = 0; i < order.length; i++) {
          invPerm[order[i] - 1] = i;
        }
        const srcShape = v.shape;
        const maxDim = Math.max(...invPerm) + 1;
        const padShape = [...srcShape];
        while (padShape.length < maxDim) padShape.push(1);
        const newShape = invPerm.map((d: number) => padShape[d]);
        const totalElems = v.data.length;
        const result = allocFloat64Array(totalElems);
        const resultImag = v.imag ? allocFloat64Array(totalElems) : undefined;

        const ndim = invPerm.length;
        const srcStrides = new Array(padShape.length);
        srcStrides[0] = 1;
        for (let d = 1; d < padShape.length; d++)
          srcStrides[d] = srcStrides[d - 1] * padShape[d - 1];
        const mappedStrides = new Array(ndim);
        for (let d = 0; d < ndim; d++)
          mappedStrides[d] = srcStrides[invPerm[d]];

        const subs = new Array(ndim).fill(0);
        let srcIdx = 0;
        for (let i = 0; i < totalElems; i++) {
          result[i] = v.data[srcIdx];
          if (resultImag) resultImag[i] = v.imag![srcIdx];
          for (let d = 0; d < ndim; d++) {
            subs[d]++;
            srcIdx += mappedStrides[d];
            if (subs[d] < newShape[d]) break;
            srcIdx -= subs[d] * mappedStrides[d];
            subs[d] = 0;
          }
        }
        return RTV.tensor(result, newShape, resultImag);
      },
    },
  ],
});

// ── ndgrid ───────────────────────────────────────────────────────────

defineBuiltin({
  name: "ndgrid",
  cases: [
    {
      match: varargMatch,
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("ndgrid requires at least 1 argument");

        const extractVec = (v: RuntimeValue): number[] => {
          if (isRuntimeNumber(v)) return [v as number];
          if (isRuntimeTensor(v)) return Array.from(v.data);
          throw new RuntimeError("ndgrid: arguments must be numeric vectors");
        };

        const vecs: number[][] = [];
        if (args.length === 1) {
          const single = extractVec(args[0]);
          for (let k = 0; k < nargout; k++) vecs.push(single);
        } else {
          for (const a of args) vecs.push(extractVec(a));
          if (vecs.length < nargout)
            throw new RuntimeError(
              "ndgrid: not enough input vectors for requested outputs"
            );
        }

        const n = nargout;
        const shape = vecs.slice(0, n).map(v2 => v2.length);
        const totalElems = shape.reduce((acc, s) => acc * s, 1);

        const outputs = [];
        for (let k = 0; k < n; k++) {
          const data = allocFloat64Array(totalElems);
          let stride = 1;
          for (let d = 0; d < k; d++) stride *= shape[d];
          const dimLen = shape[k];
          const period = stride * dimLen;
          for (let i = 0; i < totalElems; i++) {
            data[i] = vecs[k][Math.floor((i % period) / stride)];
          }
          outputs.push(RTV.tensor(data, [...shape]));
        }
        return outputs;
      },
    },
  ],
});

// ── meshgrid ─────────────────────────────────────────────────────────

defineBuiltin({
  name: "meshgrid",
  cases: [
    {
      match: varargMatch,
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("meshgrid requires at least 1 argument");

        const extractVec = (v: RuntimeValue): number[] => {
          if (isRuntimeNumber(v)) return [v as number];
          if (isRuntimeTensor(v)) return Array.from(v.data);
          throw new RuntimeError("meshgrid: arguments must be numeric vectors");
        };

        const n = Math.max(nargout, 2);
        let reordered: number[][];
        if (args.length === 1) {
          const single = extractVec(args[0]);
          reordered = Array(n).fill(single);
        } else {
          const vecs = args.map(extractVec);
          reordered = [vecs[1] ?? vecs[0], vecs[0], ...vecs.slice(2)];
          while (reordered.length < n)
            reordered.push(reordered[reordered.length - 1]);
        }

        const shape = reordered.slice(0, n).map(v2 => v2.length);
        const totalElems = shape.reduce((acc, s) => acc * s, 1);

        const ndgridOuts = [];
        for (let k = 0; k < n; k++) {
          const data = allocFloat64Array(totalElems);
          let stride = 1;
          for (let d = 0; d < k; d++) stride *= shape[d];
          const dimLen = shape[k];
          const period = stride * dimLen;
          for (let idx = 0; idx < totalElems; idx++) {
            data[idx] = reordered[k][Math.floor((idx % period) / stride)];
          }
          ndgridOuts.push(RTV.tensor(data, [...shape]));
        }

        const outputs = [...ndgridOuts];
        if (outputs.length >= 2)
          [outputs[0], outputs[1]] = [outputs[1], outputs[0]];

        return outputs;
      },
    },
  ],
});

// ── sub2ind ──────────────────────────────────────────────────────────

defineBuiltin({
  name: "sub2ind",
  cases: [
    {
      match: varargMatch2,
      apply: args => {
        if (args.length < 2)
          throw new RuntimeError("sub2ind requires at least 2 arguments");
        const szArg = args[0];
        if (!isRuntimeTensor(szArg) && !isRuntimeNumber(szArg))
          throw new RuntimeError(
            "sub2ind: first argument must be a size vector"
          );
        const shape = isRuntimeNumber(szArg)
          ? [szArg as number]
          : Array.from(szArg.data);

        const subscriptArgs = args.slice(1);

        const getValues = (v: RuntimeValue): number[] => {
          if (isRuntimeNumber(v)) return [v as number];
          if (isRuntimeLogical(v)) return [v ? 1 : 0];
          if (isRuntimeTensor(v)) return Array.from(v.data);
          throw new RuntimeError(
            "sub2ind: subscript arguments must be numeric"
          );
        };

        const allSubs = subscriptArgs.map(getValues);
        const n = allSubs[0].length;

        const strides: number[] = [1];
        for (let d = 1; d < shape.length; d++) {
          strides[d] = strides[d - 1] * shape[d - 1];
        }

        const result = allocFloat64Array(n);
        for (let i = 0; i < n; i++) {
          let idx = 0;
          for (let d = 0; d < subscriptArgs.length; d++) {
            const s = allSubs[d][i];
            if (!Number.isFinite(s) || s < 1 || s !== Math.floor(s))
              throw new RuntimeError("Out of range subscript.");
            if (d < shape.length) {
              if (s > shape[d])
                throw new RuntimeError("Out of range subscript.");
            } else {
              // MATLAB treats trailing dims as size 1, so only s==1 is valid.
              if (s !== 1) throw new RuntimeError("Out of range subscript.");
            }
            const stride = d < strides.length ? strides[d] : 0;
            idx += (s - 1) * stride;
          }
          result[i] = idx + 1;
        }

        if (n === 1) return RTV.num(result[0]);
        const firstArg = subscriptArgs[0];
        const outShape = isRuntimeTensor(firstArg)
          ? [...firstArg.shape]
          : [1, n];
        return RTV.tensor(result, outShape);
      },
    },
  ],
});

// ── ind2sub ──────────────────────────────────────────────────────────

defineBuiltin({
  name: "ind2sub",
  cases: [
    {
      match: varargMatch2,
      apply: (args, nargout) => {
        if (args.length !== 2)
          throw new RuntimeError("ind2sub requires 2 arguments");
        const szArg = args[0];
        if (!isRuntimeTensor(szArg) && !isRuntimeNumber(szArg))
          throw new RuntimeError(
            "ind2sub: first argument must be a size vector"
          );
        const shape = isRuntimeNumber(szArg)
          ? [szArg as number]
          : Array.from(szArg.data);

        const indArg = args[1];
        let indices: number[];
        let indShape: number[];
        if (isRuntimeNumber(indArg)) {
          indices = [indArg as number];
          indShape = [1, 1];
        } else if (isRuntimeLogical(indArg)) {
          indices = [indArg ? 1 : 0];
          indShape = [1, 1];
        } else if (isRuntimeTensor(indArg)) {
          indices = Array.from(indArg.data);
          indShape = [...indArg.shape];
        } else {
          throw new RuntimeError("ind2sub: second argument must be numeric");
        }

        const n = indices.length;
        const ndims = Math.max(nargout, 2);

        const strides: number[] = [1];
        for (let d = 1; d < shape.length; d++) {
          strides[d] = strides[d - 1] * shape[d - 1];
        }
        while (strides.length < ndims) {
          strides.push(
            strides[strides.length - 1] * (shape[strides.length - 1] || 1)
          );
        }

        const outputs: Float64Array[] = [];
        for (let d = 0; d < ndims; d++) outputs.push(allocFloat64Array(n));

        for (let i = 0; i < n; i++) {
          let rem = indices[i] - 1;
          for (let d = ndims - 1; d >= 0; d--) {
            if (d === 0) {
              outputs[d][i] = rem + 1;
            } else {
              const q = Math.floor(rem / strides[d]);
              outputs[d][i] = q + 1;
              rem = rem - q * strides[d];
            }
          }
        }

        if (nargout <= 1) {
          if (n === 1) return RTV.num(outputs[0][0]);
          return RTV.tensor(outputs[0], indShape);
        }

        const result = [];
        for (let d = 0; d < ndims; d++) {
          if (n === 1) {
            result.push(RTV.num(outputs[d][0]));
          } else {
            result.push(RTV.tensor(outputs[d], [...indShape]));
          }
        }
        return result;
      },
    },
  ],
});
