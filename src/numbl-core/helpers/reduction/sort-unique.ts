/**
 * Sort and unique builtins: sort, sortrows, unique, uniquetol.
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  tensorSize2D,
  RuntimeError,
} from "../../runtime/index.js";
import { register, builtinSingle } from "../registry.js";
import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { rstr } from "../../runtime/runtime.js";
import { preserveTypeCheck } from "../reduction-helpers.js";

export function registerSortUnique(): void {
  // ── sort ─────────────────────────────────────────────────────────────

  register("sort", [
    {
      check: preserveTypeCheck,
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("sort requires at least 1 argument");
        const v = args[0];

        // Parse arguments: sort(A), sort(A,dim), sort(A,direction),
        //                  sort(A,dim,direction)
        let dim: number | undefined;
        let descend = false;
        if (args.length >= 2) {
          if (isRuntimeString(args[1]) || isRuntimeChar(args[1])) {
            descend = rstr(args[1]).toLowerCase() === "descend";
          } else {
            dim = Math.round(toNumber(args[1]));
          }
        }
        if (
          args.length >= 3 &&
          (isRuntimeString(args[2]) || isRuntimeChar(args[2]))
        ) {
          descend = rstr(args[2]).toLowerCase() === "descend";
        }

        if (isRuntimeNumber(v)) {
          if (nargout > 1) return [v, RTV.num(1)];
          return v;
        }
        if (isRuntimeComplexNumber(v)) {
          if (nargout > 1) return [v, RTV.num(1)];
          return v;
        }
        if (isRuntimeTensor(v)) {
          const shape = v.shape;
          const re = v.data;
          const im = v.imag;

          if (dim === undefined) {
            const idx = shape.findIndex(d => d > 1);
            dim = idx >= 0 ? idx + 1 : 1;
          }
          const dimIdx = dim - 1;

          // Fast path: 1D/vector real, ascending, no index output
          // Works when all elements are in a single fiber (vector along the sort dim)
          if (!im && !descend && nargout <= 1 && re.length === shape[dimIdx]) {
            const sorted = new FloatXArray(re);
            sorted.sort();
            return RTV.tensor(sorted, [...shape]);
          }

          if (dimIdx >= shape.length) {
            const cp = RTV.tensor(
              new FloatXArray(re),
              [...shape],
              im ? new FloatXArray(im) : undefined
            );
            if (nargout > 1) {
              const ones = new FloatXArray(re.length).fill(1);
              return [cp, RTV.tensor(ones, [...shape])];
            }
            return cp;
          }

          const dimSize = shape[dimIdx];

          // Comparison function on flat indices
          let cmpFlatIdx: (a: number, b: number) => number;
          if (im && !im.every(x => x === 0)) {
            const mag = (i: number) => Math.sqrt(re[i] * re[i] + im[i] * im[i]);
            const phase = (i: number) => Math.atan2(im[i], re[i]);
            cmpFlatIdx = (a, b) => {
              const diff = mag(a) - mag(b);
              if (diff !== 0) return descend ? -diff : diff;
              const pDiff = phase(a) - phase(b);
              return descend ? -pDiff : pDiff;
            };
          } else {
            cmpFlatIdx = descend
              ? (a, b) => {
                  const aNaN = re[a] !== re[a];
                  const bNaN = re[b] !== re[b];
                  if (aNaN && bNaN) return 0;
                  if (aNaN) return -1;
                  if (bNaN) return 1;
                  return re[b] - re[a];
                }
              : (a, b) => {
                  const aNaN = re[a] !== re[a];
                  const bNaN = re[b] !== re[b];
                  if (aNaN && bNaN) return 0;
                  if (aNaN) return 1;
                  if (bNaN) return -1;
                  return re[a] - re[b];
                };
          }

          const resultRe = new FloatXArray(re.length);
          const resultIm = im ? new FloatXArray(re.length) : undefined;
          const resultIdx =
            nargout > 1 ? new FloatXArray(re.length) : undefined;

          // Helper to sort one fiber and write results
          const sortFiber = (
            fiberIndices: number[],
            resultBase: (k: number) => number
          ) => {
            const order = Array.from({ length: dimSize }, (_, k) => k);
            order.sort((a, b) => cmpFlatIdx(fiberIndices[a], fiberIndices[b]));
            for (let r = 0; r < dimSize; r++) {
              const dst = resultBase(r);
              resultRe[dst] = re[fiberIndices[order[r]]];
              if (resultIm) resultIm[dst] = im![fiberIndices[order[r]]];
              if (resultIdx) resultIdx[dst] = order[r] + 1;
            }
          };

          if (dimIdx === 0) {
            // Fast path: dim 1 is contiguous
            for (let slice = 0; slice < re.length / dimSize; slice++) {
              const offset = slice * dimSize;
              const indices = Array.from(
                { length: dimSize },
                (_, r) => offset + r
              );
              sortFiber(indices, k => offset + k);
            }
          } else {
            // General case: stride arithmetic
            let strideDim = 1;
            for (let d = 0; d < dimIdx; d++) strideDim *= shape[d];
            const slabSize = strideDim * dimSize;
            let numOuter = 1;
            for (let d = dimIdx + 1; d < shape.length; d++)
              numOuter *= shape[d];

            const fiberFlatIdx = new Array(dimSize);
            for (let outer = 0; outer < numOuter; outer++) {
              for (let inner = 0; inner < strideDim; inner++) {
                const base = outer * slabSize + inner;
                for (let k = 0; k < dimSize; k++) {
                  fiberFlatIdx[k] = base + k * strideDim;
                }
                sortFiber([...fiberFlatIdx], k => base + k * strideDim);
              }
            }
          }

          const imOut =
            resultIm && resultIm.some(x => x !== 0) ? resultIm : undefined;
          const sorted = RTV.tensor(resultRe, [...shape], imOut);
          if (nargout > 1) return [sorted, RTV.tensor(resultIdx!, [...shape])];
          return sorted;
        }
        throw new RuntimeError("sort: argument must be numeric");
      },
    },
  ]);

  // ── sortrows ─────────────────────────────────────────────────────────

  register(
    "sortrows",
    builtinSingle((args, nargout) => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("sortrows requires 1 or 2 arguments");
      const A = args[0];
      if (!isRuntimeTensor(A))
        throw new RuntimeError("sortrows: input must be a matrix");

      const m = A.shape[0];
      const n = A.shape.length >= 2 ? A.shape[1] : 1;
      const data = A.data;

      let cols: number[] = [];
      if (args.length >= 2) {
        const colArg = args[1];
        if (isRuntimeNumber(colArg)) {
          cols = [Math.round(colArg as number)];
        } else if (isRuntimeTensor(colArg)) {
          for (let i = 0; i < colArg.data.length; i++)
            cols.push(Math.round(colArg.data[i]));
        } else {
          throw new RuntimeError("sortrows: column argument must be numeric");
        }
      }
      if (cols.length === 0) {
        for (let j = 1; j <= n; j++) cols.push(j);
      }

      const rowIdx = Array.from({ length: m }, (_, i) => i);
      rowIdx.sort((a, b) => {
        for (const c of cols) {
          const colIdx = Math.abs(c) - 1;
          const desc = c < 0;
          const va = data[a + colIdx * m];
          const vb = data[b + colIdx * m];
          if (va !== vb) {
            const diff = va - vb;
            return desc ? -diff : diff;
          }
        }
        return 0;
      });

      const resultData = new FloatXArray(m * n);
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          resultData[i + j * m] = data[rowIdx[i] + j * m];
        }
      }
      const result = RTV.tensor(resultData, [m, n]);

      if (nargout > 1) {
        const idxData = new FloatXArray(m);
        for (let i = 0; i < m; i++) idxData[i] = rowIdx[i] + 1;
        return [result, RTV.tensor(idxData, [m, 1])];
      }
      return result;
    })
  );

  // ── unique ───────────────────────────────────────────────────────────

  register(
    "unique",
    builtinSingle((args, nargout) => {
      if (args.length < 1)
        throw new RuntimeError("unique requires at least 1 argument");
      const v = args[0];

      let byRows = false;
      let stable = false;
      for (let i = 1; i < args.length; i++) {
        const a = args[i];
        if (isRuntimeString(a) || isRuntimeChar(a)) {
          const s = rstr(a).toLowerCase();
          if (s === "rows") byRows = true;
          else if (s === "stable") stable = true;
          else if (s === "sorted") stable = false;
        }
      }

      if (isRuntimeNumber(v)) {
        if (nargout <= 1) return v;
        if (nargout === 2) return [v, RTV.num(1)];
        return [v, RTV.num(1), RTV.num(1)];
      }
      if (isRuntimeLogical(v)) {
        const r = RTV.num(v ? 1 : 0);
        if (nargout <= 1) return r;
        if (nargout === 2) return [r, RTV.num(1)];
        return [r, RTV.num(1), RTV.num(1)];
      }

      if (!isRuntimeTensor(v))
        throw new RuntimeError("unique: argument must be numeric");

      if (byRows) {
        return uniqueByRows(v, nargout, stable);
      }
      return uniqueElements(v, nargout, stable);
    })
  );

  // ── uniquetol ────────────────────────────────────────────────────────

  register(
    "uniquetol",
    builtinSingle((args, nargout) => {
      if (args.length < 1)
        throw new RuntimeError("uniquetol requires at least 1 argument");
      const v = args[0];
      if (!isRuntimeTensor(v) && !isRuntimeNumber(v))
        throw new RuntimeError("uniquetol: first argument must be numeric");

      let tol = 1e-6;
      let byRows = false;
      let startIdx = 1;

      if (
        args.length >= 2 &&
        (isRuntimeNumber(args[1]) ||
          (isRuntimeTensor(args[1]) && args[1].data.length === 1))
      ) {
        tol = toNumber(args[1]);
        startIdx = 2;
      }

      for (let i = startIdx; i < args.length; i += 2) {
        const name = args[i];
        if (
          (isRuntimeString(name) || isRuntimeChar(name)) &&
          rstr(name).toLowerCase() === "byrows"
        ) {
          byRows = i + 1 < args.length && toNumber(args[i + 1]) !== 0;
        }
      }

      if (isRuntimeNumber(v)) {
        if (nargout > 1) return [v, RTV.num(1), RTV.num(1)];
        return v;
      }

      if (byRows) {
        return uniquetolByRows(v, nargout, tol);
      }
      return uniquetolElements(v, nargout, tol);
    })
  );
}

// ── unique helpers (extracted from the monolithic closure) ──────────────

import type { RuntimeTensor } from "../../runtime/types.js";

function uniqueByRows(
  v: RuntimeTensor,
  nargout: number,
  stable: boolean
): RuntimeValue | RuntimeValue[] {
  const [rows, cols] = tensorSize2D(v);
  const rowKey =
    cols === 2
      ? (r: number): string => v.data[r] + "," + v.data[rows + r]
      : (r: number): string => {
          let key = "" + v.data[r];
          for (let c = 1; c < cols; c++) key += "," + v.data[c * rows + r];
          return key;
        };
  const rowHasNaN = (r: number): boolean => {
    for (let c = 0; c < cols; c++) {
      const val = v.data[c * rows + r];
      if (val !== val) return true;
    }
    return false;
  };
  const seen = new Map<string, number>();
  const uniqueRowOrder: number[] = [];
  const ic = new FloatXArray(rows);

  for (let r = 0; r < rows; r++) {
    // Rows containing NaN are always unique (NaN !== NaN)
    if (rowHasNaN(r)) {
      const idx = uniqueRowOrder.length;
      uniqueRowOrder.push(r);
      ic[r] = idx + 1;
      continue;
    }
    const key = rowKey(r);
    if (seen.has(key)) {
      ic[r] = seen.get(key)! + 1;
    } else {
      const idx = uniqueRowOrder.length;
      seen.set(key, idx);
      uniqueRowOrder.push(r);
      ic[r] = idx + 1;
    }
  }

  if (!stable) {
    uniqueRowOrder.sort((a, b) => {
      for (let c = 0; c < cols; c++) {
        const va = v.data[c * rows + a];
        const vb = v.data[c * rows + b];
        if (va !== vb) return va - vb;
      }
      return 0;
    });
  }

  const nUnique = uniqueRowOrder.length;
  const resultData = new FloatXArray(nUnique * cols);
  for (let c = 0; c < cols; c++) {
    for (let u = 0; u < nUnique; u++) {
      resultData[c * nUnique + u] = v.data[c * rows + uniqueRowOrder[u]];
    }
  }

  const C = RTV.tensor(resultData, [nUnique, cols]);
  if (nargout <= 1) return C;

  const ia = RTV.tensor(new FloatXArray(uniqueRowOrder.map(r => r + 1)), [
    nUnique,
    1,
  ]);

  // Rebuild ic for sorted case: map old unsorted indices to new sorted indices
  if (!stable) {
    // Build a map from old-group-index → new-sorted-position
    // Before sorting, each row was assigned an ic based on insertion order.
    // After sorting, we need to remap those to sorted positions.
    const sortedKeyToPos = new Map<string, number>();
    for (let u = 0; u < nUnique; u++) {
      sortedKeyToPos.set(rowKey(uniqueRowOrder[u]), u + 1);
    }
    for (let r = 0; r < rows; r++) {
      if (rowHasNaN(r)) {
        // NaN rows: find this specific row in uniqueRowOrder
        for (let u = 0; u < nUnique; u++) {
          if (uniqueRowOrder[u] === r) {
            ic[r] = u + 1;
            break;
          }
        }
      } else {
        ic[r] = sortedKeyToPos.get(rowKey(r))!;
      }
    }
  }

  const icTensor = RTV.tensor(ic, [rows, 1]);
  if (nargout === 2) return [C, ia];
  return [C, ia, icTensor];
}

function uniqueElements(
  v: RuntimeTensor,
  nargout: number,
  stable: boolean
): RuntimeValue | RuntimeValue[] {
  const hasImag = !!v.imag;
  const isNaNVal = (i: number): boolean =>
    v.data[i] !== v.data[i] || (hasImag && v.imag![i] !== v.imag![i]);
  const valKey = (i: number): string =>
    hasImag ? `${v.data[i]},${v.imag![i]}` : `${v.data[i]}`;
  const seen = new Map<string, number>();
  const uniqueOrder: number[] = [];
  const icArr = new FloatXArray(v.data.length);

  for (let i = 0; i < v.data.length; i++) {
    // NaN is never equal to itself — each NaN is always unique
    if (isNaNVal(i)) {
      const idx = uniqueOrder.length;
      uniqueOrder.push(i);
      icArr[i] = idx + 1;
      continue;
    }
    const key = valKey(i);
    if (seen.has(key)) {
      icArr[i] = seen.get(key)! + 1;
    } else {
      const idx = uniqueOrder.length;
      seen.set(key, idx);
      uniqueOrder.push(i);
      icArr[i] = idx + 1;
    }
  }

  let uniqueRe = uniqueOrder.map(i => v.data[i]);
  let uniqueIm = hasImag ? uniqueOrder.map(i => v.imag![i]) : null;
  if (!stable) {
    const indices = uniqueRe.map((_, i) => i);
    indices.sort((a, b) => {
      const ra = uniqueRe[a],
        rb = uniqueRe[b];
      if (ra !== ra) return 1;
      if (rb !== rb) return -1;
      if (ra !== rb) return ra - rb;
      if (uniqueIm) {
        const ia = uniqueIm[a],
          ib = uniqueIm[b];
        if (ia !== ib) return ia - ib;
      }
      return 0;
    });
    const reindex = new Array(uniqueRe.length);
    indices.forEach((origIdx, newIdx) => {
      reindex[origIdx] = newIdx;
    });
    for (let i = 0; i < icArr.length; i++) {
      icArr[i] = reindex[icArr[i] - 1] + 1;
    }
    uniqueRe = indices.map(i => uniqueRe[i]);
    if (uniqueIm) uniqueIm = indices.map(i => uniqueIm![i]);
    const sortedOrder = indices.map(i => uniqueOrder[i]);
    uniqueOrder.length = 0;
    uniqueOrder.push(...sortedOrder);
  }

  const isRow = v.shape.length === 2 && v.shape[0] === 1;
  const outShape: number[] = isRow
    ? [1, uniqueRe.length]
    : [uniqueRe.length, 1];
  const C = RTV.tensor(
    new FloatXArray(uniqueRe),
    outShape,
    uniqueIm ? new FloatXArray(uniqueIm) : undefined
  );

  if (nargout <= 1) return C;

  const ia = RTV.tensor(new FloatXArray(uniqueOrder.map(i => i + 1)), [
    uniqueRe.length,
    1,
  ]);
  const icTensor = RTV.tensor(icArr, [v.data.length, 1]);
  if (nargout === 2) return [C, ia];
  return [C, ia, icTensor];
}

function uniquetolByRows(
  v: RuntimeTensor,
  nargout: number,
  tol: number
): RuntimeValue | RuntimeValue[] {
  const [rows, cols] = tensorSize2D(v);
  const data = v.data;
  const uniqueRowIndices: number[] = [];
  const ic = new FloatXArray(rows);

  for (let r = 0; r < rows; r++) {
    let matchIdx = -1;
    for (let u = 0; u < uniqueRowIndices.length; u++) {
      const ur = uniqueRowIndices[u];
      let withinTol = true;
      for (let c = 0; c < cols; c++) {
        if (Math.abs(data[c * rows + r] - data[c * rows + ur]) > tol) {
          withinTol = false;
          break;
        }
      }
      if (withinTol) {
        matchIdx = u;
        break;
      }
    }
    if (matchIdx === -1) {
      ic[r] = uniqueRowIndices.length + 1;
      uniqueRowIndices.push(r);
    } else {
      ic[r] = matchIdx + 1;
    }
  }

  const nUnique = uniqueRowIndices.length;
  const resultData = new FloatXArray(nUnique * cols);
  for (let c = 0; c < cols; c++) {
    for (let u = 0; u < nUnique; u++) {
      resultData[c * nUnique + u] = data[c * rows + uniqueRowIndices[u]];
    }
  }

  const C = RTV.tensor(resultData, [nUnique, cols]);
  if (nargout <= 1) return C;

  const ia = RTV.tensor(new FloatXArray(uniqueRowIndices.map(r => r + 1)), [
    nUnique,
    1,
  ]);
  const icTensor = RTV.tensor(ic, [rows, 1]);
  if (nargout === 2) return [C, ia];
  return [C, ia, icTensor];
}

function uniquetolElements(
  v: RuntimeTensor,
  nargout: number,
  tol: number
): RuntimeValue | RuntimeValue[] {
  const data = v.data;
  const shape = v.shape;
  const vals = Array.from(data);
  const uniqueIndices: number[] = [];
  const icArr = new FloatXArray(vals.length);

  for (let i = 0; i < vals.length; i++) {
    let matchIdx = -1;
    for (let u = 0; u < uniqueIndices.length; u++) {
      if (Math.abs(vals[i] - vals[uniqueIndices[u]]) <= tol) {
        matchIdx = u;
        break;
      }
    }
    if (matchIdx === -1) {
      icArr[i] = uniqueIndices.length + 1;
      uniqueIndices.push(i);
    } else {
      icArr[i] = matchIdx + 1;
    }
  }

  const nUnique = uniqueIndices.length;
  const resultData = new FloatXArray(uniqueIndices.map(i => vals[i]));
  const isRow = shape.length === 2 && shape[0] === 1;
  const outShape: number[] = isRow ? [1, nUnique] : [nUnique, 1];
  const C = RTV.tensor(resultData, outShape);

  if (nargout <= 1) return C;
  const ia = RTV.tensor(new FloatXArray(uniqueIndices.map(i => i + 1)), [
    nUnique,
    1,
  ]);
  const icTensor = RTV.tensor(icArr, [vals.length, 1]);
  if (nargout === 2) return [C, ia];
  return [C, ia, icTensor];
}
