/**
 * Set and search builtins for the interpreter: find, sort, setdiff, ismember.
 */

import type {
  RuntimeValue,
  RuntimeTensor,
  RuntimeCell,
} from "../../runtime/types.js";
import {
  FloatXArray,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import {
  RTV,
  toNumber,
  toString,
  RuntimeError,
  tensorSize2D,
} from "../../runtime/index.js";
import { rstr } from "../../runtime/runtime.js";
import type { JitType } from "../jit/jitTypes.js";
import { defineBuiltin, type BuiltinCase } from "./types.js";
import { toNumArray } from "../../helpers/reduction-helpers.js";

// ── find ─────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "find",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length < 1) return null;
        const k = argTypes[0].kind;
        if (
          k !== "number" &&
          k !== "boolean" &&
          k !== "complex_or_number" &&
          k !== "tensor" &&
          k !== "sparse_matrix" &&
          k !== "unknown"
        )
          return null;
        return Array(Math.max(nargout, 1)).fill({
          kind: "tensor" as const,
          isComplex: false,
          shape: undefined,
        } as JitType);
      },
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("find requires at least 1 argument");
        const v = args[0];

        let countLimit = Infinity;
        let direction: "first" | "last" = "first";
        if (args.length >= 2) countLimit = toNumber(args[1]);
        if (args.length >= 3) {
          const dirArg = args[2];
          if (
            (isRuntimeString(dirArg) || isRuntimeChar(dirArg)) &&
            rstr(dirArg).toLowerCase() === "last"
          )
            direction = "last";
        }

        let rows: number[] = [],
          cols: number[] = [],
          vals: number[] = [];
        let imagVals: number[] = [];
        let sparseImag: Float64Array | undefined;
        let linIndices: number[] = [];

        if (isRuntimeNumber(v) || isRuntimeLogical(v)) {
          const val = isRuntimeNumber(v) ? v : v ? 1 : 0;
          if (val !== 0) {
            rows.push(1);
            cols.push(1);
            vals.push(val);
            linIndices.push(1);
          }
        } else if (isRuntimeComplexNumber(v)) {
          if (v.re !== 0 || v.im !== 0) {
            rows.push(1);
            cols.push(1);
            vals.push(v.re);
            linIndices.push(1);
          }
        } else if (isRuntimeTensor(v)) {
          const nrows = v.shape[0] ?? 1;
          for (let k = 0; k < v.data.length; k++) {
            const val = v.data[k];
            if (val !== 0 || (v.imag && v.imag[k] !== 0)) {
              rows.push((k % nrows) + 1);
              cols.push(Math.floor(k / nrows) + 1);
              vals.push(val);
              if (v.imag) imagVals.push(v.imag[k]);
              linIndices.push(k + 1);
            }
          }
          if (v.imag) sparseImag = new Float64Array(1);
        } else if (isRuntimeSparseMatrix(v)) {
          sparseImag = v.pi;
          for (let col = 0; col < v.n; col++) {
            for (let k = v.jc[col]; k < v.jc[col + 1]; k++) {
              const row = v.ir[k];
              rows.push(row + 1);
              cols.push(col + 1);
              vals.push(v.pr[k]);
              if (v.pi) imagVals.push(v.pi[k]);
              linIndices.push(col * v.m + row + 1);
            }
          }
        } else {
          throw new RuntimeError("find: argument must be numeric");
        }

        if (countLimit < rows.length) {
          if (direction === "last") {
            const start = rows.length - countLimit;
            rows = rows.slice(start);
            cols = cols.slice(start);
            vals = vals.slice(start);
            linIndices = linIndices.slice(start);
            if (sparseImag) imagVals = imagVals.slice(start);
          } else {
            rows = rows.slice(0, countLimit);
            cols = cols.slice(0, countLimit);
            vals = vals.slice(0, countLimit);
            linIndices = linIndices.slice(0, countLimit);
            if (sparseImag) imagVals = imagVals.slice(0, countLimit);
          }
        }

        const n = rows.length;
        const makeVec = (arr: number[]) =>
          n === 0
            ? RTV.tensor(new FloatXArray(0), [0, 1])
            : RTV.tensor(new FloatXArray(arr), [n, 1]);
        const makeComplexVec = (re: number[], im: number[]) =>
          n === 0
            ? RTV.tensor(new FloatXArray(0), [0, 1])
            : RTV.tensor(new FloatXArray(re), [n, 1], new FloatXArray(im));

        if (nargout <= 1) {
          const isRowVec = isRuntimeTensor(v) && v.shape[0] === 1;
          if (n === 0)
            return RTV.tensor(new FloatXArray(0), isRowVec ? [1, 0] : [0, 1]);
          if (isRowVec) return RTV.tensor(new FloatXArray(linIndices), [1, n]);
          return RTV.tensor(new FloatXArray(linIndices), [n, 1]);
        }
        if (nargout === 2) return [makeVec(rows), makeVec(cols)];
        if (sparseImag)
          return [makeVec(rows), makeVec(cols), makeComplexVec(vals, imagVals)];
        return [makeVec(rows), makeVec(cols), makeVec(vals)];
      },
    },
  ],
});

// ── sort ─────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "sort",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length < 1) return null;
        const a = argTypes[0];
        if (a.kind === "number" || a.kind === "boolean") {
          const out: JitType = { kind: "number" };
          return nargout > 1 ? [out, out] : [out];
        }
        if (a.kind === "complex_or_number") {
          const out: JitType = { kind: "complex_or_number" };
          return nargout > 1 ? [out, { kind: "number" }] : [out];
        }
        if (a.kind === "tensor") {
          const out: JitType = {
            kind: "tensor",
            isComplex: a.isComplex,
            shape: a.shape,
            ndim: a.ndim,
          };
          const idx: JitType = {
            kind: "tensor",
            isComplex: false,
            shape: a.shape,
            ndim: a.ndim,
          };
          return nargout > 1 ? [out, idx] : [out];
        }
        if (a.kind === "cell") {
          const out: JitType = { kind: "cell", shape: a.shape };
          const idx: JitType = {
            kind: "tensor",
            isComplex: false,
            shape: a.shape,
          };
          return nargout > 1 ? [out, idx] : [out];
        }
        return null;
      },
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("sort requires at least 1 argument");
        const v = args[0];

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
          return sortTensor(v, dim, descend, nargout);
        }
        if (isRuntimeCell(v)) {
          return sortCell(v as RuntimeCell, descend, nargout);
        }
        throw new RuntimeError("sort: unsupported argument type");
      },
    },
  ],
});

function sortTensor(
  v: RuntimeTensor,
  dim: number | undefined,
  descend: boolean,
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const shape = v.shape;
  const re = v.data;
  const im = v.imag;

  if (dim === undefined) {
    const idx = shape.findIndex(d => d > 1);
    dim = idx >= 0 ? idx + 1 : 1;
  }
  const dimIdx = dim - 1;

  // Fast path: 1D/vector real, ascending, no index output
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
  const resultIdx = nargout > 1 ? new FloatXArray(re.length) : undefined;

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
    for (let slice = 0; slice < re.length / dimSize; slice++) {
      const offset = slice * dimSize;
      const indices = Array.from({ length: dimSize }, (_, r) => offset + r);
      sortFiber(indices, k => offset + k);
    }
  } else {
    let strideDim = 1;
    for (let d = 0; d < dimIdx; d++) strideDim *= shape[d];
    const slabSize = strideDim * dimSize;
    let numOuter = 1;
    for (let d = dimIdx + 1; d < shape.length; d++) numOuter *= shape[d];

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

  const imOut = resultIm && resultIm.some(x => x !== 0) ? resultIm : undefined;
  const sorted = RTV.tensor(resultRe, [...shape], imOut);
  if (nargout > 1) return [sorted, RTV.tensor(resultIdx!, [...shape])];
  return sorted;
}

function sortCell(
  v: RuntimeCell,
  descend: boolean,
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const n = v.data.length;
  const strs = v.data.map(e => toString(e as RuntimeValue));
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => {
    const cmp = strs[a] < strs[b] ? -1 : strs[a] > strs[b] ? 1 : 0;
    return descend ? -cmp : cmp;
  });
  const sortedData = order.map(i => v.data[i]);
  const sorted = RTV.cell(sortedData, [...v.shape]);
  if (nargout > 1) {
    const idxData = new FloatXArray(n);
    for (let i = 0; i < n; i++) idxData[i] = order[i] + 1;
    return [sorted, RTV.tensor(idxData, [...v.shape])];
  }
  return sorted;
}

// ── setdiff ──────────────────────────────────────────────────────────────

defineBuiltin({
  name: "setdiff",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length < 2) return null;
        const out: JitType = {
          kind: "tensor",
          isComplex: false,
          shape: undefined,
        };
        return nargout > 1 ? [out, out] : [out];
      },
      apply: (args, nargout) => {
        if (args.length < 2)
          throw new RuntimeError("setdiff requires 2 arguments");
        const a = toNumArray(args[0], "setdiff");
        const bSet = new Set(toNumArray(args[1], "setdiff"));
        const seen = new Set<number>();
        const pairs: { val: number; idx: number }[] = [];
        for (let i = 0; i < a.length; i++) {
          if (!bSet.has(a[i]) && !seen.has(a[i])) {
            seen.add(a[i]);
            pairs.push({ val: a[i], idx: i + 1 });
          }
        }
        pairs.sort((x, y) => {
          if (x.val !== x.val) return 1;
          if (y.val !== y.val) return -1;
          return x.val - y.val;
        });
        const result = new FloatXArray(pairs.map(p => p.val));
        const isCol =
          isRuntimeTensor(args[0]) &&
          args[0].shape[0] > 1 &&
          (args[0].shape.length < 2 || args[0].shape[1] === 1);
        const outShape: [number, number] = isCol
          ? [result.length, 1]
          : [1, result.length];
        const c = RTV.tensor(result, outShape);
        if (nargout > 1) {
          const ia = new FloatXArray(pairs.map(p => p.idx));
          return [c, RTV.tensor(ia, [ia.length, 1])];
        }
        return c;
      },
    },
  ],
});

// ── ismember / ismembc ───────────────────────────────────────────────────

const ismemberCases: BuiltinCase[] = [
  {
    match: (argTypes, nargout) => {
      if (argTypes.length < 2) return null;
      const out: JitType[] = [{ kind: "unknown" }];
      if (nargout > 1) out.push({ kind: "unknown" });
      return out;
    },
    apply: (args, nargout) => {
      if (args.length < 2)
        throw new RuntimeError("ismember requires 2 arguments");
      const v = args[0];
      const b = args[1];

      const isStringLike = (x: RuntimeValue) =>
        isRuntimeString(x) || isRuntimeChar(x);
      const isCellOfStrings = (x: RuntimeValue) =>
        isRuntimeCell(x) &&
        x.data.every(
          (e: RuntimeValue) => isRuntimeString(e) || isRuntimeChar(e)
        );

      if (
        isStringLike(v) ||
        isCellOfStrings(v) ||
        isStringLike(b) ||
        isCellOfStrings(b)
      )
        return ismemberStrings(v, b, nargout);

      return ismemberNumeric(v, b, nargout);
    },
  },
];

defineBuiltin({ name: "ismember", cases: ismemberCases });
defineBuiltin({ name: "ismembc", cases: ismemberCases });

function ismemberStrings(
  v: RuntimeValue,
  b: RuntimeValue,
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const isStringLike = (x: RuntimeValue) =>
    isRuntimeString(x) || isRuntimeChar(x);
  const isCellOfStrings = (x: RuntimeValue) =>
    isRuntimeCell(x) &&
    x.data.every((e: RuntimeValue) => isRuntimeString(e) || isRuntimeChar(e));

  const bStrings: string[] = [];
  if (isStringLike(b)) {
    bStrings.push(toString(b));
  } else if (isCellOfStrings(b)) {
    if (!isRuntimeCell(b)) throw new RuntimeError("unexpected type");
    for (const e of b.data as RuntimeValue[]) bStrings.push(toString(e));
  } else {
    throw new RuntimeError("ismember: incompatible argument types");
  }
  const bSet = new Set(bStrings);

  if (isStringLike(v)) {
    const found = bSet.has(toString(v));
    const lia = RTV.logical(found);
    if (nargout > 1) {
      const idx = found ? bStrings.indexOf(toString(v)) + 1 : 0;
      return [lia, RTV.num(idx)];
    }
    return lia;
  }
  if (isRuntimeCell(v) && isCellOfStrings(v)) {
    const vData = v.data;
    const tfData = new FloatXArray(vData.length);
    const locData = nargout > 1 ? new FloatXArray(vData.length) : undefined;
    for (let i = 0; i < vData.length; i++) {
      const s = toString(vData[i]);
      const found = bSet.has(s);
      tfData[i] = found ? 1 : 0;
      if (locData) locData[i] = found ? bStrings.indexOf(s) + 1 : 0;
    }
    const t = RTV.tensor(tfData, [...v.shape]);
    t._isLogical = true;
    if (nargout > 1) {
      return [t, RTV.tensor(locData!, [...v.shape])];
    }
    return t;
  }
  throw new RuntimeError("ismember: incompatible argument types");
}

function ismemberNumeric(
  v: RuntimeValue,
  b: RuntimeValue,
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const bArr = toNumArray(b, "ismember");
  const bMap = new Map<number, number>();
  for (let i = 0; i < bArr.length; i++) {
    if (!bMap.has(bArr[i])) bMap.set(bArr[i], i + 1);
  }

  if (isRuntimeNumber(v)) {
    const found = bMap.has(v);
    const lia = RTV.logical(found);
    if (nargout > 1) {
      const locb = RTV.num(found ? bMap.get(v)! : 0);
      return [lia, locb];
    }
    return lia;
  }
  if (isRuntimeTensor(v)) {
    const tfData = new FloatXArray(v.data.length);
    const locData = nargout > 1 ? new FloatXArray(v.data.length) : undefined;
    for (let i = 0; i < v.data.length; i++) {
      const idx = bMap.get(v.data[i]);
      tfData[i] = idx !== undefined ? 1 : 0;
      if (locData) locData[i] = idx !== undefined ? idx : 0;
    }
    const t = RTV.tensor(tfData, [...v.shape]);
    t._isLogical = true;
    if (nargout > 1) {
      return [t, RTV.tensor(locData!, [...v.shape])];
    }
    return t;
  }
  throw new RuntimeError("ismember: first argument must be numeric");
}

// ── intersect ────────────────────────────────────────────────────────

defineBuiltin({
  name: "intersect",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const a = toNumArray(args[0], "intersect");
        const bSet = new Set(toNumArray(args[1], "intersect"));
        const result = [...new Set(a.filter(x => bSet.has(x)))].sort((x, y) => {
          if (x !== x) return 1;
          if (y !== y) return -1;
          return x - y;
        });
        const isCol =
          isRuntimeTensor(args[0]) &&
          args[0].shape[0] > 1 &&
          (args[0].shape.length < 2 || args[0].shape[1] === 1);
        const outShape: [number, number] = isCol
          ? [result.length, 1]
          : [1, result.length];
        return RTV.tensor(new FloatXArray(result), outShape);
      },
    },
  ],
});

// ── union ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "union",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const a = toNumArray(args[0], "union");
        const b = toNumArray(args[1], "union");
        const result = [...new Set([...a, ...b])].sort((x, y) => {
          if (x !== x) return 1;
          if (y !== y) return -1;
          return x - y;
        });
        const isCol =
          isRuntimeTensor(args[0]) &&
          args[0].shape[0] > 1 &&
          (args[0].shape.length < 2 || args[0].shape[1] === 1);
        const outShape: [number, number] = isCol
          ? [result.length, 1]
          : [1, result.length];
        return RTV.tensor(new FloatXArray(result), outShape);
      },
    },
  ],
});

// ── nnz ──────────────────────────────────────────────────────────────

defineBuiltin({
  name: "nnz",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "number" }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeNumber(v)) return (v as number) !== 0 ? 1 : 0;
        if (isRuntimeLogical(v)) return v ? 1 : 0;
        if (isRuntimeSparseMatrix(v)) return v.jc[v.n];
        if (isRuntimeTensor(v)) {
          let count = 0;
          for (let i = 0; i < v.data.length; i++) {
            if (v.data[i] !== 0) count++;
          }
          return count;
        }
        throw new RuntimeError("nnz: argument must be numeric");
      },
    },
  ],
});

// ── nonzeros ─────────────────────────────────────────────────────────

defineBuiltin({
  name: "nonzeros",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeSparseMatrix(v)) {
          const nnz = v.jc[v.n];
          if (nnz === 0) return RTV.tensor(new FloatXArray(0), [0, 1]);
          const pr = new FloatXArray(v.pr.subarray(0, nnz));
          const pi = v.pi ? new FloatXArray(v.pi.subarray(0, nnz)) : undefined;
          return RTV.tensor(pr, [nnz, 1], pi);
        }
        if (isRuntimeTensor(v)) {
          const nzVals: number[] = [];
          const nzImag: number[] = [];
          for (let i = 0; i < v.data.length; i++) {
            if (v.data[i] !== 0 || (v.imag && v.imag[i] !== 0)) {
              nzVals.push(v.data[i]);
              if (v.imag) nzImag.push(v.imag[i]);
            }
          }
          const n = nzVals.length;
          if (n === 0) return RTV.tensor(new FloatXArray(0), [0, 1]);
          return RTV.tensor(
            new FloatXArray(nzVals),
            [n, 1],
            v.imag ? new FloatXArray(nzImag) : undefined
          );
        }
        if (isRuntimeNumber(v)) {
          return (v as number) !== 0
            ? RTV.tensor(new FloatXArray([v as number]), [1, 1])
            : RTV.tensor(new FloatXArray(0), [0, 1]);
        }
        throw new RuntimeError("nonzeros: argument must be numeric");
      },
    },
  ],
});

// ── sortrows ─────────────────────────────────────────────────────────

defineBuiltin({
  name: "sortrows",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        const out: JitType = { kind: "unknown" };
        return nargout > 1 ? [out, out] : [out];
      },
      apply: (args, nargout) => {
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
      },
    },
  ],
});

// ── unique ───────────────────────────────────────────────────────────

defineBuiltin({
  name: "unique",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length < 1) return null;
        const out: JitType = { kind: "unknown" };
        return Array(Math.max(nargout, 1)).fill(out);
      },
      apply: (args, nargout) => {
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

        if (isRuntimeString(v) || isRuntimeChar(v)) {
          return uniqueCharVector(v, nargout, stable);
        }

        if (
          isRuntimeCell(v) &&
          v.data.every(
            (e: RuntimeValue) => isRuntimeString(e) || isRuntimeChar(e)
          )
        ) {
          return uniqueCellOfStrings(v, nargout, stable);
        }

        if (!isRuntimeTensor(v))
          throw new RuntimeError("unique: argument must be numeric");

        if (byRows) return uniqueByRows(v, nargout, stable);
        return uniqueElements(v, nargout, stable);
      },
    },
  ],
});

function uniqueCharVector(
  v: RuntimeValue,
  nargout: number,
  stable: boolean
): RuntimeValue | RuntimeValue[] {
  const s = toString(v);
  const chars = [...s];
  const seen = new Map<string, number>();
  const order: number[] = [];
  for (let i = 0; i < chars.length; i++) {
    if (!seen.has(chars[i])) {
      seen.set(chars[i], order.length);
      order.push(i);
    }
  }
  if (!stable) {
    order.sort((a, b) => {
      if (chars[a] < chars[b]) return -1;
      if (chars[a] > chars[b]) return 1;
      return 0;
    });
  }
  const resultChars = order.map(i => chars[i]);
  const result = RTV.char(resultChars.join(""));
  if (nargout <= 1) return result;
  const ia = RTV.tensor(new FloatXArray(order.map(i => i + 1)), [
    1,
    order.length,
  ]);
  if (nargout === 2) return [result, ia];
  const icData = new FloatXArray(chars.length);
  if (stable) {
    for (let i = 0; i < chars.length; i++) {
      icData[i] = seen.get(chars[i])! + 1;
    }
  } else {
    const sortedMap = new Map<string, number>();
    for (let u = 0; u < order.length; u++) {
      sortedMap.set(chars[order[u]], u + 1);
    }
    for (let i = 0; i < chars.length; i++) {
      icData[i] = sortedMap.get(chars[i])!;
    }
  }
  return [result, ia, RTV.tensor(icData, [1, chars.length])];
}

function uniqueCellOfStrings(
  v: { kind: "cell"; data: RuntimeValue[]; shape: number[]; _rc: number },
  nargout: number,
  stable: boolean
): RuntimeValue | RuntimeValue[] {
  const strs = v.data.map((e: RuntimeValue) => toString(e));
  const seen = new Map<string, number>();
  const order: number[] = [];
  for (let i = 0; i < strs.length; i++) {
    if (!seen.has(strs[i])) {
      seen.set(strs[i], order.length);
      order.push(i);
    }
  }
  if (!stable) {
    order.sort((a, b) => {
      if (strs[a] < strs[b]) return -1;
      if (strs[a] > strs[b]) return 1;
      return 0;
    });
  }
  const resultData = order.map(i => v.data[i]);
  const isRow = v.shape[0] === 1;
  const resultShape = isRow ? [1, order.length] : [order.length, 1];
  const result: RuntimeValue = {
    kind: "cell",
    data: resultData,
    shape: resultShape,
    _rc: 1,
  };
  if (nargout <= 1) return result;
  const ia = RTV.tensor(
    new FloatXArray(order.map(i => i + 1)),
    isRow ? [1, order.length] : [order.length, 1]
  );
  if (nargout === 2) return [result, ia];
  const icData = new FloatXArray(strs.length);
  if (stable) {
    for (let i = 0; i < strs.length; i++) {
      icData[i] = seen.get(strs[i])! + 1;
    }
  } else {
    const sortedMap = new Map<string, number>();
    for (let u = 0; u < order.length; u++) {
      sortedMap.set(strs[order[u]], u + 1);
    }
    for (let i = 0; i < strs.length; i++) {
      icData[i] = sortedMap.get(strs[i])!;
    }
  }
  const icShape = isRow ? [1, strs.length] : [strs.length, 1];
  return [result, ia, RTV.tensor(icData, icShape)];
}

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
      if (v.data[c * rows + r] !== v.data[c * rows + r]) return true;
    }
    return false;
  };
  const seen = new Map<string, number>();
  const uniqueRowOrder: number[] = [];
  const ic = new FloatXArray(rows);

  for (let r = 0; r < rows; r++) {
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

  if (!stable) {
    const sortedKeyToPos = new Map<string, number>();
    for (let u = 0; u < nUnique; u++) {
      sortedKeyToPos.set(rowKey(uniqueRowOrder[u]), u + 1);
    }
    for (let r = 0; r < rows; r++) {
      if (rowHasNaN(r)) {
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
        const ia2 = uniqueIm[a],
          ib = uniqueIm[b];
        if (ia2 !== ib) return ia2 - ib;
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

// ── uniquetol ────────────────────────────────────────────────────────

defineBuiltin({
  name: "uniquetol",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length < 1) return null;
        const out: JitType = { kind: "unknown" };
        return Array(Math.max(nargout, 1)).fill(out);
      },
      apply: (args, nargout) => {
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

        if (byRows) return uniquetolByRows(v, nargout, tol);
        return uniquetolElements(v, nargout, tol);
      },
    },
  ],
});

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
