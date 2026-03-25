/**
 * Set and search builtins for the interpreter: find, sort, setdiff, ismember.
 */

import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
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
import { RTV, toNumber, toString, RuntimeError } from "../../runtime/index.js";
import { rstr } from "../../runtime/runtime.js";
import type { JitType } from "../jit/jitTypes.js";
import { defineBuiltin } from "./types.js";
import { toNumArray } from "../../builtins/reduction/helpers.js";

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
        throw new RuntimeError("sort: argument must be numeric");
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
        const c = RTV.tensor(result, [1, result.length]);
        if (nargout > 1) {
          const ia = new FloatXArray(pairs.map(p => p.idx));
          return [c, RTV.tensor(ia, [ia.length, 1])];
        }
        return c;
      },
    },
  ],
});

// ── ismember ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "ismember",
  cases: [
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
  ],
});

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
