/**
 * Set and search builtins: find, intersect, union, setdiff, ismember, nnz, nonzeros.
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  toString,
  RuntimeError,
} from "../../runtime/index.js";
import { register, builtinSingle } from "../registry.js";
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
import { rstr } from "../../runtime/runtime.js";
import { toNumArray } from "./helpers.js";

export function registerSetOperations(): void {
  // ── find ─────────────────────────────────────────────────────────────

  register("find", [
    {
      check: (_argTypes: unknown[], nargout: number) => ({
        outputTypes: Array(Math.max(nargout, 1)).fill({ kind: "Unknown" }),
      }),
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("find requires at least 1 argument");
        const v = args[0];

        let countLimit = Infinity;
        let direction: "first" | "last" = "first";
        if (args.length >= 2) {
          countLimit = toNumber(args[1]);
        }
        if (args.length >= 3) {
          const dirArg = args[2];
          if (
            (isRuntimeString(dirArg) || isRuntimeChar(dirArg)) &&
            rstr(dirArg).toLowerCase() === "last"
          ) {
            direction = "last";
          }
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
          if (v.imag) {
            sparseImag = new Float64Array(1); // flag for complex return path
          }
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

        // Apply count limit and direction
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
          if (n === 0) {
            return RTV.tensor(new FloatXArray(0), isRowVec ? [1, 0] : [0, 1]);
          }
          if (isRowVec) {
            return RTV.tensor(new FloatXArray(linIndices), [1, n]);
          }
          return RTV.tensor(new FloatXArray(linIndices), [n, 1]);
        }
        if (nargout === 2) return [makeVec(rows), makeVec(cols)];
        if (sparseImag) {
          return [makeVec(rows), makeVec(cols), makeComplexVec(vals, imagVals)];
        }
        return [makeVec(rows), makeVec(cols), makeVec(vals)];
      },
    },
  ]);

  // ── intersect ────────────────────────────────────────────────────────

  register(
    "intersect",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("intersect requires 2 arguments");
      const a = toNumArray(args[0], "intersect");
      const bSet = new Set(toNumArray(args[1], "intersect"));
      const result = [...new Set(a.filter(x => bSet.has(x)))].sort((x, y) => {
        if (x !== x) return 1;
        if (y !== y) return -1;
        return x - y;
      });
      return RTV.tensor(new FloatXArray(result), [1, result.length]);
    })
  );

  // ── union ────────────────────────────────────────────────────────────

  register(
    "union",
    builtinSingle(args => {
      if (args.length < 2) throw new RuntimeError("union requires 2 arguments");
      const a = toNumArray(args[0], "union");
      const b = toNumArray(args[1], "union");
      const result = [...new Set([...a, ...b])].sort((x, y) => {
        if (x !== x) return 1;
        if (y !== y) return -1;
        return x - y;
      });
      return RTV.tensor(new FloatXArray(result), [1, result.length]);
    })
  );

  // ── setdiff ──────────────────────────────────────────────────────────

  register("setdiff", [
    {
      check: (_argTypes: unknown[], nargout: number) => ({
        outputTypes: Array(Math.max(nargout, 1)).fill({ kind: "Unknown" }),
      }),
      apply: (args: RuntimeValue[], nargout: number) => {
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
  ]);

  // ── ismember ─────────────────────────────────────────────────────────

  register("ismember", [
    {
      check: (_argTypes: unknown[], nargout: number) => ({
        outputTypes: Array(Math.max(nargout, 1)).fill({ kind: "Unknown" }),
      }),
      apply: (args: RuntimeValue[], nargout: number) => {
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
        ) {
          return ismemberStrings(v, b, nargout);
        }

        return ismemberNumeric(v, b, nargout);
      },
    },
  ]);

  // ── nnz ──────────────────────────────────────────────────────────────

  register(
    "nnz",
    builtinSingle(
      args => {
        if (args.length !== 1)
          throw new RuntimeError("nnz requires 1 argument");
        const v = args[0];
        if (isRuntimeNumber(v)) return RTV.num(v !== 0 ? 1 : 0);
        if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
        if (isRuntimeSparseMatrix(v)) return RTV.num(v.jc[v.n]);
        if (isRuntimeTensor(v)) {
          let count = 0;
          for (let i = 0; i < v.data.length; i++) {
            if (v.data[i] !== 0) count++;
          }
          return RTV.num(count);
        }
        throw new RuntimeError("nnz: argument must be numeric");
      },
      { outputType: { kind: "Number" } }
    )
  );

  // ── nonzeros ─────────────────────────────────────────────────────────

  register(
    "nonzeros",
    builtinSingle(
      args => {
        if (args.length !== 1)
          throw new RuntimeError("nonzeros requires 1 argument");
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
          return v !== 0
            ? RTV.tensor(new FloatXArray([v]), [1, 1])
            : RTV.tensor(new FloatXArray(0), [0, 1]);
        }
        throw new RuntimeError("nonzeros: argument must be numeric");
      },
      { outputType: { kind: "Unknown" } }
    )
  );
}

// ── ismember helpers ───────────────────────────────────────────────────

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
