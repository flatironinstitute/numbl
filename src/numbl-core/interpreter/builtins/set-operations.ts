/**
 * Set and search builtins: find, sort, setdiff, intersect, union, ismember,
 * ismembc, ismembc2, ismembertol, nnz, nonzeros, sortrows, unique, uniquetol,
 * groupcounts.
 *
 * Numeric values are ordered as MATLAB's sort orders them (see
 * helpers/complex-order.ts): complex values by magnitude, then angle.
 */

import type {
  RuntimeValue,
  RuntimeTensor,
  RuntimeCell,
} from "../../runtime/types.js";
import {
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeTensor,
  isRuntimeStringArray,
  stringArrayValue,
} from "../../runtime/types.js";
import {
  RTV,
  toString,
  RuntimeError,
  tensorSize2D,
} from "../../runtime/index.js";
import { rstr } from "../../runtime/runtime.js";
import {
  charToNumericTensor,
  numericTensorToChar,
} from "../../runtime/indexing.js";
import { builtin } from "./builtin.js";
import { builtinError, numericArg, realScalarArg } from "./args.js";
import { mkc } from "./types.js";
import { undefinedForTypeError } from "./value-class.js";
import { valueClass } from "../../runtime/kinds.js";
import { toNumArray } from "../../helpers/reduction-helpers.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { stripZeroImagTensor } from "../../helpers/effectively-real.js";
import {
  cmpComplex,
  cmpReal,
  isNaNValue,
} from "../../helpers/complex-order.js";

// ── Numeric operands ───────────────────────────────────────────────────────

/** A numeric, logical or char operand as flat column-major data. `im` is
 *  null for real data (an all-zero imaginary part counts as real). */
interface NumData {
  re: ArrayLike<number>;
  im: ArrayLike<number> | null;
  shape: number[];
  isChar: boolean;
  isLogical: boolean;
}

/** View a value as NumData, or null when it is not numeric, logical or
 *  char (sparse matrices are not handled here). */
function numData(v: RuntimeValue): NumData | null {
  if (typeof v === "number")
    return {
      re: [v],
      im: null,
      shape: [1, 1],
      isChar: false,
      isLogical: false,
    };
  if (typeof v === "boolean")
    return {
      re: [v ? 1 : 0],
      im: null,
      shape: [1, 1],
      isChar: false,
      isLogical: true,
    };
  if (isRuntimeComplexNumber(v))
    return {
      re: [v.re],
      im: v.im !== 0 ? [v.im] : null,
      shape: [1, 1],
      isChar: false,
      isLogical: false,
    };
  if (isRuntimeTensor(v)) {
    const t = stripZeroImagTensor(v);
    return {
      re: t.data,
      im: t.imag ?? null,
      shape: [...t.shape],
      isChar: false,
      isLogical: v.cls === "logical",
    };
  }
  if (isRuntimeChar(v)) {
    const t = charToNumericTensor(v);
    return {
      re: t.data,
      im: null,
      shape: [...t.shape],
      isChar: true,
      isLogical: false,
    };
  }
  return null;
}

/** Ascending comparison of element i of `a` with element j of `b`, using
 *  the complex order when `complex`. */
function cmpAt(
  aRe: ArrayLike<number>,
  aIm: ArrayLike<number> | null,
  i: number,
  bRe: ArrayLike<number>,
  bIm: ArrayLike<number> | null,
  j: number,
  complex: boolean
): number {
  if (!complex) return cmpReal(aRe[i], bRe[j]);
  return cmpComplex(aRe[i], aIm ? aIm[i] : 0, bRe[j], bIm ? bIm[j] : 0);
}

/** A tensor from real and (optional) imaginary parts; an all-zero
 *  imaginary part is dropped. */
function makeNumeric(
  re: ArrayLike<number>,
  im: ArrayLike<number> | null,
  shape: number[],
  logical = false
): RuntimeTensor {
  const keepIm = im !== null && Array.prototype.some.call(im, x => x !== 0);
  return RTV.tensor(
    Float64Array.from(re),
    shape,
    keepIm ? Float64Array.from(im) : undefined,
    logical ? "logical" : "double"
  );
}

const isText = (x: RuntimeValue) => isRuntimeString(x) || isRuntimeChar(x);

/** The text of an option argument, or null when it is not text. */
function optionText(x: RuntimeValue): string | null {
  return isText(x) ? toString(x) : null;
}

function notSupported(what: string): RuntimeError {
  return new RuntimeError(`${what} is not supported`);
}

// ── cellstr set-operation helpers ──────────────────────────────────────────
// union/intersect/setdiff also accept cell arrays of character vectors (and
// plain char/string scalars), matching MATLAB. Results are returned as column
// cell vectors of sorted unique strings, with optional index outputs.

function isCellstr(x: RuntimeValue): boolean {
  return (
    isRuntimeCell(x) &&
    (x as RuntimeCell).data.every(
      (e: RuntimeValue) => isRuntimeString(e) || isRuntimeChar(e)
    )
  );
}

/** True if a set op on (a, b) should run in text mode rather than numeric. */
function isTextSetOp(a: RuntimeValue, b: RuntimeValue): boolean {
  return isCellstr(a) || isCellstr(b);
}

function toStringList(x: RuntimeValue): string[] {
  if (isRuntimeString(x) || isRuntimeChar(x)) return [toString(x)];
  if (isRuntimeCell(x))
    return (x as RuntimeCell).data.map((e: RuntimeValue) => toString(e));
  throw new RuntimeError("set operation: expected text or cellstr argument");
}

function strColumnCell(strs: string[]): RuntimeValue {
  return RTV.cell(
    strs.map(s => RTV.char(s)),
    [strs.length, 1]
  );
}

function firstIndexMap(strs: string[]): Map<string, number> {
  const m = new Map<string, number>();
  for (let i = 0; i < strs.length; i++)
    if (!m.has(strs[i])) m.set(strs[i], i + 1);
  return m;
}

const colVec = (idx: number[]): RuntimeValue =>
  RTV.tensor(allocFloat64Array(idx), [idx.length, 1]);

/** Dedup a string list preserving first-appearance order. */
function uniqueStable(strs: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const s of strs) {
    if (!seen.has(s)) {
      seen.add(s);
      out.push(s);
    }
  }
  return out;
}

function unionStrings(
  a: RuntimeValue,
  b: RuntimeValue,
  nargout: number,
  order: SetOrder = "sorted"
): RuntimeValue | RuntimeValue[] {
  const aStrs = toStringList(a);
  const bStrs = toStringList(b);
  const inA = firstIndexMap(aStrs);
  const inB = firstIndexMap(bStrs);
  const all =
    order === "stable"
      ? [
          ...uniqueStable(aStrs),
          ...uniqueStable(bStrs).filter(s => !inA.has(s)),
        ]
      : [...new Set([...aStrs, ...bStrs])].sort();
  const C = strColumnCell(all);
  if (nargout <= 1) return C;
  const ia: number[] = [];
  const ib: number[] = [];
  for (const s of all) {
    if (inA.has(s)) ia.push(inA.get(s)!);
    else ib.push(inB.get(s)!);
  }
  return [C, colVec(ia), colVec(ib)];
}

function intersectStrings(
  a: RuntimeValue,
  b: RuntimeValue,
  nargout: number,
  order: SetOrder = "sorted"
): RuntimeValue | RuntimeValue[] {
  const aStrs = toStringList(a);
  const bStrs = toStringList(b);
  const inA = firstIndexMap(aStrs);
  const inB = firstIndexMap(bStrs);
  const filtered = uniqueStable(aStrs.filter(s => inB.has(s)));
  const common = order === "stable" ? filtered : [...filtered].sort();
  const C = strColumnCell(common);
  if (nargout <= 1) return C;
  return [
    C,
    colVec(common.map(s => inA.get(s)!)),
    colVec(common.map(s => inB.get(s)!)),
  ];
}

function setdiffStrings(
  a: RuntimeValue,
  b: RuntimeValue,
  nargout: number,
  order: SetOrder = "sorted"
): RuntimeValue | RuntimeValue[] {
  const aStrs = toStringList(a);
  const bSet = new Set(toStringList(b));
  const inA = firstIndexMap(aStrs);
  const filtered = uniqueStable(aStrs.filter(s => !bSet.has(s)));
  const diff = order === "stable" ? filtered : [...filtered].sort();
  const C = strColumnCell(diff);
  if (nargout <= 1) return C;
  return [C, colVec(diff.map(s => inA.get(s)!))];
}

/** MATLAB's error for a cell operand that is not a cellstr. */
function setInputClassError(
  id: string,
  a: RuntimeValue,
  b: RuntimeValue
): RuntimeError {
  return builtinError(
    `MATLAB:${id}:InputClass`,
    `Input A of class ${valueClass(a)} and input B of class ${valueClass(b)} must be cell arrays of character vectors, unless one is a character vector.`
  );
}

// ── find ─────────────────────────────────────────────────────────────────

builtin({
  name: "find",
  singleOutputs: [3],
  accepts: { sparse: true },
  nargin: [1, 3],
  nargout: 3,
  apply: (args, nargout) => {
    const v = args[0];
    if (
      !(
        typeof v === "number" ||
        typeof v === "boolean" ||
        isRuntimeComplexNumber(v) ||
        isRuntimeTensor(v) ||
        isRuntimeSparseMatrix(v) ||
        isRuntimeChar(v)
      )
    )
      throw undefinedForTypeError("find", v);

    let countLimit = Infinity;
    let direction: "first" | "last" = "first";
    if (args.length >= 2) {
      const notInt = () =>
        builtinError(
          "MATLAB:find:NotScalarInt",
          "Second argument must be a positive scalar integer."
        );
      if (isText(args[1])) throw notInt();
      countLimit = realScalarArg(args[1]);
      if (
        !(countLimit > 0) ||
        (Number.isFinite(countLimit) && !Number.isInteger(countLimit))
      )
        throw notInt();
    }
    if (args.length >= 3) {
      const dir = optionText(args[2])?.toLowerCase();
      if (dir !== "first" && dir !== "last")
        throw builtinError(
          "MATLAB:find:InvalidOption",
          "Invalid search option. Must be 'first' or 'last'."
        );
      direction = dir;
    }

    let rows: number[] = [],
      cols: number[] = [],
      vals: number[] = [];
    let imagVals: number[] = [];
    let isComplex = false;
    let linIndices: number[] = [];

    const src = isRuntimeChar(v) ? charToNumericTensor(v) : v;
    if (typeof src === "number" || typeof src === "boolean") {
      const val = typeof src === "number" ? src : src ? 1 : 0;
      if (val !== 0) {
        rows.push(1);
        cols.push(1);
        vals.push(val);
        linIndices.push(1);
      }
    } else if (isRuntimeComplexNumber(src)) {
      isComplex = true;
      if (src.re !== 0 || src.im !== 0) {
        rows.push(1);
        cols.push(1);
        vals.push(src.re);
        imagVals.push(src.im);
        linIndices.push(1);
      }
    } else if (isRuntimeTensor(src)) {
      const nrows = src.shape[0] ?? 1;
      for (let k = 0; k < src.data.length; k++) {
        const val = src.data[k];
        if (val !== 0 || (src.imag && src.imag[k] !== 0)) {
          rows.push((k % nrows) + 1);
          cols.push(Math.floor(k / nrows) + 1);
          vals.push(val);
          if (src.imag) imagVals.push(src.imag[k]);
          linIndices.push(k + 1);
        }
      }
      if (src.imag) isComplex = true;
    } else if (isRuntimeSparseMatrix(src)) {
      isComplex = !!src.pi;
      for (let col = 0; col < src.n; col++) {
        for (let k = src.jc[col]; k < src.jc[col + 1]; k++) {
          const row = src.ir[k];
          rows.push(row + 1);
          cols.push(col + 1);
          vals.push(src.pr[k]);
          if (src.pi) imagVals.push(src.pi[k]);
          linIndices.push(col * src.m + row + 1);
        }
      }
    }

    if (countLimit < rows.length) {
      const [s, e] =
        direction === "last"
          ? [rows.length - countLimit, rows.length]
          : [0, countLimit];
      rows = rows.slice(s, e);
      cols = cols.slice(s, e);
      vals = vals.slice(s, e);
      linIndices = linIndices.slice(s, e);
      if (isComplex) imagVals = imagVals.slice(s, e);
    }

    const n = rows.length;
    // A row vector gives row outputs; anything else gives columns.
    const isRowVec =
      isRuntimeTensor(src) && src.shape.length === 2 && src.shape[0] === 1;
    const vecShape = isRowVec ? [1, n] : [n, 1];
    const makeVec = (arr: number[]) =>
      RTV.tensor(allocFloat64Array(arr), [...vecShape]);
    const makeComplexVec = (re: number[], im: number[]) =>
      RTV.tensor(allocFloat64Array(re), [...vecShape], allocFloat64Array(im));

    if (nargout <= 1) return makeVec(linIndices);
    if (nargout === 2) return [makeVec(rows), makeVec(cols)];
    if (isComplex) {
      const third =
        n === 1 ? mkc(vals[0], imagVals[0]) : makeComplexVec(vals, imagVals);
      return [makeVec(rows), makeVec(cols), third];
    }
    return [makeVec(rows), makeVec(cols), makeVec(vals)];
  },
});

// ── sort ─────────────────────────────────────────────────────────────────

/** Parse sort's (dim, direction) arguments. */
function parseSortArgs(args: RuntimeValue[]): {
  dim: number | undefined;
  descend: boolean;
} {
  let dim: number | undefined;
  let descend = false;
  const dirError = () =>
    builtinError(
      "MATLAB:sort:sortDirection",
      "Sorting direction must be 'ascend' or 'descend'."
    );
  for (let i = 1; i < args.length; i++) {
    const a = args[i];
    const s = optionText(a);
    if (s !== null) {
      const lower = s.toLowerCase();
      if (lower === "ascend") descend = false;
      else if (lower === "descend") descend = true;
      else if (lower === "comparisonmethod" || lower === "missingplacement")
        throw notSupported(`sort: the '${s}' option`);
      else throw dirError();
      continue;
    }
    if (i !== 1) throw dirError();
    const notPosInt = () =>
      builtinError(
        "MATLAB:sort:notPosInt",
        "Dimension argument must be a positive integer."
      );
    let d: number;
    try {
      d = realScalarArg(a);
    } catch {
      throw notPosInt();
    }
    if (!(d >= 1) || !Number.isInteger(d)) throw notPosInt();
    dim = d;
  }
  return { dim, descend };
}

builtin({
  name: "sort",
  accepts: { sparse: true },
  nargin: [1, Infinity],
  nargout: 2,
  apply: (args, nargout) => {
    const v = args[0];
    const { dim, descend } = parseSortArgs(args);

    // Sort a string array's elements (vector orientation preserved).
    if (isRuntimeStringArray(v)) {
      const order = v.data.map((_, i) => i);
      order.sort((i, j) => {
        const x = v.data[i];
        const y = v.data[j];
        const c = x < y ? -1 : x > y ? 1 : 0;
        return descend ? -c : c;
      });
      const sorted = order.map(i => v.data[i]);
      const isRow = v.shape[0] === 1;
      const shape: [number, number] = isRow
        ? [1, sorted.length]
        : [sorted.length, 1];
      const result = stringArrayValue(sorted, shape);
      if (nargout > 1) {
        return [
          result,
          RTV.tensor(allocFloat64Array(order.map(i => i + 1)), [
            shape[0],
            shape[1],
          ]),
        ];
      }
      return result;
    }
    if (
      isRuntimeString(v) ||
      isRuntimeNumber(v) ||
      isRuntimeLogical(v) ||
      isRuntimeComplexNumber(v)
    ) {
      if (nargout > 1) return [v, RTV.num(1)];
      return v;
    }
    if (isRuntimeChar(v)) {
      const out = sortTensor(charToNumericTensor(v), dim, descend, nargout);
      if (Array.isArray(out))
        return [numericTensorToChar(out[0] as RuntimeTensor), out[1]];
      return numericTensorToChar(out as RuntimeTensor);
    }
    if (isRuntimeTensor(v)) {
      // An all-zero imaginary lane (e.g. a JIT `sqrt` that lifted to
      // complex without leaving the real domain) sorts by value, not
      // magnitude, and returns a real result.
      return sortTensor(stripZeroImagTensor(v), dim, descend, nargout);
    }
    if (isRuntimeCell(v)) {
      if (!isCellstr(v))
        throw builtinError(
          "MATLAB:sort:InputType",
          "Input argument must be a cell array of character vectors."
        );
      return sortCell(v as RuntimeCell, descend, nargout);
    }
    if (isRuntimeSparseMatrix(v)) throw notSupported("sort of a sparse matrix");
    throw undefinedForTypeError("sort", v);
  },
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

  // Fast path: 1D/vector real, ascending, no index output. A typed-array
  // sort puts -0 before 0, but MATLAB's sort is stable and treats them as
  // equal, so a vector holding -0 takes the general path.
  if (
    !im &&
    !descend &&
    nargout <= 1 &&
    re.length === shape[dimIdx] &&
    !re.some(x => Object.is(x, -0))
  ) {
    const sorted = allocFloat64Array(re);
    sorted.sort();
    return RTV.tensor(sorted, [...shape], undefined, v.cls);
  }

  if (dimIdx >= shape.length) {
    const cp = RTV.tensor(
      allocFloat64Array(re),
      [...shape],
      im ? allocFloat64Array(im) : undefined,
      v.cls
    );
    if (nargout > 1) {
      const ones = allocFloat64Array(re.length).fill(1);
      return [cp, RTV.tensor(ones, [...shape])];
    }
    return cp;
  }

  const dimSize = shape[dimIdx];

  // MATLAB's order (complex: abs, then angle; NaN last). Descending is the
  // reverse, which puts NaN first. Equal elements keep their order.
  const isComplex = !!im && !im.every(x => x === 0);
  const asc = (a: number, b: number) =>
    cmpAt(re, im ?? null, a, re, im ?? null, b, isComplex);
  const cmpFlatIdx = descend ? (a: number, b: number) => -asc(a, b) : asc;

  const resultRe = allocFloat64Array(re.length);
  const resultIm = im ? allocFloat64Array(re.length) : undefined;
  const resultIdx = nargout > 1 ? allocFloat64Array(re.length) : undefined;

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
  const sorted = RTV.tensor(resultRe, [...shape], imOut, v.cls);
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
    const idxData = allocFloat64Array(n);
    for (let i = 0; i < n; i++) idxData[i] = order[i] + 1;
    return [sorted, RTV.tensor(idxData, [...v.shape])];
  }
  return sorted;
}

// ── shared set-op engine (setdiff / intersect / union) ─────────────────────
// All three accept an optional `setOrder` ('sorted' default | 'stable') and a
// `'rows'` flag (treat each row as one element). The numeric paths share one
// engine so the option handling, NaN-last ordering, and index outputs stay
// consistent. Text/cellstr inputs are handled separately (the 'rows' option
// does not apply to cell arrays, per MATLAB).

type SetOrder = "sorted" | "stable";
type SetKind = "setdiff" | "intersect" | "union";

/** Parse the trailing options common to setdiff/intersect/union. */
function parseSetOptions(
  args: RuntimeValue[],
  kind: SetKind
): { rows: boolean; order: SetOrder } {
  let rows = false;
  let order: SetOrder = "sorted";
  for (let i = 2; i < args.length; i++) {
    const s = optionText(args[i]);
    const lower = s?.toLowerCase();
    if (lower === "rows") rows = true;
    else if (lower === "sorted") order = "sorted";
    else if (lower === "stable") order = "stable";
    else if (lower === "legacy")
      throw notSupported(`${kind}: the 'legacy' option`);
    else
      throw builtinError(
        `MATLAB:${kind.toUpperCase()}:UnknownFlag`,
        `Unrecognized flag '${s ?? ""}'. Valid flags are 'rows', 'stable', 'sorted', 'legacy'.`
      );
  }
  return { rows, order };
}

/** One operand of a set op, abstracting element-wise vs. row-wise access.
 *  Items are compared column by column. */
interface SetItems {
  n: number;
  cols: number;
  /** Membership/dedup key for item i. */
  key(i: number): string;
  /** Real and imaginary parts of the item's columns. */
  re(i: number): number[];
  im(i: number): number[] | null;
}

/** Items of `d`: its elements, or its rows when `rows`. With `complex`,
 *  keys include the imaginary part (so 2 and 2+0i are the same item). */
function setItems(d: NumData, rows: boolean, complex: boolean): SetItems {
  const [nr, nc] = rows
    ? [d.shape[0] ?? 1, d.re.length / Math.max(1, d.shape[0] ?? 1)]
    : [d.re.length, 1];
  const { re, im } = d;
  const reRow = (r: number) => {
    const out = new Array<number>(nc);
    for (let c = 0; c < nc; c++) out[c] = re[c * nr + r];
    return out;
  };
  const imRow = (r: number) => {
    if (!complex) return null;
    const out = new Array<number>(nc);
    for (let c = 0; c < nc; c++) out[c] = im ? im[c * nr + r] : 0;
    return out;
  };
  const key = (r: number) => {
    let k = "";
    for (let c = 0; c < nc; c++) {
      const idx = c * nr + r;
      if (c > 0) k += ",";
      k += complex ? `${re[idx]}:${im ? im[idx] : 0}` : `${re[idx]}`;
    }
    return k;
  };
  return { n: nr, cols: nc, key, re: reRow, im: imRow };
}

/** A selected output element and its source indices (1-based). */
interface Pick {
  re: number[];
  im: number[] | null;
  ia?: number;
  ib?: number;
}

function cmpPicks(p: Pick, q: Pick, complex: boolean): number {
  for (let c = 0; c < p.re.length; c++) {
    const r = complex
      ? cmpComplex(p.re[c], p.im![c], q.re[c], q.im![c])
      : cmpReal(p.re[c], q.re[c]);
    if (r !== 0) return r;
  }
  return 0;
}

/** The numeric operand of a set op. A char operand counts as its codes. */
function setOperand(
  v: RuntimeValue,
  other: RuntimeValue,
  kind: SetKind,
  isA: boolean
): NumData {
  const d = numData(v);
  if (d) return d;
  if (isRuntimeCell(v))
    throw setInputClassError(
      kind.toUpperCase(),
      isA ? v : other,
      isA ? other : v
    );
  if (isRuntimeSparseMatrix(v)) throw notSupported(`${kind} of sparse input`);
  throw new RuntimeError(`${kind}: arguments must be numeric arrays`);
}

/** Numeric/row set operation shared by setdiff, intersect and union. */
function runSetOp(
  kind: SetKind,
  aVal: RuntimeValue,
  bVal: RuntimeValue,
  rows: boolean,
  order: SetOrder,
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const aData = setOperand(aVal, bVal, kind, true);
  const bData = setOperand(bVal, aVal, kind, false);
  const complex = aData.im !== null || bData.im !== null;
  const A = setItems(aData, rows, complex);
  const B = setItems(bData, rows, complex);
  if (rows && A.n > 0 && B.n > 0 && A.cols !== B.cols) {
    throw builtinError(
      `MATLAB:${kind.toUpperCase()}:AandBColnumAgree`,
      "Inputs A and B must have the same number of columns in the 'rows' case."
    );
  }
  const cols = A.n > 0 ? A.cols : B.cols;
  // NaN is never equal to anything, so each NaN item gets its own key.
  const aKey = (i: number) => (isNaNKey(A, i) ? `nan:a${i}` : A.key(i));
  const bKey = (i: number) => (isNaNKey(B, i) ? `nan:b${i}` : B.key(i));

  // First-occurrence index maps (1-based) for membership/index outputs.
  const bFirst = new Map<string, number>();
  for (let i = 0; i < B.n; i++) {
    const k = bKey(i);
    if (!bFirst.has(k)) bFirst.set(k, i + 1);
  }
  const aKeys = new Set<string>();
  if (kind === "union") {
    for (let i = 0; i < A.n; i++) aKeys.add(aKey(i));
  }

  const picks: Pick[] = [];
  const seen = new Set<string>();
  if (kind === "setdiff" || kind === "intersect") {
    for (let i = 0; i < A.n; i++) {
      const k = aKey(i);
      if (seen.has(k)) continue;
      const inB = bFirst.has(k);
      if ((kind === "setdiff" && !inB) || (kind === "intersect" && inB)) {
        seen.add(k);
        picks.push({
          re: A.re(i),
          im: A.im(i),
          ia: i + 1,
          ib: kind === "intersect" ? bFirst.get(k) : undefined,
        });
      }
    }
  } else {
    // union: every unique A element, then B elements not in A.
    for (let i = 0; i < A.n; i++) {
      const k = aKey(i);
      if (seen.has(k)) continue;
      seen.add(k);
      picks.push({ re: A.re(i), im: A.im(i), ia: i + 1 });
    }
    const seenB = new Set<string>();
    for (let i = 0; i < B.n; i++) {
      const k = bKey(i);
      if (aKeys.has(k) || seenB.has(k)) continue;
      seenB.add(k);
      picks.push({ re: B.re(i), im: B.im(i), ib: i + 1 });
    }
  }

  if (order === "sorted") picks.sort((p, q) => cmpPicks(p, q, complex));

  // Materialize C.
  const n = picks.length;
  const reOut = allocFloat64Array(n * cols);
  const imOut = complex ? allocFloat64Array(n * cols) : null;
  for (let c = 0; c < cols; c++) {
    for (let u = 0; u < n; u++) {
      reOut[c * n + u] = picks[u].re[c];
      if (imOut) imOut[c * n + u] = picks[u].im![c];
    }
  }
  let C: RuntimeValue;
  if (rows) {
    C = makeNumeric(reOut, imOut, [n, cols]);
  } else {
    // Vector orientation follows A (column iff A is a column vector).
    const s = aData.shape;
    const isCol = s[0] > 1 && (s.length < 2 || s[1] === 1);
    C = makeNumeric(reOut, imOut, isCol ? [n, 1] : [1, n]);
  }
  // A char operand makes the result char (for setdiff, only a char A).
  const charResult =
    kind === "setdiff" ? aData.isChar : aData.isChar || bData.isChar;
  if (charResult && isRuntimeTensor(C) && !C.imag) C = numericTensorToChar(C);
  if (nargout <= 1) return C;

  const ia = colVec(picks.filter(p => p.ia !== undefined).map(p => p.ia!));
  if (kind === "setdiff") return [C, ia];
  const ib = colVec(picks.filter(p => p.ib !== undefined).map(p => p.ib!));
  return [C, ia, ib];
}

function defineSetOp(
  kind: SetKind,
  text: (
    a: RuntimeValue,
    b: RuntimeValue,
    nargout: number,
    order: SetOrder
  ) => RuntimeValue | RuntimeValue[]
): void {
  builtin({
    name: kind,
    singleOutputs: [1],
    nargin: [2, 4],
    nargout: kind === "setdiff" ? 2 : 3,
    mfile: true,
    accepts: { sparse: true },
    apply: (args, nargout) => {
      const { rows, order } = parseSetOptions(args, kind);
      if (!rows && isTextSetOp(args[0], args[1]))
        return text(args[0], args[1], nargout, order);
      return runSetOp(kind, args[0], args[1], rows, order, nargout);
    },
  });
}

defineSetOp("setdiff", setdiffStrings);
defineSetOp("intersect", intersectStrings);
defineSetOp("union", unionStrings);

// ── ismember / ismembc ───────────────────────────────────────────────────

function ismemberApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  let rows = false;
  for (let i = 2; i < args.length; i++) {
    const s = optionText(args[i]);
    const lower = s?.toLowerCase();
    if (lower === "rows") rows = true;
    else if (lower === "legacy")
      throw notSupported("ismember: the 'legacy' option");
    else
      throw builtinError(
        "MATLAB:ISMEMBER:UnknownFlag",
        `Unrecognized flag '${s ?? ""}'. Valid flags are 'rows', 'legacy'.`
      );
  }
  // String arrays run through the cellstr text path.
  const asCellstr = (x: RuntimeValue): RuntimeValue =>
    isRuntimeStringArray(x)
      ? RTV.cell(
          x.data.map(s => RTV.string(s)),
          [x.shape[0], x.shape[1]]
        )
      : x;
  const v = asCellstr(args[0]);
  const b = asCellstr(args[1]);

  // Text membership compares whole strings: a cellstr or string operand.
  // Two char arrays compare element-wise (by character), as in MATLAB.
  if (isCellstr(v) || isCellstr(b) || isRuntimeString(v) || isRuntimeString(b))
    return ismemberStrings(v, b, nargout);
  if (isRuntimeCell(v) || isRuntimeCell(b))
    throw setInputClassError("ISMEMBER", args[0], args[1]);

  return ismemberNumeric(v, b, rows, nargout);
}

builtin({
  name: "ismember",
  nargin: [2, 4],
  nargout: 2,
  mfile: true,
  apply: ismemberApply,
});

builtin({
  name: "ismembc",
  accepts: { sparse: true },
  nargin: 2,
  apply: (args, nargout) => ismemberApply(args, nargout),
});

function ismemberStrings(
  v: RuntimeValue,
  b: RuntimeValue,
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const isCellOfStrings = (x: RuntimeValue) => isCellstr(x);

  const bStrings: string[] = [];
  if (isText(b)) {
    bStrings.push(toString(b));
  } else if (isCellOfStrings(b)) {
    if (!isRuntimeCell(b)) throw new RuntimeError("unexpected type");
    for (const e of b.data as RuntimeValue[]) bStrings.push(toString(e));
  } else {
    throw setInputClassError("ISMEMBER", v, b);
  }
  const bSet = new Set(bStrings);

  if (isText(v)) {
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
    const tfData = allocFloat64Array(vData.length);
    const locData = nargout > 1 ? allocFloat64Array(vData.length) : undefined;
    for (let i = 0; i < vData.length; i++) {
      const s = toString(vData[i]);
      const found = bSet.has(s);
      tfData[i] = found ? 1 : 0;
      if (locData) locData[i] = found ? bStrings.indexOf(s) + 1 : 0;
    }
    const t = RTV.logicalTensor(tfData, [...v.shape]);
    if (nargout > 1) {
      return [t, RTV.tensor(locData!, [...v.shape])];
    }
    return t;
  }
  throw setInputClassError("ISMEMBER", v, b);
}

function ismemberNumeric(
  v: RuntimeValue,
  b: RuntimeValue,
  rows: boolean,
  nargout: number
): RuntimeValue | RuntimeValue[] {
  const operand = (x: RuntimeValue): NumData => {
    const d = numData(x);
    if (d) return d;
    if (isRuntimeSparseMatrix(x))
      throw notSupported("ismember of sparse input");
    throw new RuntimeError("ismember: arguments must be numeric arrays");
  };
  const aData = operand(v);
  const bData = operand(b);
  const complex = aData.im !== null || bData.im !== null;
  const A = setItems(aData, rows, complex);
  const B = setItems(bData, rows, complex);
  if (rows && A.n > 0 && B.n > 0 && A.cols !== B.cols)
    throw builtinError(
      "MATLAB:ISMEMBER:AandBColnumAgree",
      "Inputs A and B must have the same number of columns in the 'rows' case."
    );
  const bFirst = new Map<string, number>();
  for (let i = 0; i < B.n; i++) {
    const k = B.key(i);
    if (!bFirst.has(k)) bFirst.set(k, i + 1);
  }
  const tfData = allocFloat64Array(A.n);
  const locData = nargout > 1 ? allocFloat64Array(A.n) : undefined;
  for (let i = 0; i < A.n; i++) {
    const idx = isNaNKey(A, i) ? undefined : bFirst.get(A.key(i));
    tfData[i] = idx !== undefined ? 1 : 0;
    if (locData) locData[i] = idx ?? 0;
  }
  const shape = rows ? [A.n, 1] : aData.shape;
  if (shape[0] * (shape[1] ?? 1) === 1 && shape.length <= 2) {
    const lia = RTV.logical(tfData[0] === 1);
    return nargout > 1 ? [lia, RTV.num(locData![0])] : lia;
  }
  const t = RTV.logicalTensor(tfData, [...shape]);
  if (nargout > 1) return [t, RTV.tensor(locData!, [...shape])];
  return t;
}

/** NaN is never a member of anything. */
function isNaNKey(items: SetItems, i: number): boolean {
  const re = items.re(i);
  const im = items.im(i);
  for (let c = 0; c < re.length; c++)
    if (isNaNValue(re[c], im ? im[c] : 0)) return true;
  return false;
}

// ── ismembertol ──────────────────────────────────────────────────────
//
// Matches MATLAB: elements u, v are within tolerance when
// |u - v| <= tol * DataScale, with DataScale = max(abs([A(:); B(:)])) (per
// column with 'ByRows'). Among several matches, LocB is the index of the
// smallest one in sorted order (lexicographic for rows).

function ismembertolMatrix(
  v: RuntimeValue,
  which: "A" | "B"
): { data: ArrayLike<number>; rows: number; cols: number; shape: number[] } {
  const bad = () =>
    builtinError(
      `MATLAB:ismembertol:UnsupportedTypeFor${which}`,
      `Input ${which} must be a real full matrix of type single or double.`
    );
  if (isRuntimeNumber(v) || typeof v === "boolean")
    return { data: [Number(v)], rows: 1, cols: 1, shape: [1, 1] };
  if (isRuntimeTensor(v)) {
    if (v.imag) throw bad();
    const [rows, cols] = tensorSize2D(v);
    return { data: v.data, rows, cols, shape: [...v.shape] };
  }
  throw bad();
}

builtin({
  name: "ismembertol",
  accepts: { sparse: true },
  nargin: [2, Infinity],
  nargout: 2,
  apply: (args, nargout) => {
    const A = ismembertolMatrix(args[0], "A");
    const B = ismembertolMatrix(args[1], "B");
    let tol = 1e-12;
    let i = 2;
    if (i < args.length && !isText(args[i])) {
      tol = realScalarArg(args[i], "Tolerance");
      i++;
    }
    let byRows = false;
    let allIndices = false;
    let dataScale: number[] | null = null;
    for (; i < args.length; i += 2) {
      const opt = toString(args[i]).toLowerCase();
      if (i + 1 >= args.length)
        throw new RuntimeError(`ismembertol: missing value for '${opt}'`);
      const val = args[i + 1];
      if (opt === "byrows") byRows = realScalarArg(val) !== 0;
      else if (opt === "outputallindices")
        allIndices = realScalarArg(val) !== 0;
      else if (opt === "datascale")
        dataScale = Array.from(toNumArray(val, "ismembertol"));
      else throw new RuntimeError(`ismembertol: unknown option '${opt}'`);
    }

    // Treat elementwise as ByRows on column vectors.
    const nc = byRows ? A.cols : 1;
    const na = byRows ? A.rows : A.data.length;
    const nb = byRows ? B.rows : B.data.length;
    if (byRows && na > 0 && nb > 0 && B.cols !== nc)
      throw new RuntimeError(
        "ismembertol: A and B must have the same number of columns"
      );
    const aAt = (r: number, c: number) => A.data[r + c * (byRows ? A.rows : 0)];
    const bAt = (r: number, c: number) => B.data[r + c * (byRows ? B.rows : 0)];

    const t = new Array<number>(nc);
    for (let c = 0; c < nc; c++) {
      let s: number;
      if (dataScale) {
        s = dataScale.length === 1 ? dataScale[0] : dataScale[c];
      } else {
        // NaN is ignored, as in max()
        s = 0;
        for (let r = 0; r < na; r++) {
          const x = Math.abs(aAt(r, c));
          if (x > s) s = x;
        }
        for (let r = 0; r < nb; r++) {
          const x = Math.abs(bAt(r, c));
          if (x > s) s = x;
        }
      }
      t[c] = tol * s;
    }

    // B rows sorted lexicographically (stable), NaN rows dropped.
    const order: number[] = [];
    for (let r = 0; r < nb; r++) {
      let ok = true;
      for (let c = 0; c < nc; c++) if (Number.isNaN(bAt(r, c))) ok = false;
      if (ok) order.push(r);
    }
    order.sort((x, y) => {
      for (let c = 0; c < nc; c++) {
        const d = bAt(x, c) - bAt(y, c);
        if (d !== 0) return d;
      }
      return x - y;
    });

    const lia = allocFloat64Array(na);
    const locb = allocFloat64Array(na);
    const all: number[][] = [];
    for (let r = 0; r < na; r++) {
      const lo0 = aAt(r, 0) - t[0];
      const hi0 = aAt(r, 0) + t[0];
      let lo = 0;
      let hi = order.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (bAt(order[mid], 0) < lo0) lo = mid + 1;
        else hi = mid;
      }
      const hits: number[] = [];
      for (let k = lo; k < order.length && bAt(order[k], 0) <= hi0; k++) {
        let ok = true;
        for (let c = 1; c < nc && ok; c++)
          ok = Math.abs(aAt(r, c) - bAt(order[k], c)) <= t[c];
        if (ok) {
          if (hits.length === 0) locb[r] = order[k] + 1;
          hits.push(order[k] + 1);
          if (!allIndices) break;
        }
      }
      lia[r] = hits.length > 0 ? 1 : 0;
      if (allIndices) all.push(hits.length ? hits.sort((x, y) => x - y) : [0]);
    }

    const shape = byRows ? [na, 1] : A.shape;
    if (!byRows && !allIndices && isRuntimeNumber(args[0])) {
      const l = RTV.logical(lia[0] === 1);
      return nargout <= 1 ? l : [l, RTV.num(locb[0])];
    }
    const tf = RTV.logicalTensor(lia, shape);
    if (nargout <= 1) return tf;
    if (allIndices) {
      const cells = all.map(h =>
        RTV.tensor(allocFloat64Array(h), [h.length, 1])
      );
      return [tf, RTV.cell(cells, shape)];
    }
    return [tf, RTV.tensor(locb, shape)];
  },
});

// ── nnz ──────────────────────────────────────────────────────────────

builtin({
  name: "nnz",
  accepts: { sparse: true },
  nargin: 1,
  apply: args => {
    const v = numericArg(args[0]);
    if (typeof v === "number") return v !== 0 ? 1 : 0;
    if (isRuntimeComplexNumber(v)) return v.re !== 0 || v.im !== 0 ? 1 : 0;
    if (isRuntimeSparseMatrix(v)) return v.jc[v.n];
    const im = v.imag;
    let count = 0;
    for (let i = 0; i < v.data.length; i++) {
      if (v.data[i] !== 0 || (im && im[i] !== 0)) count++;
    }
    return count;
  },
});

// ── nonzeros ─────────────────────────────────────────────────────────

builtin({
  name: "nonzeros",
  singleOutputs: true,
  accepts: { sparse: true },
  nargin: 1,
  apply: args => {
    const v = args[0];
    if (isRuntimeSparseMatrix(v)) {
      const nnz = v.jc[v.n];
      if (nnz === 0) return RTV.tensor(allocFloat64Array(0), [0, 1]);
      const pr = allocFloat64Array(v.pr.subarray(0, nnz));
      const pi = v.pi ? allocFloat64Array(v.pi.subarray(0, nnz)) : undefined;
      return RTV.tensor(pr, [nnz, 1], pi);
    }
    const d = numData(v);
    if (!d) throw undefinedForTypeError("nonzeros", v);
    const nzVals: number[] = [];
    const nzImag: number[] = [];
    for (let i = 0; i < d.re.length; i++) {
      if (d.re[i] !== 0 || (d.im && d.im[i] !== 0)) {
        nzVals.push(d.re[i]);
        if (d.im) nzImag.push(d.im[i]);
      }
    }
    const n = nzVals.length;
    const out = RTV.tensor(
      allocFloat64Array(nzVals),
      [n, 1],
      d.im && n > 0 ? allocFloat64Array(nzImag) : undefined,
      d.isLogical ? "logical" : "double"
    );
    if (d.isChar) return numericTensorToChar(out);
    return out;
  },
});

// ── sortrows ─────────────────────────────────────────────────────────

/** MATLAB's sortrows(A, column, direction) argument parsing. Returns each
 *  sort column (0-based) with its direction. */
function parseSortrowsArgs(
  args: RuntimeValue[],
  ncols: number
): { col: number; desc: boolean }[] {
  let spec: number[] | null = null;
  let i = 1;
  if (i < args.length && !isText(args[i]) && !isRuntimeCell(args[i])) {
    const colErr = () =>
      builtinError(
        "MATLAB:sortrows:COLmismatchX",
        "Column sorting vector must contain integers with absolute value between 1 and the number of columns in the first argument."
      );
    const d = numData(args[i]);
    if (!d || d.im) throw colErr();
    spec = Array.from(d.re);
    for (const c of spec)
      if (!Number.isInteger(c) || c === 0 || Math.abs(c) > ncols)
        throw colErr();
    i++;
  }
  let dirs: boolean[] | null = null;
  if (i < args.length) {
    const dirErr = () =>
      builtinError(
        "MATLAB:sortrows:DIRnotRecognized",
        "Sorting direction must be 'ascend' or 'descend'."
      );
    const parseDir = (x: RuntimeValue): boolean => {
      const s = optionText(x)?.toLowerCase();
      if (s === "ascend") return false;
      if (s === "descend") return true;
      if (s === "comparisonmethod" || s === "missingplacement")
        throw notSupported(`sortrows: the '${optionText(x)}' option`);
      throw dirErr();
    };
    const d = args[i];
    dirs = isRuntimeCell(d) ? d.data.map(parseDir) : [parseDir(d)];
    i++;
    if (i < args.length) throw dirErr();
  }
  const cols = spec ?? Array.from({ length: ncols }, (_, c) => c + 1);
  if (dirs && dirs.length !== 1 && dirs.length !== cols.length)
    throw builtinError(
      "MATLAB:sortrows:DIRnotRecognized",
      "Sorting direction must be 'ascend' or 'descend'."
    );
  return cols.map((c, k) => ({
    col: Math.abs(c) - 1,
    desc: dirs ? dirs[dirs.length === 1 ? 0 : k] : c < 0,
  }));
}

builtin({
  name: "sortrows",
  singleOutputs: [1],
  accepts: { sparse: true },
  nargin: [1, Infinity],
  nargout: 2,
  mfile: true,
  apply: (args, nargout) => {
    const A = args[0];
    const d = numData(A);
    if (!d) {
      if (isRuntimeCell(A)) throw notSupported("sortrows of a cell array");
      throw new RuntimeError("sortrows: input must be a matrix");
    }
    const m = d.shape[0];
    const n = d.shape.length >= 2 ? d.shape[1] : 1;
    const keys = parseSortrowsArgs(args, n);
    const complex = d.im !== null;
    const { re, im } = d;

    const rowIdx = Array.from({ length: m }, (_, i) => i);
    rowIdx.sort((a, b) => {
      for (const { col, desc } of keys) {
        const ia = a + col * m;
        const ib = b + col * m;
        const c = cmpAt(re, im, ia, re, im, ib, complex);
        if (c !== 0) return desc ? -c : c;
      }
      return 0;
    });

    const resultRe = allocFloat64Array(m * n);
    const resultIm = im ? allocFloat64Array(m * n) : null;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) {
        resultRe[i + j * m] = re[rowIdx[i] + j * m];
        if (resultIm) resultIm[i + j * m] = im![rowIdx[i] + j * m];
      }
    }
    let result: RuntimeValue = makeNumeric(
      resultRe,
      resultIm,
      [m, n],
      d.isLogical
    );
    if (d.isChar) result = numericTensorToChar(result as RuntimeTensor);

    if (nargout > 1) {
      const idxData = allocFloat64Array(m);
      for (let i = 0; i < m; i++) idxData[i] = rowIdx[i] + 1;
      return [result, RTV.tensor(idxData, [m, 1])];
    }
    return result;
  },
});

// ── unique ───────────────────────────────────────────────────────────

builtin({
  name: "unique",
  singleOutputs: [1],
  accepts: { sparse: true },
  nargin: [1, 4],
  nargout: 3,
  mfile: true,
  apply: (args, nargout) => {
    const v = args[0];

    let byRows = false;
    let stable = false;
    let last = false;
    for (let i = 1; i < args.length; i++) {
      const s = optionText(args[i]);
      const lower = s?.toLowerCase();
      if (lower === "rows") byRows = true;
      else if (lower === "stable") stable = true;
      else if (lower === "sorted") stable = false;
      else if (lower === "first") last = false;
      else if (lower === "last") last = true;
      else if (lower === "legacy")
        throw notSupported("unique: the 'legacy' option");
      else
        throw builtinError(
          "MATLAB:UNIQUE:UnknownFlag",
          `Unrecognized option "${s ?? ""}". Valid options are "rows", "first", "last", "stable", "sorted", "legacy".`
        );
    }

    if (!last || nargout <= 1) return uniqueValues(v, nargout, byRows, stable);
    // 'last': ia indexes the last occurrence of each unique value.
    const [C, ia, ic] = uniqueValues(v, 3, byRows, stable) as RuntimeValue[];
    const iaT = ia as RuntimeTensor;
    const icData = (ic as RuntimeTensor).data;
    const lastIa = allocFloat64Array(iaT.data.length);
    for (let i = 0; i < icData.length; i++) lastIa[icData[i] - 1] = i + 1;
    const iaLast = RTV.tensor(lastIa, [...iaT.shape]);
    return nargout === 2 ? [C, iaLast] : [C, iaLast, ic];
  },
});

function uniqueValues(
  v: RuntimeValue,
  nargout: number,
  byRows: boolean,
  stable: boolean
): RuntimeValue | RuntimeValue[] {
  if (isRuntimeNumber(v) || isRuntimeLogical(v) || isRuntimeComplexNumber(v)) {
    if (nargout <= 1) return v;
    if (nargout === 2) return [v, RTV.num(1)];
    return [v, RTV.num(1), RTV.num(1)];
  }

  if (isRuntimeString(v) || isRuntimeChar(v)) {
    return uniqueCharVector(v, nargout, stable);
  }

  if (isRuntimeCell(v)) {
    if (!isCellstr(v))
      throw builtinError(
        "MATLAB:UNIQUE:InputClass",
        "Cell array input must be a cell array of character vectors."
      );
    return uniqueCellOfStrings(v, nargout, stable);
  }

  // String array: unique elements, preserving vector orientation.
  if (isRuntimeStringArray(v)) {
    const elems = v.data;
    const seen = new Map<string, number>();
    for (let i = 0; i < elems.length; i++) {
      if (!seen.has(elems[i])) seen.set(elems[i], i);
    }
    let vals = [...seen.keys()];
    if (!stable) vals = vals.sort();
    const isRow = v.shape[0] === 1;
    const shape: [number, number] = isRow ? [1, vals.length] : [vals.length, 1];
    const result = stringArrayValue(vals, shape);
    if (nargout <= 1) return result;
    const ia = RTV.tensor(allocFloat64Array(vals.map(s => seen.get(s)! + 1)), [
      vals.length,
      1,
    ]);
    if (nargout === 2) return [result, ia];
    const pos = new Map(vals.map((s, i) => [s, i + 1]));
    const ic = RTV.tensor(allocFloat64Array(elems.map(s => pos.get(s)!)), [
      elems.length,
      1,
    ]);
    return [result, ia, ic];
  }

  if (!isRuntimeTensor(v))
    throw new RuntimeError("unique: argument must be numeric");

  return byRows
    ? uniqueByRows(v, nargout, stable)
    : uniqueElements(v, nargout, stable);
}

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
  const ia = RTV.tensor(allocFloat64Array(order.map(i => i + 1)), [
    1,
    order.length,
  ]);
  if (nargout === 2) return [result, ia];
  const icData = allocFloat64Array(chars.length);
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
  v: { kind: "cell"; data: RuntimeValue[]; shape: number[] },
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
  const result: RuntimeValue = RTV.cell(resultData, resultShape);
  if (nargout <= 1) return result;
  const ia = RTV.tensor(
    allocFloat64Array(order.map(i => i + 1)),
    isRow ? [1, order.length] : [order.length, 1]
  );
  if (nargout === 2) return [result, ia];
  const icData = allocFloat64Array(strs.length);
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

/** Unique items (elements, or rows when `byRows`) of a numeric tensor.
 *  NaN items are never equal to anything, so each is kept. In sorted mode
 *  items are ordered as sort orders them (complex: abs, then angle), with
 *  NaN items last in their original order. */
function uniqueItems(
  v: RuntimeTensor,
  nargout: number,
  stable: boolean,
  byRows: boolean
): RuntimeValue | RuntimeValue[] {
  const d = numData(v)!;
  const complex = d.im !== null;
  const items = setItems(d, byRows, complex);
  const seen = new Map<string, number>();
  const uniqueOrder: number[] = [];
  const ic = allocFloat64Array(items.n);

  for (let i = 0; i < items.n; i++) {
    if (isNaNKey(items, i)) {
      ic[i] = uniqueOrder.length + 1;
      uniqueOrder.push(i);
      continue;
    }
    const key = items.key(i);
    const prev = seen.get(key);
    if (prev !== undefined) {
      ic[i] = prev + 1;
    } else {
      seen.set(key, uniqueOrder.length);
      ic[i] = uniqueOrder.length + 1;
      uniqueOrder.push(i);
    }
  }

  let order = uniqueOrder;
  if (!stable) {
    const picks = uniqueOrder.map(i => ({ re: items.re(i), im: items.im(i) }));
    const perm = uniqueOrder.map((_, u) => u);
    perm.sort((a, b) => cmpPicks(picks[a], picks[b], complex));
    const newPos = new Array<number>(perm.length);
    perm.forEach((oldU, newU) => (newPos[oldU] = newU));
    for (let i = 0; i < ic.length; i++) ic[i] = newPos[ic[i] - 1] + 1;
    order = perm.map(u => uniqueOrder[u]);
  }

  const nUnique = order.length;
  const cols = items.cols;
  const reOut = allocFloat64Array(nUnique * cols);
  const imOut = complex ? allocFloat64Array(nUnique * cols) : null;
  for (let u = 0; u < nUnique; u++) {
    const re = items.re(order[u]);
    const im = items.im(order[u]);
    for (let c = 0; c < cols; c++) {
      reOut[c * nUnique + u] = re[c];
      if (imOut) imOut[c * nUnique + u] = im![c];
    }
  }
  const isRow = !byRows && v.shape.length === 2 && v.shape[0] === 1;
  const outShape = byRows
    ? [nUnique, cols]
    : isRow
      ? [1, nUnique]
      : [nUnique, 1];
  const C = makeNumeric(reOut, imOut, outShape, d.isLogical);
  if (nargout <= 1) return C;

  const ia = RTV.tensor(allocFloat64Array(order.map(i => i + 1)), [nUnique, 1]);
  if (nargout === 2) return [C, ia];
  return [C, ia, RTV.tensor(ic, [items.n, 1])];
}

function uniqueByRows(
  v: RuntimeTensor,
  nargout: number,
  stable: boolean
): RuntimeValue | RuntimeValue[] {
  return uniqueItems(v, nargout, stable, true);
}

function uniqueElements(
  v: RuntimeTensor,
  nargout: number,
  stable: boolean
): RuntimeValue | RuntimeValue[] {
  return uniqueItems(v, nargout, stable, false);
}

// ── uniquetol ────────────────────────────────────────────────────────

builtin({
  name: "uniquetol",
  singleOutputs: [1],
  accepts: { sparse: true },
  nargin: [1, Infinity],
  nargout: 3,
  apply: (args, nargout) => {
    const v = args[0];
    if (!(isRuntimeTensor(v) && !v.imag) && !isRuntimeNumber(v))
      throw builtinError(
        "MATLAB:uniquetol:UnsupportedTypeForA",
        "First argument must be a real full matrix of type single or double."
      );

    let tol = 1e-6;
    let byRows = false;
    let startIdx = 1;

    if (
      args.length >= 2 &&
      (isRuntimeNumber(args[1]) ||
        (isRuntimeTensor(args[1]) && args[1].data.length === 1))
    ) {
      tol = realScalarArg(args[1], "Tolerance");
      startIdx = 2;
    }

    for (let i = startIdx; i < args.length; i += 2) {
      const name = args[i];
      if (isText(name) && rstr(name).toLowerCase() === "byrows") {
        byRows = i + 1 < args.length && realScalarArg(args[i + 1]) !== 0;
      }
    }

    if (isRuntimeNumber(v)) {
      if (nargout > 1) return [v, RTV.num(1), RTV.num(1)];
      return v;
    }

    if (byRows) return uniquetolByRows(v, nargout, tol);
    return uniquetolElements(v, nargout, tol);
  },
});

function uniquetolByRows(
  v: RuntimeTensor,
  nargout: number,
  tol: number
): RuntimeValue | RuntimeValue[] {
  const [rows, cols] = tensorSize2D(v);
  const data = v.data;
  const uniqueRowIndices: number[] = [];
  const ic = allocFloat64Array(rows);

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
  const resultData = allocFloat64Array(nUnique * cols);
  for (let c = 0; c < cols; c++) {
    for (let u = 0; u < nUnique; u++) {
      resultData[c * nUnique + u] = data[c * rows + uniqueRowIndices[u]];
    }
  }

  const C = RTV.tensor(resultData, [nUnique, cols]);
  if (nargout <= 1) return C;

  const ia = RTV.tensor(allocFloat64Array(uniqueRowIndices.map(r => r + 1)), [
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
  const icArr = allocFloat64Array(vals.length);

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
  const resultData = allocFloat64Array(uniqueIndices.map(i => vals[i]));
  const isRow = shape.length === 2 && shape[0] === 1;
  const outShape: number[] = isRow ? [1, nUnique] : [nUnique, 1];
  const C = RTV.tensor(resultData, outShape);

  if (nargout <= 1) return C;
  const ia = RTV.tensor(allocFloat64Array(uniqueIndices.map(i => i + 1)), [
    nUnique,
    1,
  ]);
  const icTensor = RTV.tensor(icArr, [vals.length, 1]);
  if (nargout === 2) return [C, ia];
  return [C, ia, icTensor];
}

// ismembc2(a, s): undocumented MATLAB helper — for each element of a, the
// index of its last occurrence in the sorted array s (0 if absent).
builtin({
  name: "ismembc2",
  accepts: { sparse: true },
  nargin: 2,
  apply: args => {
    const toFlat = (v: RuntimeValue): ArrayLike<number> => {
      const d = numData(v);
      if (!d || d.isChar)
        throw new RuntimeError("ismembc2: arguments must be numeric");
      if (d.im) throw notSupported("ismembc2 of complex input");
      return d.re;
    };
    const a = toFlat(args[0]);
    const s = toFlat(args[1]);
    const out = allocFloat64Array(a.length);
    for (let i = 0; i < a.length; i++) {
      const x = a[i];
      // binary search for the last index j with s[j] <= x
      let lo = 0;
      let hi = s.length - 1;
      let res = 0;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (s[mid] <= x) {
          if (s[mid] === x) res = mid + 1;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      out[i] = res;
    }
    const shape = isRuntimeTensor(args[0]) ? [...args[0].shape] : [1, 1];
    return RTV.tensor(out, shape);
  },
});

// ── groupcounts ────────────────────────────────────────────────────────────

builtin({
  name: "groupcounts",
  help: {
    signatures: [
      "GC = groupcounts(A)",
      "[GC, GR] = groupcounts(A)",
      "[GC, GR, GP] = groupcounts(A)",
    ],
    description:
      "Number of elements in each group of A. For a numeric or logical array, each column is a grouping variable and the groups are the unique rows, sorted (complex values by abs, then angle), with NaN as its own (last) group. A cell array of character vectors is one grouping variable. GC is a column of counts, GR the groups (a column for one grouping variable, otherwise a 1-by-N cell of columns) and GP the percentage of elements in each group. Table input, groupbins and name-value options are not supported.",
  },
  nargin: [1, Infinity],
  nargout: 3,
  mfile: true,
  apply: (args, nargout) => {
    if (args.length > 1)
      throw notSupported("groupcounts with groupbins or name-value options");
    const A = args[0];
    let counts: number[];
    let groups: RuntimeValue;
    let total: number;
    if (isRuntimeCell(A)) {
      const cellA = A as RuntimeCell;
      if (
        !isCellstr(cellA) ||
        (cellA.shape.length === 2 &&
          cellA.shape[1] !== 1 &&
          cellA.data.length > 0)
      )
        throw new RuntimeError(
          "groupcounts: cell input must be a column cell array of character vectors"
        );
      const strs = cellA.data.map(e => toString(e));
      total = strs.length;
      const m = new Map<string, number>();
      for (const s of strs) m.set(s, (m.get(s) ?? 0) + 1);
      const keys = [...m.keys()].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
      counts = keys.map(k => m.get(k)!);
      groups = RTV.cell(
        keys.map(k => RTV.char(k)),
        [keys.length, 1]
      );
    } else {
      const d = isRuntimeChar(A) ? null : numData(A);
      if (!d)
        throw new RuntimeError(
          "groupcounts: A must be numeric, logical or a cell array of character vectors"
        );
      if (d.shape.length > 2 && d.shape.slice(2).some(s => s !== 1))
        throw new RuntimeError("groupcounts: A must be a 2-D array");
      const rows = d.shape[0];
      const cols = d.shape.length >= 2 ? d.shape[1] : 1;
      const { re, im } = d;
      const complex = im !== null;
      total = rows;
      // Sort row indices lexicographically (NaN last), then merge equal rows
      const cmp = (a: number, b: number): number => {
        for (let j = 0; j < cols; j++) {
          const c = cmpAt(re, im, j * rows + a, re, im, j * rows + b, complex);
          if (c !== 0) return c;
        }
        return 0;
      };
      const order = Array.from({ length: rows }, (_, i) => i).sort(cmp);
      const firsts: number[] = [];
      counts = [];
      for (let k = 0; k < order.length; k++) {
        if (k > 0 && cmp(order[k - 1], order[k]) === 0)
          counts[counts.length - 1]++;
        else {
          firsts.push(order[k]);
          counts.push(1);
        }
      }
      const ng = firsts.length;
      const groupCol = (j: number): RuntimeValue => {
        const colRe = allocFloat64Array(ng);
        const colIm = complex ? allocFloat64Array(ng) : null;
        for (let g = 0; g < ng; g++) {
          colRe[g] = re[j * rows + firsts[g]] + 0; // -0 -> 0
          if (colIm) colIm[g] = im![j * rows + firsts[g]] + 0;
        }
        const t = makeNumeric(colRe, colIm, [ng, 1], d.isLogical);
        return t;
      };
      groups =
        cols === 1
          ? groupCol(0)
          : RTV.cell(
              Array.from({ length: cols }, (_, j) => groupCol(j)),
              [1, cols]
            );
    }
    const ng = counts.length;
    const gc = RTV.tensor(allocFloat64Array(counts), [ng, 1]);
    if (nargout <= 1) return gc;
    if (nargout === 2) return [gc, groups];
    const gp = allocFloat64Array(ng);
    for (let g = 0; g < ng; g++) gp[g] = (100 * counts[g]) / total;
    return [gc, groups, RTV.tensor(gp, [ng, 1])];
  },
});
