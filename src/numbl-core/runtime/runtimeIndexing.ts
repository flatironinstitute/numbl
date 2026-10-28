/**
 * Runtime indexing operations.
 *
 * This module contains all indexing-related functionality for the runtime,
 * including array indexing, cell indexing, and indexed assignment operations.
 */

import {
  type RuntimeValue,
  RTV,
  toNumber,
  mIndex,
  mIndexStore,
  mRange,
  RuntimeError,
  COLON_INDEX,
  charToNumericTensor,
  numericTensorToChar,
} from "../runtime/index.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeStruct,
  isRuntimeFunction,
  isRuntimeStructArray,
  isRuntimeComplexNumber,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeSparseMatrix,
  isRuntimeDictionary,
  isRuntimeStringArray,
  type RuntimeClassInstanceArray,
  type RuntimeStructArray,
  type RuntimeStringArray,
  type RuntimeClassInstance,
  type RuntimeStruct,
  type RuntimeTensor,
  type RuntimeChar,
  RuntimeCell,
} from "../runtime/types.js";
import {
  cellBraceElements,
  stringBraceElements,
  hasFieldOrder,
} from "./indexing.js";
import { isShared } from "./refcount.js";
import { scalarSubsOffset } from "./indexPlan.js";
import { numel } from "./utils.js";
import { valueShape } from "./kinds.js";
import { dictLookup, dictInsertSingle, dictRemove } from "./dictionary.js";
import { COLON_SENTINEL, END_SENTINEL } from "./sentinels.js";
import {
  isDeferredRange,
  isDeferredHorzcat,
  ensureRuntimeValue,
} from "./runtimeHelpers.js";
import { resolveHorzcat as _resolveHorzcat } from "./runtimeOperators.js";
import type { Runtime } from "./runtime.js";
import { allocFloat64Array } from "./alloc.js";

/**
 * Resolve indices that may contain END or COLON sentinels, deferred ranges, or functions.
 */
function resolveIndices(
  indices: unknown[],
  resolveEndForDim: (dim: number) => number
): RuntimeValue[] {
  const flat = indices.flatMap(idx => (Array.isArray(idx) ? idx : [idx]));
  const resolveSentinel =
    (dim: number) =>
    (v: RuntimeValue | typeof END_SENTINEL): number =>
      v === END_SENTINEL
        ? resolveEndForDim(dim)
        : typeof v === "number"
          ? v
          : toNumber(ensureRuntimeValue(v));

  return flat.map((idx, dim) => {
    if (idx === COLON_SENTINEL) return COLON_INDEX;
    if (idx === END_SENTINEL) return RTV.num(resolveEndForDim(dim));
    if (isDeferredRange(idx)) {
      const res = resolveSentinel(dim);
      return mRange(
        res(idx.start as RuntimeValue | typeof END_SENTINEL),
        res(idx.step as RuntimeValue | typeof END_SENTINEL),
        res(idx.end as RuntimeValue | typeof END_SENTINEL)
      );
    }
    if (isDeferredHorzcat(idx)) {
      return _resolveHorzcat(idx, resolveSentinel(dim));
    }
    if (typeof idx === "function") {
      return ensureRuntimeValue(
        (idx as (end: number) => unknown)(resolveEndForDim(dim))
      );
    }
    return ensureRuntimeValue(idx);
  });
}

/**
 * The value of `end` in subscript `dimIndex` (0-based) of an index
 * expression with `numSubscripts` subscripts. A class `end` overload is
 * called when one exists. Otherwise, with a single subscript `end` is
 * numel; in the last of several subscripts it is the product of the
 * remaining (collapsed trailing) dimensions; elsewhere it is the size of
 * that dimension.
 */
export function endOf(
  rt: Runtime,
  base: unknown,
  dimIndex: number,
  numSubscripts: number
): number {
  const v = ensureRuntimeValue(base);
  if (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v)) {
    const endFn = rt.cachedResolveClassMethod(v.className, "end");
    if (endFn) {
      const r = endFn(1, v, RTV.num(dimIndex + 1), RTV.num(numSubscripts));
      return toNumber(ensureRuntimeValue(r));
    }
  }
  const shape = valueShape(v);
  if (numSubscripts === 1) return numel(shape);
  if (dimIndex < numSubscripts - 1) {
    return dimIndex < shape.length ? shape[dimIndex] : 1;
  }
  let n = 1;
  for (let i = dimIndex; i < shape.length; i++) n *= shape[i];
  return n;
}

function endResolver(
  rt: Runtime,
  mv: RuntimeValue,
  numIndices: number
): (dim: number) => number {
  return (dim: number): number => endOf(rt, mv, dim, numIndices);
}

/** Linear offset of t(i1, ..., ik) for in-range integer scalar
 *  subscripts, else -1. */
function scalarOffset(t: RuntimeTensor, indices: unknown[]): number {
  return t.shape.length >= 2
    ? scalarSubsOffset(t.shape, t.data.length, indices)
    : -1;
}

/**
 * Index into a runtime value or call a function with arguments.
 */
export function index(
  rt: Runtime,
  base: unknown,
  indices: unknown[],
  nargout: number = 1,
  skipSubsref: boolean | string = false
): unknown {
  // ── Fast paths: one or more in-range integer scalar subscripts into a
  // tensor, struct array, object array or string array, or one into a
  // cell (paren) or a single-row char. These skip resolveIndices and the
  // planner; anything else, including errors, takes the general path.
  if (typeof base === "object" && base !== null) {
    const kind = (base as { kind?: string }).kind;
    if (kind === "tensor") {
      const t = base as RuntimeTensor;
      const lin = scalarOffset(t, indices);
      if (lin >= 0) {
        if (t.imag !== undefined) {
          const im = t.imag[lin];
          return im === 0 ? t.data[lin] : RTV.complex(t.data[lin], im);
        }
        if (t._isLogical === true) return t.data[lin] !== 0;
        return t.data[lin];
      }
    } else if (kind === "struct_array") {
      const sa = base as RuntimeStructArray;
      const lin = scalarSubsOffset(sa.shape, sa.elements.length, indices);
      if (lin >= 0) return sa.elements[lin];
    } else if (kind === "class_instance_array") {
      const oa = base as RuntimeClassInstanceArray;
      const lin = scalarSubsOffset(oa.shape, oa.elements.length, indices);
      if (lin >= 0) return oa.elements[lin];
    } else if (kind === "string_array") {
      const sa = base as RuntimeStringArray;
      const lin = scalarSubsOffset(sa.shape, sa.data.length, indices);
      if (lin >= 0) return sa.data[lin];
    } else if (indices.length === 1 && typeof indices[0] === "number") {
      const i = indices[0];
      if (kind === "cell") {
        const c = base as RuntimeCell;
        if (Number.isInteger(i) && i >= 1 && i <= c.data.length)
          return new RuntimeCell([c.data[i - 1]], [1, 1]);
      } else if (kind === "char") {
        const c = base as RuntimeChar;
        if (
          Number.isInteger(i) &&
          i >= 1 &&
          i <= c.value.length &&
          (c.shape === undefined || c.shape[0] === 1)
        )
          return RTV.char(c.value[i - 1]);
      }
    }
  }

  // Anonymous function call
  if (typeof base === "function") {
    return (base as (...args: unknown[]) => unknown)(...indices);
  }
  const mv = ensureRuntimeValue(base);
  // Function handle call
  if (isRuntimeFunction(mv)) {
    if (mv.jsFn) {
      return mv.jsFnExpectsNargout
        ? mv.jsFn(nargout, ...indices)
        : mv.jsFn(...indices);
    }
    return rt.dispatch(mv.name, nargout, indices);
  }
  // Class instance paren indexing: F(k) — dispatch to subsref or default
  if (isRuntimeClassInstance(mv)) {
    const skip =
      skipSubsref === true ||
      (typeof skipSubsref === "string" && mv.className === skipSubsref);
    return classInstanceParenIndex(rt, mv, base, indices, nargout, skip);
  }
  // Dictionary paren indexing: d(key)
  if (isRuntimeDictionary(mv)) {
    if (indices.length !== 1)
      throw new RuntimeError(
        "Dictionary indexing requires exactly one key argument"
      );
    const key = ensureRuntimeValue(indices[0]);
    return dictLookup(mv, key);
  }
  const idxMvals = resolveIndices(indices, endResolver(rt, mv, indices.length));
  return mIndex(mv, idxMvals);
}

/** Handle F(k) on a class instance — dispatch to subsref('()') or default scalar indexing */
function classInstanceParenIndex(
  rt: Runtime,
  mv: import("../runtime/types.js").RuntimeClassInstance,
  base: unknown,
  indices: unknown[],
  nargout: number,
  skipSubsref: boolean
): unknown {
  const subsrefFn = rt.cachedResolveClassMethod(mv.className, "subsref");

  if (subsrefFn && !skipSubsref) {
    const guardKey = `${mv.className}.subsref`;
    if (!rt.activeAccessors.has(guardKey)) {
      const resolvedIndices = resolveIndicesForClassInstance(rt, mv, indices);
      const subsCell = RTV.cell(resolvedIndices, [1, resolvedIndices.length]);
      const S = RTV.structArray(
        ["type", "subs"],
        [RTV.struct({ type: RTV.char("()"), subs: subsCell })]
      );
      rt.activeAccessors.add(guardKey);
      try {
        return subsrefFn(nargout, base, S);
      } finally {
        rt.activeAccessors.delete(guardKey);
      }
    }
  }

  // Default: built-in indexing of a 1x1 object.
  return mIndex(
    mv,
    resolveIndices(indices, endResolver(rt, mv, indices.length))
  );
}

/** Resolve raw index arguments for a class instance before passing to subsref.
 *  Handles END_SENTINEL (via the class's end method), deferred ranges,
 *  and arrow-function wrappers.  COLON_SENTINEL becomes RTV.char(":"). */
function resolveIndicesForClassInstance(
  rt: Runtime,
  mv: import("../runtime/types.js").RuntimeClassInstance,
  indices: unknown[]
): RuntimeValue[] {
  const numIndices = indices.length;

  const resolveEndForDim = (dim: number): number =>
    endOf(rt, mv, dim, numIndices);

  return indices.map((idx, dim) => {
    if (idx === COLON_SENTINEL) return RTV.char(":");
    if (idx === END_SENTINEL) return RTV.num(resolveEndForDim(dim));
    if (isDeferredRange(idx)) {
      const res = (v: RuntimeValue | typeof END_SENTINEL): number => {
        if (v === END_SENTINEL) return resolveEndForDim(dim);
        if (typeof v === "number") return v;
        return toNumber(ensureRuntimeValue(v));
      };
      return mRange(res(idx.start), res(idx.step), res(idx.end));
    }
    if (typeof idx === "function") {
      return ensureRuntimeValue(
        (idx as (end: number) => unknown)(resolveEndForDim(dim))
      );
    }
    return ensureRuntimeValue(idx);
  });
}

/**
 * Cell indexing with support for comma-separated lists (CSL).
 */
export function indexCell(
  rt: Runtime,
  base: unknown,
  indices: unknown[]
): unknown {
  // ── Fast path: cell with single scalar numeric index ──
  if (
    indices.length === 1 &&
    typeof indices[0] === "number" &&
    typeof base === "object" &&
    base !== null &&
    (base as { kind?: string }).kind === "cell"
  ) {
    const cell = base as import("../runtime/types.js").RuntimeCell;
    const i = Math.round(indices[0] as number) - 1;
    if (i < 0 || i >= cell.data.length)
      throw new RuntimeError("Cell index exceeds bounds");
    return cell.data[i];
  }

  const mv = ensureRuntimeValue(base);

  // Class instance brace indexing: dispatch to subsref with type '{}'
  if (isRuntimeClassInstance(mv)) {
    const subsrefFn = rt.cachedResolveClassMethod(mv.className, "subsref");
    if (subsrefFn) {
      const guardKey = `${mv.className}.subsref`;
      if (!rt.activeAccessors.has(guardKey)) {
        const resolvedIndices = resolveIndicesForClassInstance(rt, mv, indices);
        const subsCell = RTV.cell(resolvedIndices, [1, resolvedIndices.length]);
        const S = RTV.structArray(
          ["type", "subs"],
          [RTV.struct({ type: RTV.char("{}"), subs: subsCell })]
        );
        rt.activeAccessors.add(guardKey);
        try {
          return subsrefFn(1, base, S);
        } finally {
          rt.activeAccessors.delete(guardKey);
        }
      }
    }
    throw new RuntimeError("Cell indexing on non-cell");
  }

  // Dictionary curly-brace indexing: d{key} — unwrap cell contents
  if (isRuntimeDictionary(mv)) {
    if (mv.valueType && mv.valueType !== "cell")
      throw new RuntimeError(
        "Brace indexing on dictionary with non-cell values"
      );
    if (indices.length !== 1)
      throw new RuntimeError(
        "Dictionary brace indexing requires exactly one key argument"
      );
    const key = ensureRuntimeValue(indices[0]);
    const val = dictLookup(mv, key);
    // Unwrap cell
    if (isRuntimeCell(val) && val.data.length === 1) return val.data[0];
    return val;
  }

  // String brace indexing: str{i} extracts the element as a char vector
  // (MATLAB: ["one" "two"]{2} -> 'two'; "abc"{1} -> 'abc').
  if (isRuntimeString(mv) || isRuntimeStringArray(mv)) {
    const elems = stringBraceElements(
      mv,
      resolveIndices(indices, endResolver(rt, mv, indices.length))
    );
    return elems.length === 1 ? elems[0] : elems;
  }

  if (!isRuntimeCell(mv)) throw new RuntimeError("Cell indexing on non-cell");
  const idxMvals = resolveIndices(indices, endResolver(rt, mv, indices.length));
  const elems = cellBraceElements(mv, idxMvals);
  return elems.length === 1 ? elems[0] : elems;
}

/**
 * Indexed assignment `base(indices) = rhs` (deletion when rhs is `[]`).
 * `storeBack` marks the store-back of `base(indices).f = v`, which may add
 * a field to a struct array.
 */
export function indexStore(
  rt: Runtime,
  base: unknown,
  indices: unknown[],
  rhs: unknown,
  skipSubsasgn = false,
  storeBack = false
): unknown {
  // ── Fast paths: one in-range scalar subscript into an unshared target
  // that is written in place: a real tensor (number RHS, plus
  // `v(end+1) = x` appends), or a struct/object array whose element is
  // replaced by one with the same fields / class. Everything else takes
  // the general path.
  if (typeof base === "object" && base !== null) {
    const kind = (base as { kind?: string }).kind;
    if (kind === "tensor" && typeof rhs === "number") {
      const t = base as RuntimeTensor;
      if (t.imag === undefined && t._isLogical !== true) {
        if (!isShared(t)) {
          const lin = scalarOffset(t, indices);
          if (lin >= 0) {
            t.data[lin] = rhs;
            return t;
          }
        }
        if (indices.length === 1 && indices[0] === t.data.length + 1) {
          const s = t.shape;
          const n = t.data.length + 1;
          const col = s.length === 2 && s[1] === 1 && s[0] !== 1;
          if (col || n === 1 || (s.length === 2 && s[0] === 1)) {
            const data = allocFloat64Array(n);
            data.set(t.data);
            data[n - 1] = rhs;
            return RTV.tensorRaw(data, col ? [n, 1] : [1, n]);
          }
        }
      }
    } else if (
      kind === "struct_array" &&
      isRuntimeStruct(rhs as RuntimeValue) &&
      !isShared(base)
    ) {
      const sa = base as RuntimeStructArray;
      const lin = scalarSubsOffset(sa.shape, sa.elements.length, indices);
      if (lin >= 0 && hasFieldOrder(rhs as RuntimeStruct, sa.fieldNames)) {
        sa.bindElement(rt, lin, rhs as RuntimeStruct);
        return sa;
      }
    } else if (
      kind === "class_instance_array" &&
      isRuntimeClassInstance(rhs as RuntimeValue) &&
      !isShared(base)
    ) {
      const oa = base as RuntimeClassInstanceArray;
      const lin = scalarSubsOffset(oa.shape, oa.elements.length, indices);
      if (
        lin >= 0 &&
        (rhs as RuntimeClassInstance).className === oa.className
      ) {
        oa.bindElement(rt, lin, rhs as RuntimeClassInstance);
        return oa;
      }
    }
  }
  // Treat undefined/null as an empty tensor (uninitialized variable).
  // Exception: if RHS is a class instance, replace entirely
  // behaviour for `h(k) = obj` when h is an uninitialized output variable).
  if (base === undefined || base === null) {
    const rhsCheck = ensureRuntimeValue(rhs);
    if (isRuntimeClassInstance(rhsCheck)) {
      return rhsCheck;
    }
    base = RTV.tensor(allocFloat64Array(0), [0, 0]);
  }
  let mv = ensureRuntimeValue(base);
  // Dictionary paren-indexed assignment: d(key) = value or d(key) = [] for removal
  if (isRuntimeDictionary(mv)) {
    if (indices.length !== 1)
      throw new RuntimeError(
        "Dictionary assignment requires exactly one key argument"
      );
    const key = ensureRuntimeValue(indices[0]);
    const rhsMv = ensureRuntimeValue(rhs);
    // d(key) = [] means remove
    if (isRuntimeTensor(rhsMv) && rhsMv.data.length === 0) {
      return dictRemove(mv, key);
    }
    return dictInsertSingle(mv, key, rhsMv);
  }
  // A scalar object dispatches to its subsasgn overload, except when an
  // object RHS grows it into an object array (`obj(k) = other`, k > 1).
  if (isRuntimeClassInstance(mv)) {
    const rhsMv = ensureRuntimeValue(rhs);
    const grows =
      isRuntimeClassInstance(rhsMv) &&
      indices.length === 1 &&
      scalarIndexValue(indices[0]) > 1;
    const guardKey = `${mv.className}.subsasgn`;
    if (!grows && !skipSubsasgn && !rt.activeAccessors.has(guardKey)) {
      const subsasgnFn = rt.cachedResolveClassMethod(mv.className, "subsasgn");
      if (subsasgnFn) {
        const resolvedIndices = indices.map(i =>
          i === COLON_SENTINEL ? RTV.char(":") : ensureRuntimeValue(i)
        );
        const subsCell = RTV.cell(resolvedIndices, [1, resolvedIndices.length]);
        const S = RTV.structArray(
          ["type", "subs"],
          [RTV.struct({ type: RTV.char("()"), subs: subsCell })]
        );
        rt.activeAccessors.add(guardKey);
        try {
          const result = subsasgnFn(1, base, S, rhsMv);
          return result !== undefined ? result : base;
        } finally {
          rt.activeAccessors.delete(guardKey);
        }
      }
    }
  }
  // When the base is a scalar number and the RHS is a class instance or struct,
  // replace the variable entirely
  if (isRuntimeNumber(mv)) {
    const rhsMv = ensureRuntimeValue(rhs);
    if (isRuntimeClassInstance(rhsMv) || isRuntimeStruct(rhsMv)) {
      return rhsMv;
    }
  }
  // Containers other than numeric/char arrays store directly.
  if (
    isRuntimeCell(mv) ||
    isRuntimeStruct(mv) ||
    isRuntimeStructArray(mv) ||
    isRuntimeClassInstance(mv) ||
    isRuntimeClassInstanceArray(mv) ||
    isRuntimeString(mv) ||
    isRuntimeStringArray(mv) ||
    isRuntimeSparseMatrix(mv)
  ) {
    const idxMvals = resolveIndices(
      indices,
      endResolver(rt, mv, indices.length)
    );
    return mIndexStore(
      mv,
      idxMvals,
      ensureRuntimeValue(rhs),
      isRuntimeCell(mv),
      rt,
      storeBack
    );
  }
  // Convert scalar number/logical/complex to 1x1 tensor for indexed assignment
  let wasScalar = false;
  // A char base stays char: index-assign on its numeric code points, then
  // convert back (MATLAB: x='abc'; x(2)=66 -> 'aBc', class char).
  let wasChar = false;
  if (!isRuntimeTensor(mv)) {
    if (isRuntimeNumber(mv)) {
      mv = RTV.tensor(allocFloat64Array([mv]), [1, 1]);
      wasScalar = true;
    } else if (isRuntimeLogical(mv)) {
      const t = RTV.tensor(allocFloat64Array([mv ? 1 : 0]), [1, 1]);
      t._isLogical = true;
      mv = t;
      wasScalar = true;
    } else if (isRuntimeComplexNumber(mv)) {
      mv = RTV.tensor(
        allocFloat64Array([mv.re]),
        [1, 1],
        allocFloat64Array([mv.im])
      );
      wasScalar = true;
    } else if (isRuntimeChar(mv)) {
      mv = charToNumericTensor(mv);
      wasChar = true;
    } else {
      mv = RTV.tensor(allocFloat64Array(0), [0, 0]);
    }
  }
  const idxMvals = resolveIndices(indices, endResolver(rt, mv, indices.length));
  let rhsMv = ensureRuntimeValue(rhs);
  // A char RHS assigned into a numeric/char array becomes its code point(s)
  // (MATLAB: x=[1 2 3]; x(2)='A' -> [1 65 3]); '' still deletes.
  if (isRuntimeChar(rhsMv)) rhsMv = charToNumericTensor(rhsMv);
  const result = mIndexStore(mv, idxMvals, rhsMv, false, rt);
  if (wasChar && isRuntimeTensor(result)) {
    return numericTensorToChar(result);
  }
  if (wasScalar && isRuntimeTensor(result) && result.data.length === 1) {
    if (result.imag && result.imag[0] !== 0) {
      return RTV.complex(result.data[0], result.imag[0]);
    }
    if (result._isLogical === true) return result.data[0] !== 0;
    return result.data[0];
  }
  return result;
}

/** The value of a numeric scalar subscript, else 0. */
function scalarIndexValue(idx: unknown): number {
  if (typeof idx === "number") return idx;
  if (typeof idx === "object" && idx !== null) {
    const v = idx as RuntimeValue;
    if (isRuntimeTensor(v) && v.data.length === 1 && !v._isLogical)
      return v.data[0];
  }
  return 0;
}

/**
 * Cell-indexed assignment (curly braces).
 */
export function indexCellStore(
  base: unknown,
  indices: unknown[],
  rhs: unknown,
  rt: Runtime
): unknown {
  if (base === undefined || base === null) base = RTV.cell([], [0, 0]);
  let mv = ensureRuntimeValue(base);
  // Dictionary curly-brace assignment: d{key} = value — wraps in cell
  if (isRuntimeDictionary(mv)) {
    if (indices.length !== 1)
      throw new RuntimeError(
        "Dictionary brace assignment requires exactly one key argument"
      );
    const key = ensureRuntimeValue(indices[0]);
    const rhsMv = ensureRuntimeValue(rhs);
    const cellVal = RTV.cell([rhsMv], [1, 1]);
    // Use dictInsertSingle to avoid expandValues unwrapping the cell
    return dictInsertSingle(mv, key, cellVal);
  }
  // str{k} = value sets one string element.
  if (isRuntimeString(mv) || isRuntimeStringArray(mv)) {
    const idxMvals = resolveIndices(
      indices,
      endResolver(rt, mv, indices.length)
    );
    return mIndexStore(mv, idxMvals, ensureRuntimeValue(rhs), true, rt);
  }
  if (!isRuntimeCell(mv)) mv = RTV.cell([], [0, 0]);
  const idxMvals = resolveIndices(indices, endResolver(rt, mv, indices.length));
  const rhsMv = ensureRuntimeValue(rhs);
  return mIndexStore(mv, idxMvals, rhsMv, false, rt);
}

/**
 * Multi-output assignment to cell elements: [c{idx}] = func()
 * Distributes individual results to individual cell positions.
 */
export function multiOutputCellAssign(
  base: unknown,
  indices: unknown,
  results: unknown[],
  rt: Runtime
): unknown {
  if (base === undefined || base === null) base = RTV.cell([], [0, 0]);
  let mv = ensureRuntimeValue(base);
  if (!isRuntimeCell(mv)) mv = RTV.cell([], [0, 0]);

  // Get the linear indices as an array of numbers
  let idxArray: number[];
  if (indices === COLON_SENTINEL) {
    // [c{:}] — all linear indices of the cell
    const n = (mv as import("../runtime/types.js").RuntimeCell).data.length;
    idxArray = [];
    for (let i = 1; i <= n; i++) idxArray.push(i);
  } else {
    const idxMv = ensureRuntimeValue(indices);
    if (isRuntimeTensor(idxMv)) {
      idxArray = Array.from(idxMv.data);
    } else if (isRuntimeNumber(idxMv)) {
      idxArray = [idxMv];
    } else {
      idxArray = [1];
    }
  }

  // Assign each result to the corresponding cell position
  for (let i = 0; i < idxArray.length && i < results.length; i++) {
    const idx = idxArray[i];
    const rhsMv = ensureRuntimeValue(results[i]);
    const idxMvals = resolveIndices([idx], endResolver(rt, mv, 1));
    mv = mIndexStore(mv, idxMvals, rhsMv, false, rt) as typeof mv;
  }

  return mv;
}
