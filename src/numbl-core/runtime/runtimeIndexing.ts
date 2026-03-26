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
  isRuntimeSparseMatrix,
  FloatXArray,
} from "../runtime/types.js";
import { COLON_SENTINEL, END_SENTINEL } from "../executor/types.js";
import {
  isDeferredRange,
  isDeferredHorzcat,
  ensureRuntimeValue,
} from "./runtimeHelpers.js";
import { resolveHorzcat as _resolveHorzcat } from "./runtimeOperators.js";
import type { Runtime } from "./runtime.js";

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
 * Get an end resolver function for the given runtime value.
 */
function endResolver(
  mv: RuntimeValue,
  numIndices: number
): (dim: number) => number {
  return (dim: number): number => {
    if (isRuntimeTensor(mv)) {
      if (numIndices === 1) return mv.data.length;
      return dim < mv.shape.length ? mv.shape[dim] : 1;
    }
    if (isRuntimeChar(mv)) return mv.value.length;
    if (isRuntimeCell(mv)) {
      if (numIndices === 1) return mv.data.length;
      return dim < mv.shape.length ? mv.shape[dim] : 1;
    }
    if (isRuntimeStructArray(mv)) return mv.elements.length;
    if (isRuntimeSparseMatrix(mv)) {
      if (numIndices === 1) return mv.m * mv.n;
      return dim === 0 ? mv.m : dim === 1 ? mv.n : 1;
    }
    return 1;
  };
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
  // ── Fast path: tensor with scalar numeric indices ──
  // Avoids resolveIndices, ensureRuntimeValue, mIndex dispatch, and all
  // intermediate array allocations for the most common indexing pattern.
  if (
    typeof base === "object" &&
    base !== null &&
    (base as { kind?: string }).kind === "tensor"
  ) {
    const t = base as import("../runtime/types.js").RuntimeTensor;
    const nIdx = indices.length;
    if (nIdx === 1) {
      // 1D: linear index
      const idx = indices[0];
      if (typeof idx === "number") {
        const i = Math.round(idx) - 1;
        if (i < 0 || i >= t.data.length)
          throw new RuntimeError("Index exceeds array bounds");
        if (t.imag !== undefined) {
          const im = t.imag[i];
          return im === 0 ? t.data[i] : RTV.complex(t.data[i], im);
        }
        return t.data[i];
      }
    } else if (nIdx === 2) {
      // 2D: row, col
      const ri = indices[0];
      const ci = indices[1];
      if (typeof ri === "number" && typeof ci === "number") {
        const s = t.shape;
        const rows = s.length === 0 ? 1 : s.length === 1 ? 1 : s[0];
        const cols = s.length === 0 ? 1 : s.length === 1 ? s[0] : s[1];
        const r = Math.round(ri) - 1;
        const c = Math.round(ci) - 1;
        if (r < 0 || r >= rows || c < 0 || c >= cols)
          throw new RuntimeError("Index exceeds array bounds");
        const lin = c * rows + r;
        if (t.imag !== undefined) {
          const im = t.imag[lin];
          return im === 0 ? t.data[lin] : RTV.complex(t.data[lin], im);
        }
        return t.data[lin];
      }
    } else if (nIdx >= 3) {
      // N-D: all indices must be scalar numbers
      let allNumeric = true;
      for (let k = 0; k < nIdx; k++) {
        if (typeof indices[k] !== "number") {
          allNumeric = false;
          break;
        }
      }
      if (allNumeric) {
        const s = t.shape;
        let lin = 0;
        let stride = 1;
        for (let k = 0; k < nIdx; k++) {
          const dimSize = k < s.length ? s[k] : 1;
          const sub = Math.round(indices[k] as number) - 1;
          if (sub < 0 || sub >= dimSize)
            throw new RuntimeError("Index exceeds array bounds");
          lin += sub * stride;
          stride *= dimSize;
        }
        if (t.imag !== undefined) {
          const im = t.imag[lin];
          return im === 0 ? t.data[lin] : RTV.complex(t.data[lin], im);
        }
        return t.data[lin];
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
  const idxMvals = resolveIndices(indices, endResolver(mv, indices.length));
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
      const resolvedIndices = resolveIndicesForClassInstance(
        rt,
        mv,
        base,
        indices
      );
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

  // Default: scalar class instance indexing.
  // F(1), F(:), F(1,1), F(:,1), F(1,:), F(:,:) etc. all return F itself.
  {
    let valid = true;
    for (const rawIdx of indices) {
      if (rawIdx === COLON_SENTINEL) continue;
      const idx = ensureRuntimeValue(rawIdx);
      if (isRuntimeNumber(idx) && Math.round(idx) === 1) continue;
      valid = false;
      break;
    }
    if (valid) return base;
  }
  throw new RuntimeError(`Index exceeds class instance dimensions`);
}

/** Resolve raw index arguments for a class instance before passing to subsref.
 *  Handles END_SENTINEL (via the class's end method), deferred ranges,
 *  and arrow-function wrappers.  COLON_SENTINEL becomes RTV.char(":"). */
function resolveIndicesForClassInstance(
  rt: Runtime,
  mv: import("../runtime/types.js").RuntimeClassInstance,
  base: unknown,
  indices: unknown[]
): RuntimeValue[] {
  const numIndices = indices.length;

  const resolveEndForDim = (dim: number): number => {
    const endFn = rt.cachedResolveClassMethod(mv.className, "end");
    if (endFn) {
      const result = endFn(1, base, RTV.num(dim + 1), RTV.num(numIndices));
      return toNumber(ensureRuntimeValue(result));
    }
    return 1;
  };

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
        const resolvedIndices = resolveIndicesForClassInstance(
          rt,
          mv,
          base,
          indices
        );
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

  if (!isRuntimeCell(mv)) throw new RuntimeError("Cell indexing on non-cell");
  const idxMvals = resolveIndices(indices, endResolver(mv, indices.length));
  // CSL for 1D cell indexing
  if (idxMvals.length === 1) {
    const idx = idxMvals[0];
    if (isRuntimeString(idx) && idx === "__COLON__") {
      return [...mv.data];
    }
    if (isRuntimeTensor(idx)) {
      const result: unknown[] = [];
      for (let k = 0; k < idx.data.length; k++) {
        const i = Math.round(idx.data[k]) - 1;
        if (i < 0 || i >= mv.data.length)
          throw new RuntimeError("Cell index exceeds bounds");
        result.push(mv.data[i]);
      }
      return result;
    }
  }
  // Multi-index or scalar cell access: mIndex returns a cell for paren-indexing,
  // so unwrap to get the contents for curly-brace indexing.
  const result = mIndex(mv, idxMvals);
  if (isRuntimeCell(result)) {
    if (result.data.length === 1) return result.data[0];
    return [...result.data];
  }
  return result;
}

/**
 * Indexed assignment with support for struct arrays, cell deletion, and scalar expansion.
 */
export function indexStore(
  rt: Runtime,
  base: unknown,
  indices: unknown[],
  rhs: unknown,
  skipSubsasgn = false
): unknown {
  // Treat undefined/null as an empty tensor (uninitialized variable).
  // Exception: if RHS is a class instance, replace entirely
  // behaviour for `h(k) = obj` when h is an uninitialized output variable).
  if (base === undefined || base === null) {
    const rhsCheck = ensureRuntimeValue(rhs);
    if (isRuntimeClassInstance(rhsCheck)) {
      return rhsCheck;
    }
    base = RTV.tensor(new FloatXArray(0), [0, 0]);
  }
  let mv = ensureRuntimeValue(base);
  // Struct array element assignment
  if (
    indices.length === 1 &&
    (isRuntimeStructArray(mv) || (isRuntimeTensor(mv) && mv.data.length === 0))
  ) {
    const rhsMv = ensureRuntimeValue(rhs);
    if (isRuntimeStruct(rhsMv)) {
      const idxVal = ensureRuntimeValue(indices[0]);
      const k = Math.round(toNumber(idxVal)) - 1;
      const existingElements = isRuntimeStructArray(mv) ? [...mv.elements] : [];
      const existingFieldNames = isRuntimeStructArray(mv) ? mv.fieldNames : [];
      const allFieldNames = [
        ...new Set([...existingFieldNames, ...[...rhsMv.fields.keys()]]),
      ];
      while (existingElements.length <= k) {
        existingElements.push(
          RTV.struct(
            new Map(
              allFieldNames.map(f => [
                f,
                RTV.tensor(new FloatXArray(0), [0, 0]),
              ])
            )
          )
        );
      }
      existingElements[k] = rhsMv;
      return RTV.structArray(allFieldNames, existingElements);
    }
  }
  // Struct array element deletion: sa(idx) = []
  if (isRuntimeStructArray(mv) && indices.length === 1) {
    const rhsMv = ensureRuntimeValue(rhs);
    if (isRuntimeTensor(rhsMv) && rhsMv.data.length === 0) {
      const idx = ensureRuntimeValue(indices[0]);
      const toDelete = new Set<number>();
      if (isRuntimeNumber(idx)) {
        toDelete.add(Math.round(idx) - 1);
      } else if (isRuntimeTensor(idx)) {
        if (idx._isLogical) {
          for (let i = 0; i < idx.data.length; i++) {
            if (idx.data[i] !== 0) toDelete.add(i);
          }
        } else {
          for (let i = 0; i < idx.data.length; i++)
            toDelete.add(Math.round(idx.data[i]) - 1);
        }
      } else if (isRuntimeLogical(idx)) {
        if (idx) toDelete.add(0);
      }
      const newElements = mv.elements.filter((_, i) => !toDelete.has(i));
      return RTV.structArray(mv.fieldNames, newElements);
    }
  }
  // Cell element deletion
  if (isRuntimeCell(mv) && indices.length === 1) {
    const rhsMv = ensureRuntimeValue(rhs);
    if (isRuntimeTensor(rhsMv) && rhsMv.data.length === 0) {
      const idx = ensureRuntimeValue(indices[0]);
      const toDelete = new Set<number>();
      if (isRuntimeNumber(idx)) {
        toDelete.add(Math.round(idx) - 1);
      } else if (isRuntimeTensor(idx)) {
        if (idx._isLogical) {
          for (let i = 0; i < idx.data.length; i++) {
            if (idx.data[i] !== 0) toDelete.add(i);
          }
        } else {
          for (let i = 0; i < idx.data.length; i++)
            toDelete.add(Math.round(idx.data[i]) - 1);
        }
      } else if (isRuntimeLogical(idx)) {
        if (idx) toDelete.add(0);
      }
      const newData = mv.data.filter((_, i) => !toDelete.has(i));
      if (mv.shape[0] === 1) return RTV.cell(newData, [1, newData.length]);
      if (mv.shape[1] === 1) return RTV.cell(newData, [newData.length, 1]);
      return RTV.cell(newData, [1, newData.length]);
    }
  }
  // Cell paren-indexed assignment
  if (isRuntimeCell(mv)) {
    const idxMvals = resolveIndices(indices, endResolver(mv, indices.length));
    const rhsMv = ensureRuntimeValue(rhs);
    return mIndexStore(mv, idxMvals, rhsMv, true);
  }
  // Struct scalar indexed assignment
  if (isRuntimeStruct(mv)) {
    return ensureRuntimeValue(rhs);
  }
  // Class instance paren-indexed assignment: dispatch to subsasgn if available
  if (isRuntimeClassInstance(mv)) {
    const guardKey = `${mv.className}.subsasgn`;
    if (!skipSubsasgn && !rt.activeAccessors.has(guardKey)) {
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
          const result = subsasgnFn(1, base, S, ensureRuntimeValue(rhs));
          return result !== undefined ? result : base;
        } finally {
          rt.activeAccessors.delete(guardKey);
        }
      }
    }
    // No subsasgn or recursion guard active — just replace (fallback)
    return ensureRuntimeValue(rhs);
  }
  // Sparse matrix indexed assignment — delegate to storeIntoRTValueIndex
  if (isRuntimeSparseMatrix(mv)) {
    const idxMvals = resolveIndices(indices, endResolver(mv, indices.length));
    const rhsMv = ensureRuntimeValue(rhs);
    return mIndexStore(mv, idxMvals, rhsMv);
  }
  // When the base is a scalar number and the RHS is a class instance or struct,
  // replace the variable entirely
  if (isRuntimeNumber(mv)) {
    const rhsMv = ensureRuntimeValue(rhs);
    if (isRuntimeClassInstance(rhsMv) || isRuntimeStruct(rhsMv)) {
      return rhsMv;
    }
  }
  // Convert scalar number/logical/complex to 1x1 tensor for indexed assignment
  let wasScalar = false;
  if (!isRuntimeTensor(mv)) {
    if (isRuntimeNumber(mv)) {
      mv = RTV.tensor(new FloatXArray([mv]), [1, 1]);
      wasScalar = true;
    } else if (isRuntimeLogical(mv)) {
      const t = RTV.tensor(new FloatXArray([mv ? 1 : 0]), [1, 1]);
      t._isLogical = true;
      mv = t;
      wasScalar = true;
    } else if (isRuntimeComplexNumber(mv)) {
      mv = RTV.tensor(
        new FloatXArray([mv.re]),
        [1, 1],
        new FloatXArray([mv.im])
      );
      wasScalar = true;
    } else {
      mv = RTV.tensor(new FloatXArray(0), [0, 0]);
    }
  }
  const idxMvals = resolveIndices(indices, endResolver(mv, indices.length));
  const rhsMv = ensureRuntimeValue(rhs);
  const result = mIndexStore(mv, idxMvals, rhsMv);
  // Preserve _isLogical flag when assigning logical values into a logical tensor
  if (
    isRuntimeTensor(result) &&
    mv._isLogical &&
    (isRuntimeLogical(rhsMv) ||
      (isRuntimeTensor(rhsMv) && rhsMv._isLogical) ||
      (isRuntimeNumber(rhsMv) && (rhsMv === 0 || rhsMv === 1)))
  ) {
    result._isLogical = true;
  }
  if (wasScalar && isRuntimeTensor(result) && result.data.length === 1) {
    if (result.imag && result.imag[0] !== 0) {
      return RTV.complex(result.data[0], result.imag[0]);
    }
    return result.data[0];
  }
  return result;
}

/**
 * Cell-indexed assignment (curly braces).
 */
export function indexCellStore(
  base: unknown,
  indices: unknown[],
  rhs: unknown
): unknown {
  if (base === undefined || base === null) base = RTV.cell([], [0, 0]);
  let mv = ensureRuntimeValue(base);
  if (!isRuntimeCell(mv)) mv = RTV.cell([], [0, 0]);
  const idxMvals = resolveIndices(indices, endResolver(mv, indices.length));
  const rhsMv = ensureRuntimeValue(rhs);
  return mIndexStore(mv, idxMvals, rhsMv);
}

/**
 * Multi-output assignment to cell elements: [c{idx}] = func()
 * Distributes individual results to individual cell positions.
 */
export function multiOutputCellAssign(
  base: unknown,
  indices: unknown,
  results: unknown[]
): unknown {
  if (base === undefined || base === null) base = RTV.cell([], [0, 0]);
  let mv = ensureRuntimeValue(base);
  if (!isRuntimeCell(mv)) mv = RTV.cell([], [0, 0]);

  // Get the linear indices as an array of numbers
  const idxMv = ensureRuntimeValue(indices);
  let idxArray: number[];
  if (isRuntimeTensor(idxMv)) {
    idxArray = Array.from(idxMv.data);
  } else if (isRuntimeNumber(idxMv)) {
    idxArray = [idxMv];
  } else {
    idxArray = [1];
  }

  // Assign each result to the corresponding cell position
  for (let i = 0; i < idxArray.length && i < results.length; i++) {
    const idx = idxArray[i];
    const rhsMv = ensureRuntimeValue(results[i]);
    const idxMvals = resolveIndices([idx], endResolver(mv, 1));
    mv = mIndexStore(mv, idxMvals, rhsMv) as typeof mv;
  }

  return mv;
}
