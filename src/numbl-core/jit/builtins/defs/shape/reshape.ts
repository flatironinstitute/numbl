/**
 * `reshape` builtin. Two surface forms:
 *
 *   Form A — variadic scalar dims:  reshape(A, d1, d2, …, dN)
 *   Form B — vector of dims:        reshape(A, [d1, d2, …, dN])
 *
 * Discipline (mirrors numbl's array-manipulation `reshape`):
 *   - 1 ≤ N ≤ MTOC2_MAX_NDIM (8). Trailing-singletons are stripped
 *     down to a 2-axis minimum (numbl:
 *     `while (s.length>2 && s.last===1) s.pop()`), then padded back
 *     up to 2 axes if N=1 was given (mtoc2 represents every tensor
 *     as min-2D).
 *   - Form A — each `di` is a scalar real double; statically-known
 *     finite non-negative integers pin the corresponding axis, and
 *     dynamic scalars (no `exact`) leave the axis as `unknown` in
 *     the result lattice.
 *   - Form B — the dim vector must be a statically-known
 *     `Float64Array` of dims (runtime vectors are not yet supported).
 *   - Form A allows at most one `[]` auto-infer slot. The translator
 *     fills it from `numel(A)` / `prod(other dims)` when both are
 *     known statically, else codegen passes `-1L` to
 *     `mtoc2_reshape_nd` which computes the slot at runtime. Form B
 *     still rejects the placeholder (the dim vector itself must be a
 *     static constant; a single `[]` inside the bracket is ambiguous).
 *   - Input `A` must be a real numeric (scalar or tensor); complex /
 *     handle / struct / class / void / string rejected with TypeError.
 *   - Element-count check at transfer time when the input shape and
 *     every new dim are statically known; deferred to the runtime
 *     helper otherwise.
 *
 * Codegen routes through the per-op runtime helper `mtoc2_reshape_nd`
 * (declared in `src/codegen/runtime/tensor_reshape_nd.h`), which
 * already accepts a runtime `(int ndim, const long *dims)` pair plus
 * an optional `-1L` auto-infer sentinel. The result is freshly owned,
 * so the standard ANF / scope-exit-free pipeline carries it without
 * changes.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  DIM_ONE,
  EXACT_ARRAY_MAX_ELEMENTS,
  MTOC2_MAX_NDIM,
  tensorComplex,
  tensorComplexFromDims,
  tensorDouble,
  tensorDoubleFromDims,
  scalarComplex,
  scalarDouble,
  shapeNumel,
  signFromNumber,
  isNumeric,
  isScalar,
} from "../../../lowering/types.js";
import type { DimInfo, Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import {
  mtoc2_reshape_nd as jsReshape,
  mtoc2_reshape_nd_complex as jsReshapeComplex,
} from "../../runtime/snippets.gen.js";

const MAX_DIM_ARGS = MTOC2_MAX_NDIM;
import {
  exactComplex,
  exactComplexArray,
  exactDouble,
  exactRealArray,
} from "../_shared.js";

/** Per-axis resolution. `argIndex` (when present) refers to the index
 *  into the original Form A dim arg list (i.e. `argTypes.slice(1)` —
 *  argIndex 0 is `argsC[1]` in the full call's argsC array). Form B
 *  always produces all-exact axes (the dim vector itself must be
 *  statically known), so the argIndex is irrelevant there.
 *  `infer` is the `[]` auto-slot in Form A: at most one per call,
 *  filled from `numel / prod(others)` at runtime when the value can't
 *  be derived statically. */
type ResolvedAxis =
  | { kind: "exact"; value: number; argIndex?: number }
  | { kind: "dynamic"; argIndex: number }
  | { kind: "infer"; argIndex: number };

interface ResolvedNewShape {
  axes: ResolvedAxis[];
}

/** Detect numbl's `[]` placeholder: a multi-element numeric tensor with
 *  zero total elements (e.g. the empty `[]` literal lowers to a
 *  `TensorBuild` with shape `[0, 0]`). */
function isEmptyPlaceholder(t: Type): boolean {
  if (!isNumeric(t)) return false;
  if (isScalar(t)) return false;
  if (t.shape === undefined) return false;
  return shapeNumel(t.shape) === 0;
}

/** Apply numbl's strip / pad rules to a resolved axis list. Drops
 *  trailing exact-1 axes subject to a 2-axis floor; never strips a
 *  dynamic or infer axis. Pads with trailing exact-1 axes if length < 2. */
function normalizeAxes(axes: ResolvedAxis[]): ResolvedAxis[] {
  const out = axes.slice();
  while (
    out.length > 2 &&
    out[out.length - 1].kind === "exact" &&
    (out[out.length - 1] as { value: number }).value === 1
  ) {
    out.pop();
  }
  while (out.length < 2) {
    out.push({ kind: "exact", value: 1 });
  }
  return out;
}

/** Resolve the dim args (i.e. `argTypes.slice(1)`) into a per-axis
 *  list. Handles Form A (variadic scalars) and Form B (exact vector)
 *  in one pass. Form A dim args may be dynamic; Form B requires the
 *  vector itself to be statically known. Throws on any out-of-range
 *  or wrong-typed dim. */
function resolveNewShape(dimArgTypes: Type[]): ResolvedNewShape {
  if (dimArgTypes.length < 1) {
    throw new UnsupportedConstruct(
      `'reshape' requires at least one dim argument after the input tensor`
    );
  }

  let axes: ResolvedAxis[];

  if (
    dimArgTypes.length === 1 &&
    isNumeric(dimArgTypes[0]) &&
    !isScalar(dimArgTypes[0])
  ) {
    const a = dimArgTypes[0];
    if (isEmptyPlaceholder(a)) {
      throw new UnsupportedConstruct(
        `reshape: '[]' auto-infer slot is not yet supported; specify all dims explicitly`
      );
    }
    if (a.elem !== "double" || a.isComplex) {
      throw new TypeError(
        `'reshape' dim vector must be a real-double tensor (got ${a.elem}${a.isComplex ? " complex" : ""})`
      );
    }
    const arr = exactRealArray(a);
    if (arr === undefined) {
      throw new UnsupportedConstruct(
        `'reshape' dim vector must be a statically-known constant in v1`
      );
    }
    if (arr.length < 1 || arr.length > MAX_DIM_ARGS) {
      throw new UnsupportedConstruct(
        `'reshape' supports 1..${MAX_DIM_ARGS} output dims (got ${arr.length})`
      );
    }
    axes = [];
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (!Number.isFinite(v) || !Number.isInteger(v) || v < 0) {
        throw new TypeError(
          `'reshape' dim ${i + 1} must be a finite non-negative integer (got ${v})`
        );
      }
      axes.push({ kind: "exact", value: v });
    }
  } else {
    if (dimArgTypes.length > MAX_DIM_ARGS) {
      throw new UnsupportedConstruct(
        `'reshape' supports 1..${MAX_DIM_ARGS} output dims (got ${dimArgTypes.length})`
      );
    }
    axes = [];
    let sawInfer = false;
    for (let i = 0; i < dimArgTypes.length; i++) {
      const a = dimArgTypes[i];
      if (!isNumeric(a) || a.elem !== "double" || a.isComplex) {
        throw new TypeError(
          `'reshape' dim arg ${i + 1} must be a scalar real double (got ${a.kind})`
        );
      }
      if (!isScalar(a)) {
        if (isEmptyPlaceholder(a)) {
          if (sawInfer) {
            throw new UnsupportedConstruct(
              `'reshape' allows at most one '[]' auto-infer slot`
            );
          }
          sawInfer = true;
          axes.push({ kind: "infer", argIndex: i });
          continue;
        }
        throw new TypeError(
          `'reshape' dim arg ${i + 1} must be a scalar real double (got tensor)`
        );
      }
      const v = exactDouble(a);
      if (v === undefined) {
        axes.push({ kind: "dynamic", argIndex: i });
        continue;
      }
      if (!Number.isFinite(v) || !Number.isInteger(v) || v < 0) {
        throw new TypeError(
          `'reshape' dim arg ${i + 1} must be a finite non-negative integer (got ${v})`
        );
      }
      axes.push({ kind: "exact", value: v, argIndex: i });
    }
  }

  return { axes: normalizeAxes(axes) };
}

/** Concrete number[] shape iff every axis is exact, else undefined. */
function exactShape(r: ResolvedNewShape): number[] | undefined {
  const out: number[] = [];
  for (const a of r.axes) {
    if (a.kind !== "exact") return undefined;
    out.push(a.value);
  }
  return out;
}

/** Emit the C `long` expression for one axis. `dimArgsC` is sliced
 *  from the full Call argsC so its index 0 maps to `axes[*].argIndex
 *  = 0`. The `[]` auto-infer slot emits `-1L` — `mtoc2_reshape_nd`
 *  scans for the sentinel and fills it from `in_total / prod(others)`. */
function dimC(axis: ResolvedAxis, dimArgsC: string[]): string {
  if (axis.kind === "exact") return `${axis.value}L`;
  if (axis.kind === "infer") return `-1L`;
  return `(long)(${dimArgsC[axis.argIndex]})`;
}

export const reshape: Builtin = {
  name: "reshape",
  transfer(argTypes, nargout) {
    if (argTypes.length < 2 || argTypes.length > MAX_DIM_ARGS + 1) {
      throw new TypeError(
        `'reshape' expects 2..${MAX_DIM_ARGS + 1} arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'reshape' does not support multi-output (nargout=${nargout})`
      );
    }
    const a = argTypes[0];
    if (!isNumeric(a)) {
      throw new TypeError(
        `'reshape' first arg must be numeric (got ${a.kind})`
      );
    }
    if (a.elem !== "double" && a.elem !== "logical") {
      throw new TypeError(
        `'reshape' first arg must be a double or logical tensor (got ${a.elem})`
      );
    }

    const resolved = resolveNewShape(argTypes.slice(1));
    if (a.shape !== undefined) {
      const inferIdx = resolved.axes.findIndex(ax => ax.kind === "infer");
      if (inferIdx !== -1) {
        const others: number[] = [];
        let allOthersExact = true;
        for (let i = 0; i < resolved.axes.length; i++) {
          if (i === inferIdx) continue;
          const ax = resolved.axes[i];
          if (ax.kind === "exact") {
            others.push(ax.value);
          } else {
            allOthersExact = false;
            break;
          }
        }
        if (allOthersExact) {
          const inTotal = a.shape.reduce((p, d) => p * d, 1);
          const otherProd = others.reduce((p, d) => p * d, 1);
          if (otherProd === 0 && inTotal !== 0) {
            throw new TypeError(
              `'reshape' element count mismatch: input has ${inTotal} ` +
                `elements but the explicit dims around '[]' multiply to 0`
            );
          }
          if (otherProd > 0 && inTotal % otherProd !== 0) {
            throw new TypeError(
              `'reshape' element count mismatch: input has ${inTotal} ` +
                `elements, not divisible by ${otherProd} (the explicit ` +
                `dims around '[]')`
            );
          }
          const inferred = otherProd === 0 ? 0 : inTotal / otherProd;
          resolved.axes[inferIdx] = { kind: "exact", value: inferred };
        }
      }
    }
    const newShape = exactShape(resolved);

    if (newShape !== undefined) {
      const newTotal = newShape.reduce((p, d) => p * d, 1);
      if (a.shape !== undefined) {
        const inTotal = a.shape.reduce((p, d) => p * d, 1);
        if (inTotal !== newTotal) {
          throw new TypeError(
            `'reshape' element count mismatch: input has ${inTotal} elements, ` +
              `requested shape needs ${newTotal}`
          );
        }
      }

      if (newShape.every(d => d === 1)) {
        if (a.isComplex) {
          const cxScalar = exactComplex(a);
          if (cxScalar !== undefined) return [scalarComplex(cxScalar)];
          const cxArr = exactComplexArray(a);
          if (cxArr !== undefined && cxArr.re.length === 1) {
            return [scalarComplex({ re: cxArr.re[0], im: cxArr.im[0] })];
          }
          return [scalarComplex()];
        }
        if (typeof a.exact === "number") {
          return [scalarDouble(signFromNumber(a.exact), a.exact)];
        }
        return [scalarDouble("unknown")];
      }

      if (a.isComplex) {
        const cxArr = exactComplexArray(a);
        if (
          cxArr !== undefined &&
          newTotal <= EXACT_ARRAY_MAX_ELEMENTS &&
          cxArr.re.length === newTotal
        ) {
          return [tensorComplex(newShape, cxArr)];
        }
        return [tensorComplex(newShape)];
      }
      if (
        a.exact instanceof Float64Array &&
        newTotal <= EXACT_ARRAY_MAX_ELEMENTS &&
        a.exact.length === newTotal
      ) {
        return [tensorDouble(newShape, a.exact)];
      }
      return [tensorDouble(newShape)];
    }

    const dims: DimInfo[] = resolved.axes.map(axis =>
      axis.kind === "exact"
        ? axis.value === 1
          ? DIM_ONE
          : { kind: "exact", value: axis.value }
        : { kind: "unknown" }
    );
    return a.isComplex
      ? [tensorComplexFromDims(dims)]
      : [tensorDoubleFromDims(dims)];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_reshape_nd");
    useRuntime("mtoc2_reshape_nd_complex");
    const resolved = resolveNewShape(argTypes.slice(1));
    const newShape = exactShape(resolved);

    if (newShape !== undefined && newShape.every(d => d === 1)) {
      return argsC[0];
    }

    const a = argTypes[0];
    const isComplex = isNumeric(a) && a.isComplex;
    const fn = isComplex ? "mtoc2_reshape_nd_complex" : "mtoc2_reshape_nd";
    const dimArgsC = argsC.slice(1);
    const dimList = resolved.axes.map(axis => dimC(axis, dimArgsC)).join(", ");
    return `${fn}(${argsC[0]}, ${resolved.axes.length}, (long[]){${dimList}})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const resolved = resolveNewShape(argTypes.slice(1));
    const newShape = exactShape(resolved);
    if (newShape !== undefined && newShape.every(d => d === 1)) {
      return argsJs[0];
    }
    const a = argTypes[0];
    const dimArgsJs = argsJs.slice(1);
    const dimList = resolved.axes
      .map(axis =>
        axis.kind === "exact"
          ? String(axis.value)
          : axis.kind === "infer"
            ? "-1"
            : `Math.trunc(${dimArgsJs[axis.argIndex]})`
      )
      .join(", ");
    if (isNumeric(a) && a.isComplex) {
      useRuntime("mtoc2_reshape_nd_complex");
      return `mtoc2_reshape_nd_complex(${argsJs[0]}, ${resolved.axes.length}, [${dimList}])`;
    }
    useRuntime("mtoc2_reshape_nd");
    return `mtoc2_reshape_nd(${argsJs[0]}, ${resolved.axes.length}, [${dimList}])`;
  },
  call({ args, argTypes }) {
    const resolved = resolveNewShape(argTypes.slice(1));
    const a = argTypes[0];
    const dims: number[] = [];
    for (const axis of resolved.axes) {
      if (axis.kind === "exact") {
        dims.push(axis.value);
      } else if (axis.kind === "infer") {
        dims.push(-1);
      } else {
        const v = args[axis.argIndex + 1];
        dims.push(Math.trunc(typeof v === "number" ? v : Number(v as object)));
      }
    }
    const reshapeFn =
      isNumeric(a) && a.isComplex ? jsReshapeComplex : jsReshape;
    return [
      reshapeFn(
        args[0] as RuntimeTensor,
        resolved.axes.length,
        dims
      ) as unknown as RuntimeTensor,
    ];
  },
};
