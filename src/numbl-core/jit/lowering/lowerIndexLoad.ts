/**
 * Scalar index-read lowering: `v(i)` / `M(i, j)` / `T(i, j, k)` /
 * `v(end)` / `M(end, end)` / etc.
 *
 * Reached from `lowerFuncCall` whenever a `FuncCall` name resolves to
 * an in-scope multi-element numeric variable AND none of the args are
 * a range / bare colon (that case routes to `lowerIndexSlice`). Each
 * index slot must lower to a scalar real expression; an `end` token
 * inside a slot resolves through the Lowerer's `endStack` to the
 * appropriate axis size.
 */

import type { Expr, Span } from "../parser/index.js";
import { TypeError } from "./errors.js";
import type { IRExpr } from "./ir.js";
import {
  isMultiElement,
  isNumeric,
  isScalarRealNumeric,
  scalarComplex,
  scalarDouble,
  signFromNumber,
  typeToString,
} from "./types.js";
import type { NumericType } from "./types.js";
import type { Lowerer } from "./lower.js";
import { resolveIndexBase } from "./indexResolve.js";
import { lowerIndexSlice } from "./lowerIndexSlice.js";
import { columnMajorOffsetFromIndices } from "./indexFold.js";
import { exactComplexArray, exactRealArray } from "../builtins/defs/_shared.js";

/** Lower an index-read of an in-scope variable. */
export function lowerIndexLoad(
  this: Lowerer,
  name: string,
  argExprs: ReadonlyArray<Expr>,
  span: Span
): IRExpr {
  const { baseTy, baseCName, base } = resolveIndexBase.call(
    this,
    name,
    argExprs.length,
    span,
    { notInScope: "internal", operation: "read" }
  );

  const indices: IRExpr[] = [];
  const numSlots = argExprs.length;
  for (let slot = 0; slot < numSlots; slot++) {
    const axis: number | "linear" = numSlots === 1 ? "linear" : slot;
    this.endStack.push({ baseCName, baseTy, axis });
    let lowered: IRExpr;
    try {
      lowered = this.lowerExpr(argExprs[slot]);
    } finally {
      this.endStack.pop();
    }
    if (!isScalarRealNumeric(lowered.ty)) {
      // Multi-element real-numeric (logical or double) tensor in an
      // index slot is a fancy-index access — logical-mask or vector-of-
      // indices. Route to lowerIndexSlice which builds an IndexSlice IR
      // node with a LogicalMask or IndexVec slot. Re-lowering happens
      // there (deterministic; no externally-visible side effects), so
      // this is safe to bail late.
      if (
        isNumeric(lowered.ty) &&
        !lowered.ty.isComplex &&
        isMultiElement(lowered.ty) &&
        (lowered.ty.elem === "double" || lowered.ty.elem === "logical")
      ) {
        return lowerIndexSlice.call(this, name, argExprs, span);
      }
      throw new TypeError(
        `index ${slot + 1} of '${name}' must be a real scalar ` +
          `(got ${typeToString(lowered.ty)})`,
        argExprs[slot].span
      );
    }
    indices.push(lowered);
  }

  // Constant-fold the element value into the result type when the
  // base carries exact data and every index is a known positive
  // integer. The IR node is still emitted (codegen always emits the
  // runtime read; we only fold at the type level), but downstream
  // type-driven sites — function-spec key, if-cond, builtin transfer
  // (e.g. `zeros(sz(1)-1, sz(2))` after `sz = size(x)`) — can see the
  // value. Same discipline as scalarDouble exact: it propagates
  // through the type system but never substitutes a literal in C.
  const folded = foldedElemType(baseTy, indices);
  return {
    kind: "IndexLoad",
    base,
    indices,
    ty:
      folded ?? (baseTy.isComplex ? scalarComplex() : scalarDouble("unknown")),
    span,
  };
}

/** When `base` has statically-known shape AND exact element data AND
 *  every index in `indices` has a finite positive-integer `exact`,
 *  return a scalar type whose `exact` is the element at the computed
 *  column-major offset. Otherwise return `undefined`. */
function foldedElemType(
  baseTy: NumericType,
  indices: ReadonlyArray<IRExpr>
): NumericType | undefined {
  const shape = baseTy.shape;
  if (shape === undefined) return undefined;

  // Real path: data must be a flat Float64Array exact carrier.
  // Complex path: data must be the split-buffer `{re, im}` carrier.
  const realData = exactRealArray(baseTy);
  const complexData =
    realData === undefined ? exactComplexArray(baseTy) : undefined;
  if (realData === undefined && complexData === undefined) return undefined;

  const idxVals: number[] = [];
  for (const ix of indices) {
    if (!isNumeric(ix.ty) || typeof ix.ty.exact !== "number") return undefined;
    const v = ix.ty.exact;
    if (!Number.isFinite(v) || !Number.isInteger(v) || v < 1) return undefined;
    idxVals.push(v);
  }

  const offset = columnMajorOffsetFromIndices(shape, idxVals);
  if (offset === undefined) return undefined;

  if (complexData !== undefined) {
    return scalarComplex({
      re: complexData.re[offset],
      im: complexData.im[offset],
    });
  }
  const v = realData![offset];
  return scalarDouble(signFromNumber(v), v);
}
