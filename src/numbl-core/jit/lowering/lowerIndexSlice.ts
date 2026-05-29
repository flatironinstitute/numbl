/**
 * Range / colon / scalar-mix indexed-read lowering: `v(:)`, `v(a:b)`,
 * `M(:, j)`, `T(:, :, i)`, … .
 *
 * Reached from `lowerFuncCall` whenever a `FuncCall` name resolves to
 * an in-scope multi-element numeric variable AND at least one arg is a
 * `Range` or bare `Colon`. Two acceptable arities: 1 slot (linear) or
 * `base.ty.dims.length` slots (one per axis). Result-shape rules
 * mirror numbl and mtoc:
 *
 *   single-slot Colon                   → column vector, length numel(base)
 *   single-slot Range, row-vec base     → row vector, count(range)
 *   single-slot Range, col-vec base     → col vector, count(range)
 *   single-slot Range, matrix / N-D base → row vector, count(range)
 *   multi-slot Colon at axis k          → base.dims[k]
 *   multi-slot Range at axis k          → exact count when statically
 *                                          derivable, else unknown
 *   multi-slot Scalar at axis k         → exact 1
 */

import type { Expr, Span } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr, IndexSliceArg } from "./ir.js";
import {
  isColVecTy,
  isRowVecTy,
  isScalarRealNumeric,
  isNumeric,
  shapeNumel,
  tensorComplex,
  tensorComplexFromDims,
  tensorDouble,
  tensorDoubleFromDims,
  type DimInfo,
  type NumericType,
  type Type,
  typeToString,
} from "./types.js";
import type { Lowerer } from "./lower.js";
import { resolveIndexBase } from "./indexResolve.js";
import { rangeCountFromExactEnds } from "./rangeCount.js";

/** Compute the static element count of an `IndexSliceArg` (or
 *  `undefined` if it can't be known at compile time). For a Range slot
 *  every endpoint must have an exact numeric value (the step is always
 *  a NumLit by lowerSliceArg's check); the count uses the shared
 *  `rangeCountFromExactEnds` formula (same as `MakeRange` /
 *  `mtoc2_loop_count`). */
function exactRangeCount(slot: IndexSliceArg): number | undefined {
  if (slot.kind !== "Range") return undefined;
  const sExact =
    isNumeric(slot.start.ty) && typeof slot.start.ty.exact === "number"
      ? slot.start.ty.exact
      : undefined;
  const eExact =
    isNumeric(slot.end.ty) && typeof slot.end.ty.exact === "number"
      ? slot.end.ty.exact
      : undefined;
  // lowerSliceArg already required step to be a NumLit; read its value.
  if (slot.step.kind !== "NumLit") return undefined;
  const tExact = slot.step.value;
  if (sExact === undefined || eExact === undefined) return undefined;
  if (tExact === 0) return undefined;
  if (
    !Number.isFinite(sExact) ||
    !Number.isFinite(tExact) ||
    !Number.isFinite(eExact)
  ) {
    return undefined;
  }
  return rangeCountFromExactEnds(sExact, tExact, eExact);
}

/** Per-slot result-dim kind, used by the multi-slot path. Colon takes
 *  its dim from the base; Range derives an exact dim when endpoints
 *  are statically known, else `unknown`; Scalar contributes exactly 1;
 *  IndexVec contributes the slot's numel (the gathered axis size);
 *  LogicalMask contributes `sum(mask)` which is only known at runtime.
 *  Mirrors the doc-block table above. */
function resultDimForSlot(slot: IndexSliceArg, baseDim: DimInfo): DimInfo {
  if (slot.kind === "Colon") return baseDim;
  if (slot.kind === "Range") {
    const n = exactRangeCount(slot);
    return n === undefined ? { kind: "unknown" } : { kind: "exact", value: n };
  }
  if (slot.kind === "IndexVec") {
    const idxTy = slot.expr.ty;
    if (isNumeric(idxTy) && idxTy.shape !== undefined) {
      const n = shapeNumel(idxTy.shape);
      return { kind: "exact", value: n };
    }
    return { kind: "unknown" };
  }
  if (slot.kind === "LogicalMask") {
    return { kind: "unknown" };
  }
  return { kind: "exact", value: 1 };
}

export function lowerIndexSlice(
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
    { notInScope: "internal", operation: "sliceRead" }
  );

  const isSingleSlot = argExprs.length === 1;
  const slots: IndexSliceArg[] = [];
  for (let i = 0; i < argExprs.length; i++) {
    const axis: number | "linear" = isSingleSlot ? "linear" : i;
    slots.push(lowerSliceArg.call(this, baseCName, baseTy, axis, argExprs[i]));
  }

  const wantComplex = baseTy.isComplex;
  let resultTy: Type;
  if (isSingleSlot) {
    const slot = slots[0];
    if (slot.kind === "Colon") {
      // Column vector of length numel(base). If every base dim is
      // statically known, build a concrete shape; otherwise the
      // leading axis is runtime-only.
      if (baseTy.shape !== undefined) {
        const n = shapeNumel(baseTy.shape);
        resultTy = wantComplex ? tensorComplex([n, 1]) : tensorDouble([n, 1]);
      } else {
        const dims: DimInfo[] = [
          { kind: "unknown" },
          { kind: "exact", value: 1 },
        ];
        resultTy = wantComplex
          ? tensorComplexFromDims(dims)
          : tensorDoubleFromDims(dims);
      }
    } else if (slot.kind === "Range") {
      // For an index slot the parser-side range step must be a NumLit,
      // enforced by lowerSliceArg. The range count is known when the
      // endpoints are exact; otherwise the runtime axis stays unknown.
      const isRowVec = isRowVecTy(baseTy);
      const isColVec = isColVecTy(baseTy);
      const countDim: DimInfo = (() => {
        const n = exactRangeCount(slot);
        return n === undefined
          ? { kind: "unknown" }
          : { kind: "exact", value: n };
      })();
      const oneDim: DimInfo = { kind: "exact", value: 1 };
      const resultDims: DimInfo[] = isColVec
        ? [countDim, oneDim]
        : isRowVec
          ? [oneDim, countDim]
          : [oneDim, countDim];
      resultTy = wantComplex
        ? tensorComplexFromDims(resultDims)
        : tensorDoubleFromDims(resultDims);
    } else if (slot.kind === "LogicalMask") {
      // Single-slot linear logical-mask read. The length is `sum(mask)`
      // — only known at runtime. Result shape follows the same rule
      // as single-slot Range: row-vec base → row, col-vec base → col,
      // matrix / N-D base → column vector (numbl matches MATLAB here).
      const isRowVec = isRowVecTy(baseTy);
      const isColVec = isColVecTy(baseTy);
      const unkDim: DimInfo = { kind: "unknown" };
      const oneDim: DimInfo = { kind: "exact", value: 1 };
      const resultDims: DimInfo[] = isRowVec
        ? [oneDim, unkDim]
        : isColVec
          ? [unkDim, oneDim]
          : [unkDim, oneDim];
      resultTy = wantComplex
        ? tensorComplexFromDims(resultDims)
        : tensorDoubleFromDims(resultDims);
    } else {
      throw new UnsupportedConstruct(
        `internal: single-slot scalar slice should have routed through ` +
          `lowerIndexLoad`,
        span
      );
    }
  } else {
    // Multi-slot: one result axis per slot. `tensorDoubleFromDims` will
    // populate `shape` automatically when every slot pins an exact dim.
    const resultDims: DimInfo[] = slots.map((slot, k) =>
      resultDimForSlot(slot, baseTy.dims[k])
    );
    resultTy = wantComplex
      ? tensorComplexFromDims(resultDims)
      : tensorDoubleFromDims(resultDims);
  }

  return {
    kind: "IndexSlice",
    base,
    index: slots,
    ty: resultTy,
    span,
  };
}

/** Lower a single AST index slot into an `IndexSliceArg`. The
 *  `endStack` is pushed around each sub-expression so an embedded
 *  `end` token resolves against the right axis. Exported so the
 *  store-side helper reuses the same logic. */
export function lowerSliceArg(
  this: Lowerer,
  baseCName: string,
  baseTy: NumericType,
  axis: number | "linear",
  arg: Expr
): IndexSliceArg {
  if (arg.type === "Colon") {
    return { kind: "Colon", span: arg.span };
  }
  if (arg.type !== "Range") {
    this.endStack.push({ baseCName, baseTy, axis });
    let expr: IRExpr;
    try {
      expr = this.lowerExpr(arg);
    } finally {
      this.endStack.pop();
    }
    if (isScalarRealNumeric(expr.ty)) {
      return { kind: "Scalar", expr, span: arg.span };
    }
    // Tensor-valued slot — either a numeric "vector of indices" gather
    // (IndexVec) or a logical mask (LogicalMask). LogicalMask is valid
    // in both single-slot (linear) and per-axis forms; IndexVec is
    // only valid in the multi-slot per-axis form for now.
    if (
      isNumeric(expr.ty) &&
      !expr.ty.isComplex &&
      (expr.ty.elem === "double" || expr.ty.elem === "logical") &&
      !isScalarRealNumeric(expr.ty)
    ) {
      if (expr.ty.elem === "logical") {
        return { kind: "LogicalMask", expr, span: arg.span };
      }
      if (axis === "linear") {
        throw new UnsupportedConstruct(
          `linear vector-of-indices reads (got '${typeToString(expr.ty)}' in ` +
            `a single-slot context) are not yet supported`,
          arg.span
        );
      }
      return { kind: "IndexVec", expr, span: arg.span };
    }
    throw new TypeError(
      `index slot must be a real scalar or a numeric index vector ` +
        `(got ${typeToString(expr.ty)})`,
      arg.span
    );
  }
  this.endStack.push({ baseCName, baseTy, axis });
  let start: IRExpr;
  let step: IRExpr;
  let end: IRExpr;
  try {
    start = this.lowerExpr(arg.start);
    if (arg.step === null) {
      step = {
        kind: "NumLit",
        value: 1,
        ty: {
          kind: "Numeric",
          elem: "double",
          isComplex: false,
          dims: [
            { kind: "exact", value: 1 },
            { kind: "exact", value: 1 },
          ],
          shape: [1, 1],
          sign: "positive",
          exact: 1,
        },
        span: arg.span,
      };
    } else {
      step = this.lowerExpr(arg.step);
    }
    end = this.lowerExpr(arg.end);
  } finally {
    this.endStack.pop();
  }
  if (!isScalarRealNumeric(start.ty)) {
    throw new TypeError(
      `range start must be a real scalar (got ${typeToString(start.ty)})`,
      arg.start.span
    );
  }
  if (!isScalarRealNumeric(end.ty)) {
    throw new TypeError(
      `range end must be a real scalar (got ${typeToString(end.ty)})`,
      arg.end.span
    );
  }
  if (!isScalarRealNumeric(step.ty)) {
    throw new TypeError(
      `range step must be a real scalar (got ${typeToString(step.ty)})`,
      arg.step?.span ?? arg.span
    );
  }
  // Index-slot ranges require a compile-time-known step so codegen can
  // derive the loop count + source-index arithmetic. A bare NumLit
  // (`1:5:end`) is the common case; anything else is accepted as long
  // as the lowered expression's type carries an `exact` numeric value
  // — covers `1:-1:end` (parsed as `Unary(Minus, NumLit(1))`),
  // constant folds (`1:(2+1):end`), and exact-tracked variables
  // (`s = 2; a(1:s:end)`).
  if (step.kind !== "NumLit") {
    const exactStep =
      step.ty.kind === "Numeric" && typeof step.ty.exact === "number"
        ? step.ty.exact
        : undefined;
    if (exactStep === undefined) {
      throw new UnsupportedConstruct(
        `range step in an index expression must be a compile-time-known ` +
          `numeric value (got expression with unknown value)`,
        arg.step?.span ?? arg.span
      );
    }
    step = {
      kind: "NumLit",
      value: exactStep,
      ty: {
        kind: "Numeric",
        elem: "double",
        isComplex: false,
        dims: [
          { kind: "exact", value: 1 },
          { kind: "exact", value: 1 },
        ],
        shape: [1, 1],
        sign: exactStep > 0 ? "positive" : exactStep < 0 ? "negative" : "zero",
        exact: exactStep,
      },
      span: arg.step?.span ?? arg.span,
    };
  }
  if (step.value === 0) {
    throw new UnsupportedConstruct(
      `range step in an index expression must be non-zero`,
      arg.step?.span ?? arg.span
    );
  }
  return { kind: "Range", start, step, end, span: arg.span };
}
