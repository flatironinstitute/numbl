/**
 * Scalar indexed-write lowering: `v(i) = x`, `M(i, j) = x`,
 * `T(i, j, k) = x`, `v(end) = x`.
 *
 * Reached from `lowerAssignLValue` whenever the lvalue is an `Index`
 * with a simple `Ident` base AND none of the index slots are a range
 * / colon (that case routes to `lowerIndexSliceStore`).
 */

import type { Expr, LValue, Span } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr, IRStmt } from "./ir.js";
import {
  isMultiElement,
  isNumeric,
  isScalar,
  isScalarRealNumeric,
  shapeNumel,
  typeToString,
} from "./types.js";
import type { Type } from "./types.js";
import type { Lowerer } from "./lower.js";
import { resolveIndexLvalueBase } from "./indexResolve.js";
import { lowerIndexSliceStore } from "./lowerIndexSliceStore.js";

export function lowerIndexStore(
  this: Lowerer,
  lvalue: Extract<LValue, { type: "Index" }>,
  exprAst: Expr,
  span: Span
): IRStmt | IRStmt[] {
  // Resolve either a bare-Ident base (existing path) or a member-
  // rooted base (`obj.field(i) = rhs`). In the member case the IR's
  // `base` Var still names the OWNING root; codegen targets the
  // field slot via `<root.cName>.<fieldPath>` and uses the field's
  // NumericType for offset / complex-lane decisions.
  const { base, baseTy, baseCName, fieldPath, leafTy, displayName } =
    resolveIndexLvalueBase.call(this, lvalue, span, "write");

  // Range/colon slots dispatch to lowerIndexSliceStore — getting here
  // with one means the dispatcher logic is wrong.
  for (const idx of lvalue.indices) {
    if (idx.type === "Range" || idx.type === "Colon") {
      throw new UnsupportedConstruct(
        `internal: lowerIndexStore received a range/colon slot; ` +
          `should have been routed to lowerIndexSliceStore`,
        idx.span
      );
    }
  }

  const indices: IRExpr[] = [];
  const numSlots = lvalue.indices.length;
  for (let slot = 0; slot < numSlots; slot++) {
    const axis: number | "linear" = numSlots === 1 ? "linear" : slot;
    this.endStack.push({ baseCName, baseTy, axis });
    let lowered: IRExpr;
    try {
      lowered = this.lowerExpr(lvalue.indices[slot]);
    } finally {
      this.endStack.pop();
    }
    if (!isScalarRealNumeric(lowered.ty)) {
      // Multi-element tensor in an index slot of a write is a logical-
      // mask write (only the linear single-slot form is supported; the
      // multi-slot logical-mask write is rejected inside
      // lowerIndexSliceStore). Vector-of-indices writes aren't yet
      // plumbed.
      if (
        isNumeric(lowered.ty) &&
        !lowered.ty.isComplex &&
        isMultiElement(lowered.ty) &&
        lowered.ty.elem === "logical"
      ) {
        return lowerIndexSliceStore.call(this, lvalue, exprAst, span);
      }
      throw new TypeError(
        `index ${slot + 1} of '${displayName}' must be a real scalar ` +
          `(got ${typeToString(lowered.ty)})`,
        lvalue.indices[slot].span
      );
    }
    indices.push(lowered);
  }

  // Compile-time grow decline (Layer 1): if an index is a statically
  // known constant that exceeds the base's statically known bounds,
  // the store would GROW the array — valid MATLAB, but the JIT can't
  // model the new shape (the type system fixes the carrier shape). The
  // canonical idiom `v(end+1) = x` on a fixed-shape array folds to a
  // constant index that provably exceeds numel, so it's caught here
  // and declines to the interpreter from the start (no side-effect
  // replay). Dynamic OOB stores — whose index only exceeds bounds at
  // runtime — can't be proven here; those are caught by the grow-aware
  // runtime bounds-check helpers (Layer 2). We decline ONLY when growth
  // is provable, so in-bounds dynamic stores keep JITting.
  if (isNumeric(baseTy) && baseTy.shape !== undefined) {
    if (numSlots === 1) {
      const k = exactIndexVal(indices[0].ty);
      const numel = shapeNumel(baseTy.shape);
      if (k !== undefined && k > numel) {
        throw new UnsupportedConstruct(
          `indexed assignment to '${displayName}' grows the array ` +
            `(index ${k} exceeds the ${numel}-element static shape); ` +
            `array growth is not supported in the JIT`,
          span
        );
      }
    } else {
      for (let slot = 0; slot < numSlots; slot++) {
        const k = exactIndexVal(indices[slot].ty);
        const dim = slot < baseTy.shape.length ? baseTy.shape[slot] : 1;
        if (k !== undefined && k > dim) {
          throw new UnsupportedConstruct(
            `indexed assignment to '${displayName}' grows axis ${slot + 1} ` +
              `(index ${k} exceeds static dim ${dim}); array growth is not ` +
              `supported in the JIT`,
            span
          );
        }
      }
    }
  }

  const rhs = this.lowerExpr(exprAst);
  if (baseTy.isComplex) {
    // Base is complex: RHS may be either real or complex scalar.
    // The codegen splits a complex RHS via creal/cimag and writes
    // both lanes; a real RHS goes to .real with .imag = 0.
    if (!isNumeric(rhs.ty) || !isScalar(rhs.ty)) {
      throw new TypeError(
        `right-hand side of an indexed assignment must be a numeric scalar ` +
          `(got ${typeToString(rhs.ty)})`,
        exprAst.span
      );
    }
  } else {
    if (!isScalarRealNumeric(rhs.ty)) {
      if (isNumeric(rhs.ty) && rhs.ty.isComplex && isScalar(rhs.ty)) {
        throw new TypeError(
          `cannot store a complex value into a real-typed tensor '${displayName}' ` +
            `(would silently drop the imaginary part). Promote the base to ` +
            `complex first (e.g. via 'x = x + 0i' before the indexed write).`,
          exprAst.span
        );
      }
      throw new TypeError(
        `right-hand side of an indexed assignment must be a numeric scalar ` +
          `(got ${typeToString(rhs.ty)})`,
        exprAst.span
      );
    }
  }

  const stmt: IRStmt = {
    kind: "IndexStore",
    base,
    indices,
    rhs,
    span,
    ...(fieldPath !== undefined ? { fieldPath, leafTy } : {}),
  };
  return stmt;
}

/** The statically-known integer value of an index expression's type,
 *  or `undefined` when it isn't a compile-time-known scalar. Reads the
 *  NumericType `exact` carrier directly (scalar `+`/`-`/`*` fold their
 *  operands' exacts, so `end + 1` arrives here with `exact` set).
 *  Kept local rather than importing `exactDouble` from `builtins/` to
 *  avoid inverting the lowering→builtins dependency direction. */
function exactIndexVal(ty: Type): number | undefined {
  return isNumeric(ty) && typeof ty.exact === "number" ? ty.exact : undefined;
}
