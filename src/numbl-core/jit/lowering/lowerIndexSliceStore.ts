/**
 * Range / colon / scalar-mix indexed-write lowering: `v(:) = w`,
 * `v(a:b) = w`, `M(:, j) = w`, `T(:, :, i) = w`, … .
 *
 * Companion to `lowerIndexSlice` for slice writes. RHS shape:
 *   - scalar real numeric → broadcast into every slot.
 *   - named multi-element tensor (`Var`) → per-slot copy.
 *
 * Other RHS forms (a fresh `TensorBuild`, an `IndexSlice`, a tensor
 * Binary, etc.) are rejected with a clear message — the user must
 * assign the expression to a name first so the temporary lifetime is
 * explicit.
 */

import type { Expr, LValue, Span } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRStmt, IndexSliceArg } from "./ir.js";
import {
  isMultiElement,
  isNumeric,
  isScalar,
  typeToString,
  type Type,
} from "./types.js";
import type { Lowerer } from "./lower.js";
import { lowerSliceArg } from "./lowerIndexSlice.js";
import { resolveIndexLvalueBase } from "./indexResolve.js";

/** Statically-known element count of a numeric type, or null if any
 *  dimension is non-exact (runtime-only). */
function staticNumel(t: Type): number | null {
  if (!isNumeric(t)) return null;
  let n = 1;
  for (const d of t.dims) {
    if (d.kind !== "exact") return null;
    n *= d.value;
  }
  return n;
}

export function lowerIndexSliceStore(
  this: Lowerer,
  lvalue: Extract<LValue, { type: "Index" }>,
  exprAst: Expr,
  span: Span
): IRStmt | IRStmt[] {
  // Resolve either a bare-Ident base or a member-rooted base via
  // the shared dispatcher; the member case lands `baseCName` at the
  // slot path (`<root>.<field>...`) and stamps `fieldPath` / `leafTy`
  // onto the emitted IR node.
  const r = resolveIndexLvalueBase.call(this, lvalue, span, "sliceWrite");
  const { fieldPath, leafTy, displayName } = r;

  const isSingleSlot = lvalue.indices.length === 1;
  const slotHoists: IRStmt[] = [];
  const slots: IndexSliceArg[] = [];
  for (let i = 0; i < lvalue.indices.length; i++) {
    const axis: number | "linear" = isSingleSlot ? "linear" : i;
    const slot = lowerSliceArg.call(
      this,
      r.baseCName,
      r.baseTy,
      axis,
      lvalue.indices[i]
    );
    // The LogicalMask and IndexVec slot exprs are consumed at emit time as a
    // Var (codegen reads `.real[i]` / `.dims[k]` off them). ANF here so a
    // non-Var producer (a Unary `~` mask, or an inline index-vector literal
    // like `A(:,[2 4]) = ...`) lands in a named temp before the
    // IndexSliceStore is emitted. Without this, emitSliceSlotSetup throws
    // "IndexVec/LogicalMask slot expr must be a Var after ANF".
    if (slot.kind === "LogicalMask" || slot.kind === "IndexVec") {
      slots.push({
        ...slot,
        expr: this.anfRequireScalarOrVar(slot.expr, slotHoists),
      });
    } else {
      slots.push(slot);
    }
  }
  // Logical-mask store that could GROW the base — a truthy bit past the
  // end resizes the array (MATLAB: x(logical([0 0 0 1]))=9 grows it). The
  // JIT compiles against the static pre-write shape and would silently
  // drop the out-of-range write (JS) or abort (C). When both lengths are
  // statically known and the mask is the longer one, a truthy bit *could*
  // land past the end, so decline to the interpreter (which grows). The
  // common `a(a>2) = v` (mask length == base length) is unaffected.
  if (isSingleSlot && slots[0].kind === "LogicalMask") {
    const baseN = staticNumel(r.baseTy);
    const maskN = staticNumel(slots[0].expr.ty);
    if (baseN !== null && maskN !== null && maskN > baseN) {
      throw new UnsupportedConstruct(
        `logical-mask store with a statically-longer mask may grow the ` +
          `base; deferring to the interpreter`,
        slots[0].span
      );
    }
  }
  // Per-axis logical-mask writes (e.g. `M(:, mask) = rhs`) aren't yet
  // supported; only single-slot linear `a(mask) = rhs` is handled.
  if (!isSingleSlot) {
    for (const slot of slots) {
      if (slot.kind === "LogicalMask") {
        throw new UnsupportedConstruct(
          `per-axis logical-mask writes are not yet supported; only ` +
            `linear-form 'a(mask) = rhs' is handled`,
          slot.span
        );
      }
    }
  }
  // A Scalar slot that writes past the base's current dim would auto-extend
  // the tensor in MATLAB, which the JIT cannot model: it emits against the
  // pre-write static shape and cannot reallocate the buffer mid-spec. That is
  // handled at runtime rather than statically — both backends route a store's
  // Scalar slot through the grow-aware bounds check (`mtoc2_idx_axis_grow`),
  // which bails to the interpreter (full MATLAB grow semantics) on an index
  // past the extent and still aborts on a genuine sub-1 index.
  //
  // Requiring a static proof here instead, as this pass used to, meant that
  // `A(k, :) = row` inside `for k = 1:n` was never compiled: `k` is widened
  // on loop entry, so its value is never statically known, and the whole loop
  // — with everything nested inside it — fell back to the interpreter.

  const rawRhs = this.lowerExpr(exprAst);
  if (!isNumeric(rawRhs.ty)) {
    throw new TypeError(
      `right-hand side of an indexed assignment must be numeric ` +
        `(got ${typeToString(rawRhs.ty)})`,
      exprAst.span
    );
  }
  if (rawRhs.ty.elem !== "double") {
    throw new UnsupportedConstruct(
      `right-hand side of a range/colon indexed write must be a double ` +
        `(got ${typeToString(rawRhs.ty)})`,
      exprAst.span
    );
  }
  if (!r.baseTy.isComplex && rawRhs.ty.isComplex) {
    throw new TypeError(
      `cannot store a complex RHS into a real-typed tensor '${displayName}' ` +
        `(would silently drop the imaginary part). Promote the base to ` +
        `complex first (e.g. via 'x = x + 0i' before the indexed write).`,
      exprAst.span
    );
  }

  // Codegen accepts only a scalar (broadcast) or a Var (per-slot copy)
  // as RHS. Scalars pass through; multi-element non-Var RHSs (a
  // TensorBuild, an IndexSlice, a MakeRange, a tensor-returning Call,
  // a tensor Binary) hoist to a fresh `_mtoc2_t<N>` temp Assign so the
  // temporary's lifetime is named and the codegen pipeline stays
  // uniform. Same ANF rule used by every other owned consume site.
  const hoists: IRStmt[] = [];
  let rhs: typeof rawRhs;
  const rhsIsScalar = isScalar(rawRhs.ty);
  if (rhsIsScalar) {
    rhs = rawRhs;
  } else if (isMultiElement(rawRhs.ty)) {
    rhs = this.anfRequireScalarOrVar(rawRhs, hoists);
  } else {
    throw new TypeError(
      `right-hand side of a range/colon indexed write must be a scalar ` +
        `or a tensor (got ${typeToString(rawRhs.ty)})`,
      exprAst.span
    );
  }

  const store: IRStmt = {
    kind: "IndexSliceStore",
    base: r.base,
    index: slots,
    rhs,
    span,
    ...(fieldPath !== undefined ? { fieldPath, leafTy } : {}),
  };
  const allHoists = [...slotHoists, ...hoists];
  if (allHoists.length === 0) return store;
  return [...allHoists, store];
}
