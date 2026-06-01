/**
 * Brace-indexed cell read lowering: `c{i}`, `c{i, j}`.
 *
 * Routed from `lowerExpr` when the AST node is `IndexCell` AND the
 * base resolves to an in-scope cell variable. The base must be a
 * bare `Ident` for now — member-rooted `obj.field{i}` is rejected
 * pending the broader member-rooted-index work.
 *
 * Result type depends on the cell's mode:
 *   - Tuple mode + every index a `NumLit` matching an in-range
 *     slot → the specific slot's `Type`.
 *   - Tuple mode + any non-exact index → the unified slot type
 *     (rejected with `UnsupportedConstruct` if slot types diverge).
 *   - Uniform mode → the cell's `elem` type.
 *
 *  Per-axis (`c{i, j}`) reads are accepted only for tuple cells
 *  whose shape is exact 2-D; uniform cells route through the same
 *  column-major linear offset at runtime.
 */

import type { Expr, LValue, Span } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr, IRStmt } from "./ir.js";
import {
  isCell,
  isNumeric,
  isScalar,
  shapeNumel,
  storageEquivalent,
  typeToString,
  unify,
  type CellType,
  type Type,
} from "./types.js";
import type { Lowerer } from "./lower.js";

export function lowerCellIndexLoad(
  this: Lowerer,
  e: Extract<Expr, { type: "IndexCell" }>
): IRExpr {
  if (e.base.type !== "Ident") {
    throw new UnsupportedConstruct(
      `cell brace indexing supports a bare cell variable as the base; ` +
        `member-rooted forms (e.g. 'obj.f{i}') are not yet supported`,
      e.span
    );
  }
  const entry = this.env.get(e.base.name);
  if (entry === undefined) {
    throw new TypeError(`undefined variable '${e.base.name}'`, e.span);
  }
  if (!isCell(entry.ty)) {
    throw new TypeError(
      `'${e.base.name}' is not a cell (got ${typeToString(entry.ty)}); ` +
        `brace indexing 'c{...}' requires a cell-typed variable`,
      e.span
    );
  }
  const baseTy = entry.ty;
  const baseCName = entry.cName;

  if (e.indices.length === 0) {
    throw new UnsupportedConstruct(
      `cell brace indexing requires at least one index`,
      e.span
    );
  }
  if (e.indices.length > 2) {
    throw new UnsupportedConstruct(
      `cell brace indexing supports 1 or 2 indices (got ${e.indices.length})`,
      e.span
    );
  }

  // Lower the indices. Each must be a real scalar (no slice / colon /
  // logical-mask forms in phase B — those route to brace slice in a
  // later phase, or are rejected entirely).
  const indices = e.indices.map(ix => {
    if (ix.type === "Colon" || ix.type === "Range") {
      throw new UnsupportedConstruct(
        `cell brace indexing with ':' / range slices not yet supported`,
        ix.span
      );
    }
    const lowered = this.lowerExpr(ix);
    if (
      !isNumeric(lowered.ty) ||
      !isScalar(lowered.ty) ||
      lowered.ty.isComplex
    ) {
      throw new TypeError(
        `cell brace index must be a real scalar (got ${typeToString(lowered.ty)})`,
        ix.span
      );
    }
    return lowered;
  });

  const resultTy = resolveCellSlotType(baseTy, indices, e.span);
  return {
    kind: "CellIndexLoad",
    base: {
      kind: "Var",
      name: e.base.name,
      cName: baseCName,
      ty: baseTy,
      span: e.base.span,
    },
    indices,
    ty: resultTy,
    span: e.span,
  };
}

/** Determine the static type of a cell brace read result. For
 *  tuple cells the static-index fast path resolves to the specific
 *  slot's type; otherwise the slot types must unify. For uniform
 *  cells the `elem` type is returned directly. */
function resolveCellSlotType(
  baseTy: CellType,
  indices: IRExpr[],
  span: import("../parser/index.js").Span
): Type {
  if (baseTy.mode === "uniform") {
    return baseTy.elem!;
  }
  // Tuple mode — try the static-index fast path.
  const elements = baseTy.elements!;
  const shape = baseTy.shape;
  if (shape === undefined) {
    throw new UnsupportedConstruct(
      `internal: tuple cell missing exact shape`,
      span
    );
  }
  const staticOff = tryStaticCellSlotIndex(indices, baseTy);
  if (staticOff !== null) {
    if (staticOff < 0 || staticOff >= elements.length) {
      throw new TypeError(
        `cell brace index out of range: slot ${staticOff + 1} ` +
          `is outside the cell's ${shapeNumel(shape)} slots`,
        span
      );
    }
    return elements[staticOff];
  }
  // Dynamic index: require all slot types to unify.
  if (elements.length === 0) {
    throw new TypeError(`cell brace read into an empty cell {}`, span);
  }
  let unified: Type = elements[0];
  for (let i = 1; i < elements.length; i++) {
    unified = unify(unified, elements[i]);
    if (unified.kind === "Unknown") {
      throw new UnsupportedConstruct(
        `cell brace read with a non-static index requires all slot types ` +
          `to unify; got divergent slot types (slot 1: ${typeToString(elements[0])}, ` +
          `slot ${i + 1}: ${typeToString(elements[i])})`,
        span
      );
    }
  }
  return unified;
}

/** Compute the static slot offset when every index is a NumLit.
 *  Returns null for non-exact indices, returns the column-major
 *  flat index for the in-range case. */
function tryStaticCellSlotIndex(
  indices: IRExpr[],
  baseTy: CellType
): number | null {
  const shape = baseTy.shape;
  if (shape === undefined) return null;
  if (indices.length === 1) {
    if (indices[0].kind !== "NumLit") return null;
    const v = indices[0].value;
    if (!Number.isInteger(v) || v < 1) return null;
    return v - 1;
  }
  if (indices.length === 2) {
    if (indices[0].kind !== "NumLit" || indices[1].kind !== "NumLit") {
      return null;
    }
    const i = indices[0].value - 1;
    const j = indices[1].value - 1;
    if (i < 0 || j < 0) return null;
    const rows = shape[0];
    return j * rows + i;
  }
  return null;
}

/** Brace-write lowering: `c{i} = rhs`, `c{i, j} = rhs`. The base
 *  must be a bare cell-typed variable. After the store, the env's
 *  recorded type for `c` is refreshed to reflect the new slot type:
 *
 *  - Tuple mode + static index → update the slot's `Type` to the
 *    rhs's type.
 *  - Tuple mode + non-static index → require every slot type to
 *    unify with the rhs type; demote the env entry to a uniform
 *    cell with the unified elem.
 *  - Uniform mode → rhs must unify with the cell's `elem`.
 *
 *  If neither path applies, raise `UnsupportedConstruct` (no LUB /
 *  heterogeneous fallback per the design — see docs/cells_plan.md). */
export function lowerCellIndexStore(
  this: Lowerer,
  lv: Extract<LValue, { type: "IndexCell" }>,
  exprAst: Expr,
  span: Span
): IRStmt | IRStmt[] {
  if (lv.base.type !== "Ident") {
    throw new UnsupportedConstruct(
      `cell brace assignment supports a bare cell variable as the base; ` +
        `member-rooted forms (e.g. 'obj.f{i} = rhs') are not yet supported`,
      span
    );
  }
  const entry = this.env.get(lv.base.name);
  if (entry === undefined) {
    throw new TypeError(
      `undefined variable '${lv.base.name}' (must be initialised before ` +
        `brace-indexed assignment)`,
      span
    );
  }
  if (!isCell(entry.ty)) {
    throw new TypeError(
      `'${lv.base.name}' is not a cell (got ${typeToString(entry.ty)}); ` +
        `brace assignment 'c{...} = rhs' requires a cell-typed variable`,
      span
    );
  }
  const baseTy: CellType = entry.ty;
  const baseCName = entry.cName;

  if (lv.indices.length === 0) {
    throw new UnsupportedConstruct(
      `cell brace assignment requires at least one index`,
      span
    );
  }
  if (lv.indices.length > 2) {
    throw new UnsupportedConstruct(
      `cell brace assignment supports 1 or 2 indices (got ${lv.indices.length})`,
      span
    );
  }

  const hoists: IRStmt[] = [];
  const indices = lv.indices.map(ix => {
    if (ix.type === "Colon" || ix.type === "Range") {
      throw new UnsupportedConstruct(
        `cell brace assignment with ':' / range slices not yet supported`,
        ix.span
      );
    }
    const lowered = this.lowerExpr(ix);
    if (
      !isNumeric(lowered.ty) ||
      !isScalar(lowered.ty) ||
      lowered.ty.isComplex
    ) {
      throw new TypeError(
        `cell brace index must be a real scalar (got ${typeToString(lowered.ty)})`,
        ix.span
      );
    }
    return this.anfRequireScalarOrVar(lowered, hoists);
  });

  // Lower the rhs. The CellIndexStore is a direct consume site for
  // owned values — the cell takes ownership of the rhs, mirroring
  // how MemberStore handles its rhs. ANF the rhs through
  // `anfRequireScalarOrVar` for non-owned rhs (the helper is a no-op
  // for scalar / Var); for owned rhs the store consumes whatever
  // fresh producer or Var lands in.
  const rhsRaw = this.lowerExpr(exprAst);
  if (rhsRaw.ty.kind === "Void") {
    throw new TypeError(
      `cannot store the result of a zero-output function into a cell slot`,
      exprAst.span
    );
  }
  const rhs = rhsRaw;

  const staticOff = tryStaticCellSlotIndex(indices, baseTy);

  // The cell's slot storage (its per-shape C typedef) is fixed at
  // construction. Writes must be storage-equivalent with the
  // targeted slot's existing type — mtoc2 doesn't reshape a cell
  // typedef mid-lifetime. Lattice-precision narrowing / widening
  // still happens through standard env-refresh paths if a future
  // pass wants to track it, but the C representation stays put.
  if (baseTy.mode === "tuple") {
    const elements = baseTy.elements!;
    if (staticOff !== null) {
      if (staticOff < 0 || staticOff >= elements.length) {
        throw new TypeError(
          `cell brace assignment slot ${staticOff + 1} is outside the cell's ` +
            `${shapeNumel(baseTy.shape!)} slots`,
          span
        );
      }
      const slotTy = elements[staticOff];
      if (!storageEquivalent(slotTy, rhs.ty)) {
        throw new UnsupportedConstruct(
          `cell brace assignment: rhs type ${typeToString(rhs.ty)} is not ` +
            `storage-equivalent to slot ${staticOff + 1}'s type ${typeToString(slotTy)} ` +
            `(cells don't reshape mid-lifetime; introduce a new cell literal ` +
            `or construct the cell with the target slot type)`,
          span
        );
      }
    } else {
      // Dynamic index: every slot must be storage-equivalent to rhs.
      for (let i = 0; i < elements.length; i++) {
        if (!storageEquivalent(elements[i], rhs.ty)) {
          throw new UnsupportedConstruct(
            `cell brace assignment with a non-static index requires every ` +
              `slot's storage to match the rhs; slot ${i + 1} has type ` +
              `${typeToString(elements[i])}, rhs has ${typeToString(rhs.ty)}`,
            span
          );
        }
      }
    }
  } else {
    // Uniform mode: rhs storage must match elem storage.
    if (!storageEquivalent(baseTy.elem!, rhs.ty)) {
      throw new UnsupportedConstruct(
        `cell brace assignment: rhs type ${typeToString(rhs.ty)} is not ` +
          `storage-equivalent to the cell's elem type ${typeToString(baseTy.elem!)} ` +
          `(cells don't reshape mid-lifetime)`,
        span
      );
    }
  }

  const store: IRStmt = {
    kind: "CellIndexStore",
    base: {
      kind: "Var",
      name: lv.base.name,
      cName: baseCName,
      ty: entry.ty,
      span: lv.base.span,
    },
    indices,
    rhs,
    span,
  };
  if (hoists.length === 0) return store;
  return [...hoists, store];
}
