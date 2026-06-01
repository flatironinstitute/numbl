/**
 * Cell-literal lowering: `{a, b, c}`, `{1; 2; 3}`, `{1, 'hi'; [1 2], "x"}`.
 *
 * Reached from `lowerExpr` whenever the AST node is a `Cell`. The
 * literal always produces a tuple-mode `CellType` with per-slot types
 * (matching the lowered slot expressions). Empty `{}` produces a
 * 0×0 tuple cell with `elements: []`.
 *
 * Slot expressions accept any value-typed expression — scalars,
 * tensors, structs, classes, handles, strings, chars. The lowerer
 * relies on the ANF pass to hoist owned-producing slot values to
 * fresh temps before the `CellLit` becomes an Assign RHS.
 */

import type { Expr } from "../parser/index.js";
import { UnsupportedConstruct } from "./errors.js";
import type { CellLit, IRExpr } from "./ir.js";
import {
  EXACT_ARRAY_MAX_ELEMENTS,
  cellTuple,
  isVoid,
  shapeNumel,
  typeToString,
} from "./types.js";
import type { Type } from "./types.js";
import type { Lowerer } from "./lower.js";

export function lowerCellLit(
  this: Lowerer,
  e: Extract<Expr, { type: "Cell" }>
): IRExpr {
  // Empty `{}` literal → 0×0 tuple cell with no slots. Matches numbl
  // (`interpreterExec.ts:1070`: returns `RTV.cell([], [0, 0])`).
  if (e.rows.length === 0) {
    const shape = [0, 0];
    const lit: CellLit = {
      kind: "CellLit",
      elements: [],
      shape,
      ty: cellTuple(shape, []),
      span: e.span,
    };
    return lit;
  }

  // Validate row-width uniformity (the parser already enforces this
  // for tensor literals; cell literals get the same shape).
  const rowCount = e.rows.length;
  const colCount = e.rows[0].length;
  for (let r = 0; r < rowCount; r++) {
    if (e.rows[r].length !== colCount) {
      throw new UnsupportedConstruct(
        `cell literal: row ${r + 1} has ${e.rows[r].length} cells, ` +
          `expected ${colCount} (every row must have the same width)`,
        e.span
      );
    }
  }

  // Lower every slot and collect into column-major order.
  // AST `rows[r][c]` is row r, col c (source order). Cell type
  // storage matches numbl's `RuntimeCell.data`: column-major flat.
  const slotExprs: IRExpr[][] = [];
  for (let r = 0; r < rowCount; r++) {
    const rowOut: IRExpr[] = [];
    for (let c = 0; c < colCount; c++) {
      const lowered = this.lowerExpr(e.rows[r][c]);
      if (isVoid(lowered.ty)) {
        throw new UnsupportedConstruct(
          `cell literal slot: cannot use the result of a zero-output ` +
            `function as a value (got ${typeToString(lowered.ty)})`,
          e.rows[r][c].span
        );
      }
      rowOut.push(lowered);
    }
    slotExprs.push(rowOut);
  }

  const elementsColMajor: IRExpr[] = [];
  const elementTypes: Type[] = [];
  for (let c = 0; c < colCount; c++) {
    for (let r = 0; r < rowCount; r++) {
      elementsColMajor.push(slotExprs[r][c]);
      elementTypes.push(slotExprs[r][c].ty);
    }
  }

  const shape = [rowCount, colCount];

  // Cell literals always start in tuple mode. The slot count is the
  // literal's element count, which is bounded by the source. We
  // reject very large literals that would blow up the per-shape
  // typedef table — the exact cap matches the tensor-exact cap so
  // a single rule applies across the lattice.
  const total = shapeNumel(shape);
  if (total > EXACT_ARRAY_MAX_ELEMENTS) {
    throw new UnsupportedConstruct(
      `cell literal with ${total} slots exceeds the ` +
        `EXACT_ARRAY_MAX_ELEMENTS cap (${EXACT_ARRAY_MAX_ELEMENTS})`,
      e.span
    );
  }

  const lit: CellLit = {
    kind: "CellLit",
    elements: elementsColMajor,
    shape,
    ty: cellTuple(shape, elementTypes),
    span: e.span,
  };
  return lit;
}
