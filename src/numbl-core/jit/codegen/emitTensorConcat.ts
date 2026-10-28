/**
 * `TensorConcat` codegen — the bracket-concatenation IR node produced
 * by `lowerTensorLit`'s mixed-cell path (`[a, b; c, d]` where at least
 * one cell is a tensor). Two paths: a fully-static one that bakes per-
 * cell row/col offsets into literals, and a dynamic one that walks
 * runtime `.dims[k]` reads.
 *
 * Both emit a GCC statement-expression that allocs a destination
 * tensor, writes every cell's rectangle, and evaluates to the
 * destination. ANF guarantees every tensor cell is already a Var, so
 * cell reads carry no allocation cost.
 *
 * Mirrors numbl's `catAlongDim` (runtime/tensor-construction.ts) for
 * the output layout — column-major destination, per-cell rectangle.
 */

import type { IRExpr } from "../lowering/ir.js";
import { isScalar } from "../lowering/types.js";
import {
  useRuntimeByName,
  useRuntimeInline,
  type RuntimeState,
} from "./runtime.js";
import { emitExpr } from "./emitExpr.js";

/** Runtime guard mirroring the interpreter's concatenation-consistency
 *  error. Registered on demand by the dynamic path. */
const CONCAT_CHECK_SNIPPET = {
  name: "jit_concat_check",
  headers: ["<stdio.h>", "<stdlib.h>"],
  code: `static void jit_concat_check(long got, long want) {
  if (got != want) {
    fprintf(stderr,
      "Dimensions of arrays being concatenated are not consistent\\n");
    exit(1);
  }
}`,
};

/** Top-level entry: dispatch to the static or dynamic codegen path
 *  based on whether every per-cell dim and the output shape are
 *  compile-time known. */
export function emitTensorConcat(
  e: Extract<IRExpr, { kind: "TensorConcat" }>,
  state: RuntimeState
): string {
  useRuntimeByName(state, "jit_tensor_t");
  useRuntimeByName(state, "jit_tensor_alloc_nd");

  const allStatic =
    e.shape.every(s => s !== null) &&
    e.rowHeights.every(h => h !== null) &&
    e.cellCols.every(row => row.every(c => c !== null));
  if (allStatic) {
    return emitTensorConcatStatic(
      e.cells,
      e.shape as number[],
      e.rowHeights as number[],
      e.cellCols as number[][],
      state
    );
  }
  return emitTensorConcatDynamic(e, state);
}

function emitTensorConcatStatic(
  cells: IRExpr[][],
  shape: number[],
  rowHeights: number[],
  cellCols: number[][],
  state: RuntimeState
): string {
  const [totalRows, totalCols] = shape;
  void totalCols;
  const lines: string[] = [];
  lines.push(
    `jit_tensor_t _jit_t = jit_tensor_alloc_nd(2, (long[]){${totalRows}L, ${shape[1]}L});`
  );

  let rowOff = 0;
  for (let i = 0; i < cells.length; i++) {
    const row = cells[i];
    const cellRows = rowHeights[i];
    let colOff = 0;
    for (let j = 0; j < row.length; j++) {
      const cell = row[j];
      const cellColsHere = cellCols[i][j];
      const cellStr = emitExpr(cell, state);

      if (cellRows === 1 && cellColsHere === 1) {
        const dstIdx = `${rowOff}L + ${colOff}L * ${totalRows}L`;
        lines.push(`_jit_t.real[${dstIdx}] = ${cellStr};`);
      } else {
        lines.push(
          `for (long _jit_sc = 0; _jit_sc < ${cellColsHere}L; _jit_sc++) {`
        );
        lines.push(
          `  for (long _jit_sr = 0; _jit_sr < ${cellRows}L; _jit_sr++) {`
        );
        const dstIdx = `(${rowOff}L + _jit_sr) + (${colOff}L + _jit_sc) * ${totalRows}L`;
        const srcIdx = `_jit_sr + _jit_sc * ${cellRows}L`;
        lines.push(`    _jit_t.real[${dstIdx}] = ${cellStr}.real[${srcIdx}];`);
        lines.push(`  }`);
        lines.push(`}`);
      }
      colOff += cellColsHere;
    }
    rowOff += cellRows;
  }
  lines.push(`_jit_t;`);
  return `({ ${lines.join(" ")} })`;
}

function emitTensorConcatDynamic(
  e: Extract<IRExpr, { kind: "TensorConcat" }>,
  state: RuntimeState
): string {
  const lines: string[] = [];
  // Resolve every cell's emitted C expression up front and cache it.
  // After ANF every tensor cell is a `Var` (bare identifier) and
  // scalar cells are simple expressions — neither has side effects
  // we'd be doubling up on by referencing twice.
  const cellStrs: string[][] = e.cells.map(row =>
    row.map(c => emitExpr(c, state))
  );
  // Per-cell row / col extent expressions.
  const cellRowsExpr = (i: number, j: number): string => {
    const c = e.cells[i][j];
    if (c.ty.kind !== "Numeric") return "1L";
    if (c.ty.dims.length === 0 || isScalar(c.ty)) return "1L";
    const r = e.rowHeights[i];
    if (r !== null) return `${r}L`;
    return `${cellStrs[i][j]}.dims[0]`;
  };
  const cellColsExpr = (i: number, j: number): string => {
    const c = e.cells[i][j];
    if (c.ty.kind !== "Numeric") return "1L";
    if (c.ty.dims.length === 0 || isScalar(c.ty)) return "1L";
    const k = e.cellCols[i][j];
    if (k !== null) return `${k}L`;
    return `${cellStrs[i][j]}.dims[1]`;
  };
  // Emit row-height locals so we don't recompute the witness cell's
  // `.dims[0]` more than once per row.
  const rhLocals: string[] = [];
  for (let i = 0; i < e.cells.length; i++) {
    const name = `_jit_rh_${i}`;
    rhLocals.push(name);
    // Pick the first cell's height as the witness — every cell in
    // the row is required to share it (validated statically at
    // lowering when both sides are known).
    lines.push(`long ${name} = ${cellRowsExpr(i, 0)};`);
  }
  // Total height = sum of row heights.
  const trExpr = e.shape[0] !== null ? `${e.shape[0]}L` : rhLocals.join(" + ");
  lines.push(`long _jit_tr = ${trExpr};`);
  // Total width = first row's width = sum of its cells' cols.
  const widthExpr = (() => {
    if (e.shape[1] !== null) return `${e.shape[1]}L`;
    if (e.cells.length === 0) return "0L";
    return e.cells[0].map((_, j) => cellColsExpr(0, j)).join(" + ");
  })();
  lines.push(`long _jit_tc = ${widthExpr};`);
  lines.push(
    `jit_tensor_t _jit_t = jit_tensor_alloc_nd(2, (long[]){_jit_tr, _jit_tc});`
  );

  // Track destination row offset as a running long. Cells inside a
  // row use their column-offset accumulator too.
  lines.push(`long _jit_row_off = 0;`);
  for (let i = 0; i < e.cells.length; i++) {
    const row = e.cells[i];
    lines.push(`long _jit_col_off_${i} = 0;`);
    for (let j = 0; j < row.length; j++) {
      const cell = row[j];
      const cellStr = cellStrs[i][j];
      const rowsHere = cellRowsExpr(i, j);
      const colsHere = cellColsExpr(i, j);
      // Within a row every cell must share the row height (cell 0 sets
      // it); otherwise the copy walks the wrong region. Matches the
      // interpreter's consistency error.
      if (j > 0) {
        useRuntimeInline(state, CONCAT_CHECK_SNIPPET);
        lines.push(`jit_concat_check(${rowsHere}, ${rhLocals[i]});`);
      }
      const isScalarCell =
        cell.ty.kind === "Numeric" &&
        (cell.ty.dims.length === 0 || isScalar(cell.ty));
      if (isScalarCell) {
        const dstIdx = `_jit_row_off + _jit_col_off_${i} * _jit_tr`;
        lines.push(`_jit_t.real[${dstIdx}] = ${cellStr};`);
        lines.push(`_jit_col_off_${i} += 1;`);
      } else {
        const sc = `_jit_sc_${i}_${j}`;
        const sr = `_jit_sr_${i}_${j}`;
        const cw = `_jit_cw_${i}_${j}`;
        const ch = `_jit_ch_${i}_${j}`;
        lines.push(`long ${cw} = ${colsHere};`);
        lines.push(`long ${ch} = ${rowsHere};`);
        lines.push(`for (long ${sc} = 0; ${sc} < ${cw}; ${sc}++) {`);
        lines.push(`  for (long ${sr} = 0; ${sr} < ${ch}; ${sr}++) {`);
        const dstIdx = `(_jit_row_off + ${sr}) + (_jit_col_off_${i} + ${sc}) * _jit_tr`;
        const srcIdx = `${sr} + ${sc} * ${ch}`;
        lines.push(`    _jit_t.real[${dstIdx}] = ${cellStr}.real[${srcIdx}];`);
        lines.push(`  }`);
        lines.push(`}`);
        lines.push(`_jit_col_off_${i} += ${cw};`);
      }
    }
    // Each row's total width must match the result width (set by row 0).
    useRuntimeInline(state, CONCAT_CHECK_SNIPPET);
    lines.push(`jit_concat_check(_jit_col_off_${i}, _jit_tc);`);
    lines.push(`_jit_row_off += ${rhLocals[i]};`);
  }
  lines.push(`_jit_t;`);
  return `({ ${lines.join(" ")} })`;
}
