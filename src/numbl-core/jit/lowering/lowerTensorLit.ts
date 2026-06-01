/**
 * Bracket-literal lowering: `[1 2 3]`, `[1; 2; 3]`, `[A B; C D]`, etc.
 *
 * Reached from `lowerExpr` whenever the AST node is a `Tensor`. The
 * lowerer classifies every cell as scalar / tensor / empty, then takes
 * one of three paths:
 *
 *  - **All-scalar fast path** — every cell is a scalar; collapse to a
 *    single `TensorBuild` IR node with a column-major exact-data
 *    carrier when the result fits the cap.
 *  - **Singleton collapse** — `[v]` returns `v` unchanged (whether
 *    scalar or tensor), matching MATLAB.
 *  - **Concat path** — mixed scalar/tensor cells (or all-tensor
 *    cells), with per-row horzcat shape and an outer vertcat. Empty
 *    cells are dropped. Per-row width / per-cell height mismatches
 *    are caught statically when both sides are known; runtime-only
 *    axes trust the user (a future `mtoc2_check_concat_axis` could
 *    validate at runtime).
 */

import type { Expr } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr } from "./ir.js";
import {
  DIM_ONE,
  EXACT_ARRAY_MAX_ELEMENTS,
  isMultiElement,
  isNumeric,
  isScalar,
  signFromExactArray,
  tensorComplex,
  tensorDouble,
  tensorDoubleFromDims,
  typeToString,
} from "./types.js";
import type { DimInfo, NumericType } from "./types.js";
import { exactDouble, exactScalarAsComplex } from "../builtins/defs/_shared.js";
import type { Lowerer } from "./lower.js";

export function lowerTensorLit(
  this: Lowerer,
  e: Extract<Expr, { type: "Tensor" }>
): IRExpr {
  if (e.rows.length === 0) {
    // Empty `[]`. Numbl uses an empty 0×0 tensor — we mirror.
    return {
      kind: "TensorBuild",
      elements: [],
      shape: [0, 0],
      ty: tensorDouble([0, 0]),
      span: e.span,
    };
  }

  // Phase 1 — lower every cell and classify its shape.
  //   - scalar: kind=scalar, value carries the scalar IRExpr.
  //   - tensor: kind=tensor, rows/cols carry the cell's per-axis dim
  //             (number when exact, null when runtime-only).
  //   - empty:  kind=empty, contributes nothing (dropped below).
  type Cell =
    | { kind: "scalar"; expr: IRExpr; ty: NumericType }
    | {
        kind: "tensor";
        expr: IRExpr;
        ty: NumericType;
        rows: number | null;
        cols: number | null;
      }
    | { kind: "empty"; ty: NumericType };
  const grid: Cell[][] = [];
  let anyTensor = false;
  let anyComplex = false;
  for (const row of e.rows) {
    const out: Cell[] = [];
    for (const cell of row) {
      const lowered = this.lowerExpr(cell);
      const ty = lowered.ty;
      if (!isNumeric(ty)) {
        throw new UnsupportedConstruct(
          `bracket literal cell must be a numeric scalar or tensor (got ${typeToString(ty)})`,
          cell.span
        );
      }
      if (ty.elem !== "double" && ty.elem !== "logical") {
        throw new UnsupportedConstruct(
          `bracket literal cell must be a real double or logical (got ${ty.elem})`,
          cell.span
        );
      }
      if (ty.dims.length > 2) {
        throw new UnsupportedConstruct(
          `bracket concatenation requires 2-D cells (got a rank-${ty.dims.length} tensor); use 'cat'/'permute' for higher-rank inputs`,
          cell.span
        );
      }
      if (ty.isComplex) {
        if (isMultiElement(ty)) {
          // Phase 2 lands scalar-complex bracket cells and the
          // straight assembly into a complex tensor; a tensor-typed
          // complex cell would need Phase 3's complex tensor concat
          // machinery (lane-copy paths).
          throw new UnsupportedConstruct(
            `bracket literal with a complex tensor cell is not yet supported`,
            cell.span
          );
        }
        anyComplex = true;
      }
      if (isScalar(ty)) {
        // Both scalar real and scalar complex land here.
        out.push({ kind: "scalar", expr: lowered, ty });
        continue;
      }
      // Tensor cell. Per-axis dim is `number` when exact, `null`
      // when runtime-only. `dims.length === 2` is guaranteed
      // (mtoc2 normalizes to min-2D and rejected rank>2 above).
      const d0 = ty.dims[0];
      const d1 = ty.dims[1];
      const cr: number | null = d0.kind === "exact" ? d0.value : null;
      const cc: number | null = d1.kind === "exact" ? d1.value : null;
      // Statically-zero axis ⇒ empty cell. A runtime-only axis can
      // be 0 at runtime, but we can't drop it from the grid here
      // — codegen handles size-0 tensor cells inline (the copy
      // loop iterates 0 times).
      if (cr === 0 || cc === 0) {
        out.push({ kind: "empty", ty });
        continue;
      }
      anyTensor = true;
      out.push({ kind: "tensor", expr: lowered, ty, rows: cr, cols: cc });
    }
    grid.push(out);
  }

  // All-scalar fast path — preserve the existing TensorBuild shape
  // and codegen.
  if (!anyTensor && grid.every(r => r.every(c => c.kind === "scalar"))) {
    // Uniform row width is required for the scalar grid.
    const rows = grid.length;
    const cols0 = grid[0].length;
    for (const r of grid) {
      if (r.length !== cols0) {
        throw new TypeError(
          `bracket horzcat row-count mismatch: row 1 has ${cols0} cells, ` +
            `another row has ${r.length}`,
          e.span
        );
      }
    }
    const cols = cols0;
    const total = rows * cols;
    const loweredFlat: IRExpr[] = new Array(total);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = grid[r][c];
        if (cell.kind !== "scalar") {
          throw new Error(
            "internal: unexpected non-scalar in scalar fast path"
          );
        }
        loweredFlat[c * rows + r] = cell.expr;
      }
    }
    if (rows === 1 && cols === 1) {
      return loweredFlat[0];
    }
    if (anyComplex) {
      // Complex tensor literal — propagate isComplex on the result
      // type. Exact-fold via the split-buffer `{re, im}` carrier when
      // every element is exact (numeric or imaginary literals).
      let exactData: { re: Float64Array; im: Float64Array } | undefined;
      if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
        const re = new Float64Array(total);
        const im = new Float64Array(total);
        let allExact = true;
        for (let i = 0; i < total; i++) {
          const cx = exactScalarAsComplex(loweredFlat[i].ty);
          if (cx === undefined) {
            allExact = false;
            break;
          }
          re[i] = cx.re;
          im[i] = cx.im;
        }
        if (allExact) exactData = { re, im };
      }
      const ty = tensorComplex([rows, cols], exactData);
      return {
        kind: "TensorBuild",
        elements: loweredFlat,
        shape: [rows, cols],
        ty,
        span: e.span,
      };
    }
    let exactData: Float64Array | undefined;
    if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
      const data = new Float64Array(total);
      let allExact = true;
      for (let i = 0; i < total; i++) {
        const v = exactDouble(loweredFlat[i].ty);
        if (v === undefined) {
          allExact = false;
          break;
        }
        data[i] = v;
      }
      if (allExact) exactData = data;
    }
    const ty = tensorDouble([rows, cols], exactData);
    // Preserve logical element-class: `[true false true]` is a LOGICAL
    // array in MATLAB / the interpreter, so class()/islogical/isa report
    // "logical". When every cell is logical, mark the result logical —
    // the data stays the 0/1 carrier; only the elem tag changes. (A
    // single-cell `[true]` already collapsed to the scalar above.)
    if (loweredFlat.every(el => isNumeric(el.ty) && el.ty.elem === "logical")) {
      ty.elem = "logical";
      ty.sign = "nonneg";
    }
    return {
      kind: "TensorBuild",
      elements: loweredFlat,
      shape: [rows, cols],
      ty,
      span: e.span,
    };
  }

  // Phase 3 reserved for complex tensor concat lane-copies (the
  // multi-element complex cell ↔ scalar mix variant of TensorConcat
  // codegen). The Phase 2 scalar-cell fast path above covers
  // pure-scalar complex literals; phase 3 elemwise arithmetic
  // already runs without exercising this site.
  if (anyComplex) {
    throw new UnsupportedConstruct(
      `bracket literal mixing complex cells with multi-element tensor cells is not yet supported (concat lane-copy is deferred)`,
      e.span
    );
  }

  // Concat path. Compute per-row horzcat shapes, then vertcat.
  //
  // For each row, drop empty cells. The row's height is the unique
  // non-empty cell's `rows` (validated against neighbors when both
  // sides are static; mismatched runtime/static pairs trust the
  // user — `mtoc2_check_concat_axis` could later validate at
  // runtime, but the current emit just uses whichever value is
  // known). A row with no non-empty cells contributes nothing to
  // the vertcat.
  type NonEmptyCell = Exclude<Cell, { kind: "empty" }>;
  const rowsRetained: NonEmptyCell[][] = [];
  const rowHeights: (number | null)[] = [];
  const rowWidths: (number | null)[] = [];
  const cellCols: (number | null)[][] = [];
  for (let i = 0; i < grid.length; i++) {
    const row = grid[i];
    const keptCells: NonEmptyCell[] = [];
    const keptCols: (number | null)[] = [];
    let height: number | null | undefined = undefined; // undefined = no cells seen yet
    let width: number | null = 0;
    for (let j = 0; j < row.length; j++) {
      const cell = row[j];
      if (cell.kind === "empty") continue;
      const h: number | null = cell.kind === "scalar" ? 1 : cell.rows;
      const w: number | null = cell.kind === "scalar" ? 1 : cell.cols;
      if (height === undefined) {
        height = h;
      } else if (height !== null && h !== null && height !== h) {
        throw new TypeError(
          `bracket horzcat row-height mismatch: cell ${j + 1} on row ${i + 1} ` +
            `has ${h} row(s) but a neighbor in the same row has ${height}`,
          e.rows[i][j].span
        );
      } else if (height === null && h !== null) {
        height = h; // promote: prefer the static value
      }
      keptCells.push(cell);
      keptCols.push(w);
      width = width === null || w === null ? null : width + w;
    }
    if (height === undefined) continue; // entire row was empty — drop it
    rowsRetained.push(keptCells);
    rowHeights.push(height);
    rowWidths.push(width);
    cellCols.push(keptCols);
  }

  // If every row was dropped, the result is the empty 0×0 placeholder.
  if (rowsRetained.length === 0) {
    return {
      kind: "TensorBuild",
      elements: [],
      shape: [0, 0],
      ty: tensorDouble([0, 0]),
      span: e.span,
    };
  }

  // All retained rows must have the same width (statically when
  // both sides are known; otherwise trust the user / runtime).
  let staticWidth: number | null = null;
  for (const w of rowWidths) {
    if (w === null) continue;
    if (staticWidth === null) {
      staticWidth = w;
    } else if (staticWidth !== w) {
      throw new TypeError(
        `bracket vertcat column-count mismatch: a row has ${staticWidth} ` +
          `column(s), another has ${w}`,
        e.span
      );
    }
  }
  const totalCols: number | null = staticWidth;
  // Total rows = sum of row heights; null if any height is unknown.
  let totalRows: number | null = 0;
  for (const h of rowHeights) {
    if (h === null) {
      totalRows = null;
      break;
    }
    totalRows += h;
  }

  // Singleton case: one cell total, no concat needed — return the
  // cell's lowered IR unchanged. Matches MATLAB's `[v] === v`
  // (whether v is scalar or tensor).
  if (rowsRetained.length === 1 && rowsRetained[0].length === 1) {
    const only = rowsRetained[0][0];
    return only.expr;
  }

  // ANF the tensor cells so each is a Var. Scalar cells stay
  // inline. The hoist sites flow up via the Lowerer's normal
  // ANF machinery — but at this point we're returning a single
  // expression. The standard ANF rewrite in `anfChildren` will
  // catch our TensorConcat (it's owned-producing) and recurse
  // through `cells` with `anfRequireScalarOrVar`, which will
  // hoist any tensor-typed non-Var cells. So we can just hand off
  // raw cells here — they'll be hoisted by the time codegen sees
  // them.
  const cellsIR: IRExpr[][] = rowsRetained.map(row => row.map(c => c.expr));

  // Try exact-fold. Only attempted when every dim is statically
  // known (otherwise we can't allocate a fixed-size buffer or
  // address into it). Every cell must be exact; total elements
  // must fit the cap.
  let exactData: Float64Array | undefined;
  if (
    totalRows !== null &&
    totalCols !== null &&
    rowHeights.every(h => h !== null) &&
    cellCols.every(cc => cc.every(c => c !== null))
  ) {
    const total = totalRows * totalCols;
    if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
      const data = new Float64Array(total);
      let allExact = true;
      let rowOff = 0;
      for (let i = 0; i < rowsRetained.length && allExact; i++) {
        const row = rowsRetained[i];
        let colOff = 0;
        for (let j = 0; j < row.length && allExact; j++) {
          const cell = row[j];
          const cellRowsKnown =
            cell.kind === "scalar" ? 1 : (cell.rows as number);
          const cellColsKnown =
            cell.kind === "scalar" ? 1 : (cell.cols as number);
          if (cell.kind === "scalar") {
            const v = exactDouble(cell.ty);
            if (v === undefined) {
              allExact = false;
              break;
            }
            const dstIdx = rowOff + colOff * totalRows;
            data[dstIdx] = v;
          } else {
            const src = cell.ty.exact;
            if (!(src instanceof Float64Array)) {
              allExact = false;
              break;
            }
            for (let sc = 0; sc < cellColsKnown; sc++) {
              for (let sr = 0; sr < cellRowsKnown; sr++) {
                const dstIdx = rowOff + sr + (colOff + sc) * totalRows;
                const srcIdx = sr + sc * cellRowsKnown;
                data[dstIdx] = src[srcIdx];
              }
            }
          }
          colOff += cellColsKnown;
        }
        rowOff += rowHeights[i] as number;
      }
      if (allExact) exactData = data;
    }
  }

  // Build the result type. Use `tensorDoubleFromDims` so a
  // runtime-only axis lands as `{ kind: "unknown" }`.
  const resultDims: DimInfo[] = [
    totalRows === null
      ? { kind: "unknown" }
      : totalRows === 1
        ? DIM_ONE
        : { kind: "exact", value: totalRows },
    totalCols === null
      ? { kind: "unknown" }
      : totalCols === 1
        ? DIM_ONE
        : { kind: "exact", value: totalCols },
  ];
  const resultTy = tensorDoubleFromDims(resultDims);
  if (exactData !== undefined && resultTy.shape !== undefined) {
    resultTy.exact = exactData;
    resultTy.sign = signFromExactArray(exactData);
  }
  return {
    kind: "TensorConcat",
    cells: cellsIR,
    rowHeights,
    cellCols,
    shape: [totalRows, totalCols],
    ty: resultTy,
    span: e.span,
  };
}
