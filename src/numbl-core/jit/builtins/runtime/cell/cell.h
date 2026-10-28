/* mtoc2 cell-array runtime — placeholder.
 *
 * Cell helpers in the C-JIT are emitted per-shape via
 * `emitCellTypedef.ts` (sibling of `emitNamedTypedef.ts`), so there is
 * no shared mtoc2_cell_* C runtime to register. This file exists only
 * as a topic marker so the `cell/` folder participates in the runtime
 * tree the same way `tensor/`, `text/`, etc. do.
 *
 * The JS sibling (`cell.js`) carries the actual JS-JIT helpers.
 */
