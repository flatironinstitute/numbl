/* Cell-array runtime helpers for the JS-JIT.
 *
 * Cells live as plain JS objects shaped like numbl's `RuntimeCell`:
 *
 *   { jitTag: "cell", shape: number[], data: RuntimeValue[] }
 *
 * `data` is column-major flat (matching tensors). `shape` is the cell's
 * dim list.
 *
 * The C-JIT uses per-shape generated typedefs and does NOT register
 * through this file.
 */

// Numbl's scalar formatter (formatNumber), shared with disp so cell-slot
// numbers print like `disp(x)` instead of raw String(v) full precision.
// The build script strips this import when inlining all snippets.
import { jit_format_double } from "../io/format_double.js";

/** Construct a cell from a flat slot list (column-major) and a shape.
 *  Caller has already produced fresh slot values (deep_clone'd where
 *  the slot was an owned alias), so we simply build the wrapper. */
export function jit_cell_make(slots, shape) {
  return { jitTag: "cell", shape: shape.slice(), data: slots };
}

/** Empty-double sentinel slot — what every fresh `cell(n, m)` slot
 *  starts as. Matches numbl's `type-constructors.ts:442-469`. */
function jit__empty_double_tensor() {
  return {
    jitTag: "tensor",
    shape: [0, 0],
    data: new Float64Array(0),
  };
}

/** `cell(n)` / `cell(n, m)` / `cell(n, m, k, ...)`. The 1-arg square
 *  form has already been expanded to two dim args at lowering time so
 *  this helper is dim-symmetric.
 *
 *  Each axis is coerced to a non-negative integer (matching numbl —
 *  `Math.floor` then clamp at 0). The slot count is the dim product;
 *  each slot is the empty-double sentinel. */
export function jit_cell_empty(dims) {
  const shape = dims.map(d => {
    const n = Math.floor(Number(d));
    return n > 0 ? n : 0;
  });
  let total = 1;
  for (const s of shape) total *= s;
  const data = new Array(total);
  for (let i = 0; i < total; i++) data[i] = jit__empty_double_tensor();
  return { jitTag: "cell", shape, data };
}

/** Format a cell value to match numbl's `formatCell` byte-for-byte:
 *  `{e1, e2, ...}` with chars in `'...'`, strings in `"..."`, other
 *  values via `formatValue`. Empty cell renders as `{}`. */
export function jit_format_cell(c) {
  if (c.data.length === 0) return "{}";
  const parts = c.data.map(v => jit__format_cell_slot(v));
  return `{${parts.join(", ")}}`;
}

function jit__format_cell_slot(v) {
  if (v === null || typeof v !== "object") {
    if (typeof v === "string") return `"${v}"`;
    return jit__format_scalar(v);
  }
  if (v.jitTag === "char") return `'${v.value}'`;
  if (v.jitTag === "cell") return jit_format_cell(v);
  if (v.jitTag === "tensor") return jit__format_tensor(v);
  // Fallback: best-effort string.
  return String(v);
}

function jit__format_scalar(v) {
  if (typeof v === "boolean") return v ? "1" : "0";
  if (typeof v === "number") {
    // Format like disp(x) / numbl's formatNumber (pi -> "3.1416"), not
    // String(v)'s full round-trip precision (-> "3.141592653589793").
    return jit_format_double(v);
  }
  return String(v);
}

function jit__format_tensor(t) {
  // Inline tensor format (no trailing newline) — matches numbl's
  // `formatTensor` in a cell context. 2-D row/col-major; for cells we
  // expect 1×N or scalar shapes commonly. Multi-row tensors render
  // with internal `\n` between rows.
  if (t.data.length === 0) return "[]";
  const rows = t.shape[0] ?? 1;
  const cols = t.shape[1] ?? 1;
  const cell = new Array(rows * cols);
  let colWidths = new Array(cols).fill(0);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const idx = r + c * rows;
      const s = jit__format_scalar(t.data[idx]);
      cell[idx] = s;
      if (s.length > colWidths[c]) colWidths[c] = s.length;
    }
  }
  const rowsOut = [];
  for (let r = 0; r < rows; r++) {
    const parts = ["   "];
    for (let c = 0; c < cols; c++) {
      const idx = r + c * rows;
      const s = cell[idx];
      parts.push(" ".repeat(colWidths[c] - s.length));
      parts.push(s);
      if (c < cols - 1) parts.push("   ");
    }
    rowsOut.push(parts.join(""));
  }
  return rowsOut.join("\n");
}
