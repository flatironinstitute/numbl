// JS sibling of `disp_tensor.h`. Real-only tensor display, mirroring
// numbl's `format2DSlice` for the 2-D path and numbl's page-by-page
// rendering for ndim > 2. Each cell is formatted via
// `mtoc2_format_double`; columns are padded to their widest element;
// rows are separated by `\n`, columns by 3 spaces, indented by 3
// spaces.

import { mtoc2_format_double } from "./format_double.js";

// Mirrors numbl's `runtime/display.ts` format2DSlice: matrices
// wider/taller than 20 are truncated to the first/last 10 with a
// "Columns 1 through N" header and a "..." elision row/column. Without
// this the JIT printed every element on one line, diverging from the
// interpreter.
const MTOC2_MAX_DISPLAY_ROWS = 20;
const MTOC2_MAX_DISPLAY_COLS = 20;

function disp_real_slice(data, offset, rows, cols) {
  const truncRows = rows > MTOC2_MAX_DISPLAY_ROWS;
  const truncCols = cols > MTOC2_MAX_DISPLAY_COLS;
  const rHi = Math.ceil(MTOC2_MAX_DISPLAY_ROWS / 2);
  const rLo = Math.floor(MTOC2_MAX_DISPLAY_ROWS / 2);
  const cHi = Math.ceil(MTOC2_MAX_DISPLAY_COLS / 2);
  const cLo = Math.floor(MTOC2_MAX_DISPLAY_COLS / 2);
  const showRows = [];
  if (truncRows) {
    for (let i = 0; i < rHi; i++) showRows.push(i);
    for (let i = 0; i < rLo; i++) showRows.push(rows - rLo + i);
  } else {
    for (let i = 0; i < rows; i++) showRows.push(i);
  }
  const showCols = [];
  if (truncCols) {
    for (let i = 0; i < cHi; i++) showCols.push(i);
    for (let i = 0; i < cLo; i++) showCols.push(cols - cLo + i);
  } else {
    for (let i = 0; i < cols; i++) showCols.push(i);
  }

  if (truncRows || truncCols) {
    $write("  Columns 1 through " + cols + "\n");
    $write("\n");
  }

  const formatted = [];
  const colWidths = new Array(showCols.length + (truncCols ? 1 : 0)).fill(0);
  for (const r of showRows) {
    const row = [];
    let ci = 0;
    for (const c of showCols) {
      const text = mtoc2_format_double(data[offset + r + c * rows]);
      row.push(text);
      if (text.length > colWidths[ci]) colWidths[ci] = text.length;
      ci++;
      if (truncCols && ci === cHi) {
        row.push("...");
        if (colWidths[ci] < 3) colWidths[ci] = 3;
        ci++;
      }
    }
    formatted.push(row);
  }

  let fi = 0;
  for (let si = 0; si < showRows.length; si++) {
    if (truncRows && si === rHi) {
      $write("   " + colWidths.map(w => "...".padStart(w)).join("   ") + "\n");
    }
    const parts = formatted[fi].map((s, c) => s.padStart(colWidths[c]));
    $write("   " + parts.join("   ") + "\n");
    fi++;
  }
}

export function mtoc2_disp_tensor(t) {
  // Mirrors numbl: empty tensors print nothing.
  if (!t || !t.shape || t.shape.length === 0) return;
  const rows = t.shape[0] ?? 1;
  const cols = t.shape[1] ?? 1;
  let total = 1;
  for (const s of t.shape) total *= s;
  if (total <= 0) return;
  // 1-element tensors print as bare scalars (no column-aligned
  // indent) — matches numbl's `runtime/display.ts:128` special case.
  if (total === 1) {
    $write(mtoc2_format_double(t.data[0]) + "\n");
    return;
  }

  const pageSize = rows * cols;
  let numPages = 1;
  for (let i = 2; i < t.shape.length; i++) numPages *= t.shape[i];

  for (let p = 0; p < numPages; p++) {
    if (t.shape.length > 2) {
      if (p > 0) $write("\n");
      let rem = p;
      let header = "(:,:";
      for (let i = 2; i < t.shape.length; i++) {
        const d = t.shape[i];
        const s = rem % d;
        rem = Math.floor(rem / d);
        header += "," + (s + 1);
      }
      header += ") =\n\n";
      $write(header);
    }
    disp_real_slice(t.data, p * pageSize, rows, cols);
  }
}
