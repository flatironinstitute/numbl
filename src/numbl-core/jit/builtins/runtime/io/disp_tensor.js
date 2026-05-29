// JS sibling of `disp_tensor.h`. Real-only tensor display, mirroring
// numbl's `format2DSlice` for the 2-D path and numbl's page-by-page
// rendering for ndim > 2. Each cell is formatted via
// `mtoc2_format_double`; columns are padded to their widest element;
// rows are separated by `\n`, columns by 3 spaces, indented by 3
// spaces.

import { mtoc2_format_double } from "./format_double.js";

function disp_real_slice(data, offset, rows, cols) {
  const cells = new Array(rows * cols);
  const colWidths = new Array(cols).fill(0);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const idx = r + c * rows;
      const text = mtoc2_format_double(data[offset + idx]);
      cells[idx] = text;
      if (text.length > colWidths[c]) colWidths[c] = text.length;
    }
  }
  for (let r = 0; r < rows; r++) {
    let line = "   ";
    for (let c = 0; c < cols; c++) {
      const idx = r + c * rows;
      const cell = cells[idx];
      const pad = colWidths[c] - cell.length;
      for (let i = 0; i < pad; i++) line += " ";
      line += cell;
      if (c < cols - 1) line += "   ";
    }
    line += "\n";
    $write(line);
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
