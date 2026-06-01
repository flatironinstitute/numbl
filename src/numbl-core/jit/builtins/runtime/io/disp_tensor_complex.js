// JS sibling of `disp_tensor_complex.h`. Display a complex tensor
// with each cell formatted via `mtoc2_format_complex` and columns
// padded to their widest cell.

import { mtoc2_format_complex } from "./format_complex.js";

function disp_complex_slice(re, im, offset, rows, cols) {
  const cells = new Array(rows * cols);
  const colWidths = new Array(cols).fill(0);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const idx = r + c * rows;
      const text = mtoc2_format_complex({
        re: re[offset + idx],
        im: im[offset + idx],
      });
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

export function mtoc2_disp_tensor_complex(t) {
  if (!t || !t.shape || t.shape.length === 0) return;
  if (t.imag === undefined) return; // shouldn't happen for a complex tensor
  const rows = t.shape[0] ?? 1;
  const cols = t.shape[1] ?? 1;
  let total = 1;
  for (const s of t.shape) total *= s;
  if (total <= 0) return;

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
    disp_complex_slice(t.data, t.imag, p * pageSize, rows, cols);
  }
}
