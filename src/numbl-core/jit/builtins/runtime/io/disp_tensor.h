/* mtoc runtime helper: disp(t) for a multi-element real tensor.
 *
 * Mirrors numbl's `format2DSlice` for the 2-D path and numbl's
 * page-by-page N-D rendering for `ndim > 2` (each 2-D slice prefixed
 * by a `(:,:,k2,k3,...) =` header and separated by a blank line):
 *   - elements are formatted via mtoc2_format_double
 *   - each column is padded to its widest element via padStart
 *   - rows are separated by '\n', columns by 3 spaces, 3-space indent
 *
 * Allocation: per-slice malloc for the formatted-string buffer and
 * column-width array. Both freed before the next slice. The disp
 * path is not on the hot path of typical numerical code, so the
 * simplicity is worth the alloc.
 *
 * Real-only — complex-tensor disp lives in `disp_tensor_complex.h`;
 * the lowerer dispatches on `isComplex`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Render a single 2-D slice (rows × cols) starting at `data` (the
 * caller already advanced past prior pages). Each row ends with
 * '\n'; no leading or trailing framing — the caller adds page
 * headers / separators. */
static void mtoc2__disp_real_slice(const double *data, long rows, long cols) {
  enum { CELL_CAP = 32 };
  long ncells = rows * cols;
  char *cells = (char *)malloc((size_t)ncells * CELL_CAP);
  long *col_widths = (long *)calloc((size_t)cols, sizeof(long));
  if (!cells || !col_widths) {
    free(cells);
    free(col_widths);
    fprintf(stderr, "mtoc2: out of memory in mtoc2_disp_tensor\n");
    return;
  }

  for (long c = 0; c < cols; c++) {
    for (long r = 0; r < rows; r++) {
      long idx = r + c * rows;
      char *cell = cells + idx * CELL_CAP;
      mtoc2_format_double(cell, CELL_CAP, data[idx]);
      long len = (long)strlen(cell);
      if (len > col_widths[c]) col_widths[c] = len;
    }
  }

  for (long r = 0; r < rows; r++) {
    mtoc2_stdout("   ", 3);
    for (long c = 0; c < cols; c++) {
      long idx = r + c * rows;
      char *cell = cells + idx * CELL_CAP;
      long len = (long)strlen(cell);
      for (long i = 0; i < col_widths[c] - len; i++) mtoc2_stdout(" ", 1);
      mtoc2_stdout_s(cell);
      if (c < cols - 1) mtoc2_stdout("   ", 3);
    }
    mtoc2_stdout("\n", 1);
  }

  free(cells);
  free(col_widths);
}

static void mtoc2_disp_tensor(mtoc2_tensor_t t) {
  if (t.ndim == 0 || t.real == NULL) {
    /* `mtoc2_tensor_empty()` placeholder. Reaches here when a
     * conditionally-assigned tensor is `disp`ed on a path the
     * conditional didn't fire. Numbl's `disp` short-circuits empty
     * tensors silently (specialBuiltins.ts: the disp handler
     * returns before printing if `data.length === 0`), so we
     * print nothing here too. */
    return;
  }
  long rows = t.ndim >= 1 ? t.dims[0] : 1;
  long cols = t.ndim >= 2 ? t.dims[1] : 1;
  long total = 1;
  for (int i = 0; i < t.ndim; i++) total *= t.dims[i];
  if (total <= 0) {
    /* Empty tensor — numbl's `disp` prints nothing (it bails on
     * `data.length === 0` before reaching the formatter). The
     * `[]` rendering from `formatTensor` is only used by other
     * surface forms (string cast, struct field display). */
    return;
  }
  /* 1-element tensor prints as a bare scalar (no column-aligned
   * indent) — matches numbl's `display.ts:128` special case and the
   * JS sibling's `disp_tensor.js`. */
  if (total == 1) {
    char buf[32];
    mtoc2_format_double(buf, sizeof(buf), t.real[0]);
    mtoc2_stdout_s(buf);
    mtoc2_stdout("\n", 1);
    return;
  }
  long page_size = rows * cols;
  long num_pages = 1;
  for (int i = 2; i < t.ndim; i++) num_pages *= t.dims[i];

  for (long p = 0; p < num_pages; p++) {
    if (t.ndim > 2) {
      /* Blank line between pages (after the previous slice's trailing
       * '\n'). For the very first page there is no leading separator. */
      if (p > 0) mtoc2_stdout("\n", 1);
      /* Outer indices via column-major ind2sub (k2 changes fastest). */
      long rem = p;
      mtoc2_stdout("(:,:", 4);
      for (int i = 2; i < t.ndim; i++) {
        long d = t.dims[i];
        long s = rem % d;
        rem /= d;
        mtoc2_stdout_printf(",%ld", s + 1);
      }
      mtoc2_stdout_s(") =\n\n");
    }
    mtoc2__disp_real_slice(t.real + p * page_size, rows, cols);
  }
}
