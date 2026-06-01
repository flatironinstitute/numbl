/* Inline-display sibling of `mtoc2_disp_tensor`. Identical body except
 * the final '\n' is suppressed — used by per-shape cell `_disp`
 * helpers so a tensor slot inside `{...}` renders the same way numbl's
 * `formatCell` (display.ts) interpolates `formatTensor` (no trailing
 * newline).
 *
 * Empty tensor renders as `[]` (matches numbl's `formatTensor`
 * fall-through, used here because the empty-double sentinel returned
 * from `cell(n, m)` flows through this path).
 */

static void mtoc2__disp_real_slice_inline(double *data, long rows,
                                          long cols) {
  long total = rows * cols;
  if (total <= 0) return;
  enum { CELL_CAP = 32 };
  char *cells = (char *)malloc((size_t)total * CELL_CAP);
  long *col_widths = (long *)calloc((size_t)cols, sizeof(long));
  if (cells == NULL || col_widths == NULL) {
    free(cells);
    free(col_widths);
    fprintf(stderr, "mtoc2: out of memory in mtoc2_disp_tensor_inline\n");
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
    if (r < rows - 1) mtoc2_stdout("\n", 1);
  }
  free(cells);
  free(col_widths);
}

static void mtoc2_disp_tensor_inline(mtoc2_tensor_t t) {
  if (t.ndim == 0 || t.real == NULL) {
    mtoc2_stdout_s("[]");
    return;
  }
  long rows = t.ndim >= 1 ? t.dims[0] : 1;
  long cols = t.ndim >= 2 ? t.dims[1] : 1;
  long total = 1;
  for (int i = 0; i < t.ndim; i++) total *= t.dims[i];
  if (total <= 0) {
    mtoc2_stdout_s("[]");
    return;
  }
  /* Multi-page (N-D) cells flow through page headers like the
   * non-inline helper; the final page's last row drops its trailing
   * newline. This shape isn't exercised by chunkie's cell of 1×K
   * tensors but stays consistent for tests that probe it. */
  long page_size = rows * cols;
  long num_pages = 1;
  for (int i = 2; i < t.ndim; i++) num_pages *= t.dims[i];
  for (long p = 0; p < num_pages; p++) {
    if (t.ndim > 2) {
      if (p > 0) mtoc2_stdout("\n", 1);
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
    mtoc2__disp_real_slice_inline(t.real + p * page_size, rows, cols);
  }
}
