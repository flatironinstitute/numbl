/* mtoc2 runtime helper: disp(t) for a multi-element complex tensor.
 *
 * Mirrors `disp_tensor.h`'s 2-D-slice / page-by-page rendering, but
 * each cell is formatted via `mtoc2_format_complex` (which mirrors
 * numbl's `formatComplex`). Padding is computed against the
 * formatted-string lengths so a column with `1 - 2i` cells still
 * lines up with sibling cells of differing widths.
 *
 * Builds each `double _Complex` cell via `mtoc2_cmake` rather than
 * the `I` macro so the c2js backend can translate the body. The
 * actual format helper (`mtoc2_format_complex`) is on c2js's skip
 * list and substituted with a JS implementation that walks
 * `{re, im}` objects.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void mtoc2__disp_complex_slice(
    const double *re, const double *im, long rows, long cols) {
  /* Allow up to 80 chars per cell — `1.2345e+08 - 2.3456e+08i` and
   * friends comfortably fit. */
  enum { CELL_CAP = 80 };
  long ncells = rows * cols;
  char *cells = (char *)malloc((size_t)ncells * CELL_CAP);
  long *col_widths = (long *)calloc((size_t)cols, sizeof(long));
  if (!cells || !col_widths) {
    free(cells);
    free(col_widths);
    fprintf(stderr, "mtoc2: out of memory in mtoc2_disp_tensor_complex\n");
    return;
  }

  for (long c = 0; c < cols; c++) {
    for (long r = 0; r < rows; r++) {
      long idx = r + c * rows;
      char *cell = cells + idx * CELL_CAP;
      double _Complex z = mtoc2_cmake(re[idx], im[idx]);
      mtoc2_format_complex(cell, CELL_CAP, z);
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

static void mtoc2_disp_tensor_complex(mtoc2_tensor_t t) {
  if (t.ndim == 0 || t.real == NULL) return;
  long rows = t.ndim >= 1 ? t.dims[0] : 1;
  long cols = t.ndim >= 2 ? t.dims[1] : 1;
  long total = 1;
  for (int i = 0; i < t.ndim; i++) total *= t.dims[i];
  if (total <= 0) return;
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
    mtoc2__disp_complex_slice(
      t.real + p * page_size, t.imag + p * page_size, rows, cols);
  }
}
