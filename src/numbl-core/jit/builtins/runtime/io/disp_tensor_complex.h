/* @snippet name=jit_disp_tensor_complex deps=jit_tensor_t,jit_format_complex,jit_cscalar,jit_host_output */
/* JIT runtime helper: disp(t) for a multi-element complex tensor.
 *
 * Mirrors `disp_tensor.h`'s 2-D-slice / page-by-page rendering, but
 * each cell is formatted via `jit_format_complex` (which mirrors
 * numbl's `formatComplex`). Padding is computed against the
 * formatted-string lengths so a column with `1 - 2i` cells still
 * lines up with sibling cells of differing widths.
 *
 * Builds each `double _Complex` cell via `jit_cmake` rather than
 * the `I` macro. The JS sibling formats `{re, im}` objects.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void jit__disp_complex_slice(
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
    fprintf(stderr, "C-JIT: out of memory in jit_disp_tensor_complex\n");
    return;
  }

  for (long c = 0; c < cols; c++) {
    for (long r = 0; r < rows; r++) {
      long idx = r + c * rows;
      char *cell = cells + idx * CELL_CAP;
      double _Complex z = jit_cmake(re[idx], im[idx]);
      jit_format_complex(cell, CELL_CAP, z);
      long len = (long)strlen(cell);
      if (len > col_widths[c]) col_widths[c] = len;
    }
  }

  for (long r = 0; r < rows; r++) {
    jit_stdout("   ", 3);
    for (long c = 0; c < cols; c++) {
      long idx = r + c * rows;
      char *cell = cells + idx * CELL_CAP;
      long len = (long)strlen(cell);
      for (long i = 0; i < col_widths[c] - len; i++) jit_stdout(" ", 1);
      jit_stdout_s(cell);
      if (c < cols - 1) jit_stdout("   ", 3);
    }
    jit_stdout("\n", 1);
  }

  free(cells);
  free(col_widths);
}

static void jit_disp_tensor_complex(jit_tensor_t t) {
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
      if (p > 0) jit_stdout("\n", 1);
      long rem = p;
      jit_stdout("(:,:", 4);
      for (int i = 2; i < t.ndim; i++) {
        long d = t.dims[i];
        long s = rem % d;
        rem /= d;
        jit_stdout_printf(",%ld", s + 1);
      }
      jit_stdout_s(") =\n\n");
    }
    jit__disp_complex_slice(
      t.real + p * page_size, t.imag + p * page_size, rows, cols);
  }
}
