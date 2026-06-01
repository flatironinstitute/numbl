/* mtoc2 runtime helper: build a rows×cols complex tensor from two
 * parallel column-major `double[]` sources (`re` and `im`). See
 * `tensor_from_row_complex.h` for why we split rather than passing
 * `double _Complex *` (c2js can't translate that signature).
 */

#include <string.h>

static mtoc2_tensor_t mtoc2_tensor_from_matrix_complex(
    const double *re, const double *im, long rows, long cols) {
  mtoc2_tensor_t out = mtoc2_tensor_alloc_complex(rows, cols);
  long n = rows * cols;
  memcpy(out.real, re, (size_t)n * sizeof(double));
  memcpy(out.imag, im, (size_t)n * sizeof(double));
  return out;
}
