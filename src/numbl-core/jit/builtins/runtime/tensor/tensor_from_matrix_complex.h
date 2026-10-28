/* JIT runtime helper: build a rows×cols complex tensor from two
 * parallel column-major `double[]` sources (`re` and `im`). See
 * `tensor_from_row_complex.h` for why we split rather than passing
 * `double _Complex *`.
 */

#include <string.h>

static jit_tensor_t jit_tensor_from_matrix_complex(
    const double *re, const double *im, long rows, long cols) {
  jit_tensor_t out = jit_tensor_alloc_complex(rows, cols);
  long n = rows * cols;
  memcpy(out.real, re, (size_t)n * sizeof(double));
  memcpy(out.imag, im, (size_t)n * sizeof(double));
  return out;
}
