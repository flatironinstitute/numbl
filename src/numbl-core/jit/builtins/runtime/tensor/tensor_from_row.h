/* mtoc runtime helper: build a 1×n real tensor from a flat data
 * pointer. The caller typically passes a C99 compound literal
 * (`(double[]){1.0, 2.0, 3.0}`), which gives the emitted assignment
 * the same one-line shape as the numbl source.
 */

#include <string.h>

static mtoc2_tensor_t mtoc2_tensor_from_row(const double *data, long n) {
  mtoc2_tensor_t out = mtoc2_tensor_alloc(1, n);
  memcpy(out.real, data, (size_t)n * sizeof(double));
  return out;
}
