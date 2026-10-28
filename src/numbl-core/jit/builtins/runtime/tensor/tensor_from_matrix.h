/* mtoc runtime helper: build a rows×cols real tensor from a flat
 * column-major data pointer. Codegen reorders the literal's
 * row-major source cells to column-major when it builds the
 * compound-literal array, so layout matches the rest of the runtime.
 */

#include <string.h>

static jit_tensor_t jit_tensor_from_matrix(
  const double *data, long rows, long cols
) {
  jit_tensor_t out = jit_tensor_alloc(rows, cols);
  memcpy(out.real, data, (size_t)(rows * cols) * sizeof(double));
  return out;
}
