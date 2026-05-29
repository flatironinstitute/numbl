/* mtoc2 runtime helper: deep-copy a complex tensor.
 *
 * The returned tensor owns fresh `real` and `imag` buffers (memcpy'd
 * from the source). Sibling of `mtoc2_tensor_copy` for complex-typed
 * tensors; the codegen invariant ("every tensor RHS is freshly owned,
 * every Var read wraps in copy") applies identically.
 */

#include <string.h>

static mtoc2_tensor_t mtoc2_tensor_copy_complex(mtoc2_tensor_t src) {
  if (src.ndim == 0 || src.real == NULL) return mtoc2_tensor_empty();
  mtoc2_tensor_t out = src;
  long n = 1;
  for (int i = 0; i < src.ndim; i++) n *= src.dims[i];
  size_t bytes = (size_t)n * sizeof(double);
  out.real = mtoc2_alloc(bytes);
  out.imag = mtoc2_alloc(bytes);
  if (src.real) memcpy(out.real, src.real, bytes);
  if (src.imag) {
    memcpy(out.imag, src.imag, bytes);
  } else {
    /* Defensive: src claims to be complex but `imag` is NULL. Zero
     * the destination's imag lane. */
    memset(out.imag, 0, bytes);
  }
  return out;
}
