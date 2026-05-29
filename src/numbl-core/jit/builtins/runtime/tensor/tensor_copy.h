/* mtoc runtime helper: deep-copy a real tensor.
 *
 * The returned tensor owns a fresh `real` buffer (memcpy'd from the
 * source); `imag` is NULL since the caller statically knows the
 * source is real. Used by codegen to honor "copy on every
 * manipulation" — every tensor-by-name read and every user-function
 * tensor argument is wrapped in this helper, so the receiver always
 * gets an owned tensor.
 */

#include <string.h>

static mtoc2_tensor_t mtoc2_tensor_copy(mtoc2_tensor_t src) {
  /* An `mtoc2_tensor_empty()` placeholder (ndim == 0, real == NULL)
   * can flow into a copy when a conditionally-assigned tensor is read
   * on a path the conditional didn't fire — return another empty so
   * downstream consumers see the same placeholder state instead of
   * `memcpy`ing from NULL. */
  if (src.ndim == 0 || src.real == NULL) return mtoc2_tensor_empty();
  /* Struct copy preserves ndim and dims; the heap buffer is
   * replaced with a fresh allocation of the right total size. */
  mtoc2_tensor_t out = src;
  long n = 1;
  for (int i = 0; i < src.ndim; i++) n *= src.dims[i];
  out.real = mtoc2_alloc((size_t)n * sizeof(double));
  out.imag = NULL;
  memcpy(out.real, src.real, (size_t)n * sizeof(double));
  return out;
}
