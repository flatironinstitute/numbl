/* @snippet name=jit_tensor_copy_complex deps=jit_tensor_t,jit_alloc,jit_tensor_empty */
/* JIT runtime helper: deep-copy a complex tensor.
 *
 * The returned tensor owns fresh `real` and `imag` buffers (memcpy'd
 * from the source). Sibling of `jit_tensor_copy` for complex-typed
 * tensors; the codegen invariant ("every tensor RHS is freshly owned,
 * every Var read wraps in copy") applies identically.
 */

#include <string.h>

static jit_tensor_t jit_tensor_copy_complex(jit_tensor_t src) {
  if (src.ndim == 0 || src.real == NULL) return jit_tensor_empty();
  jit_tensor_t out = src;
  long n = 1;
  for (int i = 0; i < src.ndim; i++) n *= src.dims[i];
  size_t bytes = (size_t)n * sizeof(double);
  out.real = jit_alloc(bytes);
  out.imag = jit_alloc(bytes);
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
