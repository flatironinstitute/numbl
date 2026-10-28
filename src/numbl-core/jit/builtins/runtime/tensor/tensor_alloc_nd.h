/* @snippet name=jit_tensor_alloc_nd deps=jit_tensor_t,jit_alloc */
/* JIT runtime helper: allocate an uninitialized real tensor with
 * an arbitrary N-D shape.
 *
 * The 2-D fast path lives in `jit_tensor_alloc`; this helper is the
 * variadic N-D sibling that copies `ndim` dim sizes from the caller-
 * supplied `dims` array into the struct.
 *
 * `dims` is consumed read-only — the caller may pass a stack
 * `(long[]){…}` compound literal. The returned tensor owns its
 * `real` buffer; `imag` is NULL (the static-real marker). Raises
 * on `ndim` exceeding `JIT_MAX_NDIM` or on size-math overflow.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static jit_tensor_t jit_tensor_alloc_nd(int ndim, const long *dims) {
  if (ndim < 1 || ndim > JIT_MAX_NDIM) {
    /* The rest of the runtime assumes `ndim >= 1` (and codegen enforces
     * `ndim >= 2`). A zero or negative ndim would skip the dim loop and
     * leave the tensor in a malformed state with a 1-element default
     * buffer; better to raise. */
    jit_raise_bailf("tensor ndim %d out of range [1, %d]", ndim, JIT_MAX_NDIM);
  }
  jit_tensor_t out;
  size_t n = 1;
  for (int i = 0; i < ndim; i++) {
    /* MATLAB / numbl clamp negative dim values to 0 (yielding an
     * empty tensor) rather than raising. Without this the
     * `(size_t)dims[i]` cast below wraps a negative `long` to
     * SIZE_MAX and the very next mul-overflow check fires with a
     * misleading message. Trigger:
     *   `n = 0; %!numbl:opaque n; zeros(n - 1, 3);` */
    long d = dims[i] < 0 ? 0 : dims[i];
    out.dims[i] = d;
    size_t new_n;
#if defined(__has_builtin) && __has_builtin(__builtin_mul_overflow)
    if (__builtin_mul_overflow(n, (size_t)d, &new_n)) {
      jit_raise_bailf("tensor allocation overflow at dim %d (size %ld)", i, d);
    }
#else
    if ((size_t)d != 0 && n > (SIZE_MAX / sizeof(double)) / (size_t)d) {
      jit_raise_bailf("tensor allocation overflow at dim %d (size %ld)", i, d);
    }
    new_n = n * (size_t)d;
#endif
    n = new_n;
  }
  out.ndim = ndim;
  size_t bytes;
#if defined(__has_builtin) && __has_builtin(__builtin_mul_overflow)
  if (__builtin_mul_overflow(n, sizeof(double), &bytes)) {
    jit_raise_bailf("tensor allocation overflow at byte-count (%zu elements)", n);
  }
#else
  bytes = n * sizeof(double);
#endif
  out.real = jit_alloc(bytes);
  out.imag = NULL;
  return out;
}
