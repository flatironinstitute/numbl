/* @snippet name=jit_tensor_alloc_complex deps=jit_tensor_t,jit_alloc */
/* JIT runtime helper: allocate an uninitialized complex tensor of
 * the given 2-D shape. Sibling of `jit_tensor_alloc`; this one
 * allocates BOTH lanes (`real` and `imag` of equal length). The
 * returned tensor owns both buffers; caller releases via
 * `jit_tensor_free` (which frees both unconditionally — `free(NULL)`
 * being a no-op covers tensors that were re-claimed as real later).
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static jit_tensor_t jit_tensor_alloc_complex(long rows, long cols) {
  if (rows < 0) rows = 0;
  if (cols < 0) cols = 0;
  jit_tensor_t out;
  size_t n;
#if defined(__has_builtin) && __has_builtin(__builtin_mul_overflow)
  size_t bytes;
  if (__builtin_mul_overflow((size_t)rows, (size_t)cols, &n) ||
      __builtin_mul_overflow(n, sizeof(double), &bytes)) {
    jit_raise_bailf("complex tensor allocation overflow (%ldx%ld elements)",
      rows, cols);
  }
#else
  if ((size_t)cols != 0 && (size_t)rows > (SIZE_MAX / sizeof(double)) / (size_t)cols) {
    jit_raise_bailf("complex tensor allocation overflow (%ldx%ld elements)",
      rows, cols);
  }
  n = (size_t)rows * (size_t)cols;
  size_t bytes = n * sizeof(double);
#endif
  out.real = jit_alloc(bytes);
  out.imag = jit_alloc(bytes);
  out.ndim = 2;
  out.dims[0] = rows;
  out.dims[1] = cols;
  return out;
}
