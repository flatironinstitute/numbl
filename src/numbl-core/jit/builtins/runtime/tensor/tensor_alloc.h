/* mtoc runtime helper: allocate an uninitialized real tensor of the
 * given shape. Sets `imag = NULL` (the static-real marker). The
 * returned tensor owns its `real` buffer; caller must release via
 * `mtoc2_tensor_free` (or hand it to `mtoc2_tensor_assign`, which takes
 * ownership).
 *
 * `mtoc2_alloc` aborts on OOM, so the returned struct's `real` is
 * always non-NULL. Used by codegen as the workhorse for elementwise-
 * result construction at every multi-element Assign RHS that isn't a
 * plain tensor literal.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static mtoc2_tensor_t mtoc2_tensor_alloc(long rows, long cols) {
  /* MATLAB / numbl clamp negative dim values to 0 (resulting in an
   * empty tensor) rather than aborting; without this the bare
   * `(size_t)rows` cast below would wrap a negative `long` to
   * SIZE_MAX and trigger a spurious "allocation overflow" abort. */
  if (rows < 0) rows = 0;
  if (cols < 0) cols = 0;
  mtoc2_tensor_t out;
  size_t n;
#if defined(__has_builtin) && __has_builtin(__builtin_mul_overflow)
  size_t bytes;
  if (__builtin_mul_overflow((size_t)rows, (size_t)cols, &n) ||
      __builtin_mul_overflow(n, sizeof(double), &bytes)) {
    fprintf(stderr,
      "mtoc2: tensor allocation overflow (%ldx%ld elements)\n", rows, cols);
    abort();
  }
#else
  if ((size_t)cols != 0 && (size_t)rows > (SIZE_MAX / sizeof(double)) / (size_t)cols) {
    fprintf(stderr,
      "mtoc2: tensor allocation overflow (%ldx%ld elements)\n", rows, cols);
    abort();
  }
  n = (size_t)rows * (size_t)cols;
  size_t bytes = n * sizeof(double);
#endif
  out.real = mtoc2_alloc(bytes);
  out.imag = NULL;
  out.ndim = 2;
  out.dims[0] = rows;
  out.dims[1] = cols;
  return out;
}
