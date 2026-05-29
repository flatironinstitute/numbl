/* mtoc2 runtime helper: allocate an uninitialized complex tensor with
 * an arbitrary N-D shape. Sibling of `mtoc2_tensor_alloc_nd` that
 * allocates BOTH lanes. The returned tensor owns both buffers.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static mtoc2_tensor_t mtoc2_tensor_alloc_nd_complex(int ndim, const long *dims) {
  if (ndim < 1 || ndim > MTOC2_MAX_NDIM) {
    fprintf(stderr,
      "mtoc2: tensor ndim %d out of range [1, %d]\n", ndim, MTOC2_MAX_NDIM);
    abort();
  }
  mtoc2_tensor_t out;
  size_t n = 1;
  for (int i = 0; i < ndim; i++) {
    long d = dims[i] < 0 ? 0 : dims[i];
    out.dims[i] = d;
    size_t new_n;
#if defined(__has_builtin) && __has_builtin(__builtin_mul_overflow)
    if (__builtin_mul_overflow(n, (size_t)d, &new_n)) {
      fprintf(stderr,
        "mtoc2: complex tensor allocation overflow at dim %d (size %ld)\n",
        i, d);
      abort();
    }
#else
    if ((size_t)d != 0 && n > (SIZE_MAX / sizeof(double)) / (size_t)d) {
      fprintf(stderr,
        "mtoc2: complex tensor allocation overflow at dim %d (size %ld)\n",
        i, d);
      abort();
    }
    new_n = n * (size_t)d;
#endif
    n = new_n;
  }
  out.ndim = ndim;
  size_t bytes;
#if defined(__has_builtin) && __has_builtin(__builtin_mul_overflow)
  if (__builtin_mul_overflow(n, sizeof(double), &bytes)) {
    fprintf(stderr,
      "mtoc2: complex tensor allocation overflow at byte-count (%zu elements)\n", n);
    abort();
  }
#else
  bytes = n * sizeof(double);
#endif
  out.real = mtoc2_alloc(bytes);
  out.imag = mtoc2_alloc(bytes);
  return out;
}
