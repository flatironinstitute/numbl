/* mtoc2 runtime helper: reshape a complex tensor to an N-D shape.
 *
 * Sibling of `mtoc2_reshape_nd`. Same `-1` auto-infer slot, same
 * element-count check, same runtime-error surface — the only
 * difference is the output is allocated via
 * `mtoc2_tensor_alloc_nd_complex` (both lanes) and both lanes get
 * memcpy'd from the input. Reshape is a layout reinterpret, so the
 * linear element order is unchanged on either lane.
 *
 * Tolerates `in.imag == NULL` (a real tensor flowing through a
 * complex-typed reshape route) by zeroing the output imag lane.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static mtoc2_tensor_t mtoc2_reshape_nd_complex(
    mtoc2_tensor_t in, int ndim, const long *dims) {
  size_t in_total = 1;
  for (int i = 0; i < in.ndim; i++) in_total *= (size_t)in.dims[i];
  int infer_idx = -1;
  size_t explicit_prod = 1;
  for (int i = 0; i < ndim; i++) {
    if (dims[i] == -1) {
      if (infer_idx != -1) {
        fprintf(stderr,
          "mtoc2: reshape: at most one '[]' auto-infer slot allowed\n");
        abort();
      }
      infer_idx = i;
    } else if (dims[i] < 0) {
      fprintf(stderr,
        "mtoc2: reshape: dim %d must be a non-negative integer "
        "(got %ld)\n", i + 1, dims[i]);
      abort();
    } else {
      explicit_prod *= (size_t)dims[i];
    }
  }
  long resolved_dims[MTOC2_MAX_NDIM];
  for (int i = 0; i < ndim; i++) resolved_dims[i] = dims[i];
  size_t out_total;
  if (infer_idx != -1) {
    if (explicit_prod == 0 && in_total != 0) {
      fprintf(stderr,
        "mtoc2: reshape: input has %zu elements but the explicit dims "
        "around '[]' multiply to 0\n", in_total);
      abort();
    }
    if (explicit_prod > 0 && in_total % explicit_prod != 0) {
      fprintf(stderr,
        "mtoc2: reshape: input has %zu elements, not divisible by %zu "
        "(the explicit dims around '[]')\n", in_total, explicit_prod);
      abort();
    }
    resolved_dims[infer_idx] =
      (explicit_prod == 0) ? 0 : (long)(in_total / explicit_prod);
    out_total = in_total;
  } else {
    out_total = explicit_prod;
    if (in_total != out_total) {
      fprintf(stderr,
        "mtoc2: reshape: number of elements must not change "
        "(in=%zu, out=%zu)\n", in_total, out_total);
      abort();
    }
  }
  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd_complex(ndim, resolved_dims);
  if (out_total > 0) {
    memcpy(out.real, in.real, out_total * sizeof(double));
    if (in.imag != NULL) {
      memcpy(out.imag, in.imag, out_total * sizeof(double));
    } else {
      memset(out.imag, 0, out_total * sizeof(double));
    }
  }
  return out;
}
