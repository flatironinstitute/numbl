/* mtoc2 runtime helper: reshape a real tensor to an N-D shape.
 *
 * Receives the input tensor by value (`mtoc2_tensor_t`) plus a
 * caller-supplied dim list (`ndim`, `dims`). Allocates a fresh
 * output tensor via `mtoc2_tensor_alloc_nd` and copies the input's
 * column-major buffer wholesale — reshape is a layout reinterpret,
 * so the linear element order is unchanged.
 *
 * `dims[i] == -1` is the MATLAB `[]` auto-infer slot: the helper
 * scans for a single -1 and fills it from `in_total / prod(others)`.
 * Two or more sentinels, or an `in_total` not divisible by the
 * explicit dims, abort with a clear message.
 *
 * Element-count check: the lowerer enforces `prod(input.dims) ==
 * prod(dims)` at translate time when the input shape is statically
 * known. This helper is the fallback when the input shape only
 * appears at runtime (e.g. a tensor function param whose
 * specialization arg type came in without a concrete shape). On
 * mismatch it prints to stderr and aborts, matching numbl's
 * runtime-error surface.
 *
 * Real-only — mtoc2's tensor side is real-only today; the type
 * lattice rejects complex inputs at lowering. The output's `imag`
 * is NULL, set by `mtoc2_tensor_alloc_nd`.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static mtoc2_tensor_t mtoc2_reshape_nd(
    mtoc2_tensor_t in, int ndim, const long *dims) {
  size_t in_total = 1;
  for (int i = 0; i < in.ndim; i++) in_total *= (size_t)in.dims[i];
  /* Scan for at most one `-1` infer slot and the product of the
   * remaining explicit dims. */
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
  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(ndim, resolved_dims);
  if (out_total > 0)
    memcpy(out.real, in.real, out_total * sizeof(double));
  return out;
}
