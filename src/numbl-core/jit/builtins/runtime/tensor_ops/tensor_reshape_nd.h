/* JIT runtime helper: reshape a real tensor to an N-D shape.
 *
 * Receives the input tensor by value (`jit_tensor_t`) plus a
 * caller-supplied dim list (`ndim`, `dims`). Allocates a fresh
 * output tensor via `jit_tensor_alloc_nd` and copies the input's
 * column-major buffer wholesale — reshape is a layout reinterpret,
 * so the linear element order is unchanged.
 *
 * `dims[i] == -1` is the MATLAB `[]` auto-infer slot: the helper
 * scans for a single -1 and fills it from `in_total / prod(others)`.
 * Two or more sentinels, or an `in_total` not divisible by the
 * explicit dims, raise.
 *
 * Element-count check: the lowerer enforces `prod(input.dims) ==
 * prod(dims)` at translate time when the input shape is statically
 * known. This helper is the fallback when the input shape only
 * appears at runtime (e.g. a tensor function param whose
 * specialization arg type came in without a concrete shape). On
 * mismatch it raises the interpreter's error (`raise.h`), matching numbl's
 * runtime-error surface.
 *
 * Real-only — the JIT's tensor side is real-only today; the type
 * lattice rejects complex inputs at lowering. The output's `imag`
 * is NULL, set by `jit_tensor_alloc_nd`.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static jit_tensor_t jit_reshape_nd(
    jit_tensor_t in, int ndim, const long *dims) {
  size_t in_total = 1;
  for (int i = 0; i < in.ndim; i++) in_total *= (size_t)in.dims[i];
  /* Scan for at most one `-1` infer slot and the product of the
   * remaining explicit dims. */
  int infer_idx = -1;
  size_t explicit_prod = 1;
  for (int i = 0; i < ndim; i++) {
    if (dims[i] == -1) {
      if (infer_idx != -1) {
        jit_raise_bailf("reshape: at most one '[]' auto-infer slot allowed");
      }
      infer_idx = i;
    } else if (dims[i] < 0) {
      jit_raise_error_deferred();
    } else {
      explicit_prod *= (size_t)dims[i];
    }
  }
  long resolved_dims[JIT_MAX_NDIM];
  for (int i = 0; i < ndim; i++) resolved_dims[i] = dims[i];
  size_t out_total;
  if (infer_idx != -1) {
    if (explicit_prod == 0 && in_total != 0) {
      jit_raise_error_deferred();
    }
    if (explicit_prod > 0 && in_total % explicit_prod != 0) {
      jit_raise_error("", "reshape: number of elements must not change");
    }
    resolved_dims[infer_idx] =
      (explicit_prod == 0) ? 0 : (long)(in_total / explicit_prod);
    out_total = in_total;
  } else {
    out_total = explicit_prod;
    if (in_total != out_total) {
      jit_raise_error("", "reshape: number of elements must not change");
    }
  }
  jit_tensor_t out = jit_tensor_alloc_nd(ndim, resolved_dims);
  if (out_total > 0)
    memcpy(out.real, in.real, out_total * sizeof(double));
  return out;
}
