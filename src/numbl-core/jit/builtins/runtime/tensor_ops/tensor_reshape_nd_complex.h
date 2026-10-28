/* @snippet name=jit_reshape_nd_complex deps=jit_tensor_t,jit_tensor_alloc_nd_complex */
/* JIT runtime helper: reshape a complex tensor to an N-D shape.
 *
 * Sibling of `jit_reshape_nd`. Same `-1` auto-infer slot, same
 * element-count check, same runtime-error surface — the only
 * difference is the output is allocated via
 * `jit_tensor_alloc_nd_complex` (both lanes) and both lanes get
 * memcpy'd from the input. Reshape is a layout reinterpret, so the
 * linear element order is unchanged on either lane.
 *
 * Tolerates `in.imag == NULL` (a real tensor flowing through a
 * complex-typed reshape route) by zeroing the output imag lane.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static jit_tensor_t jit_reshape_nd_complex(
    jit_tensor_t in, int ndim, const long *dims) {
  size_t in_total = 1;
  for (int i = 0; i < in.ndim; i++) in_total *= (size_t)in.dims[i];
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
  jit_tensor_t out = jit_tensor_alloc_nd_complex(ndim, resolved_dims);
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
