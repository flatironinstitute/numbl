/* JIT runtime helper: build a 1×N complex tensor from two parallel
 * `double[]` sources (`re` and `im`, each length `n`). The caller
 * (the JIT's emit layer) splits each source-level cell into its real
 * and imaginary parts via `jit_creal` / `jit_cimag` before
 * stuffing into the compound literals — that avoids passing
 * `double _Complex` arrays around at all (no `<complex.h>` types in
 * the signature or body).
 */

#include <string.h>

static jit_tensor_t jit_tensor_from_row_complex(
    const double *re, const double *im, long n) {
  jit_tensor_t out = jit_tensor_alloc_complex(1, n);
  memcpy(out.real, re, (size_t)n * sizeof(double));
  memcpy(out.imag, im, (size_t)n * sizeof(double));
  return out;
}
