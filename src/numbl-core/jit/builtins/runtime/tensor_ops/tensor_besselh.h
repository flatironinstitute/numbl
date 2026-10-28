/* @snippet name=jit_tensor_besselh deps=jit_tensor_t,jit_alloc */
/* JIT runtime helpers: Hankel function of the first kind for
 * integer order, real positive argument.
 *
 * `H_n^(1)(x) = J_n(x) + i * Y_n(x)` for x > 0. We route through
 * POSIX `j0` / `j1` / `y0` / `y1` (already available via <math.h>
 * with _XOPEN_SOURCE; widely supported on Linux + macOS). For
 * tensor inputs we walk every element and write the result to a
 * fresh complex tensor.
 *
 * The JIT's `besselh` builtin enforces at lowering time that `nu`
 * and `kind` are exact integers and that `kind == 1` and
 * `nu in {0, 1}`. Other cases are deferred.
 *
 * Scalar entries return `double _Complex`; tensor entries return
 * a freshly-owned `jit_tensor_t` with both lanes allocated.
 */

#include <math.h>
#include <complex.h>

static double _Complex jit_besselh0_scalar(double x) {
  return j0(x) + I * y0(x);
}

static double _Complex jit_besselh1_scalar(double x) {
  return j1(x) + I * y1(x);
}

static jit_tensor_t jit_tensor_besselh0(jit_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t out;
  out.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) out.dims[i] = a.dims[i];
  out.real = jit_alloc((size_t)n * sizeof(double));
  out.imag = jit_alloc((size_t)n * sizeof(double));
  for (long i = 0; i < n; i++) {
    double x = a.real[i];
    out.real[i] = j0(x);
    out.imag[i] = y0(x);
  }
  return out;
}

static jit_tensor_t jit_tensor_besselh1(jit_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t out;
  out.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) out.dims[i] = a.dims[i];
  out.real = jit_alloc((size_t)n * sizeof(double));
  out.imag = jit_alloc((size_t)n * sizeof(double));
  for (long i = 0; i < n; i++) {
    double x = a.real[i];
    out.real[i] = j1(x);
    out.imag[i] = y1(x);
  }
  return out;
}
