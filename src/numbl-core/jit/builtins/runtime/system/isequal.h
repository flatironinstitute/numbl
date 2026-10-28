/* @snippet name=jit_isequal deps=jit_tensor_t */
/* JIT runtime helpers for `isequal` over numeric values.
 *
 * `emitC` rejects char / string, struct and cell argument combinations
 * with `UnsupportedConstruct`. These two helpers cover tensor-vs-tensor
 * and scalar-vs-tensor, real or complex.
 *
 * Equality follows numbl's `valuesEqualSimple` / `tensorsEqual`:
 * same rank, same per-axis extent, element-for-element `==`
 * (so `NaN != NaN`, matching MATLAB / numbl). As in MATLAB, only values
 * count: a missing imaginary part (`imag == NULL`) compares as zeros.
 */

/* Two tensors: equal iff identical shape and data. */
static double jit_isequal_tt(jit_tensor_t a, jit_tensor_t b) {
  if (a.ndim != b.ndim) return 0.0;
  long n = 1;
  for (int i = 0; i < a.ndim; i++) {
    if (a.dims[i] != b.dims[i]) return 0.0;
    n *= a.dims[i];
  }
  for (long i = 0; i < n; i++) {
    if (a.real[i] != b.real[i]) return 0.0;
  }
  if (a.imag || b.imag) {
    for (long i = 0; i < n; i++) {
      if ((a.imag ? a.imag[i] : 0.0) != (b.imag ? b.imag[i] : 0.0)) return 0.0;
    }
  }
  return 1.0;
}

/* Scalar `re + im*i` vs tensor: equal iff the tensor holds exactly one
 * element equal to it (numbl treats a scalar as a 1×1 tensor). */
static double jit_isequal_st(double re, double im, jit_tensor_t t) {
  long n = 1;
  for (int i = 0; i < t.ndim; i++) n *= t.dims[i];
  if (n != 1) return 0.0;
  return t.real[0] == re && (t.imag ? t.imag[0] : 0.0) == im ? 1.0 : 0.0;
}
