/* mtoc2 runtime helper: true (1.0) when a tensor carries no imaginary
 * content — NULL imag lane, or every imag element exactly zero. `isreal`
 * uses this for complex-typed tensors the JIT could not prove real at
 * compile time, reporting realness by value (matching the interpreter
 * and the complex-scalar `cimag(z) == 0` rule). Returns a logical double. */

static double mtoc2_tensor_imag_all_zero(mtoc2_tensor_t a) {
  if (a.imag == NULL) return 1.0;
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  for (long i = 0; i < n; i++) {
    if (a.imag[i] != 0.0) return 0.0;
  }
  return 1.0;
}
