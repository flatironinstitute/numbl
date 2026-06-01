/* mtoc2 runtime helpers: vector norms.
 *
 * `mtoc2_norm2_real(a)` / `mtoc2_norm2_complex(a)` — the 2-norm.
 * Kept as a separate, simpler path because it's by far the most
 * common case (and the codegen path doesn't need the per-element
 * `pow` of the general helper).
 *
 * `mtoc2_norm_p_real(a, p)` / `mtoc2_norm_p_complex(a, p)` — the
 * `norm(v, p)` form. `p` is a double whose value carries the order:
 *   - finite p>0 → `(sum |x|^p)^(1/p)`
 *   - +Inf       → `max |x|`
 *   - -Inf       → `min |x|`
 * 1-norm and 2-norm are handled by inline fast paths inside the
 * generic helper (avoid the `pow` rounding error and let the C
 * compiler vectorize the simple loop). All variants treat `|x|`
 * as the absolute value of a real, or `hypot(re, im)` for complex.
 *
 * Empty input (numel == 0) returns 0 for every order, matching
 * MATLAB / numbl.
 */

#include <math.h>

static double mtoc2_norm2_real(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  double acc = 0.0;
  for (long i = 0; i < n; i++) {
    double x = a.real[i];
    acc += x * x;
  }
  return sqrt(acc);
}

static double mtoc2_norm2_complex(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  double acc = 0.0;
  for (long i = 0; i < n; i++) {
    double re = a.real[i];
    double im = a.imag[i];
    acc += re * re + im * im;
  }
  return sqrt(acc);
}

static double mtoc2_norm_p_real(mtoc2_tensor_t a, double p) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  if (n == 0) return 0.0;
  if (isinf(p) && p > 0) {
    double m = 0.0;
    for (long i = 0; i < n; i++) {
      double x = fabs(a.real[i]);
      if (x > m) m = x;
    }
    return m;
  }
  if (isinf(p) && p < 0) {
    double m = fabs(a.real[0]);
    for (long i = 1; i < n; i++) {
      double x = fabs(a.real[i]);
      if (x < m) m = x;
    }
    return m;
  }
  if (p == 1.0) {
    double acc = 0.0;
    for (long i = 0; i < n; i++) acc += fabs(a.real[i]);
    return acc;
  }
  if (p == 2.0) return mtoc2_norm2_real(a);
  double acc = 0.0;
  for (long i = 0; i < n; i++) acc += pow(fabs(a.real[i]), p);
  return pow(acc, 1.0 / p);
}

static double mtoc2_norm_p_complex(mtoc2_tensor_t a, double p) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  if (n == 0) return 0.0;
  if (isinf(p) && p > 0) {
    double m = 0.0;
    for (long i = 0; i < n; i++) {
      double x = hypot(a.real[i], a.imag[i]);
      if (x > m) m = x;
    }
    return m;
  }
  if (isinf(p) && p < 0) {
    double m = hypot(a.real[0], a.imag[0]);
    for (long i = 1; i < n; i++) {
      double x = hypot(a.real[i], a.imag[i]);
      if (x < m) m = x;
    }
    return m;
  }
  if (p == 2.0) return mtoc2_norm2_complex(a);
  if (p == 1.0) {
    double acc = 0.0;
    for (long i = 0; i < n; i++) acc += hypot(a.real[i], a.imag[i]);
    return acc;
  }
  double acc = 0.0;
  for (long i = 0; i < n; i++) acc += pow(hypot(a.real[i], a.imag[i]), p);
  return pow(acc, 1.0 / p);
}
