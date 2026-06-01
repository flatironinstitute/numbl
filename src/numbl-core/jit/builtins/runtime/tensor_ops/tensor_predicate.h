/* mtoc2 runtime helpers: elementwise tensor → logical-tensor
 * predicates (`isnan`, `isinf`, `isfinite`, `logical`). Same
 * allocate-and-fill shape as `tensor_unary_real_math.h`; each element
 * maps to 1.0 / 0.0. The result is logical at the source level (the
 * buffer is the usual Float64 lane, `imag == NULL`).
 *
 * The `_complex` siblings operate per-element on `(re, im)`; for
 * `isnan` / `isinf` the predicate fires if either component
 * triggers, for `isfinite` both must be finite. They tolerate
 * `a.imag == NULL` (a real tensor that flowed through a
 * complex-typed route) by treating the imag input as zero.
 */
#include <math.h>
#include <stdlib.h>

#define MTOC2_DEFINE_UNARY_PRED(name, EXPR)              \
  static mtoc2_tensor_t name(mtoc2_tensor_t a) {         \
    long n = 1;                                          \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];     \
    mtoc2_tensor_t r;                                    \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));    \
    r.imag = NULL;                                       \
    r.ndim = a.ndim;                                     \
    for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i]; \
    MTOC2_OMP_PARFOR_N                                   \
    for (long i = 0; i < n; i++) {                       \
      double x = a.real[i];                              \
      r.real[i] = (EXPR) ? 1.0 : 0.0;                    \
    }                                                    \
    return r;                                            \
  }

MTOC2_DEFINE_UNARY_PRED(mtoc2_tensor_isnan, isnan(x))
MTOC2_DEFINE_UNARY_PRED(mtoc2_tensor_logical, x != 0.0)
MTOC2_DEFINE_UNARY_PRED(mtoc2_tensor_isinf, isinf(x))
MTOC2_DEFINE_UNARY_PRED(mtoc2_tensor_isfinite, isfinite(x))

#undef MTOC2_DEFINE_UNARY_PRED

#define MTOC2_DEFINE_UNARY_PRED_COMPLEX(name, EXPR)         \
  static mtoc2_tensor_t name(mtoc2_tensor_t a) {            \
    long n = 1;                                             \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];        \
    mtoc2_tensor_t r;                                       \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));       \
    r.imag = NULL;                                          \
    r.ndim = a.ndim;                                        \
    for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i]; \
    MTOC2_OMP_PARFOR_N                                      \
    for (long i = 0; i < n; i++) {                          \
      double re = a.real[i];                                \
      double im = (a.imag != NULL) ? a.imag[i] : 0.0;       \
      r.real[i] = (EXPR) ? 1.0 : 0.0;                       \
    }                                                       \
    return r;                                               \
  }

MTOC2_DEFINE_UNARY_PRED_COMPLEX(mtoc2_tensor_isnan_complex,
                                isnan(re) || isnan(im))
MTOC2_DEFINE_UNARY_PRED_COMPLEX(mtoc2_tensor_isinf_complex,
                                isinf(re) || isinf(im))
MTOC2_DEFINE_UNARY_PRED_COMPLEX(mtoc2_tensor_isfinite_complex,
                                isfinite(re) && isfinite(im))

#undef MTOC2_DEFINE_UNARY_PRED_COMPLEX
