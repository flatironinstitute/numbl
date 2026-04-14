/**
 * Complex unary element-wise ops (op-code dispatch), split storage.
 *
 * Caller-allocated input/output buffers; never copies.
 * a_im may be NULL → treat as zero.
 * ABS is intentionally not supported here (output is real-valued; use
 * numbl_complex_abs instead).
 */

#include "numbl_ops.h"

#include <complex.h>
#include <math.h>

/* Construct a double complex from split parts. */
static inline double _Complex cpack(double re, double im) {
  return re + im * I;
}

#define WRITE_C(out_re_ptr, out_im_ptr, i, z) do {           \
  double _Complex _z = (z);                                  \
  (out_re_ptr)[i] = creal(_z);                               \
  (out_im_ptr)[i] = cimag(_z);                               \
} while (0)

/* complex sign: z/|z| for z != 0, else 0.  Matches MATLAB's sign(z). */
static inline double _Complex csign(double _Complex z) {
  double m = cabs(z);
  if (m == 0.0) return 0.0;
  return z / m;
}

int numbl_complex_unary_elemwise(int op, size_t n,
                                 const double* a_re, const double* a_im,
                                 double* out_re, double* out_im) {
  if (!a_re || !out_re || !out_im) return NUMBL_ERR_NULL_PTR;

  switch (op) {
    case NUMBL_UNARY_EXP:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, cexp(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_LOG:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, clog(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_LOG2: {
      const double inv_ln2 = 1.0 / log(2.0);
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i,
                clog(cpack(a_re[i], a_im ? a_im[i] : 0.0)) * inv_ln2);
      return NUMBL_OK;
    }
    case NUMBL_UNARY_LOG10: {
      const double inv_ln10 = 1.0 / log(10.0);
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i,
                clog(cpack(a_re[i], a_im ? a_im[i] : 0.0)) * inv_ln10);
      return NUMBL_OK;
    }
    case NUMBL_UNARY_SQRT:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, csqrt(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_ABS:
      return NUMBL_ERR_BAD_OP; /* use numbl_complex_abs */
    case NUMBL_UNARY_FLOOR:
      for (size_t i = 0; i < n; i++) {
        out_re[i] = floor(a_re[i]);
        out_im[i] = a_im ? floor(a_im[i]) : 0.0;
      }
      return NUMBL_OK;
    case NUMBL_UNARY_CEIL:
      for (size_t i = 0; i < n; i++) {
        out_re[i] = ceil(a_re[i]);
        out_im[i] = a_im ? ceil(a_im[i]) : 0.0;
      }
      return NUMBL_OK;
    case NUMBL_UNARY_ROUND:
      for (size_t i = 0; i < n; i++) {
        out_re[i] = round(a_re[i]);
        out_im[i] = a_im ? round(a_im[i]) : 0.0;
      }
      return NUMBL_OK;
    case NUMBL_UNARY_TRUNC:
      for (size_t i = 0; i < n; i++) {
        out_re[i] = trunc(a_re[i]);
        out_im[i] = a_im ? trunc(a_im[i]) : 0.0;
      }
      return NUMBL_OK;
    case NUMBL_UNARY_SIN:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, csin(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_COS:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, ccos(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_TAN:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, ctan(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_ASIN:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, casin(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_ACOS:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, cacos(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_ATAN:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, catan(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_SINH:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, csinh(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_COSH:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, ccosh(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_TANH:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i, ctanh(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    case NUMBL_UNARY_SIGN:
      for (size_t i = 0; i < n; i++)
        WRITE_C(out_re, out_im, i,
                csign(cpack(a_re[i], a_im ? a_im[i] : 0.0)));
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}

int numbl_complex_abs(size_t n,
                      const double* a_re, const double* a_im,
                      double* out) {
  if (!a_re || !out) return NUMBL_ERR_NULL_PTR;
  if (a_im) {
    for (size_t i = 0; i < n; i++) {
      /* hypot avoids overflow/underflow that (re*re + im*im) would miss. */
      out[i] = hypot(a_re[i], a_im[i]);
    }
  } else {
    for (size_t i = 0; i < n; i++) out[i] = fabs(a_re[i]);
  }
  return NUMBL_OK;
}
