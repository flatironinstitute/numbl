/**
 * Complex element-wise binary ops, split storage (op-code dispatch).
 *
 * Caller-allocated input/output buffers; never copies.
 * a_im or b_im may be NULL → treat as zero.
 * out_re and out_im are required.
 */

#include "numbl_ops.h"

#include <math.h>

/* Complex division a / b, identical to src/numbl-core/ops/complexDivide.ts
 * (and the JIT's jit_cdiv): Smith's algorithm; a zero divisor gives NaN for
 * 0/0 and a signed Inf per nonzero numerator part; an infinite divisor with
 * a finite numerator gives 0 where Smith's ratio is Inf/Inf. */
static inline double cdiv_signed_inf(double x) {
  return x > 0 ? INFINITY : (x < 0 ? -INFINITY : (x == 0 ? 0.0 : NAN));
}

static inline void numbl_cdiv(double ar, double ai, double br, double bi,
                              double* out_re, double* out_im) {
  double re, im;
  if (br == 0.0 && bi == 0.0) {
    if (ar == 0.0 && ai == 0.0) {
      re = NAN;
      im = 0.0;
    } else {
      re = cdiv_signed_inf(ar);
      im = cdiv_signed_inf(ai);
    }
  } else if (fabs(br) >= fabs(bi)) {
    double r = bi / br;
    double d = br + bi * r;
    re = (ar + ai * r) / d;
    im = (ai - ar * r) / d;
  } else {
    double r = br / bi;
    double d = bi + br * r;
    re = (ar * r + ai) / d;
    im = (ai * r - ar) / d;
  }
  if (isnan(re) && isnan(im) && isinf(br) && isinf(bi) && isfinite(ar) &&
      isfinite(ai)) {
    re = 0.0;
    im = 0.0;
  }
  *out_re = re;
  *out_im = im;
}

int numbl_complex_binary_elemwise(int op, size_t n,
                                  const double* a_re, const double* a_im,
                                  const double* b_re, const double* b_im,
                                  double* out_re, double* out_im) {
  if (!a_re || !b_re || !out_re || !out_im) return NUMBL_ERR_NULL_PTR;

  switch (op) {
    case NUMBL_COMPLEX_BIN_ADD:
      for (size_t i = 0; i < n; i++) {
        out_re[i] = a_re[i] + b_re[i];
        out_im[i] = (a_im ? a_im[i] : 0.0) + (b_im ? b_im[i] : 0.0);
      }
      return NUMBL_OK;
    case NUMBL_COMPLEX_BIN_SUB:
      for (size_t i = 0; i < n; i++) {
        out_re[i] = a_re[i] - b_re[i];
        out_im[i] = (a_im ? a_im[i] : 0.0) - (b_im ? b_im[i] : 0.0);
      }
      return NUMBL_OK;
    case NUMBL_COMPLEX_BIN_MUL:
      for (size_t i = 0; i < n; i++) {
        double ar = a_re[i], ai = a_im ? a_im[i] : 0.0;
        double br = b_re[i], bi = b_im ? b_im[i] : 0.0;
        out_re[i] = ar * br - ai * bi;
        out_im[i] = ar * bi + ai * br;
      }
      return NUMBL_OK;
    case NUMBL_COMPLEX_BIN_DIV:
      for (size_t i = 0; i < n; i++) {
        numbl_cdiv(a_re[i], a_im ? a_im[i] : 0.0, b_re[i],
                   b_im ? b_im[i] : 0.0, &out_re[i], &out_im[i]);
      }
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}

int numbl_complex_scalar_binary_elemwise(int op, size_t n,
                                         double s_re, double s_im,
                                         const double* arr_re,
                                         const double* arr_im,
                                         int scalar_on_left,
                                         double* out_re, double* out_im) {
  if (!arr_re || !out_re || !out_im) return NUMBL_ERR_NULL_PTR;

  switch (op) {
    case NUMBL_COMPLEX_BIN_ADD:
      for (size_t i = 0; i < n; i++) {
        out_re[i] = s_re + arr_re[i];
        out_im[i] = s_im + (arr_im ? arr_im[i] : 0.0);
      }
      return NUMBL_OK;
    case NUMBL_COMPLEX_BIN_SUB:
      if (scalar_on_left) {
        for (size_t i = 0; i < n; i++) {
          out_re[i] = s_re - arr_re[i];
          out_im[i] = s_im - (arr_im ? arr_im[i] : 0.0);
        }
      } else {
        for (size_t i = 0; i < n; i++) {
          out_re[i] = arr_re[i] - s_re;
          out_im[i] = (arr_im ? arr_im[i] : 0.0) - s_im;
        }
      }
      return NUMBL_OK;
    case NUMBL_COMPLEX_BIN_MUL:
      for (size_t i = 0; i < n; i++) {
        double ar = arr_re[i];
        double ai = arr_im ? arr_im[i] : 0.0;
        out_re[i] = s_re * ar - s_im * ai;
        out_im[i] = s_re * ai + s_im * ar;
      }
      return NUMBL_OK;
    case NUMBL_COMPLEX_BIN_DIV:
      for (size_t i = 0; i < n; i++) {
        double ar = arr_re[i];
        double ai = arr_im ? arr_im[i] : 0.0;
        if (scalar_on_left)
          numbl_cdiv(s_re, s_im, ar, ai, &out_re[i], &out_im[i]);
        else
          numbl_cdiv(ar, ai, s_re, s_im, &out_re[i], &out_im[i]);
      }
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}
