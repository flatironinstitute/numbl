/**
 * Complex element-wise binary ops, split storage (op-code dispatch).
 *
 * Caller-allocated input/output buffers; never copies.
 * a_im or b_im may be NULL → treat as zero.
 * out_re and out_im are required.
 */

#include "numbl_ops.h"

/* Helper: produce JS-compatible Inf/NaN for division by zero, real part. */
static inline double cdivz_re(double r, double i) {
  if (r == 0.0 && i == 0.0) return 0.0 / 0.0;            /* NaN */
  return (r > 0 ? 1.0 : r < 0 ? -1.0 : 0.0) / 0.0;       /* ±Inf */
}

/* Imaginary-part variant of the above. */
static inline double cdivz_im(double r, double i) {
  if (r == 0.0 && i == 0.0) return 0.0;
  return (i > 0 ? 1.0 : i < 0 ? -1.0 : 0.0) / 0.0;
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
        double ar = a_re[i], ai = a_im ? a_im[i] : 0.0;
        double br = b_re[i], bi = b_im ? b_im[i] : 0.0;
        double denom = br * br + bi * bi;
        if (denom == 0.0) {
          out_re[i] = cdivz_re(ar, ai);
          out_im[i] = cdivz_im(ar, ai);
        } else {
          out_re[i] = (ar * br + ai * bi) / denom;
          out_im[i] = (ai * br - ar * bi) / denom;
        }
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
      if (scalar_on_left) {
        for (size_t i = 0; i < n; i++) {
          double ar = arr_re[i];
          double ai = arr_im ? arr_im[i] : 0.0;
          double denom = ar * ar + ai * ai;
          if (denom == 0.0) {
            out_re[i] = cdivz_re(s_re, s_im);
            out_im[i] = cdivz_im(s_re, s_im);
          } else {
            out_re[i] = (s_re * ar + s_im * ai) / denom;
            out_im[i] = (s_im * ar - s_re * ai) / denom;
          }
        }
      } else {
        double denom = s_re * s_re + s_im * s_im;
        if (denom == 0.0) {
          for (size_t i = 0; i < n; i++) {
            double ar = arr_re[i];
            double ai = arr_im ? arr_im[i] : 0.0;
            out_re[i] = cdivz_re(ar, ai);
            out_im[i] = cdivz_im(ar, ai);
          }
        } else {
          double inv_denom = 1.0 / denom;
          for (size_t i = 0; i < n; i++) {
            double ar = arr_re[i];
            double ai = arr_im ? arr_im[i] : 0.0;
            out_re[i] = (ar * s_re + ai * s_im) * inv_denom;
            out_im[i] = (ai * s_re - ar * s_im) * inv_denom;
          }
        }
      }
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}
