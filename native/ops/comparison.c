/**
 * Real + complex comparison ops.  Output is a logical tensor stored as
 * 0.0 / 1.0 in a double buffer (numbl stores logicals as FloatXArray with
 * an _isLogical flag on the runtime tensor).
 *
 * Caller-allocated input/output buffers; never copies.
 */

#include "numbl_ops.h"

static inline double rcmp(int op, double a, double b) {
  switch (op) {
    case NUMBL_CMP_EQ: return a == b ? 1.0 : 0.0;
    case NUMBL_CMP_NE: return a != b ? 1.0 : 0.0;
    case NUMBL_CMP_LT: return a <  b ? 1.0 : 0.0;
    case NUMBL_CMP_LE: return a <= b ? 1.0 : 0.0;
    case NUMBL_CMP_GT: return a >  b ? 1.0 : 0.0;
    case NUMBL_CMP_GE: return a >= b ? 1.0 : 0.0;
    default:           return -1.0;   /* sentinel; caller checks bad op */
  }
}

int numbl_real_comparison(int op, size_t n,
                          const double* a, const double* b,
                          double* out) {
  if (!a || !b || !out) return NUMBL_ERR_NULL_PTR;
  if (op < 0 || op > NUMBL_CMP_GE) return NUMBL_ERR_BAD_OP;
  switch (op) {
    case NUMBL_CMP_EQ:
      for (size_t i = 0; i < n; i++) out[i] = a[i] == b[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    case NUMBL_CMP_NE:
      for (size_t i = 0; i < n; i++) out[i] = a[i] != b[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    case NUMBL_CMP_LT:
      for (size_t i = 0; i < n; i++) out[i] = a[i] <  b[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    case NUMBL_CMP_LE:
      for (size_t i = 0; i < n; i++) out[i] = a[i] <= b[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    case NUMBL_CMP_GT:
      for (size_t i = 0; i < n; i++) out[i] = a[i] >  b[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    case NUMBL_CMP_GE:
      for (size_t i = 0; i < n; i++) out[i] = a[i] >= b[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}

int numbl_real_scalar_comparison(int op, size_t n,
                                 double scalar, const double* arr,
                                 int scalar_on_left, double* out) {
  if (!arr || !out) return NUMBL_ERR_NULL_PTR;
  if (op < 0 || op > NUMBL_CMP_GE) return NUMBL_ERR_BAD_OP;
  if (scalar_on_left) {
    for (size_t i = 0; i < n; i++) out[i] = rcmp(op, scalar, arr[i]);
  } else {
    for (size_t i = 0; i < n; i++) out[i] = rcmp(op, arr[i], scalar);
  }
  return NUMBL_OK;
}

int numbl_complex_comparison(int op, size_t n,
                             const double* a_re, const double* a_im,
                             const double* b_re, const double* b_im,
                             double* out) {
  if (!a_re || !b_re || !out) return NUMBL_ERR_NULL_PTR;
  switch (op) {
    case NUMBL_CMP_EQ:
      for (size_t i = 0; i < n; i++) {
        double ar = a_re[i], ai = a_im ? a_im[i] : 0.0;
        double br = b_re[i], bi = b_im ? b_im[i] : 0.0;
        out[i] = (ar == br && ai == bi) ? 1.0 : 0.0;
      }
      return NUMBL_OK;
    case NUMBL_CMP_NE:
      for (size_t i = 0; i < n; i++) {
        double ar = a_re[i], ai = a_im ? a_im[i] : 0.0;
        double br = b_re[i], bi = b_im ? b_im[i] : 0.0;
        out[i] = (ar != br || ai != bi) ? 1.0 : 0.0;
      }
      return NUMBL_OK;
    /* MATLAB semantics: <, <=, >, >= compare real parts only. */
    case NUMBL_CMP_LT:
      for (size_t i = 0; i < n; i++) out[i] = a_re[i] <  b_re[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    case NUMBL_CMP_LE:
      for (size_t i = 0; i < n; i++) out[i] = a_re[i] <= b_re[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    case NUMBL_CMP_GT:
      for (size_t i = 0; i < n; i++) out[i] = a_re[i] >  b_re[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    case NUMBL_CMP_GE:
      for (size_t i = 0; i < n; i++) out[i] = a_re[i] >= b_re[i] ? 1.0 : 0.0;
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}

int numbl_complex_scalar_comparison(int op, size_t n,
                                    double s_re, double s_im,
                                    const double* arr_re,
                                    const double* arr_im,
                                    int scalar_on_left, double* out) {
  if (!arr_re || !out) return NUMBL_ERR_NULL_PTR;
  switch (op) {
    case NUMBL_CMP_EQ:
      if (scalar_on_left) {
        for (size_t i = 0; i < n; i++) {
          double ar = arr_re[i], ai = arr_im ? arr_im[i] : 0.0;
          out[i] = (s_re == ar && s_im == ai) ? 1.0 : 0.0;
        }
      } else {
        for (size_t i = 0; i < n; i++) {
          double ar = arr_re[i], ai = arr_im ? arr_im[i] : 0.0;
          out[i] = (ar == s_re && ai == s_im) ? 1.0 : 0.0;
        }
      }
      return NUMBL_OK;
    case NUMBL_CMP_NE:
      if (scalar_on_left) {
        for (size_t i = 0; i < n; i++) {
          double ar = arr_re[i], ai = arr_im ? arr_im[i] : 0.0;
          out[i] = (s_re != ar || s_im != ai) ? 1.0 : 0.0;
        }
      } else {
        for (size_t i = 0; i < n; i++) {
          double ar = arr_re[i], ai = arr_im ? arr_im[i] : 0.0;
          out[i] = (ar != s_re || ai != s_im) ? 1.0 : 0.0;
        }
      }
      return NUMBL_OK;
    /* Real-part-only comparisons. */
    case NUMBL_CMP_LT:
    case NUMBL_CMP_LE:
    case NUMBL_CMP_GT:
    case NUMBL_CMP_GE:
      if (scalar_on_left) {
        for (size_t i = 0; i < n; i++) out[i] = rcmp(op, s_re, arr_re[i]);
      } else {
        for (size_t i = 0; i < n; i++) out[i] = rcmp(op, arr_re[i], s_re);
      }
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}
