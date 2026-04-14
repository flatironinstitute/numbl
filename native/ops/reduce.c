/**
 * Flat reductions: reduce an entire buffer to a single value.
 * Caller-allocated input; 1-double output.
 */

#include "numbl_ops.h"

#include <math.h>

int numbl_real_flat_reduce(int op, size_t n, const double* a, double* out) {
  if ((!a && n > 0) || !out) return NUMBL_ERR_NULL_PTR;

  switch (op) {
    case NUMBL_REDUCE_SUM: {
      double s = 0.0;
      for (size_t i = 0; i < n; i++) s += a[i];
      *out = s;
      return NUMBL_OK;
    }
    case NUMBL_REDUCE_PROD: {
      double p = 1.0;
      for (size_t i = 0; i < n; i++) p *= a[i];
      *out = p;
      return NUMBL_OK;
    }
    case NUMBL_REDUCE_MAX: {
      /* MATLAB default: omitnan.  If ALL values are NaN, result is NaN. */
      double m = -INFINITY;
      int any = 0;
      for (size_t i = 0; i < n; i++) {
        double v = a[i];
        if (isnan(v)) continue;
        if (v > m) m = v;
        any = 1;
      }
      *out = any ? m : (0.0 / 0.0);
      return NUMBL_OK;
    }
    case NUMBL_REDUCE_MIN: {
      double m = INFINITY;
      int any = 0;
      for (size_t i = 0; i < n; i++) {
        double v = a[i];
        if (isnan(v)) continue;
        if (v < m) m = v;
        any = 1;
      }
      *out = any ? m : (0.0 / 0.0);
      return NUMBL_OK;
    }
    case NUMBL_REDUCE_ANY: {
      double r = 0.0;
      for (size_t i = 0; i < n; i++) {
        if (a[i] != 0.0 || isnan(a[i])) { r = 1.0; break; }
      }
      *out = r;
      return NUMBL_OK;
    }
    case NUMBL_REDUCE_ALL: {
      double r = 1.0;
      for (size_t i = 0; i < n; i++) {
        if (a[i] == 0.0) { r = 0.0; break; }
      }
      *out = r;
      return NUMBL_OK;
    }
    case NUMBL_REDUCE_MEAN: {
      if (n == 0) { *out = 0.0 / 0.0; return NUMBL_OK; }
      double s = 0.0;
      for (size_t i = 0; i < n; i++) s += a[i];
      *out = s / (double)n;
      return NUMBL_OK;
    }
    default:
      return NUMBL_ERR_BAD_OP;
  }
}

int numbl_complex_flat_reduce(int op, size_t n,
                              const double* a_re, const double* a_im,
                              double* out_re, double* out_im) {
  if ((!a_re && n > 0) || !out_re) return NUMBL_ERR_NULL_PTR;

  switch (op) {
    case NUMBL_REDUCE_SUM: {
      if (!out_im) return NUMBL_ERR_NULL_PTR;
      double sr = 0.0, si = 0.0;
      for (size_t i = 0; i < n; i++) {
        sr += a_re[i];
        if (a_im) si += a_im[i];
      }
      *out_re = sr;
      *out_im = si;
      return NUMBL_OK;
    }
    case NUMBL_REDUCE_PROD: {
      if (!out_im) return NUMBL_ERR_NULL_PTR;
      /* Accumulate Gauss multiplication: (accRe + accIm i)(ar + ai i). */
      double ar_acc = 1.0, ai_acc = 0.0;
      for (size_t i = 0; i < n; i++) {
        double ar = a_re[i];
        double ai = a_im ? a_im[i] : 0.0;
        double nr = ar_acc * ar - ai_acc * ai;
        double ni = ar_acc * ai + ai_acc * ar;
        ar_acc = nr;
        ai_acc = ni;
      }
      *out_re = ar_acc;
      *out_im = ai_acc;
      return NUMBL_OK;
    }
    case NUMBL_REDUCE_ANY: {
      double r = 0.0;
      for (size_t i = 0; i < n; i++) {
        double ar = a_re[i];
        double ai = a_im ? a_im[i] : 0.0;
        if (ar != 0.0 || ai != 0.0 || isnan(ar) || isnan(ai)) {
          r = 1.0;
          break;
        }
      }
      *out_re = r;
      return NUMBL_OK;
    }
    case NUMBL_REDUCE_ALL: {
      double r = 1.0;
      for (size_t i = 0; i < n; i++) {
        double ar = a_re[i];
        double ai = a_im ? a_im[i] : 0.0;
        if (ar == 0.0 && ai == 0.0) {
          r = 0.0;
          break;
        }
      }
      *out_re = r;
      return NUMBL_OK;
    }
    default:
      /* MAX/MIN/MEAN on complex: ambiguous; caller handles. */
      return NUMBL_ERR_BAD_OP;
  }
}
