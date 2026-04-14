/**
 * Real unary element-wise ops (op-code dispatch).
 * Caller-allocated input/output buffers; never copies.
 */

#include "numbl_ops.h"

#include <math.h>

static inline double rsign(double x) {
  return x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : 0.0;
}

int numbl_real_unary_elemwise(int op, size_t n,
                              const double* a, double* out) {
  if (!a || !out) return NUMBL_ERR_NULL_PTR;
  switch (op) {
    case NUMBL_UNARY_EXP:
      for (size_t i = 0; i < n; i++) out[i] = exp(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_LOG:
      for (size_t i = 0; i < n; i++) out[i] = log(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_LOG2:
      for (size_t i = 0; i < n; i++) out[i] = log2(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_LOG10:
      for (size_t i = 0; i < n; i++) out[i] = log10(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_SQRT:
      for (size_t i = 0; i < n; i++) out[i] = sqrt(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ABS:
      for (size_t i = 0; i < n; i++) out[i] = fabs(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_FLOOR:
      for (size_t i = 0; i < n; i++) out[i] = floor(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_CEIL:
      for (size_t i = 0; i < n; i++) out[i] = ceil(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ROUND:
      for (size_t i = 0; i < n; i++) out[i] = round(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_TRUNC:
      for (size_t i = 0; i < n; i++) out[i] = trunc(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_SIN:
      for (size_t i = 0; i < n; i++) out[i] = sin(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_COS:
      for (size_t i = 0; i < n; i++) out[i] = cos(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_TAN:
      for (size_t i = 0; i < n; i++) out[i] = tan(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ASIN:
      for (size_t i = 0; i < n; i++) out[i] = asin(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ACOS:
      for (size_t i = 0; i < n; i++) out[i] = acos(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ATAN:
      for (size_t i = 0; i < n; i++) out[i] = atan(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_SINH:
      for (size_t i = 0; i < n; i++) out[i] = sinh(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_COSH:
      for (size_t i = 0; i < n; i++) out[i] = cosh(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_TANH:
      for (size_t i = 0; i < n; i++) out[i] = tanh(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_SIGN:
      for (size_t i = 0; i < n; i++) out[i] = rsign(a[i]);
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}
