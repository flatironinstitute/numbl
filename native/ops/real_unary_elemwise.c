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
  // `#pragma omp simd` + `-fopenmp-simd -fno-math-errno -ffast-math`
  // lets gcc emit calls to libmvec's vector math (`_ZGVdN4v_exp`, etc.)
  // instead of scalar libm, giving 2-4x on AVX2 hardware.
  switch (op) {
    case NUMBL_UNARY_EXP:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = exp(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_LOG:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = log(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_LOG2:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = log2(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_LOG10:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = log10(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_SQRT:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = sqrt(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ABS:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = fabs(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_FLOOR:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = floor(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_CEIL:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = ceil(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ROUND:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = round(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_TRUNC:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = trunc(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_SIN:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = sin(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_COS:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = cos(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_TAN:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = tan(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ASIN:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = asin(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ACOS:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = acos(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_ATAN:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = atan(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_SINH:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = sinh(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_COSH:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = cosh(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_TANH:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = tanh(a[i]);
      return NUMBL_OK;
    case NUMBL_UNARY_SIGN:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = rsign(a[i]);
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}
