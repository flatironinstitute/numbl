/**
 * Real element-wise binary ops (op-code dispatch).
 *
 * Caller-allocated input/output buffers; never copies.
 */

#include "numbl_ops.h"

int numbl_real_binary_elemwise(int op, size_t n,
                               const double* a,
                               const double* b,
                               double* out) {
  if (!a || !b || !out) return NUMBL_ERR_NULL_PTR;
  switch (op) {
    case NUMBL_REAL_BIN_ADD:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
      return NUMBL_OK;
    case NUMBL_REAL_BIN_SUB:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = a[i] - b[i];
      return NUMBL_OK;
    case NUMBL_REAL_BIN_MUL:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
      return NUMBL_OK;
    case NUMBL_REAL_BIN_DIV:
      #pragma omp simd
      for (size_t i = 0; i < n; i++) out[i] = a[i] / b[i];
      return NUMBL_OK;
    default:
      return NUMBL_ERR_BAD_OP;
  }
}

int numbl_real_scalar_binary_elemwise(int op, size_t n,
                                      double scalar,
                                      const double* arr,
                                      int scalar_on_left,
                                      double* out) {
  if (!arr || !out) return NUMBL_ERR_NULL_PTR;
  if (scalar_on_left) {
    switch (op) {
      case NUMBL_REAL_BIN_ADD:
        #pragma omp simd
        for (size_t i = 0; i < n; i++) out[i] = scalar + arr[i];
        return NUMBL_OK;
      case NUMBL_REAL_BIN_SUB:
        #pragma omp simd
        for (size_t i = 0; i < n; i++) out[i] = scalar - arr[i];
        return NUMBL_OK;
      case NUMBL_REAL_BIN_MUL:
        #pragma omp simd
        for (size_t i = 0; i < n; i++) out[i] = scalar * arr[i];
        return NUMBL_OK;
      case NUMBL_REAL_BIN_DIV:
        #pragma omp simd
        for (size_t i = 0; i < n; i++) out[i] = scalar / arr[i];
        return NUMBL_OK;
      default:
        return NUMBL_ERR_BAD_OP;
    }
  } else {
    switch (op) {
      case NUMBL_REAL_BIN_ADD:
        #pragma omp simd
        for (size_t i = 0; i < n; i++) out[i] = arr[i] + scalar;
        return NUMBL_OK;
      case NUMBL_REAL_BIN_SUB:
        #pragma omp simd
        for (size_t i = 0; i < n; i++) out[i] = arr[i] - scalar;
        return NUMBL_OK;
      case NUMBL_REAL_BIN_MUL:
        #pragma omp simd
        for (size_t i = 0; i < n; i++) out[i] = arr[i] * scalar;
        return NUMBL_OK;
      case NUMBL_REAL_BIN_DIV:
        #pragma omp simd
        for (size_t i = 0; i < n; i++) out[i] = arr[i] / scalar;
        return NUMBL_OK;
      default:
        return NUMBL_ERR_BAD_OP;
    }
  }
}
