/**
 * numbl_jit_runtime — see jit_runtime.h for the shape and invariants.
 */

#include "jit_runtime.h"
#include "numbl_ops.h"

#include <math.h>
#include <string.h>
#include <time.h>

int numbl_jit_rt_version(void) {
  return NUMBL_JIT_RT_VERSION;
}

double numbl_idx1r(const double* data, size_t len, double i, double* err_flag) {
  /* Truncation-to-zero via int64 cast mirrors JS-JIT's `(i - 1) | 0`.
   * Unsigned compare catches both negative idx and idx >= len in one
   * branch. */
  int64_t idx = (int64_t)(i - 1.0);
  if ((uint64_t)idx >= (uint64_t)len) {
    *err_flag = 1.0;
    return 0.0;
  }
  return data[idx];
}

void numbl_set1r_h(double* data, size_t len, double i, double v,
                   double* err_flag) {
  int64_t idx = (int64_t)(i - 1.0);
  if ((uint64_t)idx >= (uint64_t)len) {
    /* 2.0 = "growth needed → soft-bail to interpreter", distinct from
     * 1.0 which the JS wrapper translates into a hard bounds error. */
    *err_flag = 2.0;
    return;
  }
  data[idx] = v;
}

double numbl_idx2r(const double* data, size_t len, size_t d0,
                   double i, double j, double* err_flag) {
  int64_t r = (int64_t)(i - 1.0);
  int64_t c = (int64_t)(j - 1.0);
  /* d0 == 0 would mean an empty tensor — any index is OOB. */
  if (d0 == 0 || (uint64_t)r >= (uint64_t)d0) {
    *err_flag = 1.0;
    return 0.0;
  }
  size_t cols = len / d0;
  if ((uint64_t)c >= (uint64_t)cols) {
    *err_flag = 1.0;
    return 0.0;
  }
  return data[(size_t)c * d0 + (size_t)r];
}

double numbl_idx3r(const double* data, size_t len, size_t d0, size_t d1,
                   double i, double j, double k, double* err_flag) {
  int64_t k0 = (int64_t)(i - 1.0);
  int64_t k1 = (int64_t)(j - 1.0);
  int64_t k2 = (int64_t)(k - 1.0);
  if (d0 == 0 || d1 == 0 ||
      (uint64_t)k0 >= (uint64_t)d0 ||
      (uint64_t)k1 >= (uint64_t)d1) {
    *err_flag = 1.0;
    return 0.0;
  }
  size_t plane = d0 * d1;
  size_t d2 = len / plane;
  if ((uint64_t)k2 >= (uint64_t)d2) {
    *err_flag = 1.0;
    return 0.0;
  }
  return data[(size_t)k2 * plane + (size_t)k1 * d0 + (size_t)k0];
}

void numbl_set2r_h(double* data, size_t len, size_t d0,
                   double i, double j, double v, double* err_flag) {
  int64_t r = (int64_t)(i - 1.0);
  int64_t c = (int64_t)(j - 1.0);
  if (d0 == 0) {
    *err_flag = 2.0;
    return;
  }
  size_t cols = len / d0;
  if ((uint64_t)r >= (uint64_t)d0 || (uint64_t)c >= (uint64_t)cols) {
    *err_flag = 2.0;
    return;
  }
  data[(size_t)c * d0 + (size_t)r] = v;
}

void numbl_set3r_h(double* data, size_t len, size_t d0, size_t d1,
                   double i, double j, double k, double v,
                   double* err_flag) {
  int64_t k0 = (int64_t)(i - 1.0);
  int64_t k1 = (int64_t)(j - 1.0);
  int64_t k2 = (int64_t)(k - 1.0);
  if (d0 == 0 || d1 == 0) {
    *err_flag = 2.0;
    return;
  }
  size_t plane = d0 * d1;
  size_t d2 = len / plane;
  if ((uint64_t)k0 >= (uint64_t)d0 ||
      (uint64_t)k1 >= (uint64_t)d1 ||
      (uint64_t)k2 >= (uint64_t)d2) {
    *err_flag = 2.0;
    return;
  }
  data[(size_t)k2 * plane + (size_t)k1 * d0 + (size_t)k0] = v;
}

void numbl_setRange1r_h(double* dstData, size_t dstLen,
                        double dstStart, double dstEnd,
                        const double* srcData, size_t srcLen,
                        double srcStart, double srcEnd,
                        double* err_flag) {
  int64_t dS = (int64_t)(dstStart - 1.0);
  int64_t dE = (int64_t)(dstEnd - 1.0);
  int64_t sS = (int64_t)(srcStart - 1.0);
  int64_t sE = (int64_t)(srcEnd - 1.0);
  int64_t dN = dE - dS + 1;
  int64_t sN = sE - sS + 1;
  if (dN != sN) {
    /* 3.0 — length-mismatch error; JS wrapper translates to MATLAB's
     * "Unable to perform assignment..." message. */
    *err_flag = 3.0;
    return;
  }
  if (dN <= 0) return;
  if ((uint64_t)dS >= (uint64_t)dstLen ||
      (uint64_t)dE >= (uint64_t)dstLen ||
      (uint64_t)sS >= (uint64_t)srcLen ||
      (uint64_t)sE >= (uint64_t)srcLen) {
    *err_flag = 1.0;
    return;
  }
  /* memmove handles overlap when src and dst alias. */
  memmove(dstData + (size_t)dS, srcData + (size_t)sS,
          (size_t)dN * sizeof(double));
}

void numbl_setCol2r_h(double* dstData, size_t dstRows, size_t dstLen,
                      double col,
                      const double* srcData, size_t srcLen,
                      double* err_flag) {
  if (srcLen != dstRows) {
    *err_flag = 3.0;
    return;
  }
  int64_t j = (int64_t)(col - 1.0);
  if (j < 0) {
    *err_flag = 1.0;
    return;
  }
  if (dstRows == 0) {
    /* Empty dst with a nonempty src: growth territory — soft-bail. */
    *err_flag = 2.0;
    return;
  }
  size_t dstCols = dstLen / dstRows;
  if ((uint64_t)j >= (uint64_t)dstCols) {
    /* Growth on write — the interpreter handles growing dst; JS-JIT
     * mirrors this with JitBailToInterpreter. */
    *err_flag = 2.0;
    return;
  }
  memcpy(dstData + (size_t)j * dstRows, srcData,
         dstRows * sizeof(double));
}

void numbl_copyRange1r(const double* srcData, size_t srcLen,
                       double start, double end,
                       double* dstData,
                       double* err_flag) {
  int64_t s = (int64_t)(start - 1.0);
  int64_t e = (int64_t)(end - 1.0);
  int64_t n = e - s + 1;
  if (n <= 0) return;
  if ((uint64_t)s >= (uint64_t)srcLen ||
      (uint64_t)e >= (uint64_t)srcLen) {
    *err_flag = 1.0;
    return;
  }
  memcpy(dstData, srcData + (size_t)s, (size_t)n * sizeof(double));
}

double numbl_mod(double a, double b) {
  if (b == 0.0) return a;
  double r = fmod(a, b);
  if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
  return r;
}

double numbl_sign(double x) {
  if (x > 0.0) return 1.0;
  if (x < 0.0) return -1.0;
  return 0.0;
}

double numbl_reduce_flat(int op, const double* data, int64_t len) {
  double out = 0.0;
  numbl_real_flat_reduce(op, (size_t)len, data, &out);
  return out;
}

double numbl_monotonic_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

double numbl_tic(double* state) {
  double t = numbl_monotonic_time();
  *state = t;
  return t;
}

double numbl_toc(const double* state) {
  return numbl_monotonic_time() - *state;
}
