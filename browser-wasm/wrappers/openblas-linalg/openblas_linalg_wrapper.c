#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lapacke.h"

enum {
  NUMBL_CBLAS_COL_MAJOR = 102,
  NUMBL_CBLAS_NO_TRANS = 111
};

extern void cblas_dgemm(
  int order,
  int transa,
  int transb,
  int m,
  int n,
  int k,
  double alpha,
  const double *a,
  int lda,
  const double *b,
  int ldb,
  double beta,
  double *c,
  int ldc
);

static int to_index(size_t value, lapack_int *out) {
  if (value > (size_t)INT_MAX) {
    return 0;
  }
  *out = (lapack_int)value;
  return 1;
}

static int copy_matrix(
  double *dst,
  const double *src,
  size_t rows,
  size_t cols,
  size_t lda_dst,
  size_t lda_src
) {
  size_t col;
  if (rows == 0 || cols == 0) {
    return 1;
  }
  if (dst == NULL || src == NULL) {
    return 0;
  }
  for (col = 0; col < cols; ++col) {
    memcpy(
      dst + col * lda_dst,
      src + col * lda_src,
      rows * sizeof(double)
    );
  }
  return 1;
}

int numbl_matmul_f64(
  const double *a,
  size_t m,
  size_t k,
  const double *b,
  size_t n,
  double *out
) {
  lapack_int m_i;
  lapack_int k_i;
  lapack_int n_i;

  if (!to_index(m, &m_i) || !to_index(k, &k_i) || !to_index(n, &n_i)) {
    return -2;
  }
  if ((m > 0 && k > 0 && a == NULL) || (k > 0 && n > 0 && b == NULL)) {
    return -1;
  }
  if ((m > 0 && n > 0) && out == NULL) {
    return -1;
  }

  cblas_dgemm(
    NUMBL_CBLAS_COL_MAJOR,
    NUMBL_CBLAS_NO_TRANS,
    NUMBL_CBLAS_NO_TRANS,
    m_i,
    n_i,
    k_i,
    1.0,
    a,
    m_i,
    b,
    k_i,
    0.0,
    out,
    m_i
  );
  return 0;
}

int numbl_inv_f64(const double *data, size_t n, double *out) {
  lapack_int n_i;
  lapack_int info;
  lapack_int lwork = -1;
  lapack_int *ipiv = NULL;
  double work_query = 0.0;
  double *work = NULL;

  if (!to_index(n, &n_i)) {
    return -2;
  }
  if (n == 0) {
    return 0;
  }
  if (data == NULL || out == NULL) {
    return -1;
  }

  memcpy(out, data, n * n * sizeof(double));
  ipiv = (lapack_int *)malloc(n * sizeof(lapack_int));
  if (ipiv == NULL) {
    return -3;
  }

  info = LAPACKE_dgetrf_work(LAPACK_COL_MAJOR, n_i, n_i, out, n_i, ipiv);
  if (info == 0) {
    info = LAPACKE_dgetri_work(
      LAPACK_COL_MAJOR,
      n_i,
      out,
      n_i,
      ipiv,
      &work_query,
      lwork
    );
  }
  if (info == 0) {
    lwork = (lapack_int)(work_query > 1.0 ? work_query : 1.0);
    work = (double *)malloc((size_t)lwork * sizeof(double));
    if (work == NULL) {
      free(ipiv);
      return -3;
    }
    info = LAPACKE_dgetri_work(
      LAPACK_COL_MAJOR,
      n_i,
      out,
      n_i,
      ipiv,
      work,
      lwork
    );
  }

  free(work);
  free(ipiv);
  return info;
}

int numbl_linsolve_f64(
  const double *a,
  size_t m,
  size_t n,
  const double *b,
  size_t nrhs,
  double *out
) {
  lapack_int m_i;
  lapack_int n_i;
  lapack_int nrhs_i;
  lapack_int info = 0;
  double *a_copy = NULL;
  double *b_copy = NULL;

  if (!to_index(m, &m_i) || !to_index(n, &n_i) || !to_index(nrhs, &nrhs_i)) {
    return -2;
  }
  if ((m > 0 && n > 0 && a == NULL) || (m > 0 && nrhs > 0 && b == NULL)) {
    return -1;
  }
  if ((n > 0 && nrhs > 0) && out == NULL) {
    return -1;
  }

  a_copy = (double *)malloc(m * n * sizeof(double));
  if (a_copy == NULL) {
    return -3;
  }
  if (!copy_matrix(a_copy, a, m, n, m, m)) {
    free(a_copy);
    return -1;
  }

  if (m != n) {
    free(a_copy);
    return -4;
  }

  {
    lapack_int *ipiv = NULL;

    b_copy = (double *)malloc(n * nrhs * sizeof(double));
    ipiv = (lapack_int *)malloc(n * sizeof(lapack_int));
    if (b_copy == NULL || ipiv == NULL) {
      free(a_copy);
      free(b_copy);
      free(ipiv);
      return -3;
    }
    memcpy(b_copy, b, n * nrhs * sizeof(double));

    info = LAPACKE_dgesv_work(
      LAPACK_COL_MAJOR,
      n_i,
      nrhs_i,
      a_copy,
      n_i,
      ipiv,
      b_copy,
      n_i
    );
    if (info == 0) {
      memcpy(out, b_copy, n * nrhs * sizeof(double));
    }

    free(ipiv);
    free(b_copy);
    free(a_copy);
    return info;
  }
}
