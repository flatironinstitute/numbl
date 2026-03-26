#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int lapack_int;

extern void FLA_Init(void);
extern int dgemm_(
  const char *transa,
  const char *transb,
  const lapack_int *m,
  const lapack_int *n,
  const lapack_int *k,
  const double *alpha,
  const double *a,
  const lapack_int *lda,
  const double *b,
  const lapack_int *ldb,
  const double *beta,
  double *c,
  const lapack_int *ldc
);
extern int dgesv_(
  const lapack_int *n,
  const lapack_int *nrhs,
  double *a,
  const lapack_int *lda,
  lapack_int *ipiv,
  double *b,
  const lapack_int *ldb,
  lapack_int *info
);
extern int dgels_(
  const char *trans,
  const lapack_int *m,
  const lapack_int *n,
  const lapack_int *nrhs,
  double *a,
  const lapack_int *lda,
  double *b,
  const lapack_int *ldb,
  double *work,
  const lapack_int *lwork,
  lapack_int *info
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
    memcpy(dst + col * lda_dst, src + col * lda_src, rows * sizeof(double));
  }
  return 1;
}

static void ensure_flame_initialized(void) {
  static int initialized = 0;
  if (!initialized) {
    FLA_Init();
    initialized = 1;
  }
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
  const char trans = 'N';
  const double alpha = 1.0;
  const double beta = 0.0;

  ensure_flame_initialized();
  if (!to_index(m, &m_i) || !to_index(k, &k_i) || !to_index(n, &n_i)) {
    return -2;
  }
  if ((m > 0 && k > 0 && a == NULL) || (k > 0 && n > 0 && b == NULL)) {
    return -1;
  }
  if ((m > 0 && n > 0) && out == NULL) {
    return -1;
  }

  if (m == 0 || n == 0) {
    return 0;
  }

  dgemm_(
    &trans,
    &trans,
    &m_i,
    &n_i,
    &k_i,
    &alpha,
    a,
    &m_i,
    b,
    &k_i,
    &beta,
    out,
    &m_i
  );
  return 0;
}

int numbl_inv_f64(const double *data, size_t n, double *out) {
  lapack_int n_i;
  lapack_int info = 0;
  lapack_int *ipiv = NULL;
  double *a_copy = NULL;
  size_t col;

  ensure_flame_initialized();

  if (!to_index(n, &n_i)) {
    return -2;
  }
  if (n == 0) {
    return 0;
  }
  if (data == NULL || out == NULL) {
    return -1;
  }

  a_copy = (double *)malloc(n * n * sizeof(double));
  ipiv = (lapack_int *)malloc(n * sizeof(lapack_int));
  if (a_copy == NULL || ipiv == NULL) {
    free(a_copy);
    free(ipiv);
    return -3;
  }
  memcpy(a_copy, data, n * n * sizeof(double));
  memset(out, 0, n * n * sizeof(double));
  for (col = 0; col < n; ++col) {
    out[col * n + col] = 1.0;
  }

  dgesv_(&n_i, &n_i, a_copy, &n_i, ipiv, out, &n_i, &info);
  free(a_copy);
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

  ensure_flame_initialized();

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

  if (m == n) {
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

    dgesv_(&n_i, &nrhs_i, a_copy, &n_i, ipiv, b_copy, &n_i, &info);
    if (info == 0) {
      memcpy(out, b_copy, n * nrhs * sizeof(double));
    }

    free(ipiv);
    free(b_copy);
    free(a_copy);
    return info;
  }

  {
    const char trans = 'N';
    const size_t ldb = m > n ? m : n;
    lapack_int ldb_i;
    lapack_int lwork = -1;
    double work_query = 0.0;
    double *work = NULL;

    if (!to_index(ldb, &ldb_i)) {
      free(a_copy);
      return -2;
    }

    b_copy = (double *)calloc(ldb * nrhs, sizeof(double));
    if (b_copy == NULL) {
      free(a_copy);
      return -3;
    }
    if (!copy_matrix(b_copy, b, m, nrhs, ldb, m)) {
      free(b_copy);
      free(a_copy);
      return -1;
    }

    dgels_(
      &trans,
      &m_i,
      &n_i,
      &nrhs_i,
      a_copy,
      &m_i,
      b_copy,
      &ldb_i,
      &work_query,
      &lwork,
      &info
    );
    if (info != 0) {
      free(b_copy);
      free(a_copy);
      return info;
    }

    lwork = (lapack_int)(work_query > 1.0 ? work_query : 1.0);
    work = (double *)malloc((size_t)lwork * sizeof(double));
    if (work == NULL) {
      free(b_copy);
      free(a_copy);
      return -3;
    }

    dgels_(
      &trans,
      &m_i,
      &n_i,
      &nrhs_i,
      a_copy,
      &m_i,
      b_copy,
      &ldb_i,
      work,
      &lwork,
      &info
    );
    if (info == 0) {
      memcpy(out, b_copy, n * nrhs * sizeof(double));
    }

    free(work);
    free(b_copy);
    free(a_copy);
    return info;
  }
}
