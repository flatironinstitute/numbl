#include <cblas.h>

extern "C" {
void numbl_openblas_dgemm_f64(
    const double* A, int m, int k,
    const double* B, int n, double* C)
{
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, A, m, B, k, 0.0, C, m);
}
}
