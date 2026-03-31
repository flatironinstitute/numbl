#include <Eigen/Dense>

extern "C" {
void numbl_eigen_dgemm_f64(
    const double* A, int m, int k,
    const double* B, int n, double* C)
{
    Eigen::Map<const Eigen::MatrixXd> eA(A, m, k);
    Eigen::Map<const Eigen::MatrixXd> eB(B, k, n);
    Eigen::Map<Eigen::MatrixXd> eC(C, m, n);
    eC.noalias() = eA * eB;
}
}
