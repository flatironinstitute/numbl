/**
 * Common includes, type definitions, LAPACK/BLAS declarations, and function
 * prototypes shared across the numbl_addon source files.
 */

#pragma once

#include <napi.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// ── Complex number type used by LAPACK (interleaved real and imaginary parts) ─

struct lapack_complex_double {
  double real;
  double imag;
};

// ── LAPACK/BLAS declarations (Fortran ABI — all args passed by pointer) ───────

extern "C" {
  // ── Real matrix inversion ─────────────────────────────────────────────────
  // LU factorisation: A = P * L * U
  void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
  // Matrix inversion using LU factors produced by dgetrf
  void dgetri_(int* n, double* a, int* lda, int* ipiv,
               double* work, int* lwork, int* info);

  // ── Complex matrix inversion ──────────────────────────────────────────────
  // LU factorisation for complex matrices: A = P * L * U
  void zgetrf_(int* m, int* n, lapack_complex_double* a, int* lda,
               int* ipiv, int* info);
  // Complex matrix inversion using LU factors produced by zgetrf
  void zgetri_(int* n, lapack_complex_double* a, int* lda, int* ipiv,
               lapack_complex_double* work, int* lwork, int* info);

  // ── QR factorisation ──────────────────────────────────────────────────────
  // Compute QR factorisation of a general m×n matrix: A = Q * R
  void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau,
               double* work, int* lwork, int* info);
  // Generate the m×n (or m×m) orthogonal matrix Q from dgeqrf reflectors
  void dorgqr_(int* m, int* n, int* k, double* a, int* lda, double* tau,
               double* work, int* lwork, int* info);
  // Column-pivoted QR factorisation: A*P = Q*R
  void dgeqp3_(int* m, int* n, double* a, int* lda, int* jpvt,
               double* tau, double* work, int* lwork, int* info);

  // ── Complex QR factorisation ───────────────────────────────────────────────
  // Compute QR factorisation of a general complex m×n matrix: A = Q * R
  void zgeqrf_(int* m, int* n, lapack_complex_double* a, int* lda,
               lapack_complex_double* tau,
               lapack_complex_double* work, int* lwork, int* info);
  // Generate the m×n (or m×m) unitary matrix Q from zgeqrf reflectors
  void zungqr_(int* m, int* n, int* k, lapack_complex_double* a, int* lda,
               lapack_complex_double* tau,
               lapack_complex_double* work, int* lwork, int* info);
  // Column-pivoted complex QR factorisation: A*P = Q*R
  void zgeqp3_(int* m, int* n, lapack_complex_double* a, int* lda, int* jpvt,
               lapack_complex_double* tau, lapack_complex_double* work,
               int* lwork, double* rwork, int* info);

  // ── SVD ───────────────────────────────────────────────────────────────────
  // Compute SVD using divide-and-conquer: A = U * Sigma * V^T
  void dgesdd_(char* jobz, int* m, int* n, double* a, int* lda,
               double* s, double* u, int* ldu, double* vt, int* ldvt,
               double* work, int* lwork, int* iwork, int* info);

  // Complex SVD using divide-and-conquer: A = U * Sigma * V^H
  void zgesdd_(char* jobz, int* m, int* n, lapack_complex_double* a, int* lda,
               double* s, lapack_complex_double* u, int* ldu,
               lapack_complex_double* vt, int* ldvt,
               lapack_complex_double* work, int* lwork, double* rwork,
               int* iwork, int* info);

  // Complex SVD (standard algorithm, more robust fallback): A = U * Sigma * V^H
  void zgesvd_(char* jobu, char* jobvt, int* m, int* n,
               lapack_complex_double* a, int* lda,
               double* s, lapack_complex_double* u, int* ldu,
               lapack_complex_double* vt, int* ldvt,
               lapack_complex_double* work, int* lwork, double* rwork,
               int* info);

  // ── Matrix-matrix multiplication (BLAS) ──────────────────────────────────
  // C = alpha * op(A) * op(B) + beta * C
  void dgemm_(char* transa, char* transb,
              int* m, int* n, int* k,
              double* alpha, double* a, int* lda,
              double* b, int* ldb,
              double* beta, double* c, int* ldc);

  // Complex matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C
  void zgemm_(char* transa, char* transb,
              int* m, int* n, int* k,
              lapack_complex_double* alpha, lapack_complex_double* a, int* lda,
              lapack_complex_double* b, int* ldb,
              lapack_complex_double* beta, lapack_complex_double* c, int* ldc);

  // ── Linear solve (square) ─────────────────────────────────────────────────
  // LU factorisation + solve: A * X = B  (A is n×n, B is n×nrhs)
  // On exit A contains the LU factors; B contains X.
  void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv,
              double* b, int* ldb, int* info);

  // ── Linear least-squares / minimum-norm solve (general) ──────────────────
  // Uses QR (trans='N', m>=n) or LQ (trans='N', m<n) factorisation.
  // Solves min||A*X-B|| (overdetermined) or min||X|| s.t. A*X=B (underdetermined).
  void dgels_(char* trans, int* m, int* n, int* nrhs,
              double* a, int* lda, double* b, int* ldb,
              double* work, int* lwork, int* info);

  // ── Complex linear solve (square) ────────────────────────────────────────
  // LU factorisation + solve for complex matrices: A * X = B  (A is n×n, B is n×nrhs)
  void zgesv_(int* n, int* nrhs, lapack_complex_double* a, int* lda, int* ipiv,
              lapack_complex_double* b, int* ldb, int* info);

  // ── Complex linear least-squares / minimum-norm solve (general) ──────────
  // Uses QR (trans='N', m>=n) or LQ (trans='N', m<n) factorisation for complex matrices.
  void zgels_(char* trans, int* m, int* n, int* nrhs,
              lapack_complex_double* a, int* lda, lapack_complex_double* b, int* ldb,
              lapack_complex_double* work, int* lwork, int* info);

  // ── Eigenvalue decomposition ───────────────────────────────────────────────
  // Compute eigenvalues and optionally left/right eigenvectors of a general
  // real matrix A.  A = VR * diag(WR+i*WI) * VR^(-1)
  void dgeev_(char* jobvl, char* jobvr, int* n, double* a, int* lda,
              double* wr, double* wi,
              double* vl, int* ldvl, double* vr, int* ldvr,
              double* work, int* lwork, int* info);

  // Compute eigenvalues and optionally left/right eigenvectors of a general
  // complex matrix A.  A = VR * diag(W) * VR^(-1)
  void zgeev_(char* jobvl, char* jobvr, int* n, lapack_complex_double* a, int* lda,
              lapack_complex_double* w,
              lapack_complex_double* vl, int* ldvl,
              lapack_complex_double* vr, int* ldvr,
              lapack_complex_double* work, int* lwork, double* rwork, int* info);

  // ── Cholesky factorization ─────────────────────────────────────────────────
  // Compute the Cholesky factorization of a real symmetric positive definite
  // matrix: A = U^T * U (uplo='U') or A = L * L^T (uplo='L')
  void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);

  // Complex Cholesky factorization of a Hermitian positive definite matrix:
  // A = U^H * U (uplo='U') or A = L * L^H (uplo='L')
  void zpotrf_(char* uplo, int* n, lapack_complex_double* a, int* lda, int* info);

  // ── QZ factorization (generalized Schur decomposition) ──────────────────────
  // Compute the generalized real Schur form of (A, B):
  //   A = VSL * S * VSR^T,  B = VSL * T * VSR^T
  // where S is upper quasi-triangular, T is upper triangular,
  // and VSL, VSR are orthogonal.
  typedef int (*dgges_selctg_t)(double*, double*, double*);
  void dgges_(char* jobvsl, char* jobvsr, char* sort,
              dgges_selctg_t selctg,
              int* n, double* a, int* lda, double* b, int* ldb,
              int* sdim, double* alphar, double* alphai, double* beta,
              double* vsl, int* ldvsl, double* vsr, int* ldvsr,
              double* work, int* lwork, int* bwork, int* info);

  // ── Generalized eigenvectors from triangular pair ───────────────────────────
  // Compute some or all of the right and/or left generalized eigenvectors
  // of a pair of real upper (quasi-)triangular matrices (S, P).
  void dtgevc_(char* side, char* howmny, int* select, int* n,
               double* s, int* lds, double* p, int* ldp,
               double* vl, int* ldvl, double* vr, int* ldvr,
               int* mm, int* m, double* work, int* info);

  // ── Complex QZ factorization ───────────────────────────────────────────────
  typedef int (*zgges_selctg_t)(lapack_complex_double*, lapack_complex_double*);
  void zgges_(char* jobvsl, char* jobvsr, char* sort,
              zgges_selctg_t selctg,
              int* n, lapack_complex_double* a, int* lda,
              lapack_complex_double* b, int* ldb,
              int* sdim, lapack_complex_double* alpha,
              lapack_complex_double* beta,
              lapack_complex_double* vsl, int* ldvsl,
              lapack_complex_double* vsr, int* ldvsr,
              lapack_complex_double* work, int* lwork,
              double* rwork, int* bwork, int* info);

  // Complex generalized eigenvectors from triangular pair
  void ztgevc_(char* side, char* howmny, int* select, int* n,
               lapack_complex_double* s, int* lds,
               lapack_complex_double* p, int* ldp,
               lapack_complex_double* vl, int* ldvl,
               lapack_complex_double* vr, int* ldvr,
               int* mm, int* m,
               lapack_complex_double* work, double* rwork, int* info);
}

// ── Common helpers to reduce boilerplate across LAPACK wrappers ───────────────

// Convert split real/imag Float64Arrays into interleaved complex vector.
inline std::vector<lapack_complex_double> splitToInterleaved(
    const Napi::Float64Array& re, const Napi::Float64Array& im, int n) {
  std::vector<lapack_complex_double> out(n);
  for (int i = 0; i < n; ++i) {
    out[i].real = re[i];
    out[i].imag = im[i];
  }
  return out;
}

// Create a new Float64Array from a std::vector<double>.
inline Napi::Float64Array vecToF64(Napi::Env env, const std::vector<double>& v) {
  auto arr = Napi::Float64Array::New(env, v.size());
  std::memcpy(arr.Data(), v.data(), v.size() * sizeof(double));
  return arr;
}

// Create a new Float64Array from raw pointer + count.
inline Napi::Float64Array ptrToF64(Napi::Env env, const double* data, size_t n) {
  auto arr = Napi::Float64Array::New(env, n);
  std::memcpy(arr.Data(), data, n * sizeof(double));
  return arr;
}

// Create a new Int32Array from a std::vector<int>.
inline Napi::Int32Array vecToI32(Napi::Env env, const std::vector<int>& v) {
  auto arr = Napi::Int32Array::New(env, v.size());
  std::memcpy(arr.Data(), v.data(), v.size() * sizeof(int));
  return arr;
}

// Deinterleave complex vector into separate re/im Float64Arrays and set on obj.
inline void setSplitComplex(Napi::Env env, Napi::Object& obj,
    const char* reKey, const char* imKey,
    const lapack_complex_double* data, int n) {
  auto re = Napi::Float64Array::New(env, static_cast<size_t>(n));
  auto im = Napi::Float64Array::New(env, static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    re[i] = data[i].real;
    im[i] = data[i].imag;
  }
  obj.Set(reKey, re);
  obj.Set(imKey, im);
}

// Check LAPACK info value and throw on error. Returns true if info == 0.
inline bool checkLapackInfo(Napi::Env env, int info_val,
    const char* func, const char* routine,
    const char* singularMsg = nullptr) {
  if (info_val == 0) return true;
  std::string msg = std::string(func) + ": ";
  if (info_val < 0) {
    msg += "illegal argument passed to ";
    msg += routine;
  } else if (singularMsg) {
    msg += singularMsg;
  } else {
    msg += std::string(routine) + " failed (info=" + std::to_string(info_val) + ")";
  }
  Napi::Error::New(env, msg).ThrowAsJavaScriptException();
  return false;
}

// ── Function prototypes (implemented in their respective .cpp files) ──────────

Napi::Value Inv(const Napi::CallbackInfo& info);
Napi::Value InvComplex(const Napi::CallbackInfo& info);
Napi::Value Qr(const Napi::CallbackInfo& info);
Napi::Value QrPivot(const Napi::CallbackInfo& info);
Napi::Value QrPivotComplex(const Napi::CallbackInfo& info);
Napi::Value QrComplex(const Napi::CallbackInfo& info);
Napi::Value Lu(const Napi::CallbackInfo& info);
Napi::Value LuComplex(const Napi::CallbackInfo& info);
Napi::Value Svd(const Napi::CallbackInfo& info);
Napi::Value SvdComplex(const Napi::CallbackInfo& info);
Napi::Value Matmul(const Napi::CallbackInfo& info);
Napi::Value MatmulComplex(const Napi::CallbackInfo& info);
Napi::Value Linsolve(const Napi::CallbackInfo& info);
Napi::Value LinsolveComplex(const Napi::CallbackInfo& info);
Napi::Value Eig(const Napi::CallbackInfo& info);
Napi::Value EigComplex(const Napi::CallbackInfo& info);
Napi::Value Chol(const Napi::CallbackInfo& info);
Napi::Value CholComplex(const Napi::CallbackInfo& info);
Napi::Value Qz(const Napi::CallbackInfo& info);
Napi::Value QzComplex(const Napi::CallbackInfo& info);
Napi::Value Fft1d(const Napi::CallbackInfo& info);
Napi::Value Fft1dComplex(const Napi::CallbackInfo& info);
Napi::Value FftAlongDim(const Napi::CallbackInfo& info);
Napi::Value Elemwise(const Napi::CallbackInfo& info);
Napi::Value ElemwiseScalar(const Napi::CallbackInfo& info);
Napi::Value ElemwiseComplex(const Napi::CallbackInfo& info);
Napi::Value FillRandn(const Napi::CallbackInfo& info);
Napi::Value UnaryElemwise(const Napi::CallbackInfo& info);
