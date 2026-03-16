/**
 * Native Node.js addon exposing LAPACK/BLAS routines for efficient linear algebra.
 *
 * Exported functions (see individual .cpp files for full documentation):
 *
 *   inv(data, n)                      — real matrix inversion      (lapack_inv.cpp)
 *   invComplex(dataRe, dataIm, n)     — complex matrix inversion   (lapack_inv.cpp)
 *   qr(data, m, n, econ, wantQ)       — QR decomposition           (lapack_qr.cpp)
 *   qrComplex(dataRe, dataIm, m, n, econ, wantQ) — complex QR      (lapack_qr.cpp)
 *   lu(data, m, n)                    — LU factorization           (lapack_lu.cpp)
 *   luComplex(dataRe, dataIm, m, n)  — complex LU factorization   (lapack_lu.cpp)
 *   svd(data, m, n, econ, computeUV)  — Singular Value Decomp.     (lapack_svd.cpp)
 *   matmul(A, m, k, B, n)             — matrix-matrix multiply     (lapack_matmul.cpp)
 *   linsolve(A, m, n, B, nrhs)        — linear solve / least-sq    (lapack_linsolve.cpp)
 *   linsolveComplex(ARe, AIm, m, n, BRe, BIm, nrhs) — complex linear solve (lapack_linsolve.cpp)
 *   eig(data, n, computeVL, computeVR, balance) — eigenvalue decomp. (lapack_eig.cpp)
 *   eigComplex(dataRe, dataIm, n, computeVL, computeVR) — complex eigenvalue decomp. (lapack_eig.cpp)
 *   chol(data, n, upper)             — Cholesky factorization    (lapack_chol.cpp)
 *   cholComplex(dataRe, dataIm, n, upper) — complex Cholesky     (lapack_chol.cpp)
 */

#include "lapack_common.h"

// ── Module initialisation ─────────────────────────────────────────────────────

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "inv"),
              Napi::Function::New(env, Inv));
  exports.Set(Napi::String::New(env, "invComplex"),
              Napi::Function::New(env, InvComplex));
  exports.Set(Napi::String::New(env, "qr"),
              Napi::Function::New(env, Qr));
  exports.Set(Napi::String::New(env, "qrComplex"),
              Napi::Function::New(env, QrComplex));
  exports.Set(Napi::String::New(env, "lu"),
              Napi::Function::New(env, Lu));
  exports.Set(Napi::String::New(env, "luComplex"),
              Napi::Function::New(env, LuComplex));
  exports.Set(Napi::String::New(env, "svd"),
              Napi::Function::New(env, Svd));
  exports.Set(Napi::String::New(env, "svdComplex"),
              Napi::Function::New(env, SvdComplex));
  exports.Set(Napi::String::New(env, "matmul"),
              Napi::Function::New(env, Matmul));
  exports.Set(Napi::String::New(env, "linsolve"),
              Napi::Function::New(env, Linsolve));
  exports.Set(Napi::String::New(env, "linsolveComplex"),
              Napi::Function::New(env, LinsolveComplex));
  exports.Set(Napi::String::New(env, "eig"),
              Napi::Function::New(env, Eig));
  exports.Set(Napi::String::New(env, "eigComplex"),
              Napi::Function::New(env, EigComplex));
  exports.Set(Napi::String::New(env, "chol"),
              Napi::Function::New(env, Chol));
  exports.Set(Napi::String::New(env, "cholComplex"),
              Napi::Function::New(env, CholComplex));
  exports.Set(Napi::String::New(env, "qz"),
              Napi::Function::New(env, Qz));
  exports.Set(Napi::String::New(env, "qzComplex"),
              Napi::Function::New(env, QzComplex));
  exports.Set(Napi::String::New(env, "fft1d"),
              Napi::Function::New(env, Fft1d));
  exports.Set(Napi::String::New(env, "fft1dComplex"),
              Napi::Function::New(env, Fft1dComplex));
  exports.Set(Napi::String::New(env, "fftAlongDim"),
              Napi::Function::New(env, FftAlongDim));
  return exports;
}

NODE_API_MODULE(lapack_addon, Init)
