/**
 * numbl native addon — LAPACK/BLAS, FFT, element-wise arithmetic, and more.
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

#include "numbl_addon_common.h"
#include <cstdlib>

extern "C" {
  void openblas_set_num_threads(int num_threads);
}

// ── Addon version ────────────────────────────────────────────────────────────
// Bump this integer whenever the addon's API changes (new functions, signature
// changes, etc.) so that the JS side can detect stale builds.
static const int ADDON_VERSION = 7;

// ── New tensor-ops layer (native/ops/) ───────────────────────────────────────
Napi::Value TensorOpRealBinary(const Napi::CallbackInfo& info);
Napi::Value TensorOpRealScalarBinary(const Napi::CallbackInfo& info);
Napi::Value TensorOpComplexBinary(const Napi::CallbackInfo& info);
Napi::Value TensorOpComplexScalarBinary(const Napi::CallbackInfo& info);
Napi::Value TensorOpRealUnary(const Napi::CallbackInfo& info);
Napi::Value TensorOpComplexUnary(const Napi::CallbackInfo& info);
Napi::Value TensorOpComplexAbs(const Napi::CallbackInfo& info);
Napi::Value TensorOpRealComparison(const Napi::CallbackInfo& info);
Napi::Value TensorOpRealScalarComparison(const Napi::CallbackInfo& info);
Napi::Value TensorOpComplexComparison(const Napi::CallbackInfo& info);
Napi::Value TensorOpComplexScalarComparison(const Napi::CallbackInfo& info);
Napi::Value TensorOpRealFlatReduce(const Napi::CallbackInfo& info);
Napi::Value TensorOpComplexFlatReduce(const Napi::CallbackInfo& info);
Napi::Value TensorOpDumpCodes(const Napi::CallbackInfo& info);

static Napi::Value AddonVersion(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), ADDON_VERSION);
}

// ── Module initialisation ─────────────────────────────────────────────────────

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  // Use single-threaded BLAS unless the user explicitly set the env var.
  // Multi-threaded BLAS adds overhead for the many small matmuls in numbl.
  if (!std::getenv("OPENBLAS_NUM_THREADS")) {
    openblas_set_num_threads(1);
  }
  exports.Set(Napi::String::New(env, "addonVersion"),
              Napi::Function::New(env, AddonVersion));
  exports.Set(Napi::String::New(env, "inv"),
              Napi::Function::New(env, Inv));
  exports.Set(Napi::String::New(env, "invComplex"),
              Napi::Function::New(env, InvComplex));
  exports.Set(Napi::String::New(env, "qr"),
              Napi::Function::New(env, Qr));
  exports.Set(Napi::String::New(env, "qrPivot"),
              Napi::Function::New(env, QrPivot));
  exports.Set(Napi::String::New(env, "qrPivotComplex"),
              Napi::Function::New(env, QrPivotComplex));
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
  exports.Set(Napi::String::New(env, "matmulComplex"),
              Napi::Function::New(env, MatmulComplex));
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
  exports.Set(Napi::String::New(env, "elemwise"),
              Napi::Function::New(env, Elemwise));
  exports.Set(Napi::String::New(env, "elemwiseScalar"),
              Napi::Function::New(env, ElemwiseScalar));
  exports.Set(Napi::String::New(env, "elemwiseComplex"),
              Napi::Function::New(env, ElemwiseComplex));
  exports.Set(Napi::String::New(env, "elemwiseComplexScalar"),
              Napi::Function::New(env, ElemwiseComplexScalar));
  exports.Set(Napi::String::New(env, "fillRandn"),
              Napi::Function::New(env, FillRandn));
  exports.Set(Napi::String::New(env, "unaryElemwise"),
              Napi::Function::New(env, UnaryElemwise));
  exports.Set(Napi::String::New(env, "gmres"),
              Napi::Function::New(env, Gmres));
  exports.Set(Napi::String::New(env, "gmresComplex"),
              Napi::Function::New(env, GmresComplex));

  // ── New tensor-ops layer ──────────────────────────────────────────────────
  exports.Set(Napi::String::New(env, "tensorOpRealBinary"),
              Napi::Function::New(env, TensorOpRealBinary));
  exports.Set(Napi::String::New(env, "tensorOpRealScalarBinary"),
              Napi::Function::New(env, TensorOpRealScalarBinary));
  exports.Set(Napi::String::New(env, "tensorOpComplexBinary"),
              Napi::Function::New(env, TensorOpComplexBinary));
  exports.Set(Napi::String::New(env, "tensorOpComplexScalarBinary"),
              Napi::Function::New(env, TensorOpComplexScalarBinary));
  exports.Set(Napi::String::New(env, "tensorOpRealUnary"),
              Napi::Function::New(env, TensorOpRealUnary));
  exports.Set(Napi::String::New(env, "tensorOpComplexUnary"),
              Napi::Function::New(env, TensorOpComplexUnary));
  exports.Set(Napi::String::New(env, "tensorOpComplexAbs"),
              Napi::Function::New(env, TensorOpComplexAbs));
  exports.Set(Napi::String::New(env, "tensorOpRealComparison"),
              Napi::Function::New(env, TensorOpRealComparison));
  exports.Set(Napi::String::New(env, "tensorOpRealScalarComparison"),
              Napi::Function::New(env, TensorOpRealScalarComparison));
  exports.Set(Napi::String::New(env, "tensorOpComplexComparison"),
              Napi::Function::New(env, TensorOpComplexComparison));
  exports.Set(Napi::String::New(env, "tensorOpComplexScalarComparison"),
              Napi::Function::New(env, TensorOpComplexScalarComparison));
  exports.Set(Napi::String::New(env, "tensorOpRealFlatReduce"),
              Napi::Function::New(env, TensorOpRealFlatReduce));
  exports.Set(Napi::String::New(env, "tensorOpComplexFlatReduce"),
              Napi::Function::New(env, TensorOpComplexFlatReduce));
  exports.Set(Napi::String::New(env, "tensorOpDumpCodes"),
              Napi::Function::New(env, TensorOpDumpCodes));
  return exports;
}

NODE_API_MODULE(numbl_addon, Init)
