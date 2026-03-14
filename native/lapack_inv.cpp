/**
 * inv() and invComplex() — real and complex matrix inversion via LAPACK.
 *
 *   inv(data: Float64Array, n: number): Float64Array
 *     Inverts an n×n real matrix (column-major) using dgetrf + dgetri.
 *     Throws if the matrix is singular.
 *
 *   invComplex(dataRe: Float64Array, dataIm: Float64Array, n: number):
 *             {re: Float64Array, im: Float64Array}
 *     Inverts an n×n complex matrix (column-major, split re/im) using
 *     zgetrf + zgetri.  Throws if the matrix is singular.
 */

#include "lapack_common.h"

// ── inv() ─────────────────────────────────────────────────────────────────────

Napi::Value Inv(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()) {
    Napi::TypeError::New(env, "inv: expected (Float64Array data, number n)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::TypedArray>();
  if (arr.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "inv: data must be a Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[1].As<Napi::Number>().Int32Value();

  if (n <= 0 || static_cast<int>(arr.ElementLength()) != n * n) {
    Napi::RangeError::New(env, "inv: data.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arr = info[0].As<Napi::Float64Array>();

  // Copy into a working buffer (dgetrf/dgetri overwrite A in-place)
  std::vector<double> a(n * n);
  std::memcpy(a.data(), float64arr.Data(), n * n * sizeof(double));

  std::vector<int> ipiv(n);
  int info_val = 0;

  // Step 1: LU factorisation
  dgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info_val);
  if (!checkLapackInfo(env, info_val, "inv", "dgetrf",
      "matrix is singular (dgetrf)"))
    return env.Null();

  // Step 2: Query optimal workspace size
  int lwork = -1;
  double work_query = 0.0;
  dgetri_(&n, a.data(), &n, ipiv.data(), &work_query, &lwork, &info_val);
  lwork = static_cast<int>(work_query);
  if (lwork < 1) lwork = n;

  // Step 3: Compute inverse
  std::vector<double> work(lwork);
  dgetri_(&n, a.data(), &n, ipiv.data(), work.data(), &lwork, &info_val);
  if (!checkLapackInfo(env, info_val, "inv", "dgetri",
      "matrix is singular (dgetri)"))
    return env.Null();

  return vecToF64(env, a);
}

// ── invComplex() ──────────────────────────────────────────────────────────────

Napi::Value InvComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()) {
    Napi::TypeError::New(env,
      "invComplex: expected (Float64Array dataRe, Float64Array dataIm, number n)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrRe = info[0].As<Napi::TypedArray>();
  auto arrIm = info[1].As<Napi::TypedArray>();

  if (arrRe.TypedArrayType() != napi_float64_array ||
      arrIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "invComplex: dataRe and dataIm must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[2].As<Napi::Number>().Int32Value();

  if (n <= 0 ||
      static_cast<int>(arrRe.ElementLength()) != n * n ||
      static_cast<int>(arrIm.ElementLength()) != n * n) {
    Napi::RangeError::New(env,
      "invComplex: dataRe.length and dataIm.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = splitToInterleaved(
      info[0].As<Napi::Float64Array>(),
      info[1].As<Napi::Float64Array>(), n * n);

  std::vector<int> ipiv(n);
  int info_val = 0;

  // Step 1: Complex LU factorisation
  zgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info_val);
  if (!checkLapackInfo(env, info_val, "invComplex", "zgetrf",
      "matrix is singular (zgetrf)"))
    return env.Null();

  // Step 2: Query optimal workspace size
  int lwork = -1;
  lapack_complex_double work_query;
  zgetri_(&n, a.data(), &n, ipiv.data(), &work_query, &lwork, &info_val);
  lwork = static_cast<int>(work_query.real);
  if (lwork < 1) lwork = n;

  // Step 3: Compute inverse
  std::vector<lapack_complex_double> work(lwork);
  zgetri_(&n, a.data(), &n, ipiv.data(), work.data(), &lwork, &info_val);
  if (!checkLapackInfo(env, info_val, "invComplex", "zgetri",
      "matrix is singular (zgetri)"))
    return env.Null();

  auto result = Napi::Object::New(env);
  setSplitComplex(env, result, "re", "im", a.data(), n * n);
  return result;
}
