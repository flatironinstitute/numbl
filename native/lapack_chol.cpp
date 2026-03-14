/**
 * chol() and cholComplex() — Cholesky factorization via LAPACK.
 *
 *   chol(data: Float64Array, n: number, upper: boolean):
 *       {R: Float64Array, info: number}
 *
 *   cholComplex(dataRe: Float64Array, dataIm: Float64Array, n: number, upper: boolean):
 *       {RRe: Float64Array, RIm: Float64Array, info: number}
 *
 *     Cholesky factorization of an n×n symmetric (Hermitian) positive definite
 *     matrix stored in column-major order.
 *     Uses LAPACK dpotrf (real) / zpotrf (complex).
 *     Returns the triangular factor and info (0 = success, >0 = not pos def).
 */

#include "lapack_common.h"

// Zero out the opposite triangle of an n×n column-major matrix.
template<typename T>
static void zeroOppositeTriangle(T* a, int n, bool upper) {
  if (upper) {
    for (int j = 0; j < n; j++)
      for (int i = j + 1; i < n; i++)
        a[i + j * n] = T{};
  } else {
    for (int j = 0; j < n; j++)
      for (int i = 0; i < j; i++)
        a[i + j * n] = T{};
  }
}

// ── chol() ───────────────────────────────────────────────────────────────────

Napi::Value Chol(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsBoolean()) {
    Napi::TypeError::New(env,
      "chol: expected (Float64Array data, number n, boolean upper)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::TypedArray>();
  if (arr.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "chol: data must be a Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[1].As<Napi::Number>().Int32Value();
  bool upper = info[2].As<Napi::Boolean>().Value();

  if (n <= 0 || static_cast<int>(arr.ElementLength()) != n * n) {
    Napi::RangeError::New(env, "chol: data.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arr = info[0].As<Napi::Float64Array>();

  std::vector<double> a(n * n);
  std::memcpy(a.data(), float64arr.Data(), n * n * sizeof(double));

  char uplo = upper ? 'U' : 'L';
  int info_val = 0;

  dpotrf_(&uplo, &n, a.data(), &n, &info_val);

  if (info_val < 0) {
    checkLapackInfo(env, info_val, "chol", "dpotrf");
    return env.Null();
  }

  zeroOppositeTriangle(a.data(), n, upper);

  auto result = Napi::Object::New(env);
  result.Set("R", vecToF64(env, a));
  result.Set("info", Napi::Number::New(env, info_val));
  return result;
}

// ── cholComplex() ────────────────────────────────────────────────────────────

Napi::Value CholComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 4
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()) {
    Napi::TypeError::New(env,
      "cholComplex: expected (Float64Array dataRe, Float64Array dataIm, "
      "number n, boolean upper)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrRe = info[0].As<Napi::TypedArray>();
  auto arrIm = info[1].As<Napi::TypedArray>();

  if (arrRe.TypedArrayType() != napi_float64_array ||
      arrIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "cholComplex: dataRe and dataIm must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[2].As<Napi::Number>().Int32Value();
  bool upper = info[3].As<Napi::Boolean>().Value();

  if (n <= 0 ||
      static_cast<int>(arrRe.ElementLength()) != n * n ||
      static_cast<int>(arrIm.ElementLength()) != n * n) {
    Napi::RangeError::New(env,
      "cholComplex: dataRe.length and dataIm.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = splitToInterleaved(
      info[0].As<Napi::Float64Array>(),
      info[1].As<Napi::Float64Array>(), n * n);

  char uplo = upper ? 'U' : 'L';
  int info_val = 0;

  zpotrf_(&uplo, &n, a.data(), &n, &info_val);

  if (info_val < 0) {
    checkLapackInfo(env, info_val, "cholComplex", "zpotrf");
    return env.Null();
  }

  zeroOppositeTriangle(a.data(), n, upper);

  auto result = Napi::Object::New(env);
  setSplitComplex(env, result, "RRe", "RIm", a.data(), n * n);
  result.Set("info", Napi::Number::New(env, info_val));
  return result;
}
