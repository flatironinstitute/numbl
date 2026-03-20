/**
 * lu() and luComplex() — LU factorization with partial pivoting via LAPACK.
 *
 *   lu(data: Float64Array, m: number, n: number):
 *       {LU: Float64Array, ipiv: Int32Array}
 *
 *   luComplex(dataRe: Float64Array, dataIm: Float64Array, m: number, n: number):
 *       {LURe: Float64Array, LUIm: Float64Array, ipiv: Int32Array}
 *
 *     LU factorization of an m×n matrix stored in column-major order.
 *     Uses LAPACK dgetrf (real) / zgetrf (complex) with partial pivoting.
 *     Returns the packed LU matrix and 1-based pivot indices.
 */

#include "numbl_addon_common.h"

// ── lu() ─────────────────────────────────────────────────────────────────────

Napi::Value Lu(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsNumber()) {
    Napi::TypeError::New(env,
      "lu: expected (Float64Array data, number m, number n)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::TypedArray>();
  if (arr.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "lu: data must be a Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m = info[1].As<Napi::Number>().Int32Value();
  int n = info[2].As<Napi::Number>().Int32Value();

  if (m <= 0 || n <= 0 || static_cast<int>(arr.ElementLength()) != m * n) {
    Napi::RangeError::New(env, "lu: data.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arr = info[0].As<Napi::Float64Array>();
  int k = m < n ? m : n;

  std::vector<double> a(m * n);
  std::memcpy(a.data(), float64arr.Data(), m * n * sizeof(double));

  std::vector<int> ipiv(k);
  int info_val = 0;

  dgetrf_(&m, &n, a.data(), &m, ipiv.data(), &info_val);

  if (info_val < 0) {
    checkLapackInfo(env, info_val, "lu", "dgetrf");
    return env.Null();
  }

  auto result = Napi::Object::New(env);
  result.Set("LU", vecToF64(env, a));
  result.Set("ipiv", vecToI32(env, ipiv));
  return result;
}

// ── luComplex() ──────────────────────────────────────────────────────────────

Napi::Value LuComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 4
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsNumber()) {
    Napi::TypeError::New(env,
      "luComplex: expected (Float64Array dataRe, Float64Array dataIm, "
      "number m, number n)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrRe = info[0].As<Napi::TypedArray>();
  auto arrIm = info[1].As<Napi::TypedArray>();

  if (arrRe.TypedArrayType() != napi_float64_array ||
      arrIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "luComplex: dataRe and dataIm must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m = info[2].As<Napi::Number>().Int32Value();
  int n = info[3].As<Napi::Number>().Int32Value();

  if (m <= 0 || n <= 0 ||
      static_cast<int>(arrRe.ElementLength()) != m * n ||
      static_cast<int>(arrIm.ElementLength()) != m * n) {
    Napi::RangeError::New(env,
      "luComplex: dataRe.length and dataIm.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int k = m < n ? m : n;

  auto a = splitToInterleaved(
      info[0].As<Napi::Float64Array>(),
      info[1].As<Napi::Float64Array>(), m * n);

  std::vector<int> ipiv(k);
  int info_val = 0;

  zgetrf_(&m, &n, a.data(), &m, ipiv.data(), &info_val);

  if (info_val < 0) {
    checkLapackInfo(env, info_val, "luComplex", "zgetrf");
    return env.Null();
  }

  auto result = Napi::Object::New(env);
  setSplitComplex(env, result, "LURe", "LUIm", a.data(), m * n);
  result.Set("ipiv", vecToI32(env, ipiv));
  return result;
}
