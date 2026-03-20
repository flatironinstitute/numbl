/**
 * matmulComplex() — Complex matrix-matrix multiplication via BLAS zgemm.
 *
 *   matmulComplex(ARe: Float64Array, AIm: Float64Array,
 *                 m: number, k: number,
 *                 BRe: Float64Array, BIm: Float64Array,
 *                 n: number): { re: Float64Array, im: Float64Array }
 *
 *     Computes C = A * B where:
 *       A is an m×k complex matrix (split re/im) stored in column-major order
 *       B is a  k×n complex matrix (split re/im) stored in column-major order
 *       C is an m×n complex matrix returned as {re, im} in column-major order
 */

#include "numbl_addon_common.h"

Napi::Value MatmulComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  // matmulComplex(ARe, AIm, m, k, BRe, BIm, n)
  if (info.Length() < 7
      || !info[0].IsTypedArray()   // ARe
      || !info[1].IsTypedArray()   // AIm
      || !info[2].IsNumber()       // m
      || !info[3].IsNumber()       // k
      || !info[4].IsTypedArray()   // BRe
      || !info[5].IsTypedArray()   // BIm
      || !info[6].IsNumber()) {    // n
    Napi::TypeError::New(env,
      "matmulComplex: expected (Float64Array ARe, Float64Array AIm, "
      "number m, number k, Float64Array BRe, Float64Array BIm, number n)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrARe = info[0].As<Napi::Float64Array>();
  auto arrAIm = info[1].As<Napi::Float64Array>();
  int m = info[2].As<Napi::Number>().Int32Value();
  int k = info[3].As<Napi::Number>().Int32Value();
  auto arrBRe = info[4].As<Napi::Float64Array>();
  auto arrBIm = info[5].As<Napi::Float64Array>();
  int n = info[6].As<Napi::Number>().Int32Value();

  if (m < 0 || k < 0 || n < 0) {
    Napi::RangeError::New(env, "matmulComplex: m, k, n must be non-negative")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int mk = m * k;
  int kn = k * n;
  int mn = m * n;

  // Handle empty-dimension multiply
  if (m == 0 || k == 0 || n == 0) {
    auto result = Napi::Object::New(env);
    result.Set("re", Napi::Float64Array::New(env, static_cast<size_t>(mn)));
    result.Set("im", Napi::Float64Array::New(env, static_cast<size_t>(mn)));
    return result;
  }

  // Interleave into complex arrays for zgemm
  std::vector<lapack_complex_double> a(mk);
  for (int i = 0; i < mk; ++i) {
    a[i].real = arrARe[i];
    a[i].imag = arrAIm[i];
  }

  std::vector<lapack_complex_double> b(kn);
  for (int i = 0; i < kn; ++i) {
    b[i].real = arrBRe[i];
    b[i].imag = arrBIm[i];
  }

  std::vector<lapack_complex_double> c(mn, {0.0, 0.0});

  char transa = 'N';
  char transb = 'N';
  lapack_complex_double alpha = {1.0, 0.0};
  lapack_complex_double beta  = {0.0, 0.0};
  int lda = m;
  int ldb = k;
  int ldc = m;

  zgemm_(&transa, &transb,
         &m, &n, &k,
         &alpha, a.data(), &lda,
                 b.data(), &ldb,
         &beta,  c.data(), &ldc);

  // Deinterleave result
  auto result = Napi::Object::New(env);
  auto outRe = Napi::Float64Array::New(env, static_cast<size_t>(mn));
  auto outIm = Napi::Float64Array::New(env, static_cast<size_t>(mn));
  for (int i = 0; i < mn; ++i) {
    outRe[i] = c[i].real;
    outIm[i] = c[i].imag;
  }

  // Check if result is purely real
  bool isReal = true;
  for (int i = 0; i < mn; ++i) {
    if (outIm[i] != 0.0) { isReal = false; break; }
  }
  result.Set("re", outRe);
  if (!isReal) {
    result.Set("im", outIm);
  }
  return result;
}
