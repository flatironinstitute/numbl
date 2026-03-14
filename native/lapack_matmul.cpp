/**
 * matmul() — Real matrix-matrix multiplication via BLAS dgemm.
 *
 *   matmul(A: Float64Array, m: number, k: number,
 *          B: Float64Array, n: number): Float64Array
 *
 *     Computes C = A * B where:
 *       A is an m×k matrix stored in column-major order
 *       B is a  k×n matrix stored in column-major order
 *       C is an m×n matrix returned in column-major order
 */

#include "lapack_common.h"

// ── matmul() ──────────────────────────────────────────────────────────────────

Napi::Value Matmul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsNumber()
      || !info[3].IsTypedArray()
      || !info[4].IsNumber()) {
    Napi::TypeError::New(env,
      "matmul: expected (Float64Array A, number m, number k, Float64Array B, number n)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrA = info[0].As<Napi::TypedArray>();
  auto arrB = info[3].As<Napi::TypedArray>();

  if (arrA.TypedArrayType() != napi_float64_array ||
      arrB.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "matmul: A and B must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m = info[1].As<Napi::Number>().Int32Value();
  int k = info[2].As<Napi::Number>().Int32Value();
  int n = info[4].As<Napi::Number>().Int32Value();

  if (m < 0 || k < 0 || n < 0) {
    Napi::RangeError::New(env, "matmul: m, k, n must be non-negative")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Handle empty-dimension multiply without calling dgemm.
  if (m == 0 || k == 0 || n == 0) {
    return Napi::Float64Array::New(env, static_cast<size_t>(m * n));
  }
  if (static_cast<int>(arrA.ElementLength()) != m * k) {
    Napi::RangeError::New(env, "matmul: A.length must equal m*k")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (static_cast<int>(arrB.ElementLength()) != k * n) {
    Napi::RangeError::New(env, "matmul: B.length must equal k*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64A = info[0].As<Napi::Float64Array>();
  auto float64B = info[3].As<Napi::Float64Array>();

  char transa = 'N';
  char transb = 'N';
  double alpha = 1.0;
  double beta  = 0.0;
  int lda = m;
  int ldb = k;
  int ldc = m;

  std::vector<double> c(m * n);

  dgemm_(&transa, &transb,
         &m, &n, &k,
         &alpha, float64A.Data(), &lda,
                 float64B.Data(), &ldb,
         &beta,  c.data(),        &ldc);

  return vecToF64(env, c);
}
