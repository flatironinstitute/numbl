/**
 * linsolve() — solve A * X = B via LAPACK (real).
 * linsolveComplex() — solve A * X = B via LAPACK (complex).
 *
 *   linsolve(A: Float64Array, m: number, n: number,
 *            B: Float64Array, nrhs: number): Float64Array
 *
 *   linsolveComplex(ARe: Float64Array, AIm: Float64Array, m: number, n: number,
 *                   BRe: Float64Array, BIm: Float64Array, nrhs: number): {re, im}
 *
 *   A is m×n in column-major order; B is m×nrhs.
 *   Returns X (n×nrhs) in a new Float64Array in column-major order.
 *
 *   Square (m == n):
 *     Uses dgesv / zgesv (LU with partial pivoting).  Throws if A is singular.
 *
 *   Non-square:
 *     Uses dgels / zgels (QR for overdetermined, LQ for underdetermined).
 *     Overdetermined (m > n): least-squares solution minimising ||A*X - B||₂.
 *     Underdetermined (m < n): minimum-norm solution minimising ||X||₂.
 */

#include "lapack_common.h"

// ── linsolve() ────────────────────────────────────────────────────────────────

Napi::Value Linsolve(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsNumber()
      || !info[3].IsTypedArray()
      || !info[4].IsNumber()) {
    Napi::TypeError::New(env,
      "linsolve: expected (Float64Array A, number m, number n,"
      " Float64Array B, number nrhs)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrA = info[0].As<Napi::TypedArray>();
  auto arrB = info[3].As<Napi::TypedArray>();

  if (arrA.TypedArrayType() != napi_float64_array ||
      arrB.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "linsolve: A and B must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m    = info[1].As<Napi::Number>().Int32Value();
  int n    = info[2].As<Napi::Number>().Int32Value();
  int nrhs = info[4].As<Napi::Number>().Int32Value();

  if (m <= 0 || n <= 0 || nrhs <= 0
      || static_cast<int>(arrA.ElementLength()) != m * n
      || static_cast<int>(arrB.ElementLength()) != m * nrhs) {
    Napi::RangeError::New(env,
      "linsolve: A.length must equal m*n and B.length must equal m*nrhs")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto fA = info[0].As<Napi::Float64Array>();
  auto fB = info[3].As<Napi::Float64Array>();

  int info_val = 0;

  if (m == n) {
    // ── Square: dgesv (LU + solve) ──────────────────────────────────────────
    std::vector<double> a(n * n), b(n * nrhs);
    std::memcpy(a.data(), fA.Data(), n * n * sizeof(double));
    std::memcpy(b.data(), fB.Data(), n * nrhs * sizeof(double));

    std::vector<int> ipiv(n);
    dgesv_(&n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info_val);

    if (!checkLapackInfo(env, info_val, "linsolve", "dgesv",
        "matrix is singular (dgesv)"))
      return env.Null();

    return ptrToF64(env, b.data(), n * nrhs);

  } else {
    // ── Non-square: dgels (QR / LQ least-squares / min-norm solve) ─────────
    int ldb = m > n ? m : n;
    std::vector<double> a(m * n), b(ldb * nrhs, 0.0);
    std::memcpy(a.data(), fA.Data(), m * n * sizeof(double));

    for (int c = 0; c < nrhs; c++) {
      std::memcpy(b.data() + c * ldb, fB.Data() + c * m, m * sizeof(double));
    }

    char trans = 'N';

    // Workspace query
    int lwork = -1;
    double work_query = 0.0;
    dgels_(&trans, &m, &n, &nrhs,
           a.data(), &m, b.data(), &ldb,
           &work_query, &lwork, &info_val);
    lwork = static_cast<int>(work_query);
    if (lwork < 1) lwork = std::max(1, std::max(m, n));

    std::vector<double> work(lwork);
    dgels_(&trans, &m, &n, &nrhs,
           a.data(), &m, b.data(), &ldb,
           work.data(), &lwork, &info_val);

    if (!checkLapackInfo(env, info_val, "linsolve", "dgels",
        "A does not have full rank (dgels)"))
      return env.Null();

    // Solution is in the first n rows of b
    auto result = Napi::Float64Array::New(env, static_cast<size_t>(n * nrhs));
    for (int c = 0; c < nrhs; c++) {
      std::memcpy(result.Data() + c * n, b.data() + c * ldb,
                  n * sizeof(double));
    }
    return result;
  }
}

// ── linsolveComplex() ─────────────────────────────────────────────────────────

Napi::Value LinsolveComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 7
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsNumber()
      || !info[4].IsTypedArray()
      || !info[5].IsTypedArray()
      || !info[6].IsNumber()) {
    Napi::TypeError::New(env,
      "linsolveComplex: expected (Float64Array ARe, Float64Array AIm,"
      " number m, number n, Float64Array BRe, Float64Array BIm, number nrhs)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrARe = info[0].As<Napi::TypedArray>();
  auto arrAIm = info[1].As<Napi::TypedArray>();
  auto arrBRe = info[4].As<Napi::TypedArray>();
  auto arrBIm = info[5].As<Napi::TypedArray>();

  if (arrARe.TypedArrayType() != napi_float64_array ||
      arrAIm.TypedArrayType() != napi_float64_array ||
      arrBRe.TypedArrayType() != napi_float64_array ||
      arrBIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env,
      "linsolveComplex: ARe, AIm, BRe, BIm must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m    = info[2].As<Napi::Number>().Int32Value();
  int n    = info[3].As<Napi::Number>().Int32Value();
  int nrhs = info[6].As<Napi::Number>().Int32Value();

  if (m <= 0 || n <= 0 || nrhs <= 0
      || static_cast<int>(arrARe.ElementLength()) != m * n
      || static_cast<int>(arrAIm.ElementLength()) != m * n
      || static_cast<int>(arrBRe.ElementLength()) != m * nrhs
      || static_cast<int>(arrBIm.ElementLength()) != m * nrhs) {
    Napi::RangeError::New(env,
      "linsolveComplex: array lengths must match m*n (A) and m*nrhs (B)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto fARe = info[0].As<Napi::Float64Array>();
  auto fAIm = info[1].As<Napi::Float64Array>();
  auto fBRe = info[4].As<Napi::Float64Array>();
  auto fBIm = info[5].As<Napi::Float64Array>();

  int info_val = 0;

  if (m == n) {
    // ── Square: zgesv (LU + solve) ──────────────────────────────────────────
    auto a = splitToInterleaved(fARe, fAIm, n * n);
    auto b = splitToInterleaved(fBRe, fBIm, n * nrhs);

    std::vector<int> ipiv(n);
    zgesv_(&n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info_val);

    if (!checkLapackInfo(env, info_val, "linsolveComplex", "zgesv",
        "matrix is singular (zgesv)"))
      return env.Null();

    auto result = Napi::Object::New(env);
    setSplitComplex(env, result, "re", "im", b.data(), n * nrhs);
    return result;

  } else {
    // ── Non-square: zgels (QR / LQ least-squares / min-norm solve) ─────────
    int ldb = m > n ? m : n;
    auto a = splitToInterleaved(fARe, fAIm, m * n);
    std::vector<lapack_complex_double> b(ldb * nrhs, {0.0, 0.0});

    for (int c = 0; c < nrhs; ++c) {
      for (int r = 0; r < m; ++r) {
        b[r + c * ldb].real = fBRe[r + c * m];
        b[r + c * ldb].imag = fBIm[r + c * m];
      }
    }

    char trans = 'N';

    // Workspace query
    int lwork = -1;
    lapack_complex_double work_query;
    zgels_(&trans, &m, &n, &nrhs,
           a.data(), &m, b.data(), &ldb,
           &work_query, &lwork, &info_val);
    lwork = static_cast<int>(work_query.real);
    if (lwork < 1) lwork = std::max(1, std::max(m, n));

    std::vector<lapack_complex_double> work(lwork);
    zgels_(&trans, &m, &n, &nrhs,
           a.data(), &m, b.data(), &ldb,
           work.data(), &lwork, &info_val);

    if (!checkLapackInfo(env, info_val, "linsolveComplex", "zgels",
        "A does not have full rank (zgels)"))
      return env.Null();

    // Solution is in the first n rows of b
    auto resultRe = Napi::Float64Array::New(env, static_cast<size_t>(n * nrhs));
    auto resultIm = Napi::Float64Array::New(env, static_cast<size_t>(n * nrhs));
    for (int c = 0; c < nrhs; ++c) {
      for (int r = 0; r < n; ++r) {
        resultRe[r + c * n] = b[r + c * ldb].real;
        resultIm[r + c * n] = b[r + c * ldb].imag;
      }
    }

    auto result = Napi::Object::New(env);
    result.Set("re", resultRe);
    result.Set("im", resultIm);
    return result;
  }
}
