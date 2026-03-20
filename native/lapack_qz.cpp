/**
 * qz() and qzComplex() — Generalized Schur (QZ) factorization via LAPACK.
 *
 *   qz(dataA, dataB, n, computeEigvecs) — real case via dgges/dtgevc
 *   qzComplex(ARe, AIm, BRe, BIm, n, computeEigvecs) — complex case via zgges/ztgevc
 *
 *   Computes the generalized Schur decomposition of (A, B):
 *     Q*A*Z = AA, Q*B*Z = BB
 *   where Q and Z are unitary (orthogonal for real case).
 */

#include "numbl_addon_common.h"

Napi::Value Qz(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 4
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()) {
    Napi::TypeError::New(env,
      "qz: expected (Float64Array dataA, Float64Array dataB, number n, "
      "boolean computeEigvecs)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrA = info[0].As<Napi::TypedArray>();
  auto arrB = info[1].As<Napi::TypedArray>();
  if (arrA.TypedArrayType() != napi_float64_array ||
      arrB.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "qz: data must be Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[2].As<Napi::Number>().Int32Value();
  bool computeEigvecs = info[3].As<Napi::Boolean>().Value();

  if (n <= 0
      || static_cast<int>(arrA.ElementLength()) != n * n
      || static_cast<int>(arrB.ElementLength()) != n * n) {
    Napi::RangeError::New(env, "qz: data.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto f64A = info[0].As<Napi::Float64Array>();
  auto f64B = info[1].As<Napi::Float64Array>();

  std::vector<double> a(n * n), b(n * n);
  std::memcpy(a.data(), f64A.Data(), n * n * sizeof(double));
  std::memcpy(b.data(), f64B.Data(), n * n * sizeof(double));

  std::vector<double> alphar(n), alphai(n), beta(n);
  std::vector<double> vsl(n * n), vsr(n * n);

  char jobvsl = 'V', jobvsr = 'V', sort = 'N';
  int sdim = 0, info_val = 0;
  std::vector<int> bwork(n);

  // Workspace query
  int lwork = -1;
  double work_query = 0.0;
  dgges_(&jobvsl, &jobvsr, &sort, nullptr,
         &n, a.data(), &n, b.data(), &n,
         &sdim, alphar.data(), alphai.data(), beta.data(),
         vsl.data(), &n, vsr.data(), &n,
         &work_query, &lwork, bwork.data(), &info_val);

  lwork = static_cast<int>(work_query);
  if (lwork < 1) lwork = std::max(1, 8 * n + 16);
  std::vector<double> work(lwork);

  // Reset a and b (workspace query may have modified them)
  std::memcpy(a.data(), f64A.Data(), n * n * sizeof(double));
  std::memcpy(b.data(), f64B.Data(), n * n * sizeof(double));

  dgges_(&jobvsl, &jobvsr, &sort, nullptr,
         &n, a.data(), &n, b.data(), &n,
         &sdim, alphar.data(), alphai.data(), beta.data(),
         vsl.data(), &n, vsr.data(), &n,
         work.data(), &lwork, bwork.data(), &info_val);

  if (!checkLapackInfo(env, info_val, "qz", "dgges",
      "QZ iteration failed to converge in dgges"))
    return env.Null();

  auto result = Napi::Object::New(env);
  result.Set("AA", vecToF64(env, a));
  result.Set("BB", vecToF64(env, b));

  // Q = VSL^T (MATLAB convention: Q*A*Z = AA)
  auto q_arr = Napi::Float64Array::New(env, static_cast<size_t>(n * n));
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      q_arr[i + j * n] = vsl[j + i * n];
  result.Set("Q", q_arr);

  result.Set("Z", vecToF64(env, vsr));
  result.Set("alphar", vecToF64(env, alphar));
  result.Set("alphai", vecToF64(env, alphai));
  result.Set("beta", vecToF64(env, beta));

  if (computeEigvecs) {
    std::vector<double> vr(vsr), vl(vsl);
    char side = 'B', howmny = 'B';
    std::vector<int> select(n);
    int mm = n, m_out = 0;
    std::vector<double> work_tgevc(6 * n);

    dtgevc_(&side, &howmny, select.data(), &n,
            a.data(), &n, b.data(), &n,
            vl.data(), &n, vr.data(), &n,
            &mm, &m_out, work_tgevc.data(), &info_val);

    if (!checkLapackInfo(env, info_val, "qz", "dtgevc",
        "dtgevc failed to compute eigenvectors"))
      return env.Null();

    result.Set("V", vecToF64(env, vr));
    result.Set("W", vecToF64(env, vl));
  }

  return result;
}

// ── qzComplex() ──────────────────────────────────────────────────────────────

Napi::Value QzComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 6
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsTypedArray()
      || !info[3].IsTypedArray()
      || !info[4].IsNumber()
      || !info[5].IsBoolean()) {
    Napi::TypeError::New(env,
      "qzComplex: expected (Float64Array ARe, Float64Array AIm, "
      "Float64Array BRe, Float64Array BIm, number n, boolean computeEigvecs)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[4].As<Napi::Number>().Int32Value();
  bool computeEigvecs = info[5].As<Napi::Boolean>().Value();

  auto f64ARe = info[0].As<Napi::Float64Array>();
  auto f64AIm = info[1].As<Napi::Float64Array>();
  auto f64BRe = info[2].As<Napi::Float64Array>();
  auto f64BIm = info[3].As<Napi::Float64Array>();

  if (n <= 0
      || static_cast<int>(f64ARe.ElementLength()) != n * n
      || static_cast<int>(f64AIm.ElementLength()) != n * n
      || static_cast<int>(f64BRe.ElementLength()) != n * n
      || static_cast<int>(f64BIm.ElementLength()) != n * n) {
    Napi::RangeError::New(env, "qzComplex: data.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = splitToInterleaved(f64ARe, f64AIm, n * n);
  auto b = splitToInterleaved(f64BRe, f64BIm, n * n);

  std::vector<lapack_complex_double> alpha(n), beta_c(n);
  std::vector<lapack_complex_double> vsl(n * n), vsr(n * n);

  char jobvsl = 'V', jobvsr = 'V', sort = 'N';
  int sdim = 0, info_val = 0;
  std::vector<int> bwork(n);

  // Workspace query
  int lwork = -1;
  lapack_complex_double work_query;
  std::vector<double> rwork(8 * n);
  zgges_(&jobvsl, &jobvsr, &sort, nullptr,
         &n, a.data(), &n, b.data(), &n,
         &sdim, alpha.data(), beta_c.data(),
         vsl.data(), &n, vsr.data(), &n,
         &work_query, &lwork, rwork.data(), bwork.data(), &info_val);

  lwork = static_cast<int>(work_query.real);
  if (lwork < 1) lwork = std::max(1, 2 * n);
  std::vector<lapack_complex_double> work(lwork);

  // Reset a and b
  a = splitToInterleaved(f64ARe, f64AIm, n * n);
  b = splitToInterleaved(f64BRe, f64BIm, n * n);

  zgges_(&jobvsl, &jobvsr, &sort, nullptr,
         &n, a.data(), &n, b.data(), &n,
         &sdim, alpha.data(), beta_c.data(),
         vsl.data(), &n, vsr.data(), &n,
         work.data(), &lwork, rwork.data(), bwork.data(), &info_val);

  if (!checkLapackInfo(env, info_val, "qzComplex", "zgges",
      "QZ iteration failed to converge in zgges"))
    return env.Null();

  auto result = Napi::Object::New(env);
  size_t nn = static_cast<size_t>(n * n);

  setSplitComplex(env, result, "AARe", "AAIm", a.data(), n * n);
  setSplitComplex(env, result, "BBRe", "BBIm", b.data(), n * n);

  // Q = VSL^H (conjugate transpose) for MATLAB convention
  auto qRe = Napi::Float64Array::New(env, nn);
  auto qIm = Napi::Float64Array::New(env, nn);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      qRe[i + j * n] =  vsl[j + i * n].real;
      qIm[i + j * n] = -vsl[j + i * n].imag;
    }
  }
  result.Set("QRe", qRe);
  result.Set("QIm", qIm);

  setSplitComplex(env, result, "ZRe", "ZIm", vsr.data(), n * n);
  setSplitComplex(env, result, "alphaRe", "alphaIm", alpha.data(), n);
  setSplitComplex(env, result, "betaRe", "betaIm", beta_c.data(), n);

  if (computeEigvecs) {
    std::vector<lapack_complex_double> vr(vsr), vl(vsl);
    char side = 'B', howmny = 'B';
    std::vector<int> select(n);
    int mm = n, m_out = 0;
    std::vector<lapack_complex_double> work_tgevc(2 * n);
    std::vector<double> rwork_tgevc(2 * n);

    ztgevc_(&side, &howmny, select.data(), &n,
            a.data(), &n, b.data(), &n,
            vl.data(), &n, vr.data(), &n,
            &mm, &m_out, work_tgevc.data(), rwork_tgevc.data(), &info_val);

    if (!checkLapackInfo(env, info_val, "qzComplex", "ztgevc"))
      return env.Null();

    setSplitComplex(env, result, "VRe", "VIm", vr.data(), n * n);
    setSplitComplex(env, result, "WRe", "WIm", vl.data(), n * n);
  }

  return result;
}
