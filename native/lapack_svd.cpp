/**
 * svd() — Singular Value Decomposition via LAPACK dgesdd (divide-and-conquer).
 *
 *   svd(data: Float64Array, m: number, n: number, econ: boolean,
 *       computeUV: boolean): {U?: Float64Array, S: Float64Array, V?: Float64Array}
 *
 *   svdComplex(dataRe, dataIm, m, n, econ, computeUV):
 *       {S, URe?, UIm?, VRe?, VIm?}
 *
 *     econ=true,  computeUV=true:  economy SVD — U is m×k, S is k, V is n×k
 *     econ=false, computeUV=true:  full SVD    — U is m×m, S is k, V is n×n
 *     computeUV=false:             singular values only — S is k
 *     (k = min(m, n))
 */

#include "lapack_common.h"
#include <string>

// ── svd() ─────────────────────────────────────────────────────────────────────

Napi::Value Svd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()
      || !info[4].IsBoolean()) {
    Napi::TypeError::New(env,
      "svd: expected (Float64Array data, number m, number n, boolean econ, boolean computeUV)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::TypedArray>();
  if (arr.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "svd: data must be a Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m          = info[1].As<Napi::Number>().Int32Value();
  int n          = info[2].As<Napi::Number>().Int32Value();
  bool econ      = info[3].As<Napi::Boolean>().Value();
  bool computeUV = info[4].As<Napi::Boolean>().Value();

  if (m <= 0 || n <= 0 || static_cast<int>(arr.ElementLength()) != m * n) {
    Napi::RangeError::New(env, "svd: data.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arr = info[0].As<Napi::Float64Array>();
  int k = m < n ? m : n;

  std::vector<double> a(m * n);
  std::memcpy(a.data(), float64arr.Data(), m * n * sizeof(double));

  std::vector<double> s(k);
  int info_val = 0;

  char jobz;
  if (!computeUV) jobz = 'N';
  else if (econ)  jobz = 'S';
  else            jobz = 'A';

  int ldu, ldvt;
  std::vector<double> u_vec, vt_vec;

  if (jobz == 'N') {
    ldu = m; ldvt = n;
  } else if (jobz == 'S') {
    ldu = m; ldvt = k;
    u_vec.resize(m * k);
    vt_vec.resize(k * n);
  } else {
    ldu = m; ldvt = n;
    u_vec.resize(m * m);
    vt_vec.resize(n * n);
  }

  double* u_ptr  = (jobz == 'N') ? nullptr : u_vec.data();
  double* vt_ptr = (jobz == 'N') ? nullptr : vt_vec.data();

  int lwork = -1;
  double work_query = 0.0;
  std::vector<int> iwork(8 * k);
  dgesdd_(&jobz, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu, vt_ptr, &ldvt,
          &work_query, &lwork, iwork.data(), &info_val);

  lwork = static_cast<int>(work_query);
  if (lwork < 1) lwork = 3 * k + std::max(m, n);

  std::vector<double> work(lwork);
  dgesdd_(&jobz, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu, vt_ptr, &ldvt,
          work.data(), &lwork, iwork.data(), &info_val);

  if (!checkLapackInfo(env, info_val, "svd", "dgesdd"))
    return env.Null();

  auto result = Napi::Object::New(env);
  result.Set("S", vecToF64(env, s));

  if (computeUV) {
    result.Set("U", vecToF64(env, u_vec));

    // V = VT^T (column-major transpose)
    int vt_rows = (jobz == 'S') ? k : n;
    int vt_cols = n;
    std::vector<double> v_vec(vt_rows * vt_cols);
    for (int i = 0; i < vt_rows; i++)
      for (int j = 0; j < vt_cols; j++)
        v_vec[j + i * vt_cols] = vt_vec[i + j * vt_rows];

    result.Set("V", vecToF64(env, v_vec));
  }

  return result;
}

// ── svdComplex() ─────────────────────────────────────────────────────────────

Napi::Value SvdComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 6
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsNumber()
      || !info[4].IsBoolean()
      || !info[5].IsBoolean()) {
    Napi::TypeError::New(env,
      "svdComplex: expected (Float64Array dataRe, Float64Array dataIm, "
      "number m, number n, boolean econ, boolean computeUV)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrRe = info[0].As<Napi::TypedArray>();
  auto arrIm = info[1].As<Napi::TypedArray>();
  if (arrRe.TypedArrayType() != napi_float64_array ||
      arrIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "svdComplex: data must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m          = info[2].As<Napi::Number>().Int32Value();
  int n          = info[3].As<Napi::Number>().Int32Value();
  bool econ      = info[4].As<Napi::Boolean>().Value();
  bool computeUV = info[5].As<Napi::Boolean>().Value();

  if (m <= 0 || n <= 0
      || static_cast<int>(arrRe.ElementLength()) != m * n
      || static_cast<int>(arrIm.ElementLength()) != m * n) {
    Napi::RangeError::New(env, "svdComplex: data.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int k = m < n ? m : n;

  auto a = splitToInterleaved(
      info[0].As<Napi::Float64Array>(),
      info[1].As<Napi::Float64Array>(), m * n);

  std::vector<double> s(k);
  int info_val = 0;

  char jobz;
  if (!computeUV) jobz = 'N';
  else if (econ)  jobz = 'S';
  else            jobz = 'A';

  int ldu, ldvt;
  std::vector<lapack_complex_double> u_vec, vt_vec;

  if (jobz == 'N') {
    ldu = m; ldvt = n;
  } else if (jobz == 'S') {
    ldu = m; ldvt = k;
    u_vec.resize(m * k);
    vt_vec.resize(k * n);
  } else {
    ldu = m; ldvt = n;
    u_vec.resize(m * m);
    vt_vec.resize(n * n);
  }

  lapack_complex_double* u_ptr  = (jobz == 'N') ? nullptr : u_vec.data();
  lapack_complex_double* vt_ptr = (jobz == 'N') ? nullptr : vt_vec.data();

  // rwork size for zgesdd
  int rwork_size;
  if (jobz == 'N') {
    rwork_size = 7 * k;
  } else {
    int t1 = 5 * k + 7;
    int t2 = 2 * std::max(m, n) + 2 * k + 1;
    rwork_size = k * std::max(t1, t2);
  }
  std::vector<double> rwork(rwork_size);
  std::vector<int> iwork(8 * k);

  // Keep a copy for potential zgesvd fallback
  std::vector<lapack_complex_double> a_backup(a);

  // Workspace query
  int lwork = -1;
  lapack_complex_double work_query;
  zgesdd_(&jobz, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu, vt_ptr, &ldvt,
          &work_query, &lwork, rwork.data(), iwork.data(), &info_val);

  lwork = static_cast<int>(work_query.real);
  if (lwork < 1) lwork = 3 * k + std::max(m, n);

  std::vector<lapack_complex_double> work(lwork);
  zgesdd_(&jobz, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu, vt_ptr, &ldvt,
          work.data(), &lwork, rwork.data(), iwork.data(), &info_val);

  // If zgesdd fails, fall back to zgesvd (standard algorithm, more robust)
  if (info_val != 0) {
    a = a_backup;
    info_val = 0;

    char jobu, jobvt;
    if (jobz == 'N') { jobu = 'N'; jobvt = 'N'; }
    else if (jobz == 'S') { jobu = 'S'; jobvt = 'S'; }
    else { jobu = 'A'; jobvt = 'A'; }

    std::vector<double> rwork_svd(5 * k);

    lwork = -1;
    zgesvd_(&jobu, &jobvt, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu,
            vt_ptr, &ldvt, &work_query, &lwork, rwork_svd.data(), &info_val);

    lwork = static_cast<int>(work_query.real);
    if (lwork < 1) lwork = 2 * k + std::max(m, n);

    work.resize(lwork);
    info_val = 0;
    zgesvd_(&jobu, &jobvt, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu,
            vt_ptr, &ldvt, work.data(), &lwork, rwork_svd.data(), &info_val);

    if (info_val != 0) {
      std::string msg = "svdComplex: zgesvd failed (info=" + std::to_string(info_val)
                      + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ")";
      Napi::Error::New(env, msg).ThrowAsJavaScriptException();
      return env.Null();
    }
  }

  auto result = Napi::Object::New(env);
  result.Set("S", vecToF64(env, s));

  if (computeUV) {
    int u_size = (jobz == 'S') ? m * k : m * m;
    setSplitComplex(env, result, "URe", "UIm", u_vec.data(), u_size);

    // V = conj(VT^T): conjugate transpose
    int vt_rows = (jobz == 'S') ? k : n;
    int vt_cols = n;
    int v_size = vt_rows * vt_cols;
    auto VRe = Napi::Float64Array::New(env, static_cast<size_t>(v_size));
    auto VIm = Napi::Float64Array::New(env, static_cast<size_t>(v_size));
    for (int i = 0; i < vt_rows; i++) {
      for (int j = 0; j < vt_cols; j++) {
        int v_idx = j + i * vt_cols;
        int vt_idx = i + j * vt_rows;
        VRe[v_idx] = vt_vec[vt_idx].real;
        VIm[v_idx] = -vt_vec[vt_idx].imag;  // conjugate
      }
    }
    result.Set("VRe", VRe);
    result.Set("VIm", VIm);
  }

  return result;
}
