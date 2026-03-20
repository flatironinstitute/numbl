/**
 * eig() — Eigenvalue decomposition via LAPACK dgeev.
 *
 *   eig(data: Float64Array, n: number, computeVL: boolean,
 *       computeVR: boolean, balance: boolean):
 *       {wr: Float64Array, wi: Float64Array, VL?: Float64Array, VR?: Float64Array}
 *
 *   eigComplex(dataRe, dataIm, n, computeVL, computeVR):
 *       {wRe, wIm, VLRe?, VLIm?, VRRe?, VRIm?}
 *
 *   Note: balance parameter is accepted but ignored — dgeev always balances.
 *   The ts-lapack bridge handles the nobalance case.
 */

#include "numbl_addon_common.h"

// ── eig() ─────────────────────────────────────────────────────────────────────

Napi::Value Eig(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsBoolean()
      || !info[3].IsBoolean()
      || !info[4].IsBoolean()) {
    Napi::TypeError::New(env,
      "eig: expected (Float64Array data, number n, boolean computeVL, "
      "boolean computeVR, boolean balance)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::TypedArray>();
  if (arr.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "eig: data must be a Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n           = info[1].As<Napi::Number>().Int32Value();
  bool computeVL  = info[2].As<Napi::Boolean>().Value();
  bool computeVR  = info[3].As<Napi::Boolean>().Value();
  // balance param (info[4]) accepted but not used — dgeev always balances.

  if (n <= 0 || static_cast<int>(arr.ElementLength()) != n * n) {
    Napi::RangeError::New(env, "eig: data.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arr = info[0].As<Napi::Float64Array>();

  std::vector<double> a(n * n);
  std::memcpy(a.data(), float64arr.Data(), n * n * sizeof(double));

  std::vector<double> wr(n), wi(n);

  char jobvl = computeVL ? 'V' : 'N';
  char jobvr = computeVR ? 'V' : 'N';
  int ldvl = computeVL ? n : 1;
  int ldvr = computeVR ? n : 1;
  std::vector<double> vl(computeVL ? n * n : 0);
  std::vector<double> vr(computeVR ? n * n : 0);

  int info_val = 0;

  // Workspace query
  int lwork = -1;
  double work_query = 0.0;
  dgeev_(&jobvl, &jobvr, &n, a.data(), &n,
         wr.data(), wi.data(),
         computeVL ? vl.data() : nullptr, &ldvl,
         computeVR ? vr.data() : nullptr, &ldvr,
         &work_query, &lwork, &info_val);

  lwork = static_cast<int>(work_query);
  if (lwork < 1) lwork = std::max(1, (computeVL || computeVR) ? 4 * n : 3 * n);

  std::vector<double> work(lwork);

  dgeev_(&jobvl, &jobvr, &n, a.data(), &n,
         wr.data(), wi.data(),
         computeVL ? vl.data() : nullptr, &ldvl,
         computeVR ? vr.data() : nullptr, &ldvr,
         work.data(), &lwork, &info_val);

  if (!checkLapackInfo(env, info_val, "eig", "dgeev",
      "QR algorithm failed to converge in dgeev"))
    return env.Null();

  auto result = Napi::Object::New(env);
  result.Set("wr", vecToF64(env, wr));
  result.Set("wi", vecToF64(env, wi));

  if (computeVL)
    result.Set("VL", vecToF64(env, vl));
  if (computeVR)
    result.Set("VR", vecToF64(env, vr));

  return result;
}

// ── eigComplex() ─────────────────────────────────────────────────────────────

Napi::Value EigComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()
      || !info[4].IsBoolean()) {
    Napi::TypeError::New(env,
      "eigComplex: expected (Float64Array dataRe, Float64Array dataIm, "
      "number n, boolean computeVL, boolean computeVR)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrRe = info[0].As<Napi::TypedArray>();
  auto arrIm = info[1].As<Napi::TypedArray>();
  if (arrRe.TypedArrayType() != napi_float64_array
      || arrIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "eigComplex: data must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n           = info[2].As<Napi::Number>().Int32Value();
  bool computeVL  = info[3].As<Napi::Boolean>().Value();
  bool computeVR  = info[4].As<Napi::Boolean>().Value();

  if (n <= 0
      || static_cast<int>(arrRe.ElementLength()) != n * n
      || static_cast<int>(arrIm.ElementLength()) != n * n) {
    Napi::RangeError::New(env, "eigComplex: data.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = splitToInterleaved(
      info[0].As<Napi::Float64Array>(),
      info[1].As<Napi::Float64Array>(), n * n);

  std::vector<lapack_complex_double> w(n);

  char jobvl = computeVL ? 'V' : 'N';
  char jobvr = computeVR ? 'V' : 'N';
  int ldvl = computeVL ? n : 1;
  int ldvr = computeVR ? n : 1;
  std::vector<lapack_complex_double> vl(computeVL ? n * n : 0);
  std::vector<lapack_complex_double> vr(computeVR ? n * n : 0);
  std::vector<double> rwork(2 * n);

  int info_val = 0;

  // Workspace query
  int lwork = -1;
  lapack_complex_double work_query;
  zgeev_(&jobvl, &jobvr, &n, a.data(), &n,
         w.data(),
         computeVL ? vl.data() : nullptr, &ldvl,
         computeVR ? vr.data() : nullptr, &ldvr,
         &work_query, &lwork, rwork.data(), &info_val);

  lwork = static_cast<int>(work_query.real);
  if (lwork < 1) lwork = std::max(1, 2 * n);

  std::vector<lapack_complex_double> work(lwork);

  zgeev_(&jobvl, &jobvr, &n, a.data(), &n,
         w.data(),
         computeVL ? vl.data() : nullptr, &ldvl,
         computeVR ? vr.data() : nullptr, &ldvr,
         work.data(), &lwork, rwork.data(), &info_val);

  if (!checkLapackInfo(env, info_val, "eigComplex", "zgeev",
      "QR algorithm failed to converge in zgeev"))
    return env.Null();

  auto result = Napi::Object::New(env);
  setSplitComplex(env, result, "wRe", "wIm", w.data(), n);

  if (computeVL)
    setSplitComplex(env, result, "VLRe", "VLIm", vl.data(), n * n);
  if (computeVR)
    setSplitComplex(env, result, "VRRe", "VRIm", vr.data(), n * n);

  return result;
}
