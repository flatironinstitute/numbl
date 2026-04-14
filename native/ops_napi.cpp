/**
 * N-API wrappers for the new tensor-ops layer (native/ops/).
 *
 * Each export is a thin shim: validate args, get raw pointers via Data(),
 * call the C core, throw on negative return.  The output buffer is
 * caller-allocated (passed in as the last argument).
 *
 * Categories exposed:
 *   tensorOpRealBinary(op, n, a, b, out)
 *   tensorOpRealScalarBinary(op, n, scalar, arr, scalarOnLeft, out)
 *   tensorOpComplexBinary(op, n, aRe, aIm|null, bRe, bIm|null, outRe, outIm)
 *   tensorOpComplexScalarBinary(op, n, sRe, sIm, arrRe, arrIm|null, scalarOnLeft, outRe, outIm)
 *   tensorOpDumpCodes()  →  string (used only by drift-detection test)
 */

#include <napi.h>

extern "C" {
  #include "ops/numbl_ops.h"
}

namespace {

inline void ThrowOnError(Napi::Env env, int code, const char* op_name) {
  if (code != NUMBL_OK) {
    std::string msg = std::string(op_name) + ": " + numbl_strerror(code);
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
  }
}

inline const double* AsF64(const Napi::Value& v) {
  return v.As<Napi::Float64Array>().Data();
}

inline double* AsF64Mut(const Napi::Value& v) {
  return v.As<Napi::Float64Array>().Data();
}

inline const double* AsF64OrNull(const Napi::Value& v) {
  if (v.IsNull() || v.IsUndefined()) return nullptr;
  return v.As<Napi::Float64Array>().Data();
}

}  // namespace

Napi::Value TensorOpRealBinary(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 5) {
    Napi::TypeError::New(env,
      "tensorOpRealBinary(op, n, a, b, out)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  int op = info[0].As<Napi::Number>().Int32Value();
  size_t n = (size_t)info[1].As<Napi::Number>().Int64Value();
  int rc = numbl_real_binary_elemwise(
      op, n, AsF64(info[2]), AsF64(info[3]), AsF64Mut(info[4]));
  ThrowOnError(env, rc, "tensorOpRealBinary");
  return env.Undefined();
}

Napi::Value TensorOpRealScalarBinary(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 6) {
    Napi::TypeError::New(env,
      "tensorOpRealScalarBinary(op, n, scalar, arr, scalarOnLeft, out)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  int op = info[0].As<Napi::Number>().Int32Value();
  size_t n = (size_t)info[1].As<Napi::Number>().Int64Value();
  double scalar = info[2].As<Napi::Number>().DoubleValue();
  int scalar_on_left = info[4].As<Napi::Boolean>().Value() ? 1 : 0;
  int rc = numbl_real_scalar_binary_elemwise(
      op, n, scalar, AsF64(info[3]), scalar_on_left, AsF64Mut(info[5]));
  ThrowOnError(env, rc, "tensorOpRealScalarBinary");
  return env.Undefined();
}

Napi::Value TensorOpComplexBinary(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 8) {
    Napi::TypeError::New(env,
      "tensorOpComplexBinary(op, n, aRe, aIm|null, bRe, bIm|null, outRe, outIm)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  int op = info[0].As<Napi::Number>().Int32Value();
  size_t n = (size_t)info[1].As<Napi::Number>().Int64Value();
  int rc = numbl_complex_binary_elemwise(
      op, n,
      AsF64(info[2]), AsF64OrNull(info[3]),
      AsF64(info[4]), AsF64OrNull(info[5]),
      AsF64Mut(info[6]), AsF64Mut(info[7]));
  ThrowOnError(env, rc, "tensorOpComplexBinary");
  return env.Undefined();
}

Napi::Value TensorOpComplexScalarBinary(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 9) {
    Napi::TypeError::New(env,
      "tensorOpComplexScalarBinary(op, n, sRe, sIm, arrRe, arrIm|null, scalarOnLeft, outRe, outIm)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  int op = info[0].As<Napi::Number>().Int32Value();
  size_t n = (size_t)info[1].As<Napi::Number>().Int64Value();
  double s_re = info[2].As<Napi::Number>().DoubleValue();
  double s_im = info[3].As<Napi::Number>().DoubleValue();
  int scalar_on_left = info[6].As<Napi::Boolean>().Value() ? 1 : 0;
  int rc = numbl_complex_scalar_binary_elemwise(
      op, n, s_re, s_im,
      AsF64(info[4]), AsF64OrNull(info[5]),
      scalar_on_left,
      AsF64Mut(info[7]), AsF64Mut(info[8]));
  ThrowOnError(env, rc, "tensorOpComplexScalarBinary");
  return env.Undefined();
}

Napi::Value TensorOpDumpCodes(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t need = numbl_dump_op_codes(nullptr, 0);
  std::string buf(need, '\0');
  numbl_dump_op_codes(&buf[0], need + 1);
  return Napi::String::New(env, buf);
}
