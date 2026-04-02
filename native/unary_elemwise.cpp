/**
 * Native unary element-wise math operations on Float64Arrays.
 *
 *   unaryElemwise(arr: Float64Array, op: number): Float64Array
 *     op codes:
 *       0=exp, 1=log, 2=log2, 3=log10,
 *       4=sqrt, 5=abs, 6=floor, 7=ceil, 8=round, 9=trunc (fix),
 *       10=sin, 11=cos, 12=tan, 13=asin, 14=acos, 15=atan,
 *       16=sinh, 17=cosh, 18=tanh, 19=sign
 */

#include "numbl_addon_common.h"
#include <cmath>

Napi::Value UnaryElemwise(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()) {
    Napi::TypeError::New(env,
      "unaryElemwise: expected (Float64Array arr, number op)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::Float64Array>();
  int op = info[1].As<Napi::Number>().Int32Value();

  size_t n = arr.ElementLength();
  auto result = Napi::Float64Array::New(env, n);
  const double* a = arr.Data();
  double* out = result.Data();

  switch (op) {
    case 0:  for (size_t i = 0; i < n; i++) out[i] = std::exp(a[i]);   break;
    case 1:  for (size_t i = 0; i < n; i++) out[i] = std::log(a[i]);   break;
    case 2:  for (size_t i = 0; i < n; i++) out[i] = std::log2(a[i]);  break;
    case 3:  for (size_t i = 0; i < n; i++) out[i] = std::log10(a[i]); break;
    case 4:  for (size_t i = 0; i < n; i++) out[i] = std::sqrt(a[i]);  break;
    case 5:  for (size_t i = 0; i < n; i++) out[i] = std::abs(a[i]);   break;
    case 6:  for (size_t i = 0; i < n; i++) out[i] = std::floor(a[i]); break;
    case 7:  for (size_t i = 0; i < n; i++) out[i] = std::ceil(a[i]);  break;
    case 8:  for (size_t i = 0; i < n; i++) out[i] = std::round(a[i]); break;
    case 9:  for (size_t i = 0; i < n; i++) out[i] = std::trunc(a[i]); break;
    case 10: for (size_t i = 0; i < n; i++) out[i] = std::sin(a[i]);   break;
    case 11: for (size_t i = 0; i < n; i++) out[i] = std::cos(a[i]);   break;
    case 12: for (size_t i = 0; i < n; i++) out[i] = std::tan(a[i]);   break;
    case 13: for (size_t i = 0; i < n; i++) out[i] = std::asin(a[i]);  break;
    case 14: for (size_t i = 0; i < n; i++) out[i] = std::acos(a[i]);  break;
    case 15: for (size_t i = 0; i < n; i++) out[i] = std::atan(a[i]);  break;
    case 16: for (size_t i = 0; i < n; i++) out[i] = std::sinh(a[i]);  break;
    case 17: for (size_t i = 0; i < n; i++) out[i] = std::cosh(a[i]);  break;
    case 18: for (size_t i = 0; i < n; i++) out[i] = std::tanh(a[i]);  break;
    case 19:
      for (size_t i = 0; i < n; i++) {
        out[i] = (a[i] > 0.0) ? 1.0 : (a[i] < 0.0) ? -1.0 : 0.0;
      }
      break;
    default:
      Napi::RangeError::New(env, "unaryElemwise: unknown op code")
          .ThrowAsJavaScriptException();
      return env.Null();
  }

  return result;
}
