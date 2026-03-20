/**
 * Element-wise binary operations on Float64Arrays.
 *
 * Real:
 *   elemwise(a: Float64Array, b: Float64Array, op: number): Float64Array
 *     op: 0=add, 1=sub, 2=mul, 3=div
 *
 * Complex:
 *   elemwiseComplex(aRe: Float64Array, aIm: Float64Array,
 *                   bRe: Float64Array, bIm: Float64Array,
 *                   op: number): { re: Float64Array, im: Float64Array }
 *     op: 0=add, 1=sub, 2=mul, 3=div
 *     Pass null for aIm or bIm to treat as zero (mixed real/complex).
 */

#include "numbl_addon_common.h"

// ── elemwise() — real element-wise binary op ────────────────────────────────

Napi::Value Elemwise(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()) {
    Napi::TypeError::New(env,
      "elemwise: expected (Float64Array a, Float64Array b, number op)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrA = info[0].As<Napi::Float64Array>();
  auto arrB = info[1].As<Napi::Float64Array>();
  int op = info[2].As<Napi::Number>().Int32Value();

  size_t n = arrA.ElementLength();
  if (arrB.ElementLength() != n) {
    Napi::RangeError::New(env, "elemwise: arrays must have same length")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto result = Napi::Float64Array::New(env, n);
  const double* a = arrA.Data();
  const double* b = arrB.Data();
  double* out = result.Data();

  switch (op) {
    case 0: // add
      for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
      break;
    case 1: // sub
      for (size_t i = 0; i < n; i++) out[i] = a[i] - b[i];
      break;
    case 2: // mul
      for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
      break;
    case 3: // div
      for (size_t i = 0; i < n; i++) out[i] = a[i] / b[i];
      break;
    default:
      Napi::RangeError::New(env, "elemwise: op must be 0-3")
          .ThrowAsJavaScriptException();
      return env.Null();
  }

  return result;
}

// ── elemwiseComplex() — complex element-wise binary op ──────────────────────

Napi::Value ElemwiseComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  // (aRe, aIm_or_null, bRe, bIm_or_null, op)
  if (info.Length() < 5 || !info[0].IsTypedArray() || !info[2].IsTypedArray()
      || !info[4].IsNumber()) {
    Napi::TypeError::New(env,
      "elemwiseComplex: expected (Float64Array aRe, Float64Array|null aIm, "
      "Float64Array bRe, Float64Array|null bIm, number op)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrARe = info[0].As<Napi::Float64Array>();
  auto arrBRe = info[2].As<Napi::Float64Array>();
  int op = info[4].As<Napi::Number>().Int32Value();

  size_t n = arrARe.ElementLength();
  if (arrBRe.ElementLength() != n) {
    Napi::RangeError::New(env, "elemwiseComplex: arrays must have same length")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  const double* aRe = arrARe.Data();
  const double* bRe = arrBRe.Data();

  // aIm and bIm may be null (treat as zero)
  bool hasAIm = info[1].IsTypedArray();
  bool hasBIm = info[3].IsTypedArray();
  const double* aIm = hasAIm ? info[1].As<Napi::Float64Array>().Data() : nullptr;
  const double* bIm = hasBIm ? info[3].As<Napi::Float64Array>().Data() : nullptr;

  auto outRe = Napi::Float64Array::New(env, n);
  auto outIm = Napi::Float64Array::New(env, n);
  double* oRe = outRe.Data();
  double* oIm = outIm.Data();

  switch (op) {
    case 0: // add
      for (size_t i = 0; i < n; i++) {
        oRe[i] = aRe[i] + bRe[i];
        oIm[i] = (aIm ? aIm[i] : 0.0) + (bIm ? bIm[i] : 0.0);
      }
      break;
    case 1: // sub
      for (size_t i = 0; i < n; i++) {
        oRe[i] = aRe[i] - bRe[i];
        oIm[i] = (aIm ? aIm[i] : 0.0) - (bIm ? bIm[i] : 0.0);
      }
      break;
    case 2: { // mul: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
      for (size_t i = 0; i < n; i++) {
        double ar = aRe[i], ai = aIm ? aIm[i] : 0.0;
        double br = bRe[i], bi = bIm ? bIm[i] : 0.0;
        oRe[i] = ar * br - ai * bi;
        oIm[i] = ar * bi + ai * br;
      }
      break;
    }
    case 3: { // div: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
      for (size_t i = 0; i < n; i++) {
        double ar = aRe[i], ai = aIm ? aIm[i] : 0.0;
        double br = bRe[i], bi = bIm ? bIm[i] : 0.0;
        double denom = br * br + bi * bi;
        if (denom == 0.0) {
          oRe[i] = (ar == 0.0 && ai == 0.0) ? 0.0 / 0.0 /* NaN */
                    : (ar > 0 ? 1.0 : ar < 0 ? -1.0 : 0.0) / 0.0 /* ±Inf */;
          oIm[i] = (ar == 0.0 && ai == 0.0) ? 0.0
                    : (ai > 0 ? 1.0 : ai < 0 ? -1.0 : 0.0) / 0.0;
        } else {
          oRe[i] = (ar * br + ai * bi) / denom;
          oIm[i] = (ai * br - ar * bi) / denom;
        }
      }
      break;
    }
    default:
      Napi::RangeError::New(env, "elemwiseComplex: op must be 0-3")
          .ThrowAsJavaScriptException();
      return env.Null();
  }

  // Check if result is purely real
  bool isReal = true;
  for (size_t i = 0; i < n; i++) {
    if (oIm[i] != 0.0) { isReal = false; break; }
  }

  auto result = Napi::Object::New(env);
  result.Set("re", outRe);
  if (!isReal) {
    result.Set("im", outIm);
  }
  return result;
}
