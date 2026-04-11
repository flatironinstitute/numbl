/**
 * Element-wise binary operations on Float64Arrays.
 *
 * Real:
 *   elemwise(a: Float64Array, b: Float64Array, op: number): Float64Array
 *     op: 0=add, 1=sub, 2=mul, 3=div
 *
 *   elemwiseScalar(scalar: number, arr: Float64Array, op: number, scalarOnLeft: boolean): Float64Array
 *     op: 0=add, 1=sub, 2=mul, 3=div
 *     scalarOnLeft=true:  result[i] = scalar op arr[i]
 *     scalarOnLeft=false: result[i] = arr[i] op scalar
 *
 * Complex:
 *   elemwiseComplex(aRe: Float64Array, aIm: Float64Array,
 *                   bRe: Float64Array, bIm: Float64Array,
 *                   op: number): { re: Float64Array, im: Float64Array }
 *     op: 0=add, 1=sub, 2=mul, 3=div
 *     Pass null for aIm or bIm to treat as zero (mixed real/complex).
 *
 *   elemwiseComplexScalar(scalarRe: number, scalarIm: number,
 *                         arrRe: Float64Array, arrIm: Float64Array|null,
 *                         op: number, scalarOnLeft: boolean)
 *     : { re: Float64Array, im: Float64Array }
 *     op: 0=add, 1=sub, 2=mul, 3=div
 *     Complex scalar with a possibly-complex tensor.  When scalar is purely
 *     real (scalarIm == 0) and the tensor is real (arrIm == null), prefer
 *     elemwiseScalar for better throughput.
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

// ── elemwiseScalar() — scalar-tensor element-wise binary op ────────────────

Napi::Value ElemwiseScalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 4
      || !info[0].IsNumber()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()) {
    Napi::TypeError::New(env,
      "elemwiseScalar: expected (number scalar, Float64Array arr, number op, boolean scalarOnLeft)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  double scalar = info[0].As<Napi::Number>().DoubleValue();
  auto arr = info[1].As<Napi::Float64Array>();
  int op = info[2].As<Napi::Number>().Int32Value();
  bool scalarOnLeft = info[3].As<Napi::Boolean>().Value();

  size_t n = arr.ElementLength();
  auto result = Napi::Float64Array::New(env, n);
  const double* a = arr.Data();
  double* out = result.Data();

  if (scalarOnLeft) {
    switch (op) {
      case 0: for (size_t i = 0; i < n; i++) out[i] = scalar + a[i]; break;
      case 1: for (size_t i = 0; i < n; i++) out[i] = scalar - a[i]; break;
      case 2: for (size_t i = 0; i < n; i++) out[i] = scalar * a[i]; break;
      case 3: for (size_t i = 0; i < n; i++) out[i] = scalar / a[i]; break;
      default:
        Napi::RangeError::New(env, "elemwiseScalar: op must be 0-3")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
  } else {
    switch (op) {
      case 0: for (size_t i = 0; i < n; i++) out[i] = a[i] + scalar; break;
      case 1: for (size_t i = 0; i < n; i++) out[i] = a[i] - scalar; break;
      case 2: for (size_t i = 0; i < n; i++) out[i] = a[i] * scalar; break;
      case 3: for (size_t i = 0; i < n; i++) out[i] = a[i] / scalar; break;
      default:
        Napi::RangeError::New(env, "elemwiseScalar: op must be 0-3")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
  }

  return result;
}

// ── elemwiseComplexScalar() — complex-scalar-tensor element-wise binary op ──

Napi::Value ElemwiseComplexScalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  // (scalarRe, scalarIm, arrRe, arrIm_or_null, op, scalarOnLeft)
  if (info.Length() < 6
      || !info[0].IsNumber()
      || !info[1].IsNumber()
      || !info[2].IsTypedArray()
      || !info[4].IsNumber()
      || !info[5].IsBoolean()) {
    Napi::TypeError::New(env,
      "elemwiseComplexScalar: expected (number sRe, number sIm, "
      "Float64Array arrRe, Float64Array|null arrIm, number op, "
      "boolean scalarOnLeft)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  double sRe = info[0].As<Napi::Number>().DoubleValue();
  double sIm = info[1].As<Napi::Number>().DoubleValue();
  auto arrRe = info[2].As<Napi::Float64Array>();
  int op = info[4].As<Napi::Number>().Int32Value();
  bool scalarOnLeft = info[5].As<Napi::Boolean>().Value();

  size_t n = arrRe.ElementLength();
  const double* aRe = arrRe.Data();
  const bool hasAIm = info[3].IsTypedArray();
  const double* aIm = hasAIm ? info[3].As<Napi::Float64Array>().Data() : nullptr;

  auto outRe = Napi::Float64Array::New(env, n);
  auto outIm = Napi::Float64Array::New(env, n);
  double* oRe = outRe.Data();
  double* oIm = outIm.Data();

  switch (op) {
    case 0: // add: result is (sRe + aRe) + (sIm + aIm) i
      for (size_t i = 0; i < n; i++) {
        oRe[i] = sRe + aRe[i];
        oIm[i] = sIm + (aIm ? aIm[i] : 0.0);
      }
      break;
    case 1: // sub
      if (scalarOnLeft) {
        for (size_t i = 0; i < n; i++) {
          oRe[i] = sRe - aRe[i];
          oIm[i] = sIm - (aIm ? aIm[i] : 0.0);
        }
      } else {
        for (size_t i = 0; i < n; i++) {
          oRe[i] = aRe[i] - sRe;
          oIm[i] = (aIm ? aIm[i] : 0.0) - sIm;
        }
      }
      break;
    case 2: // mul: (sRe + sIm i)(aRe + aIm i)
      for (size_t i = 0; i < n; i++) {
        double ar = aRe[i];
        double ai = aIm ? aIm[i] : 0.0;
        oRe[i] = sRe * ar - sIm * ai;
        oIm[i] = sRe * ai + sIm * ar;
      }
      break;
    case 3: // div
      if (scalarOnLeft) {
        // (sRe + sIm i) / (ar + ai i)
        for (size_t i = 0; i < n; i++) {
          double ar = aRe[i];
          double ai = aIm ? aIm[i] : 0.0;
          double denom = ar * ar + ai * ai;
          if (denom == 0.0) {
            // Match JS: produce Inf/NaN components.
            oRe[i] = (sRe == 0.0 && sIm == 0.0) ? 0.0 / 0.0
                      : (sRe > 0 ? 1.0 : sRe < 0 ? -1.0 : 0.0) / 0.0;
            oIm[i] = (sRe == 0.0 && sIm == 0.0) ? 0.0
                      : (sIm > 0 ? 1.0 : sIm < 0 ? -1.0 : 0.0) / 0.0;
          } else {
            oRe[i] = (sRe * ar + sIm * ai) / denom;
            oIm[i] = (sIm * ar - sRe * ai) / denom;
          }
        }
      } else {
        // (ar + ai i) / (sRe + sIm i)
        double denom = sRe * sRe + sIm * sIm;
        if (denom == 0.0) {
          for (size_t i = 0; i < n; i++) {
            double ar = aRe[i];
            double ai = aIm ? aIm[i] : 0.0;
            oRe[i] = (ar == 0.0 && ai == 0.0) ? 0.0 / 0.0
                      : (ar > 0 ? 1.0 : ar < 0 ? -1.0 : 0.0) / 0.0;
            oIm[i] = (ar == 0.0 && ai == 0.0) ? 0.0
                      : (ai > 0 ? 1.0 : ai < 0 ? -1.0 : 0.0) / 0.0;
          }
        } else {
          double invDenom = 1.0 / denom;
          for (size_t i = 0; i < n; i++) {
            double ar = aRe[i];
            double ai = aIm ? aIm[i] : 0.0;
            oRe[i] = (ar * sRe + ai * sIm) * invDenom;
            oIm[i] = (ai * sRe - ar * sIm) * invDenom;
          }
        }
      }
      break;
    default:
      Napi::RangeError::New(env, "elemwiseComplexScalar: op must be 0-3")
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
