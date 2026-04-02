/**
 * Native bulk fill for randn (Marsaglia polar method + xoshiro128**).
 *
 * fillRandn(state: Uint32Array(4), n: number, spare: number, hasSpare: boolean)
 *   → { data: Float64Array, spare: number, hasSpare: boolean }
 *
 * The xoshiro128** state is mutated in-place through the Uint32Array.
 * The caller passes in any cached Box-Muller spare and gets the updated spare back.
 */

#include "numbl_addon_common.h"
#include <cmath>

// Inline xoshiro128** — returns a uint32 and advances state in place.
static inline uint32_t xoshiro128ss(uint32_t* s) {
  uint32_t tmp = s[1] * 5;
  uint32_t result = ((tmp << 7) | (tmp >> 25)) * 9;
  uint32_t t = s[1] << 9;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = (s[3] << 11) | (s[3] >> 21);
  return result;
}

static inline double rngUniform(uint32_t* s) {
  return static_cast<double>(xoshiro128ss(s)) / 4294967296.0;
}

Napi::Value FillRandn(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 4
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()) {
    Napi::TypeError::New(env,
      "fillRandn: expected (Uint32Array state, number n, number spare, boolean hasSpare)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto stateArr = info[0].As<Napi::Uint32Array>();
  int n = info[1].As<Napi::Number>().Int32Value();
  double spare = info[2].As<Napi::Number>().DoubleValue();
  bool hasSpare = info[3].As<Napi::Boolean>().Value();

  if (stateArr.ElementLength() < 4) {
    Napi::RangeError::New(env, "fillRandn: state must have 4 elements")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  uint32_t* state = stateArr.Data();
  auto result = Napi::Float64Array::New(env, static_cast<size_t>(n));
  double* out = result.Data();

  int i = 0;

  // Drain cached spare
  if (hasSpare && i < n) {
    out[i++] = spare;
    hasSpare = false;
  }

  // Generate pairs via Marsaglia polar method
  for (; i + 1 < n; i += 2) {
    double u, v, s;
    do {
      u = 2.0 * rngUniform(state) - 1.0;
      v = 2.0 * rngUniform(state) - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    double mul = std::sqrt((-2.0 * std::log(s)) / s);
    out[i] = u * mul;
    out[i + 1] = v * mul;
  }

  // Handle odd trailing element
  if (i < n) {
    double u, v, s;
    do {
      u = 2.0 * rngUniform(state) - 1.0;
      v = 2.0 * rngUniform(state) - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    double mul = std::sqrt((-2.0 * std::log(s)) / s);
    out[i] = u * mul;
    spare = v * mul;
    hasSpare = true;
  }

  auto obj = Napi::Object::New(env);
  obj.Set("data", result);
  obj.Set("spare", Napi::Number::New(env, spare));
  obj.Set("hasSpare", Napi::Boolean::New(env, hasSpare));
  return obj;
}
