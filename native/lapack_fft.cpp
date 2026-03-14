/**
 * fft1d() and fft1dComplex() — 1D FFT via FFTW3.
 *
 *   fft1d(re: Float64Array, n: number, inverse: boolean):
 *         { re: Float64Array, im: Float64Array }
 *     Forward/inverse 1D FFT of real-only input. Imaginary part is assumed zero.
 *     Does NOT normalize for inverse (caller handles 1/n scaling).
 *
 *   fft1dComplex(re: Float64Array, im: Float64Array, n: number, inverse: boolean):
 *               { re: Float64Array, im: Float64Array }
 *     Forward/inverse 1D FFT of complex input (split re/im).
 *     Does NOT normalize for inverse (caller handles 1/n scaling).
 */

#include "lapack_common.h"
#include <fftw3.h>

// Shared core: run FFTW on pre-filled input, return {re, im} result object.
static Napi::Value fftCore(Napi::Env env, fftw_complex* in, int n, bool inverse) {
  fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);

  int sign = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
  fftw_plan plan = fftw_plan_dft_1d(n, in, out, sign, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  fftw_free(in);

  auto resultRe = Napi::Float64Array::New(env, static_cast<size_t>(n));
  auto resultIm = Napi::Float64Array::New(env, static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    resultRe[i] = out[i][0];
    resultIm[i] = out[i][1];
  }
  fftw_free(out);

  auto result = Napi::Object::New(env);
  result.Set("re", resultRe);
  result.Set("im", resultIm);
  return result;
}

// ── fft1d() ───────────────────────────────────────────────────────────────────

Napi::Value Fft1d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsBoolean()) {
    Napi::TypeError::New(env,
      "fft1d: expected (Float64Array re, number n, boolean inverse)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::TypedArray>();
  if (arr.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "fft1d: re must be a Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[1].As<Napi::Number>().Int32Value();
  bool inverse = info[2].As<Napi::Boolean>().Value();

  if (n <= 0 || static_cast<int>(arr.ElementLength()) != n) {
    Napi::RangeError::New(env, "fft1d: re.length must equal n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arr = info[0].As<Napi::Float64Array>();

  fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
  for (int i = 0; i < n; ++i) {
    in[i][0] = float64arr[i];
    in[i][1] = 0.0;
  }

  return fftCore(env, in, n, inverse);
}

// ── fft1dComplex() ──────────────────────────────────────────────────────────

Napi::Value Fft1dComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 4
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()) {
    Napi::TypeError::New(env,
      "fft1dComplex: expected (Float64Array re, Float64Array im, number n, boolean inverse)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrRe = info[0].As<Napi::TypedArray>();
  auto arrIm = info[1].As<Napi::TypedArray>();

  if (arrRe.TypedArrayType() != napi_float64_array ||
      arrIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env,
      "fft1dComplex: re and im must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[2].As<Napi::Number>().Int32Value();
  bool inverse = info[3].As<Napi::Boolean>().Value();

  if (n <= 0 ||
      static_cast<int>(arrRe.ElementLength()) != n ||
      static_cast<int>(arrIm.ElementLength()) != n) {
    Napi::RangeError::New(env,
      "fft1dComplex: re.length and im.length must equal n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arrRe = info[0].As<Napi::Float64Array>();
  auto float64arrIm = info[1].As<Napi::Float64Array>();

  fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
  for (int i = 0; i < n; ++i) {
    in[i][0] = float64arrRe[i];
    in[i][1] = float64arrIm[i];
  }

  return fftCore(env, in, n, inverse);
}
