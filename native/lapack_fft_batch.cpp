/**
 * fftAlongDim() — Batch FFT along a single dimension of a column-major tensor.
 */

#include "numbl_addon_common.h"

#ifdef NUMBL_USE_DUCC0

#include <algorithm>
#include <complex>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#include "ducc0/fft/fft.h"
#include "ducc0/infra/mav.h"

namespace {

using Complex64 = std::complex<double>;
using Shape = ducc0::cfmav<Complex64>::shape_t;
using Stride = ducc0::cfmav<Complex64>::stride_t;

Shape toShape(const std::vector<int>& shape) {
  Shape out(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    out[i] = static_cast<size_t>(shape[i]);
  }
  return out;
}

Stride columnMajorStrides(const Shape& shape) {
  Stride strides(shape.size(), 1);
  for (size_t i = 1; i < shape.size(); ++i) {
    strides[i] = strides[i - 1] * static_cast<std::ptrdiff_t>(shape[i - 1]);
  }
  return strides;
}

size_t totalSize(const Shape& shape) {
  size_t total = 1;
  for (size_t extent : shape) {
    total *= extent;
  }
  return total;
}

void copyAxisInput(
  const double* re,
  const double* im,
  const Shape& shape,
  int dim,
  int outAxisLength,
  const Stride& outStride,
  std::vector<Complex64>& dst
) {
  const size_t axisLength = shape[static_cast<size_t>(dim)];
  const size_t strideDim =
    static_cast<size_t>(outStride[static_cast<size_t>(dim)]);
  const size_t copyLength =
    std::min(axisLength, static_cast<size_t>(outAxisLength));
  size_t numAbove = 1;
  for (int d = dim + 1; d < static_cast<int>(shape.size()); ++d) {
    numAbove *= shape[static_cast<size_t>(d)];
  }

  for (size_t outer = 0; outer < numAbove; ++outer) {
    for (size_t inner = 0; inner < strideDim; ++inner) {
      const size_t inBase = inner + outer * strideDim * axisLength;
      const size_t outBase =
        inner + outer * strideDim * static_cast<size_t>(outAxisLength);
      for (size_t k = 0; k < copyLength; ++k) {
        const size_t inIdx = inBase + k * strideDim;
        const size_t outIdx = outBase + k * strideDim;
        dst[outIdx] = Complex64(re[inIdx], im == nullptr ? 0.0 : im[inIdx]);
      }
    }
  }
}

}  // namespace

#else

#include <fftw3.h>

#endif

Napi::Value FftAlongDim(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 6) {
    Napi::TypeError::New(
      env,
      "fftAlongDim: expected (re, im|null, shape, dim, n, inverse)"
    )
      .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto inReArr = info[0].As<Napi::Float64Array>();
  double* pInRe = inReArr.Data();
  int inLen = static_cast<int>(inReArr.ElementLength());

  bool hasImag = !info[1].IsNull() && !info[1].IsUndefined();
  double* pInIm = nullptr;
  Napi::Float64Array inImArr;
  if (hasImag) {
    inImArr = info[1].As<Napi::Float64Array>();
    pInIm = inImArr.Data();
  }

  auto shapeArr = info[2].As<Napi::Array>();
  int dim = info[3].As<Napi::Number>().Int32Value();
  int n = info[4].As<Napi::Number>().Int32Value();
  bool inverse = info[5].As<Napi::Boolean>().Value();

  const int ndim = static_cast<int>(shapeArr.Length());
  std::vector<int> shape(static_cast<size_t>(ndim));
  for (int i = 0; i < ndim; ++i) {
    shape[static_cast<size_t>(i)] =
      ((Napi::Value)shapeArr[static_cast<uint32_t>(i)])
        .As<Napi::Number>()
        .Int32Value();
  }

  if (ndim <= 0 || dim < 0 || dim >= ndim || n <= 0) {
    Napi::RangeError::New(env, "fftAlongDim: invalid shape, dim, or n")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  for (int extent : shape) {
    if (extent <= 0) {
      Napi::RangeError::New(env, "fftAlongDim: shape extents must be positive")
        .ThrowAsJavaScriptException();
      return env.Null();
    }
  }
  if (hasImag && inImArr.ElementLength() != inReArr.ElementLength()) {
    Napi::RangeError::New(
      env,
      "fftAlongDim: imaginary input length must match real input length"
    )
      .ThrowAsJavaScriptException();
    return env.Null();
  }

  size_t expectedInputLength = 1;
  for (int extent : shape) {
    const size_t extentSize = static_cast<size_t>(extent);
    if (expectedInputLength > std::numeric_limits<size_t>::max() / extentSize) {
      Napi::RangeError::New(env, "fftAlongDim: shape is too large")
        .ThrowAsJavaScriptException();
      return env.Null();
    }
    expectedInputLength *= extentSize;
  }
  if (expectedInputLength != static_cast<size_t>(inLen)) {
    Napi::RangeError::New(
      env,
      "fftAlongDim: input length must match prod(shape)"
    )
      .ThrowAsJavaScriptException();
    return env.Null();
  }

#ifdef NUMBL_USE_DUCC0
  try {
    const Shape inShape = toShape(shape);
    Shape outShape = inShape;
    outShape[static_cast<size_t>(dim)] = static_cast<size_t>(n);
    const Stride outStride = columnMajorStrides(outShape);
    const size_t outLen = totalSize(outShape);

    std::vector<Complex64> padded(outLen, Complex64(0.0, 0.0));
    std::vector<Complex64> output(outLen);
    copyAxisInput(pInRe, pInIm, inShape, dim, n, outStride, padded);

    const Shape axes{static_cast<size_t>(dim)};
    ducc0::c2c(
      ducc0::cfmav<Complex64>(padded.data(), outShape, outStride),
      ducc0::vfmav<Complex64>(output.data(), outShape, outStride),
      axes,
      !inverse,
      1.0,
      1
    );

    auto outReArr = Napi::Float64Array::New(env, outLen);
    auto outImArr = Napi::Float64Array::New(env, outLen);
    double* pOutRe = outReArr.Data();
    double* pOutIm = outImArr.Data();
    for (size_t i = 0; i < outLen; ++i) {
      pOutRe[i] = output[i].real();
      pOutIm[i] = output[i].imag();
    }

    auto result = Napi::Object::New(env);
    result.Set("re", outReArr);
    result.Set("im", outImArr);
    return result;
  } catch (const std::exception& error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
#else
  int dimSize = shape[static_cast<size_t>(dim)];

  int strideDim = 1;
  for (int d = 0; d < dim; ++d) strideDim *= shape[static_cast<size_t>(d)];

  int numAbove = 1;
  for (int d = dim + 1; d < ndim; ++d) {
    numAbove *= shape[static_cast<size_t>(d)];
  }

  int outTotal = strideDim * n * numAbove;

  auto outReArr = Napi::Float64Array::New(env, static_cast<size_t>(outTotal));
  auto outImArr = Napi::Float64Array::New(env, static_cast<size_t>(outTotal));
  double* pOutRe = outReArr.Data();
  double* pOutIm = outImArr.Data();
  std::memset(pOutRe, 0, outTotal * sizeof(double));
  std::memset(pOutIm, 0, outTotal * sizeof(double));

  std::vector<double> zeroImBuf;
  if (!pInIm) {
    zeroImBuf.resize(static_cast<size_t>(inLen), 0.0);
    pInIm = zeroImBuf.data();
  }

  if (n == dimSize) {
    fftw_iodim transformDim;
    transformDim.n = dimSize;
    transformDim.is = strideDim;
    transformDim.os = strideDim;

    fftw_iodim batchDims[2];
    batchDims[0].n = strideDim;
    batchDims[0].is = 1;
    batchDims[0].os = 1;
    batchDims[1].n = numAbove;
    batchDims[1].is = strideDim * dimSize;
    batchDims[1].os = strideDim * dimSize;

    fftw_plan plan;
    if (!inverse) {
      plan = fftw_plan_guru_split_dft(
        1,
        &transformDim,
        2,
        batchDims,
        pInRe,
        pInIm,
        pOutRe,
        pOutIm,
        FFTW_ESTIMATE
      );
    } else {
      plan = fftw_plan_guru_split_dft(
        1,
        &transformDim,
        2,
        batchDims,
        pInIm,
        pInRe,
        pOutIm,
        pOutRe,
        FFTW_ESTIMATE
      );
    }

    if (plan) {
      fftw_execute(plan);
      fftw_destroy_plan(plan);
    }
  } else {
    int copyLen = std::min(n, dimSize);
    std::vector<double> workRe(static_cast<size_t>(outTotal), 0.0);
    std::vector<double> workIm(static_cast<size_t>(outTotal), 0.0);

    for (int outer = 0; outer < numAbove; ++outer) {
      for (int inner = 0; inner < strideDim; ++inner) {
        int inBase = inner + outer * strideDim * dimSize;
        int outBase = inner + outer * strideDim * n;
        for (int k = 0; k < copyLen; ++k) {
          workRe[static_cast<size_t>(outBase + k * strideDim)] =
            pInRe[inBase + k * strideDim];
          workIm[static_cast<size_t>(outBase + k * strideDim)] =
            pInIm[inBase + k * strideDim];
        }
      }
    }

    fftw_iodim transformDim;
    transformDim.n = n;
    transformDim.is = strideDim;
    transformDim.os = strideDim;

    fftw_iodim batchDims[2];
    batchDims[0].n = strideDim;
    batchDims[0].is = 1;
    batchDims[0].os = 1;
    batchDims[1].n = numAbove;
    batchDims[1].is = strideDim * n;
    batchDims[1].os = strideDim * n;

    fftw_plan plan;
    if (!inverse) {
      plan = fftw_plan_guru_split_dft(
        1,
        &transformDim,
        2,
        batchDims,
        workRe.data(),
        workIm.data(),
        pOutRe,
        pOutIm,
        FFTW_ESTIMATE
      );
    } else {
      plan = fftw_plan_guru_split_dft(
        1,
        &transformDim,
        2,
        batchDims,
        workIm.data(),
        workRe.data(),
        pOutIm,
        pOutRe,
        FFTW_ESTIMATE
      );
    }

    if (plan) {
      fftw_execute(plan);
      fftw_destroy_plan(plan);
    }
  }

  auto result = Napi::Object::New(env);
  result.Set("re", outReArr);
  result.Set("im", outImArr);
  return result;
#endif
}
