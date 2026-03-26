#include <algorithm>
#include <complex>
#include <cstdint>
#include <exception>
#include <vector>

#include "ducc0/fft/fft.h"
#include "ducc0/infra/mav.h"

namespace {

using Complex64 = std::complex<double>;
using Shape = ducc0::cfmav<Complex64>::shape_t;
using Stride = ducc0::cfmav<Complex64>::stride_t;

Shape toShape(const int32_t *shape, int ndim) {
  Shape out(static_cast<size_t>(ndim));
  for (int i = 0; i < ndim; ++i) {
    out[static_cast<size_t>(i)] = static_cast<size_t>(shape[i]);
  }
  return out;
}

Stride columnMajorStrides(const Shape &shape) {
  Stride strides(shape.size(), 1);
  for (size_t i = 1; i < shape.size(); ++i) {
    strides[i] = strides[i - 1] * static_cast<std::ptrdiff_t>(shape[i - 1]);
  }
  return strides;
}

size_t totalSize(const Shape &shape) {
  size_t total = 1;
  for (size_t extent : shape) {
    total *= extent;
  }
  return total;
}

bool isValidShape(const int32_t *shape, int ndim) {
  if (shape == nullptr || ndim <= 0) return false;
  for (int i = 0; i < ndim; ++i) {
    if (shape[i] <= 0) return false;
  }
  return true;
}

void packComplex(const double *re, const double *im, size_t len, Complex64 *out) {
  for (size_t i = 0; i < len; ++i) {
    out[i] = Complex64(re[i], im == nullptr ? 0.0 : im[i]);
  }
}

void unpackComplex(const Complex64 *buffer, size_t len, double *outRe, double *outIm) {
  for (size_t i = 0; i < len; ++i) {
    outRe[i] = buffer[i].real();
    outIm[i] = buffer[i].imag();
  }
}

void copyAxisInput(
  const double *re,
  const double *im,
  const Shape &shape,
  int dim,
  int outAxisLength,
  const Stride &outStride,
  std::vector<Complex64> &dst
) {
  const size_t axisLength = shape[static_cast<size_t>(dim)];
  const size_t strideDim = static_cast<size_t>(outStride[static_cast<size_t>(dim)]);
  const size_t copyLength = std::min(axisLength, static_cast<size_t>(outAxisLength));
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

template <typename Func>
int guarded(Func &&func) {
  try {
    return func();
  } catch (const std::exception &) {
    return -1;
  } catch (...) {
    return -1;
  }
}

}  // namespace

extern "C" int numbl_fft1d_f64(
  const double *re,
  const double *im,
  int n,
  int inverse,
  double *outRe,
  double *outIm
) {
  return guarded([&]() -> int {
    if (re == nullptr || outRe == nullptr || outIm == nullptr || n <= 0) {
      return -1;
    }

    const size_t len = static_cast<size_t>(n);
    std::vector<Complex64> input(len);
    std::vector<Complex64> output(len);
    packComplex(re, im, len, input.data());

    const Shape shape{len};
    const Stride stride{1};
    const Shape axes{0};
    ducc0::c2c(
      ducc0::cfmav<Complex64>(input.data(), shape, stride),
      ducc0::vfmav<Complex64>(output.data(), shape, stride),
      axes,
      inverse == 0,
      1.0,
      1
    );

    unpackComplex(output.data(), len, outRe, outIm);
    return 0;
  });
}

extern "C" int numbl_fft_along_dim_f64(
  const double *re,
  const double *im,
  const int32_t *shape,
  int ndim,
  int dim,
  int n,
  int inverse,
  double *outRe,
  double *outIm
) {
  return guarded([&]() -> int {
    if (re == nullptr || outRe == nullptr || outIm == nullptr) {
      return -1;
    }
    if (!isValidShape(shape, ndim) || dim < 0 || dim >= ndim || n <= 0) {
      return -1;
    }

    const Shape inShape = toShape(shape, ndim);
    Shape outShape = inShape;
    outShape[static_cast<size_t>(dim)] = static_cast<size_t>(n);
    const Stride outStride = columnMajorStrides(outShape);
    const size_t outLen = totalSize(outShape);

    std::vector<Complex64> padded(outLen, Complex64(0.0, 0.0));
    std::vector<Complex64> output(outLen);
    copyAxisInput(re, im, inShape, dim, n, outStride, padded);

    const Shape axes{static_cast<size_t>(dim)};
    ducc0::c2c(
      ducc0::cfmav<Complex64>(padded.data(), outShape, outStride),
      ducc0::vfmav<Complex64>(output.data(), outShape, outStride),
      axes,
      inverse == 0,
      1.0,
      1
    );

    unpackComplex(output.data(), outLen, outRe, outIm);
    return 0;
  });
}
