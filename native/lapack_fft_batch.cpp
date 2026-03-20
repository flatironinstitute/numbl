/**
 * fftAlongDim() — Batch FFT along a single dimension of a column-major tensor.
 *
 * Uses FFTW's guru split-complex interface (fftw_plan_guru_split_dft) to
 * transform ALL fibers in one or two FFTW calls, avoiding per-fiber overhead.
 *
 *   fftAlongDim(re, im_or_null, shape, dim, n, inverse):
 *     re:      Float64Array — real part of input tensor (column-major)
 *     im:      Float64Array | null — imaginary part (null → real input)
 *     shape:   Array<number> — tensor dimensions
 *     dim:     number — 0-based dimension to transform along
 *     n:       number — FFT length (may differ from shape[dim] for pad/truncate)
 *     inverse: boolean — true for inverse FFT
 *
 *   Returns { re: Float64Array, im: Float64Array } — output tensor (column-major)
 *   with the same shape except shape[dim] replaced by n.
 *   Does NOT normalize for inverse — caller handles 1/n scaling.
 */

#include "numbl_addon_common.h"
#include <fftw3.h>
#include <cmath>

Napi::Value FftAlongDim(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  // ── Parse arguments ──────────────────────────────────────────────────────

  if (info.Length() < 6) {
    Napi::TypeError::New(env,
      "fftAlongDim: expected (re, im|null, shape, dim, n, inverse)")
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

  int ndim = shapeArr.Length();
  std::vector<int> shape(ndim);
  for (int i = 0; i < ndim; i++) {
    shape[i] = ((Napi::Value)shapeArr[static_cast<uint32_t>(i)])
                   .As<Napi::Number>().Int32Value();
  }

  int dimSize = shape[dim];

  // ── Compute strides ──────────────────────────────────────────────────────

  // strideDim = product of dims before dim (inner stride)
  int strideDim = 1;
  for (int d = 0; d < dim; d++) strideDim *= shape[d];

  // numAbove = product of dims after dim (outer batches)
  int numAbove = 1;
  for (int d = dim + 1; d < ndim; d++) numAbove *= shape[d];

  // Output total elements
  int outTotal = strideDim * n * numAbove;

  // ── Allocate output ──────────────────────────────────────────────────────

  auto outReArr = Napi::Float64Array::New(env, static_cast<size_t>(outTotal));
  auto outImArr = Napi::Float64Array::New(env, static_cast<size_t>(outTotal));
  double* pOutRe = outReArr.Data();
  double* pOutIm = outImArr.Data();

  // Zero output (needed for padding case)
  std::memset(pOutRe, 0, outTotal * sizeof(double));
  std::memset(pOutIm, 0, outTotal * sizeof(double));

  // ── Prepare zero imaginary buffer if input is real ───────────────────────

  std::vector<double> zeroImBuf;
  if (!pInIm) {
    zeroImBuf.resize(inLen, 0.0);
    pInIm = zeroImBuf.data();
  }

  // ── Case 1: n == dimSize — direct guru transform on input data ──────────

  if (n == dimSize) {
    // FFTW guru: 1 transform dimension + 2 batch dimensions
    fftw_iodim transformDim;
    transformDim.n  = dimSize;
    transformDim.is = strideDim;
    transformDim.os = strideDim;

    fftw_iodim batchDims[2];
    // Inner batch: strideDim fibers with dist=1
    batchDims[0].n  = strideDim;
    batchDims[0].is = 1;
    batchDims[0].os = 1;
    // Outer batch: numAbove slabs
    batchDims[1].n  = numAbove;
    batchDims[1].is = strideDim * dimSize;
    batchDims[1].os = strideDim * dimSize;

    int howmanyRank = 2;

    fftw_plan plan;
    if (!inverse) {
      plan = fftw_plan_guru_split_dft(
        1, &transformDim, howmanyRank, batchDims,
        pInRe, pInIm, pOutRe, pOutIm, FFTW_ESTIMATE);
    } else {
      // Inverse DFT with split format: swap re<->im on both input and output
      plan = fftw_plan_guru_split_dft(
        1, &transformDim, howmanyRank, batchDims,
        pInIm, pInRe, pOutIm, pOutRe, FFTW_ESTIMATE);
    }

    if (plan) {
      fftw_execute(plan);
      fftw_destroy_plan(plan);
    }
  }

  // ── Case 2: n != dimSize — copy to work buffer with padding/truncation ──

  else {
    int copyLen = std::min(n, dimSize);

    // Allocate work buffers for input (padded/truncated to n per fiber)
    std::vector<double> workRe(outTotal, 0.0);
    std::vector<double> workIm(outTotal, 0.0);

    // Copy input fibers into work buffer with padding/truncation
    for (int outer = 0; outer < numAbove; outer++) {
      for (int inner = 0; inner < strideDim; inner++) {
        int inBase  = inner + outer * strideDim * dimSize;
        int outBase = inner + outer * strideDim * n;
        for (int k = 0; k < copyLen; k++) {
          workRe[outBase + k * strideDim] = pInRe[inBase + k * strideDim];
          workIm[outBase + k * strideDim] = pInIm[inBase + k * strideDim];
        }
        // Remaining elements are already zero from initialization
      }
    }

    // FFTW guru on the work buffer (now has n elements per fiber)
    fftw_iodim transformDim;
    transformDim.n  = n;
    transformDim.is = strideDim;
    transformDim.os = strideDim;

    fftw_iodim batchDims[2];
    batchDims[0].n  = strideDim;
    batchDims[0].is = 1;
    batchDims[0].os = 1;
    batchDims[1].n  = numAbove;
    batchDims[1].is = strideDim * n;
    batchDims[1].os = strideDim * n;

    fftw_plan plan;
    if (!inverse) {
      plan = fftw_plan_guru_split_dft(
        1, &transformDim, 2, batchDims,
        workRe.data(), workIm.data(), pOutRe, pOutIm, FFTW_ESTIMATE);
    } else {
      plan = fftw_plan_guru_split_dft(
        1, &transformDim, 2, batchDims,
        workIm.data(), workRe.data(), pOutIm, pOutRe, FFTW_ESTIMATE);
    }

    if (plan) {
      fftw_execute(plan);
      fftw_destroy_plan(plan);
    }
  }

  // ── Return result ────────────────────────────────────────────────────────

  auto result = Napi::Object::New(env);
  result.Set("re", outReArr);
  result.Set("im", outImArr);
  return result;
}
