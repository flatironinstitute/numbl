#include <fftw3.h>

#include <stddef.h>

int numbl_fft1d_f64(
  const double *re,
  const double *im,
  int n,
  int inverse,
  double *outRe,
  double *outIm
) {
  fftw_complex *input = NULL;
  fftw_complex *output = NULL;
  fftw_plan plan = NULL;
  int direction;
  int i;

  if (re == NULL || outRe == NULL || outIm == NULL || n <= 0) {
    return -1;
  }

  input = (fftw_complex *)fftw_malloc((size_t)n * sizeof(fftw_complex));
  output = (fftw_complex *)fftw_malloc((size_t)n * sizeof(fftw_complex));
  if (input == NULL || output == NULL) {
    fftw_free(input);
    fftw_free(output);
    return -2;
  }

  for (i = 0; i < n; ++i) {
    input[i][0] = re[i];
    input[i][1] = im == NULL ? 0.0 : im[i];
  }

  direction = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
  plan = fftw_plan_dft_1d(n, input, output, direction, FFTW_ESTIMATE);
  if (plan == NULL) {
    fftw_free(input);
    fftw_free(output);
    return -3;
  }

  fftw_execute(plan);

  for (i = 0; i < n; ++i) {
    outRe[i] = output[i][0];
    outIm[i] = output[i][1];
  }

  fftw_destroy_plan(plan);
  fftw_free(input);
  fftw_free(output);
  return 0;
}
