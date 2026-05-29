/* mtoc2 runtime helper: MATLAB-style coordinate grid (`meshgrid`).
 *
 * Three entry points:
 *
 *   mtoc2_meshgrid_x(x, y)
 *     `X = meshgrid(x, y)` — returns the row-replicated grid only.
 *     The single-output codegen path emits a call here; the 1-arg
 *     shorthand `X = meshgrid(x)` passes `x` twice (the tensor struct
 *     is consumed by value, so the duplicate is safe).
 *
 *   mtoc2_meshgrid(x, y, &X, &Y)
 *     `[X, Y] = meshgrid(x, y)` — fills both outputs.
 *
 *   mtoc2_meshgrid_1arg(x, &X, &Y)
 *     `[X, Y] = meshgrid(x)` — shorthand for `meshgrid(x, x)`.
 *
 * Input args are real-double 1-D vectors (1×N row or N×1 column); the
 * type system enforces that. Each helper walks the column-major flat
 * `.real` buffer of length `numel(x)` / `numel(y)`, so the row/column
 * orientation of the input is irrelevant.
 *
 * Output shape is `[N, M]` with `N = numel(y)`, `M = numel(x)`:
 *   X.real[i + j*N] = x[j]   (each row is x)
 *   Y.real[i + j*N] = y[i]   (each column is y)
 */

static long mtoc2_meshgrid_numel(mtoc2_tensor_t t) {
  long n = 1;
  for (int k = 0; k < t.ndim; k++) n *= t.dims[k];
  return n;
}

static mtoc2_tensor_t mtoc2_meshgrid_x(mtoc2_tensor_t x, mtoc2_tensor_t y) {
  long M = mtoc2_meshgrid_numel(x);
  long N = mtoc2_meshgrid_numel(y);
  mtoc2_tensor_t X = mtoc2_tensor_alloc(N, M);
  for (long j = 0; j < M; j++) {
    double xj = x.real[j];
    for (long i = 0; i < N; i++) {
      X.real[i + j * N] = xj;
    }
  }
  return X;
}

static void mtoc2_meshgrid(mtoc2_tensor_t x, mtoc2_tensor_t y,
                           mtoc2_tensor_t *out_X, mtoc2_tensor_t *out_Y) {
  long M = mtoc2_meshgrid_numel(x);
  long N = mtoc2_meshgrid_numel(y);
  mtoc2_tensor_t X = mtoc2_tensor_alloc(N, M);
  mtoc2_tensor_t Y = mtoc2_tensor_alloc(N, M);
  for (long j = 0; j < M; j++) {
    double xj = x.real[j];
    for (long i = 0; i < N; i++) {
      X.real[i + j * N] = xj;
      Y.real[i + j * N] = y.real[i];
    }
  }
  mtoc2_tensor_assign(out_X, X);
  mtoc2_tensor_assign(out_Y, Y);
}

static void mtoc2_meshgrid_1arg(mtoc2_tensor_t x, mtoc2_tensor_t *out_X,
                                mtoc2_tensor_t *out_Y) {
  mtoc2_meshgrid(x, x, out_X, out_Y);
}
