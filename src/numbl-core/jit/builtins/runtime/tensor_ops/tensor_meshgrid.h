/* @snippet name=jit_meshgrid deps=jit_tensor_t,jit_tensor_alloc,jit_tensor_assign */
/* JIT runtime helper: MATLAB-style coordinate grid (`meshgrid`).
 *
 * Three entry points:
 *
 *   jit_meshgrid_x(x, y)
 *     `X = meshgrid(x, y)` — returns the row-replicated grid only.
 *     The single-output codegen path emits a call here; the 1-arg
 *     shorthand `X = meshgrid(x)` passes `x` twice (the tensor struct
 *     is consumed by value, so the duplicate is safe).
 *
 *   jit_meshgrid(x, y, &X, &Y)
 *     `[X, Y] = meshgrid(x, y)` — fills both outputs.
 *
 *   jit_meshgrid_1arg(x, &X, &Y)
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

static long jit_meshgrid_numel(jit_tensor_t t) {
  long n = 1;
  for (int k = 0; k < t.ndim; k++) n *= t.dims[k];
  return n;
}

static jit_tensor_t jit_meshgrid_x(jit_tensor_t x, jit_tensor_t y) {
  long M = jit_meshgrid_numel(x);
  long N = jit_meshgrid_numel(y);
  jit_tensor_t X = jit_tensor_alloc(N, M);
  for (long j = 0; j < M; j++) {
    double xj = x.real[j];
    for (long i = 0; i < N; i++) {
      X.real[i + j * N] = xj;
    }
  }
  return X;
}

static void jit_meshgrid(jit_tensor_t x, jit_tensor_t y,
                           jit_tensor_t *out_X, jit_tensor_t *out_Y) {
  long M = jit_meshgrid_numel(x);
  long N = jit_meshgrid_numel(y);
  jit_tensor_t X = jit_tensor_alloc(N, M);
  jit_tensor_t Y = jit_tensor_alloc(N, M);
  for (long j = 0; j < M; j++) {
    double xj = x.real[j];
    for (long i = 0; i < N; i++) {
      X.real[i + j * N] = xj;
      Y.real[i + j * N] = y.real[i];
    }
  }
  jit_tensor_assign(out_X, X);
  jit_tensor_assign(out_Y, Y);
}

static void jit_meshgrid_1arg(jit_tensor_t x, jit_tensor_t *out_X,
                                jit_tensor_t *out_Y) {
  jit_meshgrid(x, x, out_X, out_Y);
}
