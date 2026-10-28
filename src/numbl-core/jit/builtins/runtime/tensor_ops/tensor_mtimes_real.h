/* @snippet name=jit_tensor_mtimes_real deps=jit_tensor_t,jit_alloc,jit_tensor_alloc */
/* JIT runtime helper: real 2-D matrix multiplication `A * B`.
 *
 * A is m×k, B is k×n; result is m×n, freshly owned. Column-major in,
 * column-major out — A's element at (i, p) lives at `a.real[i + p*m]`;
 * B's element at (p, j) lives at `b.real[p + j*k]`; result at (i, j)
 * lives at `r.real[i + j*m]`.
 *
 * The translate-time check guarantees both inputs are 2-D and
 * `a.dims[1] == b.dims[0]` when both shapes are statically known.
 * When one or both shapes are partly unknown, the check fires at
 * runtime; mismatch raises with a clear message (the JIT doesn't have
 * an exception-throwing channel).
 *
 * The inner-product accumulator runs over the shared dimension k. The
 * loop order (i outermost, then j, with p innermost) traverses A
 * column-by-column on the inside and writes to r column-by-column on
 * the outside, mirroring numbl's `mtimesCore`.
 *
 * The vector-special-cases (column×row outer product, row×column
 * inner product, row×matrix, matrix×column) all reduce to the same
 * triple-loop — no separate fast path.
 */

#include <stdio.h>
#include <stdlib.h>

/* Scalar-return variant for the 1×k * k×1 inner-product case. The
 * tensor-return helper above always allocates m*n doubles even when
 * m=n=1; this path skips the allocation and just returns the dot
 * product. The translate-time type system classifies a 1×1 result
 * as a scalar (`isMultiElement` returns false), and `jit_disp_double`
 * / scalar consumers need a `double`-typed expression. */
static double jit_tensor_mtimes_real_scalar(jit_tensor_t a,
                                               jit_tensor_t b) {
  if (a.ndim != 2 || b.ndim != 2) {
    jit_raise_bailf("jit_tensor_mtimes_real_scalar: inputs must be 2-D (got %d-D, %d-D)",
            a.ndim, b.ndim);
  }
  if (a.dims[0] != 1 || b.dims[1] != 1) {
    jit_raise_bailf("jit_tensor_mtimes_real_scalar: scalar-result form requires 1×k * k×1 (got %ld×%ld * %ld×%ld)",
            a.dims[0], a.dims[1], b.dims[0], b.dims[1]);
  }
  if (a.dims[1] != b.dims[0]) {
    jit_raise_errorf("", "Inner matrix dimensions must agree: %ld vs %ld",
            a.dims[1], b.dims[0]);
  }
  long k = a.dims[1];
  double acc = 0.0;
  for (long p = 0; p < k; p++) {
    acc += a.real[p] * b.real[p];
  }
  return acc;
}

static jit_tensor_t jit_tensor_mtimes_real(jit_tensor_t a,
                                                jit_tensor_t b) {
  if (a.ndim != 2 || b.ndim != 2) {
    jit_raise_bailf("jit_tensor_mtimes_real: inputs must be 2-D (got %d-D, %d-D)",
            a.ndim, b.ndim);
  }
  long m = a.dims[0];
  long k = a.dims[1];
  long k2 = b.dims[0];
  long n = b.dims[1];
  if (k != k2) {
    jit_raise_errorf("", "Inner matrix dimensions must agree: %ld vs %ld",
            k, k2);
  }
  jit_tensor_t r;
  r.real = jit_alloc((size_t)m * (size_t)n * sizeof(double));
  r.imag = NULL;
  r.ndim = 2;
  r.dims[0] = m;
  r.dims[1] = n;
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i++) {
      double acc = 0.0;
      for (long p = 0; p < k; p++) {
        acc += a.real[i + p * m] * b.real[p + j * k];
      }
      r.real[i + j * m] = acc;
    }
  }
  return r;
}
