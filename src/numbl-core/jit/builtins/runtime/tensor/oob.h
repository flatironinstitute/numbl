/* @snippet name=jit_oob_abort deps=jit_raise,jit_tensor_t */
/* JIT runtime helpers: index bounds checks for tensor access.
 *
 * numbl raises an indexing RuntimeError (runtime/indexPlan.ts) on any
 * invalid or out-of-range scalar or slice access; the JIT must match — without these
 * checks the emitter would compile to a raw `base.real[off]` access
 * that silently reads/writes past the buffer.
 *
 * Three helpers cover the access shapes:
 *   - `jit_idx_axis`: per-axis check used by 2-arg or N-arg scalar
 *     indexing (`M(i,j)`, `T(i,j,k)`). Catches `M(4, 1)` on a 2x3
 *     matrix where the linear-total check would erroneously pass.
 *   - `jit_idx_lin`: linear-index check used by 1-arg indexing
 *     (`v(i)`). Validates against numel(t).
 *   - `jit_check_axis_range`: closed-interval check for slice
 *     setup. The slice helpers compute `first`/`last` once at setup
 *     time and call this before iterating.
 *
 * Each helper returns the 0-based offset (or 0-based index) on
 * success and raises via `jit_oob_abort` on failure.
 */

#include <math.h>

/* Convert a double-valued index expression to the long the bounds
 * checkers below expect. A non-integer (or NaN) index is invalid in numbl
 * (runtime/indexPlan.ts), so it maps to 0, which the checkers reject with
 * the interpreter's invalid-index error. */
static long jit_to_idx(double v) {
  return v == floor(v) && fabs(v) < 9.0e18 ? (long)v : 0;
}

/* Raise the interpreter's error (runtime/indexPlan.ts) for index `got`
 * outside `lo..hi`: `axis` is the 0-based subscript position, or -1 for
 * linear indexing. The error's line comes from the statement location
 * (`raise.h`); `loc` names the index expression for diagnostics. */
static void jit_oob_abort(
  const char *loc, int axis, long got, long lo, long hi
) {
  (void)loc;
  (void)lo;
  if (got < 1) {
    if (axis < 0)
      jit_raise_error("", "Array indices must be positive integers or logical values.");
    jit_raise_errorf("", "Index in position %d is invalid. Array indices must "
                     "be positive integers or logical values.", axis + 1);
  }
  if (axis < 0)
    jit_raise_errorf("", "Index exceeds the number of array elements. Index "
                     "must not exceed %ld.", hi);
  jit_raise_errorf("", "Index in position %d exceeds array bounds. Index must "
                   "not exceed %ld.", axis + 1, hi);
}

/* A logical index with a true value past the extent. */
static void jit_mask_oob(int axis) {
  if (axis < 0)
    jit_raise_error("", "The logical indices contain a true value outside of "
                    "the array bounds.");
  jit_raise_errorf("", "The logical indices in position %d contain a true "
                   "value outside of the array bounds.", axis + 1);
}

static long jit_idx_axis(
  const jit_tensor_t *t, int axis, long got1, const char *loc
) {
  long dim = t->dims[axis];
  if (got1 < 1 || got1 > dim) jit_oob_abort(loc, axis, got1, 1, dim);
  return got1 - 1;
}

static long jit_idx_lin(
  const jit_tensor_t *t, long got1, const char *loc
) {
  long total = 1;
  for (int i = 0; i < t->ndim; i++) total *= t->dims[i];
  if (got1 < 1 || got1 > total) jit_oob_abort(loc, -1, got1, 1, total);
  return got1 - 1;
}

static void jit_check_axis_range(
  const jit_tensor_t *t, int axis, long first1, long last1, const char *loc
) {
  long dim = t->dims[axis];
  /* An invalid index is reported before an out-of-range one. */
  if (first1 < 1) jit_oob_abort(loc, axis, first1, 1, dim);
  if (last1 < 1) jit_oob_abort(loc, axis, last1, 1, dim);
  if (first1 > dim) jit_oob_abort(loc, axis, first1, 1, dim);
  if (last1 > dim) jit_oob_abort(loc, axis, last1, 1, dim);
}

/* Linear (single-slot) range bounds check: validates against
 * `numel(t)` rather than a per-axis dim. Used by `v(a:b)` /
 * `M(a:b)` style single-slot range slices, where MATLAB semantics
 * are linear-flatten regardless of base shape. */
static void jit_check_linear_range(
  const jit_tensor_t *t, long first1, long last1, const char *loc
) {
  long total = 1;
  for (int i = 0; i < t->ndim; i++) total *= t->dims[i];
  if (first1 < 1) jit_oob_abort(loc, -1, first1, 1, total);
  if (last1 < 1) jit_oob_abort(loc, -1, last1, 1, total);
  if (first1 > total) jit_oob_abort(loc, -1, first1, 1, total);
  if (last1 > total) jit_oob_abort(loc, -1, last1, 1, total);
}

/* Whether a tensor RHS fits a subscripted store selecting `counts[0..k)`
 * elements per subscript: the non-singleton dimensions must agree (the
 * interpreter's rule for `A(i, j) = B`). */
static int jit_store_shape_ok(int k, const long *counts,
                              const jit_tensor_t *rhs) {
  int i = 0, j = 0;
  for (;;) {
    while (i < k && counts[i] == 1) i++;
    while (j < rhs->ndim && rhs->dims[j] == 1) j++;
    if (i == k || j == rhs->ndim) return i == k && j == rhs->ndim;
    if (counts[i] != rhs->dims[j]) return 0;
    i++;
    j++;
  }
}
