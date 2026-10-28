/* JIT runtime helpers: index bounds checks for tensor access.
 *
 * Numbl raises a `RuntimeError("Index exceeds array bounds")` on any
 * out-of-range scalar or slice access; the JIT must match — without these
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
 * success and aborts via `jit_oob_abort` on failure. The `loc`
 * string is "<file>:<offset>" from the index's source span — formatted
 * by the emitter so the user sees the violating access site.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Convert a double-valued index expression to the long the bounds
 * checkers below expect. Mirrors numbl's `Math.round(idx)` convention
 * (see runtime/runtimeIndexing.ts) so `a(2.5)` reads element 3, not 2.
 * A plain `(long)` cast would truncate and silently diverge from
 * numbl on every non-integer index. */
static long jit_to_idx(double v) {
  return lround(v);
}

static void jit_oob_abort(
  const char *loc, int axis, long got, long lo, long hi
) {
  if (axis < 0) {
    fprintf(stderr,
      "%s: Index exceeds array bounds (got %ld, valid %ld..%ld)\n",
      loc, got, lo, hi);
  } else {
    fprintf(stderr,
      "%s: Index in position %d exceeds array bounds (got %ld, valid %ld..%ld)\n",
      loc, axis + 1, got, lo, hi);
  }
  /* exit(1) rather than abort(): a clean non-zero exit status
   * rather than SIGABRT. */
  exit(1);
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
  if (first1 < 1 || first1 > dim) jit_oob_abort(loc, axis, first1, 1, dim);
  if (last1 < 1 || last1 > dim) jit_oob_abort(loc, axis, last1, 1, dim);
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
  if (first1 < 1 || first1 > total) jit_oob_abort(loc, -1, first1, 1, total);
  if (last1 < 1 || last1 > total) jit_oob_abort(loc, -1, last1, 1, total);
}
