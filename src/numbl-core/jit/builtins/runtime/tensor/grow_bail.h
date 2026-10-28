/* JIT runtime: grow-aware indexed-store bounds checks.
 *
 * MATLAB indexed STORES grow the array when the index exceeds the
 * current bounds (`v(k) = x` with k > numel, `M(i, j) = x` with
 * i > rows), but the JIT fixes each tensor's shape at compile time and
 * can't model the grown carrier. A store past the end therefore raises a
 * bail (`jit_raise_bail`, see `raise.h`): the host re-runs the unit on the
 * interpreter, which has full MATLAB grow semantics.
 *
 * Index < 1 is NOT a grow (`v(0) = x` is a genuine error in MATLAB), so
 * it raises the same error as a read's OOB (`jit_oob_abort`).
 *
 * Statically-provable grows (`v(end+1) = x` on a fixed-shape array)
 * are declined at compile time (see `lowering/lowerIndexStore.ts`)
 * and never reach here; this is the runtime safety net for grows that
 * are only detectable at runtime (a dynamic index exceeding the
 * runtime extent).
 */

/* Linear (1-arg) store bounds check. Returns the 0-based offset for an
 * in-bounds index; bails on a grow (index past numel); raises on a
 * genuine sub-1 index. */
static long jit_idx_lin_grow(
  const jit_tensor_t *t, long got1, const char *loc
) {
  long total = 1;
  for (int i = 0; i < t->ndim; i++) total *= t->dims[i];
  if (got1 < 1) jit_oob_abort(loc, -1, got1, 1, total);
  if (got1 > total) jit_raise_bail("indexed-store array growth");
  return got1 - 1;
}

/* Per-axis (N-arg) store bounds check. Same contract as
 * `jit_idx_lin_grow` but against a single axis's dim. */
static long jit_idx_axis_grow(
  const jit_tensor_t *t, int axis, long got1, const char *loc
) {
  long dim = t->dims[axis];
  if (got1 < 1) jit_oob_abort(loc, axis, got1, 1, dim);
  if (got1 > dim) jit_raise_bail("indexed-store array growth");
  return got1 - 1;
}
