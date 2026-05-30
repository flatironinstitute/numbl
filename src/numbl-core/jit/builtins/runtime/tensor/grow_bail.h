/* mtoc2 runtime: grow-aware indexed-store bounds checks + JIT bail.
 *
 * MATLAB indexed STORES grow the array when the index exceeds the
 * current bounds (`v(k) = x` with k > numel, `M(i, j) = x` with
 * i > rows) — but the JIT fixes each tensor's shape at compile time
 * and can't model the grown carrier. Rather than abort (as a READ's
 * OOB does) or silently corrupt the buffer, a store past the end
 * signals a *bail*: it sets a flag and `longjmp`s back to the
 * `setjmp` guard the emitter places at the top of the host-entry
 * function, which then returns normally. The host inspects the flag
 * after the call (`mtoc2_grow_bail_check`) and re-runs the whole
 * scope on the interpreter, which has full MATLAB grow semantics.
 *
 * Index < 1 is NOT a grow — `v(0) = x` is a genuine error in MATLAB —
 * so it still aborts via `mtoc2_oob_abort`, matching a read's OOB.
 *
 * Statically-provable grows (`v(end+1) = x` on a fixed-shape array)
 * are declined at compile time (see `lowering/lowerIndexStore.ts`)
 * and never reach here; this is the runtime safety net for grows that
 * are only detectable at runtime (a dynamic index exceeding the
 * runtime extent).
 *
 * The `jmp_buf` / flag are file-scope singletons: the JIT runs one
 * spec call at a time, and only the host-entry function (storage
 * class `""`) arms a `setjmp` — inner specialized callees let their
 * grow-`longjmp` propagate up to it (their owned locals leak on the
 * bail path, which is acceptable for this rare fallback). The
 * `_reset` / `_check` entry points are exported (non-static) so the
 * koffi host can clear the flag before a call and read it after.
 */

#include <setjmp.h>

static jmp_buf mtoc2_grow_bail_buf;
static int mtoc2_grow_bailed = 0;

void mtoc2_grow_bail_reset(void) { mtoc2_grow_bailed = 0; }
int mtoc2_grow_bail_check(void) { return mtoc2_grow_bailed; }

/* Linear (1-arg) store bounds check. Returns the 0-based offset for an
 * in-bounds index; bails on a grow (index past numel); aborts on a
 * genuine sub-1 index. */
static long mtoc2_idx_lin_grow(
  const mtoc2_tensor_t *t, long got1, const char *loc
) {
  long total = 1;
  for (int i = 0; i < t->ndim; i++) total *= t->dims[i];
  if (got1 < 1) mtoc2_oob_abort(loc, -1, got1, 1, total);
  if (got1 > total) {
    mtoc2_grow_bailed = 1;
    longjmp(mtoc2_grow_bail_buf, 1);
  }
  return got1 - 1;
}

/* Per-axis (N-arg) store bounds check. Same contract as
 * `mtoc2_idx_lin_grow` but against a single axis's dim. */
static long mtoc2_idx_axis_grow(
  const mtoc2_tensor_t *t, int axis, long got1, const char *loc
) {
  long dim = t->dims[axis];
  if (got1 < 1) mtoc2_oob_abort(loc, axis, got1, 1, dim);
  if (got1 > dim) {
    mtoc2_grow_bailed = 1;
    longjmp(mtoc2_grow_bail_buf, 1);
  }
  return got1 - 1;
}
