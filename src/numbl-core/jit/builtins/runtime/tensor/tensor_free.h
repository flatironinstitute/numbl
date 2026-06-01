/* mtoc runtime helper: release a tensor's backing buffers and reset
 * the struct to the empty/zero state.
 *
 * Both `free`s are unconditional. `free(NULL)` is well-defined, so
 * the imag-side free is a no-op for real tensors (where `imag` is
 * always NULL) and a real release for complex tensors. There is no
 * runtime branch on `imag != NULL`. Resetting the struct lets a
 * freed local be safely re-used by `mtoc2_tensor_assign` without
 * leaking. Called at every scope-exit site (end of `main`, end of a
 * function body, every `IRStmt.ReturnFromFunction`).
 */

#include <stdlib.h>

static void mtoc2_tensor_free(mtoc2_tensor_t *t) {
  free(t->real);
  free(t->imag);
  t->real = NULL;
  t->imag = NULL;
  t->ndim = 0;
}
