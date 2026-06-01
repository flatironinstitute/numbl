/* mtoc runtime helper: malloc wrapper that aborts on allocation
 * failure with a clear diagnostic. Used for every tensor backing
 * buffer (mtoc allocates tensor storage uniformly on the heap so the
 * codegen path is exercised by every test, not just large ones).
 *
 * `n_bytes` is the size in bytes; the call site computes
 * `numel * sizeof(double)`. Returns a non-NULL pointer or aborts.
 */

#include <stdio.h>
#include <stdlib.h>

static double *mtoc2_alloc(size_t n_bytes) {
  /* `malloc(0)` is implementation-defined: glibc returns a non-NULL
   * sentinel, but some C libraries return NULL — which would trip
   * the OOM abort below on a perfectly valid empty-tensor request.
   * Clamp to a 1-byte allocation so the contract "return non-NULL or
   * abort on OOM" holds uniformly. The caller never reads from a
   * zero-element buffer, so the wasted byte is irrelevant. */
  double *p = (double *)malloc(n_bytes == 0 ? 1 : n_bytes);
  if (!p) {
    fprintf(stderr, "mtoc2: out of memory (mtoc2_alloc requested %zu bytes)\n",
            n_bytes);
    abort();
  }
  return p;
}
