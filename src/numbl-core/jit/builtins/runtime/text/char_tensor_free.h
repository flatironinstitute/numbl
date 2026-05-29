/* mtoc2 runtime helper: release a char tensor's backing buffer and reset
 * the struct to the empty state.
 *
 * The owned flag distinguishes literal-pointing handles (no free) from
 * heap-allocated ones (must free). After this call the struct matches
 * `mtoc2_char_tensor_empty()`, so calling free a second time is a safe
 * no-op — useful for the scope-exit safety net at branch merges.
 */

#include <stdlib.h>

static void mtoc2_char_tensor_free(mtoc2_char_tensor_t *t) {
  if (t->owned && t->data) {
    free(t->data);
  }
  t->data = (char *)0;
  t->rows = 0;
  t->cols = 0;
  t->owned = 0;
}
