/* mtoc2 runtime helper: release a string handle's backing buffer if
 * owned, then reset it to the empty state.
 *
 * The owned flag distinguishes literal-pointing handles (no free) from
 * heap-allocated ones (must free). After this call the struct matches
 * `mtoc2_string_empty()`, so calling free a second time is a safe
 * no-op — useful for the scope-exit safety net at branch merges.
 */

#include <stdlib.h>

static void mtoc2_string_free(mtoc2_string_t *s) {
  if (s->owned && s->data) {
    free((void *)s->data);
  }
  s->data = (const char *)0;
  s->len = 0;
  s->owned = 0;
}
