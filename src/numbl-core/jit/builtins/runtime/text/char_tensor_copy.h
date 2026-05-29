/* mtoc2 runtime helper: deep-copy a char tensor into a fresh heap buffer.
 *
 * Returns an owned copy of `src`. Used for `c = d;` where `d` is a
 * char-tensor variable — the copy lets `d` remain usable while giving
 * the new binding its own buffer to free independently.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mtoc2_char_tensor_t mtoc2_char_tensor_copy(mtoc2_char_tensor_t src) {
  long n = src.rows * src.cols;
  mtoc2_char_tensor_t out;
  out.data = (char *)malloc((size_t)(n > 0 ? n : 1));
  if (!out.data) {
    fprintf(stderr, "mtoc2: out of memory in mtoc2_char_tensor_copy\n");
    abort();
  }
  out.rows = src.rows;
  out.cols = src.cols;
  out.owned = 1;
  if (n > 0 && src.data) {
    memcpy(out.data, src.data, (size_t)n);
  }
  return out;
}
