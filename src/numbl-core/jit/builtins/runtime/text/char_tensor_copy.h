/* JIT runtime helper: deep-copy a char tensor into a fresh heap buffer.
 *
 * Returns an owned copy of `src`. Used for `c = d;` where `d` is a
 * char-tensor variable — the copy lets `d` remain usable while giving
 * the new binding its own buffer to free independently.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static jit_char_tensor_t jit_char_tensor_copy(jit_char_tensor_t src) {
  long n = src.rows * src.cols;
  jit_char_tensor_t out;
  out.data = (char *)malloc((size_t)(n > 0 ? n : 1));
  if (!out.data) {
    jit_raise_bailf("out of memory in jit_char_tensor_copy");
  }
  out.rows = src.rows;
  out.cols = src.cols;
  out.owned = 1;
  if (n > 0 && src.data) {
    memcpy(out.data, src.data, (size_t)n);
  }
  return out;
}
