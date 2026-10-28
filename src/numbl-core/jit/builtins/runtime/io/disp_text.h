/* @snippet name=jit_disp_text deps=jit_text_view_t,jit_host_output */
/* JIT runtime helper: disp on a text view.
 *
 * Mirrors numbl's `displayValue` for string / char — the raw bytes
 * followed by a newline. Empty input prints just the newline. Bytes
 * are written verbatim through stdout so UTF-8 sequences pass through
 * cleanly. Used for both `disp("hi")` (string) and `disp('hi')` (char)
 * — the caller wraps either source in `jit_text_from_string` /
 * `jit_text_from_char_tensor`.
 */

#include <stdio.h>

static void jit_disp_text(jit_text_view_t t) {
  if (t.data && t.len > 0) {
    jit_stdout(t.data, (long)t.len);
  }
  jit_stdout("\n", 1);
}
