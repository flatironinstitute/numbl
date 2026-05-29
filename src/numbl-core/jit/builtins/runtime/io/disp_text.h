/* mtoc2 runtime helper: disp on a text view.
 *
 * Mirrors numbl's `displayValue` for string / char — the raw bytes
 * followed by a newline. Empty input prints just the newline. Bytes
 * are written verbatim through stdout so UTF-8 sequences pass through
 * cleanly. Used for both `disp("hi")` (string) and `disp('hi')` (char)
 * — the caller wraps either source in `mtoc2_text_from_string` /
 * `mtoc2_text_from_char_tensor`.
 */

#include <stdio.h>

static void mtoc2_disp_text(mtoc2_text_view_t t) {
  if (t.data && t.len > 0) {
    fwrite(t.data, 1, (size_t)t.len, stdout);
  }
  putchar('\n');
}
