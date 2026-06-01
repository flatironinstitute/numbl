/* mtoc2 text view: a non-owning (data, len) pair used as the common
 * argument type for runtime helpers that read text byte-for-byte
 * (`disp`, future `error`/`strcmp`/`fprintf`).
 *
 * numbl distinguishes `char` (1×N row-vector of bytes, single-quoted)
 * from `string` (scalar handle, double-quoted), but the helpers above
 * only need to walk the bytes. `mtoc2_text_view_t` is what we pass to
 * them; `mtoc2_text_from_string` / `mtoc2_text_from_char_tensor` are
 * zero-copy adapters the call site uses to bridge either source struct
 * into the view.
 *
 * The view is *non-owning*: the underlying storage stays with the
 * caller (literal in `.rodata`, owned `mtoc2_string_t`, owned
 * `mtoc2_char_tensor_t`).
 */

typedef struct {
  const char *data;
  long len;
} mtoc2_text_view_t;

static mtoc2_text_view_t mtoc2_text_from_string(mtoc2_string_t s) {
  mtoc2_text_view_t v;
  v.data = s.data;
  v.len = s.len > 0 ? s.len : 0;
  return v;
}

static mtoc2_text_view_t mtoc2_text_from_char_tensor(mtoc2_char_tensor_t c) {
  mtoc2_text_view_t v;
  v.data = c.data;
  v.len = c.rows * c.cols;
  return v;
}
