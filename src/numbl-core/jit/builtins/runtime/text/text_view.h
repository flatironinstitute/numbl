/* JIT text view: a non-owning (data, len) pair used as the common
 * argument type for runtime helpers that read text byte-for-byte
 * (`disp`, future `error`/`strcmp`/`fprintf`).
 *
 * numbl distinguishes `char` (1×N row-vector of bytes, single-quoted)
 * from `string` (scalar handle, double-quoted), but the helpers above
 * only need to walk the bytes. `jit_text_view_t` is what we pass to
 * them; `jit_text_from_string` / `jit_text_from_char_tensor` are
 * zero-copy adapters the call site uses to bridge either source struct
 * into the view.
 *
 * The view is *non-owning*: the underlying storage stays with the
 * caller (literal in `.rodata`, owned `jit_string_t`, owned
 * `jit_char_tensor_t`).
 */

typedef struct {
  const char *data;
  long len;
} jit_text_view_t;

static jit_text_view_t jit_text_from_string(jit_string_t s) {
  jit_text_view_t v;
  v.data = s.data;
  v.len = s.len > 0 ? s.len : 0;
  return v;
}

static jit_text_view_t jit_text_from_char_tensor(jit_char_tensor_t c) {
  jit_text_view_t v;
  v.data = c.data;
  v.len = c.rows * c.cols;
  return v;
}
