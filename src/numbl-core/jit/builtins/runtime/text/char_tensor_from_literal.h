/* JIT runtime helper: build a non-owning char tensor from a C string
 * literal in .rodata.
 *
 * Returns a 1×n handle with owned=0. `n` is the byte count of the
 * string body, excluding the trailing NUL that C string literals
 * carry. Codegen always passes a literal whose storage duration is
 * static, so the non-owning handle is safe for the program's lifetime.
 */

static jit_char_tensor_t jit_char_tensor_from_literal(const char *src,
                                                            long n) {
  jit_char_tensor_t t;
  t.data = (char *)src; /* discards const; we never write through it */
  t.rows = 1;
  t.cols = n;
  t.owned = 0;
  return t;
}
