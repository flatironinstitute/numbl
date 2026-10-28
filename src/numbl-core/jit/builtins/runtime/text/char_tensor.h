/* @snippet name=jit_char_tensor_t */
/* JIT char-tensor runtime: the C representation for every numbl char
 * array (single-quoted) value the JIT emits.
 *
 * numbl's `char` type is a 1×N row-vector of code units (so
 * `length('hi') == 2`). v1 only supports 1×N rows; multi-row char
 * matrices aren't constructable yet. Each value lives in this struct
 * with a byte buffer, a (rows, cols) shape, and an `owned` flag.
 *
 * The `owned` flag follows the same idiom as `jit_string_t`: 0 for
 * literal-pointing non-owning handles (from
 * `jit_char_tensor_from_literal`), 1 for heap-allocated buffers.
 * `jit_char_tensor_free` inspects this flag before calling `free`.
 */

typedef struct {
  char *data;   /* pointer to char bytes; NULL only in the empty state */
  long rows;
  long cols;
  int owned;    /* 1 iff data is heap-allocated and must be free()'d */
} jit_char_tensor_t;
