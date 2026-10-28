/* JIT runtime helper: logical-mask indexing support.
 *
 * `jit_logical_mask_indices` scans `mask` column-major and fills
 * `out_indices` with the 0-based positions where the mask is truthy.
 * Each truthy position must be less than `axis_len`; otherwise this
 * raises the interpreter's logical-index error via
 * `jit_mask_oob`. Returns the truthy count, which is also the
 * number of entries written into `out_indices`. `out_indices` must
 * have room for at least `mask.numel()` longs.
 *
 * `axis` is the axis number for the diagnostic (0-based) when this
 * helper is called for a per-axis slot (`M(:, mask)` → axis = 1), or
 * `-1` for the linear single-slot form (`a(mask)`).
 *
 * Used by both reads (`IndexSlice` with a `LogicalMask` slot) and
 * linear-form writes (`IndexSliceStore` with a single `LogicalMask`
 * slot). The caller allocates the index buffer with `jit_alloc`,
 * passes it in, and frees it after the iteration that consumes it.
 */

static long jit_logical_mask_indices(
  jit_tensor_t mask, long axis_len, int axis, const char *loc,
  long *out_indices
) {
  long mask_n = 1;
  for (int d = 0; d < mask.ndim; d++) mask_n *= mask.dims[d];
  long count = 0;
  for (long i = 0; i < mask_n; i++) {
    if (mask.real[i] != 0.0) {
      if (i >= axis_len) {
        (void)loc;
        jit_mask_oob(axis);
      }
      out_indices[count++] = i;
    }
  }
  return count;
}
