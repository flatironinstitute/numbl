/* mtoc2 runtime helper: logical-mask indexing support.
 *
 * `mtoc2_logical_mask_indices` scans `mask` column-major and fills
 * `out_indices` with the 0-based positions where the mask is truthy.
 * Each truthy position must be less than `axis_len`; otherwise this
 * aborts with a numbl-style "Index exceeds array bounds" message via
 * `mtoc2_oob_abort`. Returns the truthy count, which is also the
 * number of entries written into `out_indices`. `out_indices` must
 * have room for at least `mask.numel()` longs.
 *
 * `axis` is the axis number for the diagnostic (0-based) when this
 * helper is called for a per-axis slot (`M(:, mask)` → axis = 1), or
 * `-1` for the linear single-slot form (`a(mask)`).
 *
 * Used by both reads (`IndexSlice` with a `LogicalMask` slot) and
 * linear-form writes (`IndexSliceStore` with a single `LogicalMask`
 * slot). The caller allocates the index buffer with `mtoc2_alloc`,
 * passes it in, and frees it after the iteration that consumes it.
 */

static long mtoc2_logical_mask_indices(
  mtoc2_tensor_t mask, long axis_len, int axis, const char *loc,
  long *out_indices
) {
  long mask_n = 1;
  for (int d = 0; d < mask.ndim; d++) mask_n *= mask.dims[d];
  long count = 0;
  for (long i = 0; i < mask_n; i++) {
    if (mask.real[i] != 0.0) {
      if (i >= axis_len) {
        mtoc2_oob_abort(loc, axis, i + 1, 1, axis_len);
      }
      out_indices[count++] = i;
    }
  }
  return count;
}
