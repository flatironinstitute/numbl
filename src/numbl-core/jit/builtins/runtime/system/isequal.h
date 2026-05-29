/* mtoc2 runtime helpers for `isequal` over real numeric values.
 *
 * Complex tensors, char / string, struct and cell comparison are
 * handled by the interpreter's `call` hook; the AOT `emitC` rejects
 * those argument combinations with `UnsupportedConstruct`. These two
 * helpers cover the dominant cases: tensor-vs-tensor and
 * scalar-vs-tensor over real data.
 *
 * Equality follows numbl's `valuesEqualSimple` / `tensorsEqual`:
 * same rank, same per-axis extent, element-for-element `==`
 * (so `NaN != NaN`, matching MATLAB / numbl).
 */

/* Two real tensors: equal iff identical shape and data. */
static double mtoc2_isequal_tt(mtoc2_tensor_t a, mtoc2_tensor_t b) {
  if (a.ndim != b.ndim) return 0.0;
  long n = 1;
  for (int i = 0; i < a.ndim; i++) {
    if (a.dims[i] != b.dims[i]) return 0.0;
    n *= a.dims[i];
  }
  for (long i = 0; i < n; i++) {
    if (a.real[i] != b.real[i]) return 0.0;
  }
  return 1.0;
}

/* Real scalar vs real tensor: equal iff the tensor holds exactly one
 * element equal to `s` (numbl treats a scalar as a 1×1 tensor). */
static double mtoc2_isequal_st(double s, mtoc2_tensor_t t) {
  long n = 1;
  for (int i = 0; i < t.ndim; i++) n *= t.dims[i];
  if (n != 1) return 0.0;
  return t.real[0] == s ? 1.0 : 0.0;
}
