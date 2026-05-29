/* mtoc2 runtime helper: i-th element of a numbl-style `start:step:end`
 * range, with snap-to-end on the last element.
 *
 * Mirrors numbl's `makeRangeTensor` step-of-i computation:
 *   - element i is `start + step * i`
 *   - the LAST element (i == count - 1) snaps to exactly `end` when
 *     the computed value is within `|step| * 1e-10` of `end`. Without
 *     the snap, ranges like `0.1:0.1:0.3` produce a last element of
 *     `0.3000000000000000444...`, so `v(end) == 0.3` evaluates to
 *     false even though the user wrote `:0.3`.
 *
 * Used by `mtoc2_tensor_make_range`, slice-read range generation, and
 * the For-loop emitter so every range-yielding site agrees on the
 * snapped values numbl produces.
 */

#include <math.h>

static double mtoc2_range_value(
  double start, double step, double end, long count, long i
) {
  double v = start + step * (double)i;
  if (i == count - 1 && fabs(v - end) < fabs(step) * 1e-10) return end;
  return v;
}
