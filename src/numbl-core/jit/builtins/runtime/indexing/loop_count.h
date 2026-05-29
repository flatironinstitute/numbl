/* mtoc2 runtime helper: compute the iteration count for a numbl-style
 * `start : step : end` range as a non-negative `long`.
 *
 * Mirrors numbl's `makeRangeTensor` formula
 * (`Math.max(0, Math.floor((end - start) / step + 1 + 1e-10))`) so
 * that float-imprecision ranges like `0:0.1:0.3` produce the same 4
 * elements both runners. The naive `floor((e-s)/step) + 1` would give
 * 3 because `(0.3 - 0) / 0.1` evaluates to 2.99999999..., one ulp
 * below 3. The `+ 1e-10` cushion is small enough that mathematically
 * non-integer quotients still floor down (the cushion only matters
 * within an ulp of an integer).
 *
 * `step == 0` short-circuits to an empty range — division-by-zero on
 * IEEE doubles would give ±Inf, then `floor(±Inf) + 1` is ±Inf, and
 * the `(long)` cast on Inf is undefined. The early return matches
 * numbl, which also treats step == 0 as empty.
 *
 * The same formula appears at every range-emitting codegen site
 * (single-slot slice reads, multi-slot slice setup, slice range writes,
 * MakeRange, and the For loop); routing through one helper centralizes
 * the cast and protects against undefined behavior when `floor(...)`
 * returns a value outside the `long` range.
 *
 * Returns `0` on:
 *   - step == 0
 *   - non-finite floor result (NaN — e.g. start/end NaN)
 *   - empty range (negative count)
 *   - count that doesn't fit in `long` (clamped rather than aborting:
 *     the user gets an empty range, which is the safest fallback
 *     under static translation — a 1e19-iteration loop is almost
 *     certainly a typo).
 */

#include <limits.h>
#include <math.h>

static long mtoc2_loop_count(double start, double end, double step) {
  if (step == 0.0) return 0;
  double n = floor((end - start) / step + 1.0 + 1e-10);
  if (!isfinite(n)) return 0;
  if (n <= 0.0) return 0;
  /* `(long)n` is undefined when n > LONG_MAX. Compare in double
   * space to keep the guard well-defined. */
  if (n > (double)LONG_MAX) return 0;
  return (long)n;
}
