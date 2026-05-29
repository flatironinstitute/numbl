/**
 * Compile-time range-element-count formula. Single source of truth for
 * the `n = floor((end - start) / step + 1 + 1e-10)` formula used in
 * three places: `lowerRangeAsValue` (range-as-value `1:n`), the
 * `exactRangeCount` helper in `lowerIndexSlice.ts`, and the C-side
 * `mtoc2_loop_count` runtime helper. All three must agree byte-for-
 * byte so the statically-known shape, the runtime-allocated buffer
 * length, and the for-loop iteration count match.
 *
 * The `+ 1e-10` cushion absorbs the IEEE-ulp underflow on ranges like
 * `0:0.1:0.3` (where `(0.3-0)/0.1` evaluates to 2.99999... rather
 * than 3.0) without affecting genuinely-non-integer quotients.
 */

/** Compute the count of elements in a range `start : step : end`,
 *  given that every endpoint is a finite real number and `step` is
 *  non-zero. Returns 0 for empty ranges (e.g. `5:1:0`). The caller
 *  is responsible for checking finiteness and `step !== 0` before
 *  calling. */
export function rangeCountFromExactEnds(
  start: number,
  step: number,
  end: number
): number {
  const raw = Math.floor((end - start) / step + 1 + 1e-10);
  return raw > 0 ? raw : 0;
}
