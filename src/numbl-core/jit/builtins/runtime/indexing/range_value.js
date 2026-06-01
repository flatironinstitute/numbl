// JS sibling of `range_value.h`. Snap-to-end for the last element
// of a `start:step:end` range so cross-runner test output matches
// numbl byte-for-byte.

export function mtoc2_range_value(start, step, end, count, i) {
  const v = start + step * i;
  // Snap the last element to exactly `end` — but only for multi-element
  // ranges, matching makeRangeTensor's `n > 1` guard. A single-element
  // range (count === 1) is just `start`; snapping it would wrongly pull
  // it to `end` (e.g. `0:1000:1e-9` is [0], not [1e-9]).
  if (
    count > 1 &&
    i === count - 1 &&
    Math.abs(v - end) < Math.abs(step) * 1e-10
  ) {
    return end;
  }
  return v;
}
