// JS sibling of `range_value.h`. Snap-to-end for the last element
// of a `start:step:end` range so cross-runner test output matches
// numbl byte-for-byte.

export function mtoc2_range_value(start, step, end, count, i) {
  const v = start + step * i;
  if (i === count - 1 && Math.abs(v - end) < Math.abs(step) * 1e-10) {
    return end;
  }
  return v;
}
