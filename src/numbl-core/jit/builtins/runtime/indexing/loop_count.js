// JS sibling of `loop_count.h`. Mirrors numbl's `makeRangeTensor`
// count formula: `floor((end - start) / step + 1 + 1e-10)`, clamped
// to non-negative.

export function mtoc2_loop_count(start, end, step) {
  if (step === 0) return 0;
  const n = Math.floor((end - start) / step + 1 + 1e-10);
  if (!Number.isFinite(n)) return 0;
  if (n <= 0) return 0;
  return n;
}
