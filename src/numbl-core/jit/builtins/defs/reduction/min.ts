import { defineReducer, minMaxSign } from "./_shape.js";

export const min = defineReducer({
  name: "min",
  // min(A, [], dim) — the [] placeholder sits in slot 2, so the
  // dim arg lives at slot 3.
  dimArgIndex: 2,
  // Seeded with NaN at runtime; this `emptyValue` is unused by the
  // fold path (min/max take a dedicated NaN-aware branch).
  emptyValue: NaN,
  step: (acc, x) => {
    if (Number.isNaN(x)) return acc;
    if (Number.isNaN(acc) || x < acc) return x;
    return acc;
  },
  outputElem: "double",
  resultSign: minMaxSign,
});
