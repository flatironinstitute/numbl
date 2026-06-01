import { defineReducer } from "./_shape.js";

export const any = defineReducer({
  name: "any",
  dimArgIndex: 1,
  // any of an empty fiber is false. Fold path short-circuits the
  // accumulator on the first nonzero element; this `step` is only
  // used as a fallback / sanity check.
  emptyValue: 0,
  step: (acc, x) => (x !== 0 ? 1 : acc),
  outputElem: "logical",
  // Result is always {0, 1} → nonneg.
  resultSign: () => "nonneg",
});
