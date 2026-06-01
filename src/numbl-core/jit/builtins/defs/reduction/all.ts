import { defineReducer } from "./_shape.js";

export const all = defineReducer({
  name: "all",
  dimArgIndex: 1,
  emptyValue: 1,
  step: (acc, x) => (x === 0 ? 0 : acc),
  outputElem: "logical",
  resultSign: () => "nonneg",
});
