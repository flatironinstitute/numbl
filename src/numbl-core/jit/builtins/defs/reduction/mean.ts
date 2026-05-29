import { defineReducer, meanSign } from "./_shape.js";

export const mean = defineReducer({
  name: "mean",
  dimArgIndex: 1,
  emptyValue: 0,
  step: (acc, x) => acc + x,
  finalize: (acc, n) => acc / n,
  outputElem: "double",
  resultSign: (t, nonEmpty) => meanSign(t, nonEmpty),
});
