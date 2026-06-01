import { defineReducer, sumSign } from "./_shape.js";

export const sum = defineReducer({
  name: "sum",
  dimArgIndex: 1,
  emptyValue: 0,
  step: (acc, x) => acc + x,
  outputElem: "double",
  resultSign: (t, nonEmpty) => sumSign(t, nonEmpty),
});
