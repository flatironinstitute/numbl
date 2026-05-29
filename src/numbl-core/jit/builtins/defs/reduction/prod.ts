import { defineReducer, prodSign } from "./_shape.js";

export const prod = defineReducer({
  name: "prod",
  dimArgIndex: 1,
  emptyValue: 1,
  step: (acc, x) => acc * x,
  outputElem: "double",
  resultSign: (t, nonEmpty) => prodSign(t, nonEmpty),
});
