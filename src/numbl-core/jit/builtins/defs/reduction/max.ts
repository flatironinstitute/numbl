import { defineReducer, minMaxSign } from "./_shape.js";

export const max = defineReducer({
  name: "max",
  dimArgIndex: 2,
  emptyValue: NaN,
  step: (acc, x) => {
    if (Number.isNaN(x)) return acc;
    if (Number.isNaN(acc) || x > acc) return x;
    return acc;
  },
  outputElem: "double",
  resultSign: minMaxSign,
});
