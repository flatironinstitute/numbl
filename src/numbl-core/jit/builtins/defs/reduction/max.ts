import { defineReducer, minMaxSign } from "./_shape.js";
import { keepsLogical } from "../_shared.js";

const maxBase = defineReducer({
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

export const max = keepsLogical(maxBase, argTypes =>
  argTypes.length === 2 ? argTypes : argTypes.slice(0, 1)
);
