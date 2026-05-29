import { cumsumSign, defineCumulative } from "./_cumulative.js";

export const cumsum = defineCumulative({
  name: "cumsum",
  init: 0,
  step: (acc, x) => acc + x,
  stepComplex: (aRe, aIm, xRe, xIm) => [aRe + xRe, aIm + xIm],
  signRule: cumsumSign,
});
