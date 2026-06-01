import { cumprodSign, defineCumulative } from "./_cumulative.js";

export const cumprod = defineCumulative({
  name: "cumprod",
  init: 1,
  step: (acc, x) => acc * x,
  stepComplex: (aRe, aIm, xRe, xIm) => [
    aRe * xRe - aIm * xIm,
    aRe * xIm + aIm * xRe,
  ],
  signRule: cumprodSign,
});
