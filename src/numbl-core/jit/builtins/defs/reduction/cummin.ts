import { cumsumSign, defineCumulative } from "./_cumulative.js";

// Prefix minimum. Uses `Math.min` (NaN-propagating, matching numbl's
// cumOp) seeded with +Infinity so the first element passes through.
// Real-only (see cummax.ts).
export const cummin = defineCumulative({
  name: "cummin",
  init: Infinity,
  step: (acc, x) => Math.min(acc, x),
  signRule: cumsumSign,
  supportsComplex: false,
});
