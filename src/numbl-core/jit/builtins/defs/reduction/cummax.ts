import { cumsumSign, defineCumulative } from "./_cumulative.js";

// Prefix maximum. Uses `Math.max` (NaN-propagating, matching numbl's
// cumOp) seeded with -Infinity so the first element passes through.
// cummax preserves the input's sign class (running max of like-signed
// values stays in that class), so cumsumSign (pass-through) applies.
// Real-only: numbl's complex cummax is a component-wise quirk.
export const cummax = defineCumulative({
  name: "cummax",
  init: -Infinity,
  step: (acc, x) => Math.max(acc, x),
  signRule: cumsumSign,
  supportsComplex: false,
});
