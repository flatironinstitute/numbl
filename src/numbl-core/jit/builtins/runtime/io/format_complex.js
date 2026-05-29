// JS sibling of `format_complex.h`. Format a scalar complex `{re, im}`
// the way numbl's `formatComplex` does. Each component is rendered
// through `mtoc2_format_double` so the scalar formatting stays in
// lockstep with numbl's `formatNumber`.

import { mtoc2_format_double } from "./format_double.js";

export function mtoc2_format_complex(z) {
  const re = z.re;
  const im = z.im;
  if (im === 0) return mtoc2_format_double(re);
  if (re === 0) return mtoc2_format_double(im) + "i";
  if (im < 0) {
    return mtoc2_format_double(re) + " - " + mtoc2_format_double(-im) + "i";
  }
  return mtoc2_format_double(re) + " + " + mtoc2_format_double(im) + "i";
}
