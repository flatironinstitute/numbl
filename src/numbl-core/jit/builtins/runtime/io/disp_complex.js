// JS sibling of `disp_complex.h` — disp(z) for a scalar complex
// `{re, im}`. Mirrors numbl's `formatComplex` via
// `mtoc2_format_complex` and adds the trailing newline.

import { mtoc2_format_complex } from "./format_complex.js";

export function mtoc2_disp_complex(z) {
  $write(mtoc2_format_complex(z) + "\n");
}
