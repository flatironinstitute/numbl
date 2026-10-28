// JS sibling of `disp_complex.h` — disp(z) for a scalar complex
// `{re, im}`. Mirrors numbl's `formatComplex` via
// `jit_format_complex` and adds the trailing newline.

import { jit_format_complex } from "./format_complex.js";

export function jit_disp_complex(z) {
  $write(jit_format_complex(z) + "\n");
}
