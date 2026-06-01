// JS sibling of `sprintf.h`. Two output variants:
//   - `mtoc2_sprintf_str(fmt, args)` returns the formatted text as a
//     plain JS string (the equivalent of numbl's `String` kind).
//   - `mtoc2_sprintf_char(fmt, args)` returns it as a `mtoc2_char_tensor_t`
//     placeholder — we keep the JS shape as a `{mtoc2Tag: "char", value}`
//     object so disp/length etc. dispatch correctly.

import { mtoc2_sprintf_format } from "./format_engine.js";

export function mtoc2_sprintf_str(fmt, ...args) {
  return mtoc2_sprintf_format(fmt, args);
}

export function mtoc2_sprintf_char(fmt, ...args) {
  return { mtoc2Tag: "char", value: mtoc2_sprintf_format(fmt, args) };
}
