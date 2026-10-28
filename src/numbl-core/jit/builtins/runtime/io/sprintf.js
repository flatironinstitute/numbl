// JS sibling of `sprintf.h`. Two output variants:
//   - `jit_sprintf_str(fmt, args)` returns the formatted text as a
//     plain JS string (the equivalent of numbl's `String` kind).
//   - `jit_sprintf_char(fmt, args)` returns it as a `jit_char_tensor_t`
//     placeholder — we keep the JS shape as a `{jitTag: "char", value}`
//     object so disp/length etc. dispatch correctly.

import { jit_sprintf_format } from "./format_engine.js";

export function jit_sprintf_str(fmt, ...args) {
  return jit_sprintf_format(fmt, args);
}

export function jit_sprintf_char(fmt, ...args) {
  return { jitTag: "char", value: jit_sprintf_format(fmt, args) };
}
