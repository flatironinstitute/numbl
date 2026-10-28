// JS sibling of `assert_fmt.h`. Format-string variant of `assert`.
// Numbl-compatible: format `msg` with `args`, throw on falsy `cond`.

import { jit_sprintf_format } from "./format_engine.js";

export function jit_assert_scalar_fmt(cond, fmt, ...args) {
  if (cond) return;
  throw new Error(jit_sprintf_format(fmt, args));
}
