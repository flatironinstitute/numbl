// JS sibling of `assert_fmt.h`: raise a MATLAB error when `cond` is not
// truthy. `jit_assert_text` takes the message literally; the `_fmt` form
// formats it. `identifier` is "" for none.

import { jit_error_fmt, jit_error_text } from "./error_fmt.js";

export function jit_assert_text(cond, identifier, msg) {
  if (cond) return;
  jit_error_text(identifier, msg);
}

export function jit_assert_scalar_fmt(cond, identifier, fmt, ...args) {
  if (cond) return;
  jit_error_fmt(identifier, fmt, ...args);
}
