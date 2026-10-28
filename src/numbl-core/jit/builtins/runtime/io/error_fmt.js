// JS sibling of `error_fmt.h`: raise a MATLAB error through the raise
// channel (`raise.js`), which the host turns into numbl's RuntimeError
// with the same message and identifier. `jit_error_fmt` formats the
// message; `jit_error_text` takes it literally (the interpreter's rule for
// a lone `error(msg)`).

import { jit_sprintf_format } from "./format_engine.js";
import { jit_raise_error } from "../system/raise.js";

export function jit_error_fmt(identifier, fmt, ...args) {
  jit_raise_error(identifier, jit_sprintf_format(fmt, args));
}

export function jit_error_text(identifier, msg) {
  jit_raise_error(
    identifier,
    typeof msg === "object" && msg !== null ? msg.value : String(msg)
  );
}
