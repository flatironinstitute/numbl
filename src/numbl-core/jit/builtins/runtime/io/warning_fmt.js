// JS sibling of `warning_fmt.h`: hand a warning to the host (`$warning`),
// which checks the warning state for `identifier` ("" for none) and writes
// `Warning: <msg>\n`. `jit_warning_fmt` formats the message;
// `jit_warning_text` takes it literally (a lone `warning(msg)`).

import { jit_sprintf_format } from "./format_engine.js";

export function jit_warning_fmt(identifier, fmt, ...args) {
  $warning(identifier, jit_sprintf_format(fmt, args));
}

export function jit_warning_text(identifier, msg) {
  $warning(
    identifier,
    typeof msg === "object" && msg !== null ? msg.value : String(msg)
  );
}
