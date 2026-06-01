// JS sibling of `warning_fmt.h`. Format the message and write
// `Warning: <msg>\n` via `$write`. Matches numbl
// (`runtime/specialBuiltins.ts:190` — `rt.output("Warning: " + msg + "\n")`).

import { mtoc2_sprintf_format } from "./format_engine.js";

export function mtoc2_warning_fmt(fmt, ...args) {
  $write("Warning: " + mtoc2_sprintf_format(fmt, args) + "\n");
}

export function mtoc2_warning_fmt_id(_id, fmt, ...args) {
  // The id is informational — numbl threads it to lastwarn state,
  // which mtoc2 doesn't expose. Drop it here so the visible output
  // matches the bare form.
  $write("Warning: " + mtoc2_sprintf_format(fmt, args) + "\n");
}
