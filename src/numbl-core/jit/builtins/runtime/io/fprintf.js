// JS sibling of `fprintf.h`. Routes formatted output through
// `mtoc2_sprintf_format` then writes via `$write` (the JS analog of
// C's stdout). Numbl-compatible spec set & semantics.

import { mtoc2_sprintf_format } from "./format_engine.js";

export function mtoc2_fprintf(fmt, ...args) {
  $write(mtoc2_sprintf_format(fmt, args));
}
